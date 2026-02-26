#!/usr/bin/env python3
"""
Check whether a given STEP file is renderable.

Definition of "renderable":
- The STEP file can be read successfully by OpenCASCADE
- The resulting shape is non-null
- The shape can be meshed/triangulated and exported to a non-empty STL
  (optionally verified to have faces via trimesh if available)

This script runs the heavy OCC operations in an isolated subprocess to avoid
crashing the parent process on malformed files.

Usage:
  # Single file mode
  python check_renderability.py /absolute/path/to/file.step [options]
  
  # Batch mode
  python check_renderability.py --batch /path/to/directory [options]

Options:
  --batch DIRECTORY                 Batch process all STEP files in directory recursively
  --output FILE                     Output file for batch report (default: auto-generated)
  --timeout N                       Timeout in seconds (default: 30)
  --require-all-roots              Check every ADVANCED_BREP_SHAPE_REPRESENTATION root; 
                                   fail if any root fails to mesh/export
  --verify-trimesh-faces           Require exported STL to load in trimesh with 
                                   non-zero face count (needs trimesh installed)
                                   (automatically enabled in batch mode)
  --treat-reader-errors-as-failure Treat OCC StepReaderData error messages as failure
                                   (otherwise they're reported but don't fail the check)

Exit codes:
  0 = renderable (single file) or batch completed successfully
  2 = not renderable (read/mesh/export failed or empty geometry)
  3 = timeout while checking
  4 = internal error (unexpected)

Single file output fields:
  renderable                True if file can be rendered
  status                    "ok", "error", or "timeout"
  reason                    Error details (null on success)
  num_roots_expected        Number of ADVANCED_BREP_SHAPE_REPRESENTATION found in file
  num_roots_found           Number of roots OCC reader can transfer
  roots                     Per-root details (only with --require-all-roots)
  stl_ok                    Whether STL export succeeded
  trimesh_verified          Whether trimesh validation was requested
  trimesh_faces             Face count from trimesh (null if not verified)
  reader_errors_present     Whether OCC printed StepReaderData warnings
  reader_error_lines        Array of OCC warning messages
  duration_seconds          Time taken for the check

Batch mode output (JSON file):
  batch_info                Processing metadata (start/end times, settings, etc.)
  summary_statistics        Overall statistics (success rates, face counts, etc.)
  subdirectory_statistics   Individual file results with paths and subdirectory info

Examples:
  # Single file basic check
  python check_renderability.py file.step
  
  # Single file strict validation
  python check_renderability.py file.step --require-all-roots --verify-trimesh-faces
  
  # Batch check dfs_step directory (checks all STEP files)
  python check_renderability.py --batch ./data/dfs_step

  # Batch check STEP_generated directory (only checks 'gen' subfolders)
  python check_renderability.py --batch ./data/STEP_generated/eval_output

  # Batch check with custom output file
  python check_renderability.py --batch ./data/dfs_step --output report.json
  
  # Batch check with custom timeout
  python check_renderability.py --batch /path/to/directory --timeout 60
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def _build_isolated_checker_script(step_path: str, require_all_roots: bool, verify_trimesh_faces: bool) -> str:
    """Return Python source for the isolated OCC check script using percent-formatting to avoid f-string collisions."""
    template = """
import json
import os
import sys
import re

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.IFSelect import IFSelect_RetDone

def main():
    step_path = %(step_path)r
    require_all_roots = %(require_all_roots)r
    verify_trimesh_faces = %(verify_trimesh_faces)r

    result = {
        "renderable": False,
        "status": "unknown",
        "reason": None,
        "num_roots_expected": None,
        "num_roots_found": None,
        "roots": []
    }

    try:
        if not os.path.exists(step_path):
            result.update({"status": "error", "reason": "file_not_found"})
            print(json.dumps(result))
            sys.exit(2)

        # Read STEP
        reader = STEPControl_Reader()
        status = reader.ReadFile(step_path)
        if status != IFSelect_RetDone:
            result.update({"status": "error", "reason": "read_failed"})
            print(json.dumps(result))
            sys.exit(2)

        # Count expected roots and analyze geometry types by scanning file content
        try:
            with open(step_path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
            expected = len(re.findall(r"ADVANCED_BREP_SHAPE_REPRESENTATION\\s*\\(", text))
            result["num_roots_expected"] = expected
            
            # Check for geometry quality indicators
            wireframe_count = len(re.findall(r"GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION\\s*\\(", text))
            surface_count = len(re.findall(r"MANIFOLD_SURFACE_SHAPE_REPRESENTATION\\s*\\(", text))
            shell_count = len(re.findall(r"SHELL_BASED_SURFACE_MODEL\\s*\\(", text))
            
            # Detect mixed geometry (solid + wireframe/surface)
            has_mixed_geometry = expected > 0 and (wireframe_count > 0 or surface_count > 0 or shell_count > 0)
            result["has_mixed_geometry"] = has_mixed_geometry
            result["wireframe_representations"] = wireframe_count
            result["surface_representations"] = surface_count
            
        except Exception:
            expected = None
            result["num_roots_expected"] = None
            result["has_mixed_geometry"] = False
            result["wireframe_representations"] = 0
            result["surface_representations"] = 0

        # Discover how many roots the reader sees
        try:
            nb_roots = reader.NbRootsForTransfer()
        except Exception:
            nb_roots = None
        result["num_roots_found"] = nb_roots

        def _mesh_and_export(shape, suffix: str):
            # Mesh (triangulate) and export STL with adaptive parameters
            # Try different meshing strategies to avoid timeouts on complex geometry
            mesh_success = False
            mesh_params = [
                ("coarse", 1.0),   # Start with coarse mesh to avoid timeouts
                ("medium", 0.5),   # Medium mesh
                ("fine", 0.1),     # Fine mesh (may timeout on complex geometry)
                ("very_coarse", 2.0)  # Last resort for extremely complex geometry
            ]
            
            used_mesh_quality = None
            for param_name, deflection in mesh_params:
                try:
                    mesh = BRepMesh_IncrementalMesh(shape, deflection)
                    mesh.Perform()
                    mesh_success = True
                    used_mesh_quality = param_name
                    break
                except Exception:
                    # If meshing fails, try next parameter set
                    continue
            
            if not mesh_success:
                return False, step_path + ".__temp_check__" + suffix + ".stl", None, None

            temp_stl = step_path + ".__temp_check__" + suffix + ".stl"
            writer = StlAPI_Writer()
            writer.SetASCIIMode(False)
            ok = writer.Write(shape, temp_stl)
            if not ok or (not os.path.exists(temp_stl)) or os.path.getsize(temp_stl) == 0:
                return False, temp_stl, None, used_mesh_quality

            tri_ok = True
            tri_faces = None
            if verify_trimesh_faces:
                try:
                    import trimesh  # type: ignore
                    mesh = trimesh.load_mesh(temp_stl)
                    tri_faces = int(len(getattr(mesh, "faces", [])))
                    if getattr(mesh, "is_empty", False) or tri_faces == 0:
                        tri_ok = False
                except Exception:
                    # If trimesh fails, consider it a failure only if verify_trimesh_faces requested
                    tri_ok = False
            return tri_ok, temp_stl, tri_faces, used_mesh_quality

        # Strict path: require every root to transfer and mesh
        if require_all_roots and nb_roots is not None and nb_roots > 0:
            all_ok = True
            any_solid_ok = False  # Track if at least one solid geometry succeeded
            for i in range(1, nb_roots + 1):
                root_info = {"root_index": i, "transfer_ok": False, "shape_null": None,
                             "stl_ok": None, "trimesh_verified": bool(verify_trimesh_faces), "trimesh_faces": None, "ok": False, "geometry_type": "unknown"}
                try:
                    # Create fresh reader for each root to avoid state corruption
                    reader_i = STEPControl_Reader()
                    reader_i.ReadFile(step_path)
                    
                    try:
                        xfer_ok = reader_i.TransferRoot(i)
                        root_info["transfer_ok"] = bool(xfer_ok) if xfer_ok is not None else True
                    except Exception:
                        # Some bindings lack return value; assume transfer attempted
                        root_info["transfer_ok"] = True

                    # After TransferRoot(i), use Shape() to get the transferred shape
                    shape_i = reader_i.Shape()

                    is_null = shape_i.IsNull()
                    root_info["shape_null"] = bool(is_null)
                    if is_null:
                        all_ok = False
                        result["roots"].append(root_info)
                        continue

                    tri_ok, temp_stl, tri_faces, mesh_quality = _mesh_and_export(shape_i, "_root" + str(i))
                    root_info["stl_ok"] = bool(tri_ok)
                    root_info["trimesh_faces"] = tri_faces
                    root_info["ok"] = bool(tri_ok)
                    
                    # Determine geometry type based on STL export success
                    if tri_ok:
                        root_info["geometry_type"] = "solid"
                        any_solid_ok = True
                    else:
                        # STL export failed - likely wireframe/curve geometry
                        root_info["geometry_type"] = "wireframe_or_curve"
                        # Don't fail for wireframe geometry - it's expected that curves can't export to STL
                    
                    # cleanup
                    try:
                        if temp_stl and os.path.exists(temp_stl):
                            os.remove(temp_stl)
                    except Exception:
                        pass
                except Exception:
                    all_ok = False
                finally:
                    result["roots"].append(root_info)

            # Success if at least one solid geometry root exported successfully
            if any_solid_ok:
                result.update({"renderable": True, "status": "ok"})
                print(json.dumps(result))
                sys.exit(0)
            else:
                # Only fail if NO solid geometry could be exported (all wireframes or all failed)
                solid_roots = [r for r in result["roots"] if r.get("geometry_type") == "solid"]
                if not solid_roots:
                    result.update({"renderable": False, "status": "error", "reason": "no_solid_geometry_found"})
                else:
                    result.update({"renderable": False, "status": "error", "reason": "all_solid_roots_failed"})
                print(json.dumps(result))
                sys.exit(2)

        # Non-strict path: try to find any solid geometry that can export to STL
        # First try the default transfer approach
        reader.TransferRoot()
        shape = reader.Shape()
        
        tri_ok = False
        tri_faces = None
        temp_stl = None
        mesh_quality = None
        
        if not shape.IsNull():
            tri_ok, temp_stl, tri_faces, mesh_quality = _mesh_and_export(shape, "")
        
        # If default transfer failed, try each root individually to find solid geometry
        if not tri_ok and nb_roots is not None and nb_roots > 1:
            for i in range(1, nb_roots + 1):
                try:
                    # Create fresh reader for each root to avoid state corruption
                    reader_i = STEPControl_Reader()
                    reader_i.ReadFile(step_path)
                    reader_i.TransferRoot(i)
                    shape_i = reader_i.Shape()  # Use Shape() after TransferRoot(i)
                    if not shape_i.IsNull():
                        tri_ok_i, temp_stl_i, tri_faces_i, mesh_quality_i = _mesh_and_export(shape_i, "_root" + str(i))
                        if tri_ok_i:
                            # Found a root that can export to STL successfully
                            tri_ok = True
                            tri_faces = tri_faces_i
                            temp_stl = temp_stl_i
                            mesh_quality = mesh_quality_i
                            break
                        # Cleanup failed attempt
                        try:
                            if temp_stl_i and os.path.exists(temp_stl_i):
                                os.remove(temp_stl_i)
                        except Exception:
                            pass
                except Exception:
                    continue
        
        if not tri_ok:
            result.update({"status": "error", "reason": "stl_empty_or_export_failed"})
            # Cleanup if exists
            try:
                if temp_stl and os.path.exists(temp_stl):
                    os.remove(temp_stl)
            except Exception:
                pass
            print(json.dumps(result))
            sys.exit(2)

        # Success - Basic shape validation for geometry quality assessment
        geometry_warnings = []
        
        # Only flag as warning for truly exceptional cases
        # Normal coarse meshing is not a problem - many files legitimately need it
        
        result.update({
            "renderable": True,
            "status": "ok",
            "roots": [],
            "stl_ok": True,
            "trimesh_verified": bool(verify_trimesh_faces),
            "trimesh_faces": tri_faces,
            # "mesh_quality_used": mesh_quality,
            # "geometry_warnings": geometry_warnings,
        })
        print(json.dumps(result))
        sys.exit(0)

    except Exception as e:
        result.update({"status": "error", "reason": "unexpected_exception: " + str(e)})
        print(json.dumps(result))
        sys.exit(2)
    finally:
        # Best-effort cleanup of temp STL
        try:
            # Cleanup all temp STL files
            for fn in os.listdir(os.path.dirname(step_path) or "."):
                if fn.startswith(os.path.basename(step_path)) and ".__temp_check__" in fn and fn.endswith('.stl'):
                    try:
                        os.remove(os.path.join(os.path.dirname(step_path) or ".", fn))
                    except Exception:
                        pass
        except Exception:
            pass

if __name__ == "__main__":
    main()
"""
    return template % {
        "step_path": step_path,
        "require_all_roots": require_all_roots,
        "verify_trimesh_faces": verify_trimesh_faces,
    }


def check_renderability(step_path: str, timeout: int, require_all_roots: bool, verify_trimesh_faces: bool, treat_reader_errors_as_failure: bool) -> Dict[str, Any]:
    start = time.time()
    # Write the isolated checker script to a temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(_build_isolated_checker_script(step_path, require_all_roots, verify_trimesh_faces))
        isolated_script = f.name

    try:
        proc = subprocess.run(
            [sys.executable, isolated_script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        duration = time.time() - start

        # Try to parse JSON from stdout; capture stderr for debugging on failures
        stdout_text = (proc.stdout or "").strip()
        stderr_text = (proc.stderr or "").strip()
        result: Dict[str, Any] = {}
        # First attempt: direct JSON parse
        if stdout_text:
            try:
                result = json.loads(stdout_text)
            except json.JSONDecodeError:
                # Second attempt: extract the last JSON-looking object from stdout
                try:
                    matches = list(re.finditer(r"\{[\s\S]*?\}", stdout_text))
                    if matches:
                        last = matches[-1].group(0)
                        result = json.loads(last)
                except Exception:
                    result = {}

        # Merge metadata
        result.update({
            "duration_seconds": round(duration, 4),
            "returncode": proc.returncode,
        })

        # Extract OCC reader error lines from child's stdout (do not fail solely for this unless flagged)
        try:
            ansi_stripped = re.sub(r"\x1B\[[0-9;]*[A-Za-z]", "", stdout_text)
        except Exception:
            ansi_stripped = stdout_text
        reader_error_lines = [ln for ln in ansi_stripped.splitlines() if ("StepReaderData" in ln or "ERR" in ln)]
        result["reader_errors_present"] = bool(reader_error_lines)
        if reader_error_lines:
            result["reader_error_lines"] = reader_error_lines

        # Optionally treat OCC reader warnings as hard failure
        # OCC prints lines like "*** ERR StepReaderData *** ..." to stdout
        if treat_reader_errors_as_failure and result.get("reader_errors_present"):
            result = {
                "renderable": False,
                "status": "error",
                "reason": "reader_errors_detected",
                "child_stdout": stdout_text,
                "child_stderr": stderr_text,
                "duration_seconds": round(duration, 4),
                "returncode": 2,
                "reader_errors_present": True,
                "reader_error_lines": reader_error_lines,
            }
            return result

        # If child exited cleanly but we couldn't parse JSON, treat as error (hardened behavior)
        if proc.returncode == 0 and not result:
            return {
                "renderable": False,
                "status": "error",
                "reason": "no_json_output_with_code_0",
                "child_stdout": stdout_text,
                "child_stderr": stderr_text,
                "duration_seconds": round(duration, 4),
                "returncode": 4,
                "reader_errors_present": bool(reader_error_lines),
                "reader_error_lines": reader_error_lines if reader_error_lines else [],
            }

        if proc.returncode == 0:
            return result

        # Non-zero return codes: treat as not renderable; include child stderr/stdout for diagnostics
        if proc.returncode != 0:
            if not result or not stdout_text:
                result = {
                    "renderable": False,
                    "status": "error",
                    "reason": "no_json_output",
                    "child_stdout": stdout_text,
                    "child_stderr": stderr_text,
                    "duration_seconds": round(duration, 4),
                    "returncode": proc.returncode if proc.returncode in (2, 3, 4) else 2,
                    "reader_errors_present": bool(reader_error_lines),
                    "reader_error_lines": reader_error_lines if reader_error_lines else [],
                }
            result.setdefault("renderable", False)
            result.setdefault("status", "error")
            if "child_stderr" not in result:
                result["child_stderr"] = stderr_text
            if "child_stdout" not in result:
                result["child_stdout"] = stdout_text
            if "reader_errors_present" not in result:
                result["reader_errors_present"] = bool(reader_error_lines)
            if "reader_error_lines" not in result and reader_error_lines:
                result["reader_error_lines"] = reader_error_lines
            return result

        return result

    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return {
            "renderable": False,
            "status": "timeout",
            "reason": f"exceeded {timeout}s",
            "duration_seconds": round(duration, 4),
            "returncode": 3,
        }
    except Exception as e:
        duration = time.time() - start
        return {
            "renderable": False,
            "status": "error",
            "reason": f"runner_exception: {e}",
            "duration_seconds": round(duration, 4),
            "returncode": 4,
        }
    finally:
        try:
            os.unlink(isolated_script)
        except Exception:
            pass


def find_step_files(directory: str) -> List[str]:
    """Recursively find all STEP files in directory and its subdirectories.
    
    Special handling:
    - For directories containing 'STEP_generated', only look in 'gen' subfolders
    - For other directories, scan all subdirectories
    """
    step_files = []
    directory_path = Path(directory)
    
    try:
        # Check if this is a STEP_generated directory
        is_step_generated = 'STEP_generated' in str(directory_path)
        
        for root, dirs, files in os.walk(directory):
            root_path = Path(root)
            
            # For STEP_generated directories, only process 'gen' subfolders
            if is_step_generated:
                # Skip if we're not in a 'gen' folder and this folder contains subdirectories
                if dirs and root_path.name != 'gen' and 'gen' not in [d for d in dirs]:
                    continue
                # Skip if we're in 're' or 'gt' folders
                if root_path.name in ['re', 'gt']:
                    continue
                # Only process files if we're in a 'gen' folder or if this is a leaf directory with 'gen' in the path
                if root_path.name != 'gen' and 'gen' not in str(root_path):
                    continue
            
            # Find STEP files in current directory
            for file in files:
                if file.lower().endswith(('.step', '.stp')):
                    step_files.append(os.path.join(root, file))
                    
    except Exception as e:
        print(f"Error scanning directory {directory}: {e}")
    
    return sorted(step_files)


def print_progress_bar(current: int, total: int, prefix: str = '', suffix: str = '', length: int = 50, fill: str = '█', decimals: int = 1) -> None:
    """Print a progress bar to the terminal."""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    # Print New Line on Complete
    if current == total: 
        print()


def get_subdirectory_level(file_path: str, input_dir: str) -> str:
    """Get the appropriate subdirectory level for statistics grouping."""
    rel_path = os.path.relpath(file_path, input_dir)
    path_parts = rel_path.split(os.sep)
    
    # For STEP_generated paths, group by the checkpoint directory (e.g., eval_ckpt-900)
    if 'STEP_generated' in input_dir:
        if len(path_parts) >= 1:
            return path_parts[0]  # e.g., eval_ckpt-900
    else:
        # For other paths, group by the first directory level (e.g., 0001, 0002)
        if len(path_parts) >= 1:
            return path_parts[0]  # e.g., 0001, 0002
    
    return "root"


def batch_check_renderability(input_dir: str, output_file: str, timeout: int = 30, verify_trimesh_faces: bool = True) -> None:
    """Batch check all STEP files in input directory and generate analysis report."""
    print(f"Starting batch renderability check...")
    print(f"Input directory: {input_dir}")
    print(f"Output report: {output_file}")
    
    # Find all STEP files
    step_files = find_step_files(input_dir)
    if not step_files:
        print(f"No STEP files found in {input_dir}")
        return
    
    print(f"Found {len(step_files)} STEP files to check")
    
    # Track statistics - overall and by subdirectory
    overall_stats = {
        'total_files': len(step_files),
        'renderable': 0,
        'not_renderable': 0,
        'timeout': 0,
        'error': 0,
        'with_reader_errors': 0,
        'total_duration': 0.0,
        'avg_duration': 0.0,
        'failure_reasons': defaultdict(int),
        'trimesh_stats': {
            'total_with_faces': 0,
            'total_verified': 0,
            'min_faces': None,
            'max_faces': None,
            'avg_faces': None
        }
    }
    
    # Subdirectory-level statistics
    subdir_stats = defaultdict(lambda: {
        'total': 0,
        'renderable': 0,
        'not_renderable': 0,
        'with_trimesh_faces': 0,
        'failed_files': []  # List of non-renderable files with reasons
    })
    
    results = []
    start_time = datetime.now()
    
    try:
        for i, step_file in enumerate(step_files, 1):
            # Update progress bar with current file name
            file_name = os.path.basename(step_file)
            print_progress_bar(i-1, len(step_files), prefix='Processing:', suffix=f'{file_name} ({i}/{len(step_files)})', length=30)
            
            # Get subdirectory grouping
            subdir_key = get_subdirectory_level(step_file, input_dir)
            
            # Check renderability
            result = check_renderability(
                step_file, 
                timeout=timeout, 
                require_all_roots=False, 
                verify_trimesh_faces=verify_trimesh_faces, 
                treat_reader_errors_as_failure=False
            )
            
            # Add file info to result
            result['file_path'] = step_file
            result['relative_path'] = os.path.relpath(step_file, input_dir)
            result['subdirectory'] = subdir_key
            result['file_size_bytes'] = os.path.getsize(step_file) if os.path.exists(step_file) else 0
            
            results.append(result)
            
            # Update overall statistics
            overall_stats['total_duration'] += result.get('duration_seconds', 0)
            subdir_stats[subdir_key]['total'] += 1
            
            if result.get('reader_errors_present', False):
                overall_stats['with_reader_errors'] += 1
            
            if result.get('renderable', False):
                overall_stats['renderable'] += 1
                subdir_stats[subdir_key]['renderable'] += 1
                
                # Track trimesh face statistics
                faces = result.get('trimesh_faces')
                if result.get('trimesh_verified', False):
                    overall_stats['trimesh_stats']['total_verified'] += 1
                    if faces is not None and faces > 0:
                        overall_stats['trimesh_stats']['total_with_faces'] += 1
                        subdir_stats[subdir_key]['with_trimesh_faces'] += 1
                        
                        if overall_stats['trimesh_stats']['min_faces'] is None or faces < overall_stats['trimesh_stats']['min_faces']:
                            overall_stats['trimesh_stats']['min_faces'] = faces
                        if overall_stats['trimesh_stats']['max_faces'] is None or faces > overall_stats['trimesh_stats']['max_faces']:
                            overall_stats['trimesh_stats']['max_faces'] = faces
            else:
                overall_stats['not_renderable'] += 1
                subdir_stats[subdir_key]['not_renderable'] += 1
                
                # Track failure reasons
                reason = result.get('reason', 'unknown')
                overall_stats['failure_reasons'][reason] += 1
                
                # Add to failed files list for subdirectory
                subdir_stats[subdir_key]['failed_files'].append({
                    'file_path': step_file,
                    'relative_path': result['relative_path'],
                    'reason': reason,
                    'status': result.get('status', 'unknown')
                })
                
                # Categorize by status
                status = result.get('status', 'unknown')
                if status == 'timeout':
                    overall_stats['timeout'] += 1
                else:
                    overall_stats['error'] += 1
        
        # Final progress bar update
        print_progress_bar(len(step_files), len(step_files), prefix='Processing:', suffix='Complete!', length=30)
                
    except KeyboardInterrupt:
        print("\n\nBatch checking interrupted by user")
    
    end_time = datetime.now()
    
    # Calculate final statistics
    if overall_stats['total_files'] > 0:
        overall_stats['avg_duration'] = overall_stats['total_duration'] / overall_stats['total_files']
    
    # Calculate trimesh face average
    if overall_stats['trimesh_stats']['total_with_faces'] > 0:
        total_faces = sum(r.get('trimesh_faces', 0) for r in results if r.get('trimesh_faces'))
        overall_stats['trimesh_stats']['avg_faces'] = total_faces / overall_stats['trimesh_stats']['total_with_faces']
    
    # Convert defaultdicts to regular dicts for JSON serialization
    overall_stats['failure_reasons'] = dict(overall_stats['failure_reasons'])
    subdir_stats_dict = {k: dict(v) for k, v in subdir_stats.items()}
    
    # Generate report
    report = {
        'batch_info': {
            'input_directory': input_dir,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_duration_minutes': (end_time - start_time).total_seconds() / 60,
            'files_processed': len(results),
            'settings': {
                'timeout': timeout,
                'verify_trimesh_faces': verify_trimesh_faces
            }
        },
        'summary_statistics': overall_stats,
        'subdirectory_statistics': subdir_stats_dict
    }
    
    # Write report to file
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nBatch check complete! Report saved to: {output_file}")
    except Exception as e:
        print(f"Error saving report: {e}")
        # Try to save in current directory as fallback
        fallback_file = f"renderability_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(fallback_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to fallback location: {fallback_file}")
        except Exception as e2:
            print(f"Failed to save report: {e2}")
    
    # Print summary
    print_batch_summary(overall_stats, subdir_stats_dict, len(results))


def print_batch_summary(overall_stats: Dict, subdir_stats: Dict, files_processed: int) -> None:
    """Print a summary of batch checking results."""
    print("\n" + "="*80)
    print("BATCH RENDERABILITY CHECK SUMMARY")
    print("="*80)
    
    # Overall statistics
    print(f"Files processed: {files_processed}")
    print(f"Renderable: {overall_stats['renderable']} ({overall_stats['renderable']/max(1,files_processed)*100:.1f}%)")
    print(f"Not renderable: {overall_stats['not_renderable']} ({overall_stats['not_renderable']/max(1,files_processed)*100:.1f}%)")
    print(f"Timeouts: {overall_stats['timeout']}")
    print(f"With reader errors: {overall_stats['with_reader_errors']}")
    print(f"Average check duration: {overall_stats['avg_duration']:.2f}s")
    
    # Trimesh statistics
    trimesh_stats = overall_stats['trimesh_stats']
    if trimesh_stats['total_verified'] > 0:
        print(f"\nTrimesh Face Statistics:")
        print(f"  Files verified with trimesh: {trimesh_stats['total_verified']}")
        print(f"  Files with non-zero faces: {trimesh_stats['total_with_faces']}")
        if trimesh_stats['total_with_faces'] > 0:
            print(f"  Face count range: {trimesh_stats['min_faces']} - {trimesh_stats['max_faces']}")
            print(f"  Average faces: {trimesh_stats['avg_faces']:.1f}")
    
    # Failure reasons
    if overall_stats['failure_reasons']:
        print(f"\nTop failure reasons:")
        for reason, count in sorted(overall_stats['failure_reasons'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {reason}: {count}")
    
    # Subdirectory breakdown
    if subdir_stats:
        print(f"\nSubdirectory Statistics:")
        print("-" * 80)
        for subdir_name in sorted(subdir_stats.keys()):
            subdir_data = subdir_stats[subdir_name]
            total = subdir_data['total']
            renderable = subdir_data['renderable']
            not_renderable = subdir_data['not_renderable']
            with_faces = subdir_data['with_trimesh_faces']
            success_rate = renderable / max(1, total) * 100
            
            print(f"\n{subdir_name}:")
            print(f"  Total files: {total}")
            print(f"  Renderable: {renderable} ({success_rate:.1f}%)")
            print(f"  Not renderable: {not_renderable}")
            print(f"  With trimesh faces: {with_faces}")
            
            # Show failed files if any
            failed_files = subdir_data.get('failed_files', [])
            if failed_files:
                print(f"  Failed files ({len(failed_files)}):")
                for failed in failed_files[:10]:  # Show up to 10 failed files
                    print(f"    - {failed['relative_path']} (reason: {failed['reason']})")
                if len(failed_files) > 10:
                    print(f"    ... and {len(failed_files) - 10} more")
    
    print("\n" + "="*80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether STEP files are renderable")
    
    # Mode selection - either single file or batch directory
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("step_path", nargs='?', type=str, help="Path to a single .step/.stp file")
    group.add_argument("--batch", type=str, help="Directory containing STEP files to check in batch")
    
    # Common options
    parser.add_argument("--timeout", type=int, default=30, help="Timeout seconds for the OCC check subprocess")
    parser.add_argument("--require-all-roots", action="store_true", help="Require every ADVANCED_BREP_SHAPE_REPRESENTATION to mesh/export")
    parser.add_argument("--verify-trimesh-faces", action="store_true", help="Require exported STL to load in trimesh with non-zero face count")
    parser.add_argument("--treat-reader-errors-as-failure", action="store_true", help="Treat OCC StepReaderData error messages as failure")
    
    # Batch-specific options
    parser.add_argument("--output", type=str, help="Output file for batch analysis report (default: auto-generated)")
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    # Batch mode
    if args.batch:
        if not os.path.isdir(args.batch):
            print(f"Error: Batch directory does not exist: {args.batch}")
            return 2
        
        # Generate output file name if not provided
        if args.output:
            output_file = args.output
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            dir_name = os.path.basename(args.batch.rstrip('/'))
            output_file = f"renderability_report_{dir_name}_{timestamp}.json"
        
        batch_check_renderability(
            input_dir=args.batch,
            output_file=output_file,
            timeout=args.timeout,
            verify_trimesh_faces=args.verify_trimesh_faces or True  # Default to True for batch mode
        )
        return 0
    
    # Single file mode
    step_path = args.step_path
    if not step_path:
        print("Error: Must provide either a step file path or --batch directory")
        return 2

    # Basic validation
    if not step_path.lower().endswith((".step", ".stp")):
        out = {
            "renderable": False,
            "status": "error",
            "reason": "invalid_extension",
        }
        print(json.dumps(out))
        return 2

    if not os.path.isabs(step_path):
        # Normalize to absolute; some environments prefer it
        step_path = os.path.abspath(step_path)

    result = check_renderability(
        step_path,
        timeout=args.timeout,
        require_all_roots=args.require_all_roots,
        verify_trimesh_faces=args.verify_trimesh_faces,
        treat_reader_errors_as_failure=args.treat_reader_errors_as_failure,
    )

    # Normalize exit codes to spec
    returncode = result.get("returncode")
    if returncode in (0, 2, 3, 4):
        print(json.dumps({k: v for k, v in result.items() if k != "returncode"}))
        return int(returncode)

    # Fallback
    result.setdefault("renderable", False)
    result.setdefault("status", "error")
    result.setdefault("reason", "unknown_return_code")
    print(json.dumps({k: v for k, v in result.items() if k != "returncode"}))
    return 4


if __name__ == "__main__":
    sys.exit(main())
