#!/usr/bin/env python3
"""
STEP File Chamfer Distance and Reward Calculator

This script computes chamfer distance between two STEP files and calculates a reward based on
user-defined thresholds. It wraps the complete pipeline:
1. STEP → STL → Point Cloud conversion with deterministic sampling
2. Point cloud alignment (center + global registration + ICP)
3. Chamfer distance calculation with scale normalization
4. Reward calculation based on thresholds

Usage:
    python step_chamfer_reward.py step_file1.step step_file2.step --lower-bound 0.5 --upper-bound 2.0
    
Requirements:
    - Conda environment: brepgen_env (contains pyocc)
    - Libraries: OpenCASCADE, trimesh, open3d, chamferdist
"""

import os
import sys
import tempfile
import hashlib
import argparse
import numpy as np
import trimesh
import torch
import open3d as o3d
from pathlib import Path
from plyfile import PlyData, PlyElement
from chamferdist import ChamferDistance

# OpenCASCADE imports
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StlAPI import StlAPI_Writer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.IFSelect import IFSelect_RetDone

# Constants
N_POINTS = 2000
DEFAULT_TIMEOUT = 30


def deterministic_seed_from_content(content1, content2):
    """Generate deterministic seed from STEP file contents for reproducible sampling."""
    combined_content = content1 + content2
    hash_object = hashlib.md5(combined_content.encode())
    seed = int(hash_object.hexdigest()[:8], 16)
    return seed


def write_ply(points, filename):
    """Write points to PLY file."""
    points_list = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points_list, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=False).write(f)


def step_to_stl(step_content, output_stl_path):
    """
    Convert STEP content to STL file.
    
    Args:
        step_content (str): STEP file content
        output_stl_path (str): Path to output STL file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Write STEP content to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.step', delete=False) as f:
            f.write(step_content)
            temp_step_path = f.name
        
        try:
            # Read STEP file
            reader = STEPControl_Reader()
            status = reader.ReadFile(temp_step_path)
            
            if status != IFSelect_RetDone:
                print(f"ERROR: Cannot read STEP file")
                return False
            
            reader.TransferRoot()
            shape = reader.Shape()
            
            if shape.IsNull():
                print(f"ERROR: Null shape")
                return False
            
            # Apply meshing
            mesh = BRepMesh_IncrementalMesh(shape, 0.1)
            mesh.Perform()
            
            # Write STL
            writer = StlAPI_Writer()
            writer.SetASCIIMode(False)
            success = writer.Write(shape, output_stl_path)
            
            if not success or not os.path.exists(output_stl_path) or os.path.getsize(output_stl_path) == 0:
                print(f"ERROR: STL conversion failed")
                return False
                
            return True
            
        finally:
            # Clean up temporary STEP file
            try:
                os.unlink(temp_step_path)
            except:
                pass
                
    except Exception as e:
        print(f"ERROR in STEP to STL conversion: {e}")
        return False


def stl_to_pointcloud(stl_path, seed=None):
    """
    Convert STL file to point cloud with deterministic sampling.
    
    Args:
        stl_path (str): Path to STL file
        seed (int, optional): Random seed for deterministic sampling
        
    Returns:
        numpy.ndarray or None: Point cloud array (N_POINTS x 3) or None if failed
    """
    try:
        # Set random seed for deterministic sampling
        if seed is not None:
            np.random.seed(seed)
        
        # Load mesh
        mesh = trimesh.load_mesh(stl_path)
        
        if mesh.is_empty or len(mesh.faces) == 0:
            print(f"ERROR: Empty mesh")
            return None
        
        # Sample points from surface
        points, _ = trimesh.sample.sample_surface(mesh, N_POINTS)
        
        if len(points) == 0:
            print(f"ERROR: No points sampled")
            return None
            
        return points
        
    except Exception as e:
        print(f"ERROR in STL to point cloud conversion: {e}")
        return None


def read_ply(path):
    """Read PLY file and return point cloud."""
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)
        x = np.array(plydata['vertex']['x'])
        y = np.array(plydata['vertex']['y'])
        z = np.array(plydata['vertex']['z'])
        vertex = np.stack([x, y, z], axis=1)
    return vertex


def to_open3d_pc(points):
    """Convert numpy array to Open3D point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def global_registration(pcd1, pcd2, voxel_size=1.0):
    """Perform global registration using RANSAC."""
    def compute_fpfh(pcd, voxel_size):
        radius_normal = voxel_size * 2
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        radius_feature = voxel_size * 5
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return fpfh
    
    fpfh1 = compute_fpfh(pcd1, voxel_size)
    fpfh2 = compute_fpfh(pcd2, voxel_size)
    
    # RANSAC registration
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd2, pcd1, fpfh2, fpfh1, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    return result.transformation


def align_point_clouds(pc1, pc2, threshold=10.0, iterations=3, use_global=True):
    """
    Align point clouds using center alignment + global registration + ICP.
    
    Returns:
        numpy.ndarray: Aligned pc2
    """
    pcd1 = to_open3d_pc(pc1)
    pcd2 = to_open3d_pc(pc2)
    
    # Global registration for better initial alignment
    if use_global:
        try:
            # Estimate appropriate voxel size based on point cloud scale
            voxel_size = np.mean(np.max(pc1, axis=0) - np.min(pc1, axis=0)) * 0.05
            global_trans = global_registration(pcd1, pcd2, voxel_size)
            pcd2.transform(global_trans)
        except Exception as e:
            print(f"Global registration failed: {e}, proceeding with ICP only")
    
    trans_init = np.identity(4)
    current_threshold = threshold
    
    for i in range(iterations):
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd2, pcd1, current_threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        pcd2.transform(reg_p2p.transformation)
        trans_init = np.identity(4)  # Reset for next iteration
        
        # Decrease threshold for next iteration
        current_threshold *= 0.5
    
    pc2_aligned = np.asarray(pcd2.points)
    return pc2_aligned


def align_centers(pc1, pc2):
    """Align point cloud centers."""
    center1 = np.mean(pc1, axis=0)
    center2 = np.mean(pc2, axis=0)
    pc2_aligned = pc2 + (center1 - center2)
    return pc2_aligned


def compute_scale_metrics(pc):
    """Compute various scale metrics for normalization."""
    # Center the point cloud
    centered_pc = pc - np.mean(pc, axis=0)
    
    # RMS distance from center
    distances_from_center = np.linalg.norm(centered_pc, axis=1)
    rms_distance = np.sqrt(np.mean(distances_from_center**2))
    
    return rms_distance


def compute_chamfer_distance(pc1, pc2, scale_normalize=True, use_gt_scale_only=True):
    """
    Compute Chamfer Distance between two point clouds with alignment and optional scale normalization.
    
    Args:
        pc1 (numpy.ndarray): First point cloud (ground truth when use_gt_scale_only=True)
        pc2 (numpy.ndarray): Second point cloud (generated when use_gt_scale_only=True)
        scale_normalize (bool): Whether to apply scale normalization
        use_gt_scale_only (bool): If True, use only pc1's scale for normalization (recommended for GT vs Generated evaluation)
        
    Returns:
        float: Chamfer distance value
    """
    try:
        # Step 1: Center alignment
        pc2_centered = align_centers(pc1, pc2)
        
        # Step 2: Full alignment (global registration + ICP)
        pc2_aligned = align_point_clouds(pc1, pc2_centered)
        
        # Step 3: Compute Chamfer Distance
        try:
            pc1_tensor = torch.tensor(pc1, dtype=torch.float32).unsqueeze(0).cuda()
            pc2_tensor = torch.tensor(pc2_aligned, dtype=torch.float32).unsqueeze(0).cuda()
        except Exception:
            pc1_tensor = torch.tensor(pc1, dtype=torch.float32).unsqueeze(0)
            pc2_tensor = torch.tensor(pc2_aligned, dtype=torch.float32).unsqueeze(0)
        
        chamfer_dist = ChamferDistance()
        with torch.no_grad():
            dist = chamfer_dist(pc1_tensor, pc2_tensor, point_reduction="mean")
            raw_cd = dist.mean().item()
        
        # Step 4: Scale normalization (optional)
        if scale_normalize:
            if use_gt_scale_only:
                # Use only first point cloud's scale (ground truth) for fair evaluation
                scale_factor = compute_scale_metrics(pc1)
            else:
                # Use average of both scales (for general comparison)
                scale1 = compute_scale_metrics(pc1)
                scale2 = compute_scale_metrics(pc2_aligned)
                scale_factor = (scale1 + scale2) / 2
            
            normalized_cd = raw_cd / (scale_factor**2) if scale_factor > 0 else raw_cd
            return normalized_cd
        else:
            return raw_cd
            
    except Exception as e:
        print(f"ERROR in chamfer distance computation: {e}")
        return None


def calculate_reward(chamfer_distance, lower_bound, upper_bound):
    """
    Calculate reward based on chamfer distance and thresholds.
    
    Args:
        chamfer_distance (float or None): Chamfer distance value
        lower_bound (float): Lower threshold
        upper_bound (float): Upper threshold
        
    Returns:
        float: Reward value between 0 and 1
    """
    if chamfer_distance is None or chamfer_distance > upper_bound:
        return 0.0
    elif chamfer_distance <= lower_bound:
        return 1.0
    else:
        # Linear interpolation between lower_bound and upper_bound
        reward = 1.0 - (chamfer_distance - lower_bound) / (upper_bound - lower_bound)
        return max(0.0, min(1.0, reward))


def process_step_files(step_content1, step_content2, lower_bound, upper_bound, 
                      scale_normalize=True, use_gt_scale_only=True, verbose=False):
    """
    Main processing function for two STEP file contents.
    
    Args:
        step_content1 (str): First STEP file content (ground truth when use_gt_scale_only=True)
        step_content2 (str): Second STEP file content (generated when use_gt_scale_only=True)
        lower_bound (float): Lower threshold for reward calculation
        upper_bound (float): Upper threshold for reward calculation
        scale_normalize (bool): Whether to apply scale normalization
        use_gt_scale_only (bool): If True, use only first file's scale for normalization (recommended for GT vs Generated)
        verbose (bool): Whether to print detailed information
        
    Returns:
        tuple: (chamfer_distance, reward)
    """
    temp_files = []
    
    try:
        # Generate deterministic seed from file contents
        seed = deterministic_seed_from_content(step_content1, step_content2)
        
        if verbose:
            print(f"Using deterministic seed: {seed}")
            print("Converting STEP files to point clouds...")
        
        # Create temporary files
        temp_stl1 = tempfile.NamedTemporaryFile(suffix='_1.stl', delete=False)
        temp_stl2 = tempfile.NamedTemporaryFile(suffix='_2.stl', delete=False)
        temp_files.extend([temp_stl1.name, temp_stl2.name])
        temp_stl1.close()
        temp_stl2.close()
        
        # Convert STEP to STL
        if not step_to_stl(step_content1, temp_stl1.name):
            if verbose:
                print("Failed to convert first STEP file to STL")
            return None, 0.0
            
        if not step_to_stl(step_content2, temp_stl2.name):
            if verbose:
                print("Failed to convert second STEP file to STL")
            return None, 0.0
        
        # Convert STL to point clouds with deterministic sampling
        pc1 = stl_to_pointcloud(temp_stl1.name, seed)
        pc2 = stl_to_pointcloud(temp_stl2.name, seed + 1)  # Different seed for second file
        
        if pc1 is None or pc2 is None:
            if verbose:
                print("Failed to convert STL to point cloud")
            return None, 0.0
        
        if verbose:
            print(f"Generated point clouds: {pc1.shape}, {pc2.shape}")
            print("Computing chamfer distance with alignment...")
        
        # Compute chamfer distance
        chamfer_distance = compute_chamfer_distance(pc1, pc2, scale_normalize, use_gt_scale_only)
        
        if chamfer_distance is None:
            if verbose:
                print("Failed to compute chamfer distance")
            return None, 0.0
        
        # Calculate reward
        reward = calculate_reward(chamfer_distance, lower_bound, upper_bound)
        
        if verbose:
            print(f"Chamfer Distance: {chamfer_distance:.6f}")
            print(f"Reward: {reward:.6f}")
        
        return chamfer_distance, reward
        
    except Exception as e:
        if verbose:
            print(f"ERROR in processing: {e}")
        return None, 0.0
        
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Compute chamfer distance and reward between two STEP files')
    parser.add_argument('step_file1', type=str, help='Path to first STEP file')
    parser.add_argument('step_file2', type=str, help='Path to second STEP file')
    parser.add_argument('--lower-bound', type=float, required=True, help='Lower threshold for reward calculation')
    parser.add_argument('--upper-bound', type=float, required=True, help='Upper threshold for reward calculation')
    parser.add_argument('--no-scale-normalize', action='store_true', help='Disable scale normalization')
    parser.add_argument('--use-gt-scale-only', action='store_true', help='Use only first file (GT) scale for normalization (recommended for GT vs Generated evaluation)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.lower_bound >= args.upper_bound:
        print("ERROR: Lower bound must be less than upper bound")
        sys.exit(1)
    
    # Read STEP files
    try:
        with open(args.step_file1, 'r') as f:
            step_content1 = f.read()
        with open(args.step_file2, 'r') as f:
            step_content2 = f.read()
    except Exception as e:
        print(f"ERROR reading STEP files: {e}")
        sys.exit(1)
    
    if args.verbose:
        print(f"Processing files: {args.step_file1}, {args.step_file2}")
        print(f"Thresholds: lower={args.lower_bound}, upper={args.upper_bound}")
        print(f"Scale normalization: {not args.no_scale_normalize}")
        print(f"GT-only scale normalization: {args.use_gt_scale_only}")
    
    # Process files
    chamfer_distance, reward = process_step_files(
        step_content1, step_content2, 
        args.lower_bound, args.upper_bound,
        scale_normalize=not args.no_scale_normalize,
        use_gt_scale_only=args.use_gt_scale_only,
        verbose=args.verbose
    )
    
    # Output results
    if chamfer_distance is not None:
        print(f"Chamfer Distance: {chamfer_distance:.6f}")
    else:
        print("Chamfer Distance: FAILED")
    
    print(f"Reward: {reward:.6f}")


if __name__ == "__main__":
    main()
