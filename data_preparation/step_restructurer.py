#!/usr/bin/env python3
"""
STEP File Restructurer

This script restructures STEP files by reorganizing entities according to specific rules
for different root entity types (PRODUCT_CATEGORY_RELATIONSHIP, APPLICATION_PROTOCOL_DEFINITION,
MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION, SHAPE_DEFINITION_REPRESENTATION).

The restructuring uses a depth-first traversal (DFS) to eliminate forward references,
which is crucial for LLM training (the model cannot "look ahead" in the file).
Annotations (/* ... */) are inserted to help the model understand structure.
Run restore_step_valid.py afterwards to strip annotations and produce a valid STEP file.

Usage:
    python step_restructurer.py <step_file_path> [-o <output_directory>]

Pipeline:
    1. round_step_numbers.py   — normalise floating-point precision in raw STEP files
    2. step_restructurer.py    — DFS reorder + annotate (produces training-format STEP)
    3. restore_step_valid.py   — strip annotations → valid STEP file
    4. dataset_construct_rag.py — build RAG training dataset (pairs each STEP with retrieval)
    5. data_split.py            — split into train / val / test
"""

import re
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional


class StepRestructurer:
    def __init__(self):
        self.header_section = ""
        self.entity_map = {}  # entity_id -> entity_content
        self.entity_types = {}  # entity_id -> entity_type
        self.entity_references = defaultdict(list)  # entity_id -> [referenced_entity_ids]
        self.referenced_by = defaultdict(set)  # entity_id -> {entities_that_reference_it}
        self.complex_entities = {}  # entity_id -> complex_entity_content

        # For renumbering and post-processing annotations
        self._last_id_mapping: Dict[int, int] = {}
        self._mdgpr_placeholder_plans: List[Dict[str, object]] = []

        # Expected root entity types
        self.expected_root_types = {
            'PRODUCT_CATEGORY_RELATIONSHIP',
            'SHAPE_DEFINITION_REPRESENTATION',
            'APPLICATION_PROTOCOL_DEFINITION',
            'MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION'
        }

        # Output sections
        self.output_sections = []
        self.processed_entities = set()  # Track entities already included in output

    def parse_step_file(self, file_path):
        """Parse a STEP file and extract entity information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()

        # Extract HEADER section
        header_match = re.search(r'(ISO-10303-21;.*?DATA;)', content, re.DOTALL)
        if header_match:
            self.header_section = header_match.group(1)

        # Extract DATA section
        data_match = re.search(r'DATA;(.*?)ENDSEC;', content, re.DOTALL)
        if not data_match:
            raise ValueError("No DATA section found in STEP file")

        data_section = data_match.group(1)

        # Parse entities - collect all entity lines first
        all_entity_lines = re.findall(r'#\d+\s*=\s*[^;]+;', data_section, re.MULTILINE)

        print(f"Found {len(all_entity_lines)} total entity lines in STEP file")

        # First pass: collect entities
        for line in all_entity_lines:
            # Check if it's a complex entity (starts with parentheses)
            if re.search(r'#\d+\s*=\s*\(', line):
                # Parse complex entity
                complex_match = re.search(r'#(\d+)\s*=\s*(.+);', line)
                if complex_match:
                    entity_id = int(complex_match.group(1))
                    self.complex_entities[entity_id] = line.strip()
                continue

            # Extract entity ID and type for simple entities
            entity_match = re.search(r'#(\d+)\s*=\s*([A-Z_][A-Z0-9_]*)\s*\([^;]*\)\s*;', line)
            if entity_match:
                entity_id = int(entity_match.group(1))
                entity_type = entity_match.group(2)

                # Store entity information
                self.entity_map[entity_id] = line.strip()
                self.entity_types[entity_id] = entity_type

        # Second pass: analyze references
        for entity_id, entity_content in self.entity_map.items():
            param_section_match = re.search(r'=\s*[A-Z_][A-Z0-9_]*\s*\((.+)\)\s*;', entity_content, re.DOTALL)
            if param_section_match:
                param_section = param_section_match.group(1)
                references = re.findall(r'#(\d+)', param_section)

                valid_references = []
                for ref_id_str in references:
                    ref_id = int(ref_id_str)
                    if (ref_id in self.entity_map or ref_id in self.complex_entities) and ref_id != entity_id:
                        valid_references.append(ref_id)
                        self.referenced_by[ref_id].add(entity_id)

                self.entity_references[entity_id] = valid_references

        # Third pass: analyze references from complex entities
        for complex_id, complex_content in self.complex_entities.items():
            references = re.findall(r'#(\d+)', complex_content)
            for ref_id_str in references:
                ref_id = int(ref_id_str)
                if (ref_id in self.entity_map or ref_id in self.complex_entities) and ref_id != complex_id:
                    self.referenced_by[ref_id].add(complex_id)

        print(f"Parsed {len(self.entity_map)} simple entities and {len(self.complex_entities)} complex entities")

    def find_root_entities(self):
        """Find root entities by analyzing pointer relationships."""
        simple_roots = []
        complex_roots = []

        # Find simple entities with no incoming references
        for entity_id in self.entity_map.keys():
            if not self.referenced_by[entity_id]:
                simple_roots.append(entity_id)

        # Find complex entities with no incoming references
        for complex_id in self.complex_entities.keys():
            is_referenced = False

            # Check if any simple entity references this complex entity
            for entity_id, refs in self.entity_references.items():
                if complex_id in refs:
                    is_referenced = True
                    break

            # Check if any complex entity references this complex entity
            if not is_referenced:
                for other_complex_id, complex_content in self.complex_entities.items():
                    if other_complex_id != complex_id:
                        references = re.findall(r'#(\d+)', complex_content)
                        if str(complex_id) in references:
                            is_referenced = True
                            break

            if not is_referenced:
                complex_roots.append(complex_id)

        all_roots = simple_roots + complex_roots
        print(f"Found {len(simple_roots)} simple root entities and {len(complex_roots)} complex root entities")

        return sorted(all_roots)

    def get_ordinal_suffix(self, number):
        """Get ordinal suffix for a number (st, nd, rd, th)."""
        if 10 <= number % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')
        return suffix

    def validate_root_types(self, root_entities):
        """Validate that all root entities are of expected types."""
        unexpected_roots = []

        for root_id in root_entities:
            if root_id in self.entity_types:
                root_type = self.entity_types[root_id]
                if root_type not in self.expected_root_types:
                    unexpected_roots.append((root_id, root_type))
            elif root_id in self.complex_entities:
                # Complex entities are acceptable as roots
                pass

        if unexpected_roots:
            print("WARNING: Found unexpected root entity types:")
            for root_id, root_type in unexpected_roots:
                print(f"  #{root_id} - {root_type}")

        return unexpected_roots

    def dfs_traverse_tree(self, start_entity_id, visited_in_path=None, depth=0):
        """Perform DFS traversal and return the tree structure."""
        if visited_in_path is None:
            visited_in_path = set()

        if depth > 20 or start_entity_id in visited_in_path:
            return [(depth, start_entity_id, "CIRCULAR_REF")]

        tree = [(depth, start_entity_id, "NORMAL")]
        visited_in_path.add(start_entity_id)

        # Get references
        if start_entity_id in self.entity_map:
            references = self.entity_references.get(start_entity_id, [])
        elif start_entity_id in self.complex_entities:
            complex_content = self.complex_entities[start_entity_id]
            ref_ids = re.findall(r'#(\d+)', complex_content)
            references = [int(r) for r in ref_ids if int(r) != start_entity_id]
        else:
            references = []

        # Recursively traverse references
        for ref_id in references:
            if ref_id in self.entity_map or ref_id in self.complex_entities:
                subtree = self.dfs_traverse_tree(ref_id, visited_in_path.copy(), depth + 1)
                tree.extend(subtree)

        visited_in_path.discard(start_entity_id)
        return tree

    def merge_trees_by_depth(self, trees):
        """Merge multiple trees by combining entities at the same depth level."""
        depth_groups = defaultdict(list)

        # Group entities by depth
        for tree in trees:
            for depth, entity_id, ref_type in tree:
                if ref_type != "CIRCULAR_REF":
                    depth_groups[depth].append(entity_id)

        # Remove duplicates at each depth level
        merged_tree = []
        for depth in sorted(depth_groups.keys()):
            unique_entities = []
            seen = set()
            for entity_id in depth_groups[depth]:
                if entity_id not in seen:
                    unique_entities.append(entity_id)
                    seen.add(entity_id)
            merged_tree.append((depth, unique_entities))

        return merged_tree

    def format_entity_for_output(self, entity_id, indent_level=0):
        """Format an entity for output."""
        indent = "  " * indent_level if indent_level > 0 else ""

        if entity_id in self.entity_map:
            content = self.entity_map[entity_id]
        elif entity_id in self.complex_entities:
            content = self.complex_entities[entity_id]
        else:
            return f"{indent}#{entity_id}=UNKNOWN_ENTITY"

        return f"{indent}{content}"

    def process_product_category_relationship(self, root_entities):
        """Process PRODUCT_CATEGORY_RELATIONSHIP subtrees."""
        pcr_roots = [eid for eid in root_entities
                     if eid in self.entity_types and
                     self.entity_types[eid] == 'PRODUCT_CATEGORY_RELATIONSHIP']

        if not pcr_roots:
            return

        print(f"Processing {len(pcr_roots)} PRODUCT_CATEGORY_RELATIONSHIP subtrees...")

        # Generate trees for all PCR roots
        trees = []
        for root_id in pcr_roots:
            tree = self.dfs_traverse_tree(root_id)
            trees.append(tree)

        # Merge trees by depth
        merged_tree = self.merge_trees_by_depth(trees)

        # Add section annotation
        self.output_sections.append(f"/* PRODUCT_CATEGORY_RELATIONSHIP Section - {len(pcr_roots)} root entities */")

        # Add entities from merged tree
        for depth, entity_ids in merged_tree:
            for entity_id in entity_ids:
                if entity_id not in self.processed_entities:
                    self.output_sections.append(self.format_entity_for_output(entity_id))
                    self.processed_entities.add(entity_id)

    def process_application_protocol_definition(self, root_entities):
        """Process APPLICATION_PROTOCOL_DEFINITION subtrees."""
        apd_roots = [eid for eid in root_entities
                     if eid in self.entity_types and
                     self.entity_types[eid] == 'APPLICATION_PROTOCOL_DEFINITION']

        if not apd_roots:
            return

        print(f"Processing {len(apd_roots)} APPLICATION_PROTOCOL_DEFINITION subtrees...")

        for root_id in apd_roots:
            tree = self.dfs_traverse_tree(root_id)

            # Check if it has exactly 2 layers
            max_depth = max(depth for depth, _, _ in tree)
            if max_depth != 1:
                print(f"WARNING: APPLICATION_PROTOCOL_DEFINITION #{root_id} has {max_depth + 1} layers instead of 2")
                # Add as separate section
                self.output_sections.append(f"/* APPLICATION_PROTOCOL_DEFINITION Section - Unusual depth */")
                for depth, entity_id, ref_type in tree:
                    if ref_type != "CIRCULAR_REF" and entity_id not in self.processed_entities:
                        self.output_sections.append(self.format_entity_for_output(entity_id))
                        self.processed_entities.add(entity_id)
            else:
                # Add directly to previous section (no separate annotation)
                for depth, entity_id, ref_type in tree:
                    if ref_type != "CIRCULAR_REF" and entity_id not in self.processed_entities:
                        self.output_sections.append(self.format_entity_for_output(entity_id))
                        self.processed_entities.add(entity_id)

    def count_styled_items_and_complex_entities(self, root_id):
        """Count STYLED_ITEM and COMPLEX_ENTITY references from root."""
        if root_id not in self.entity_references:
            return 0, 0

        styled_items = 0
        complex_entities = 0

        for ref_id in self.entity_references[root_id]:
            if ref_id in self.entity_types and self.entity_types[ref_id] == 'STYLED_ITEM':
                styled_items += 1
            elif ref_id in self.complex_entities:
                complex_entities += 1

        return styled_items, complex_entities

    def find_and_unify_colors(self, tree):
        """Find all COLOUR_RGB entities in tree and unify them."""
        color_entities = []

        for depth, entity_id, ref_type in tree:
            if (ref_type != "CIRCULAR_REF" and
                entity_id in self.entity_types and
                self.entity_types[entity_id] == 'COLOUR_RGB'):
                color_entities.append(entity_id)

        if not color_entities:
            return None, {}

        # Use the first color entity as the unified one
        unified_color_id = color_entities[0]

        # Create unified color entity with standard values
        unified_color_content = f"#{unified_color_id}=COLOUR_RGB( , 0.5, 0.5, 0.5 );"

        # Create mapping for replacements
        color_replacements = {}
        for color_id in color_entities[1:]:  # Skip the first one
            color_replacements[color_id] = unified_color_id

        return unified_color_id, unified_color_content, color_replacements

    def apply_color_replacements(self, content, color_replacements):
        """Replace color entity references in content."""
        for old_id, new_id in color_replacements.items():
            content = re.sub(rf'#{old_id}\b', f'#{new_id}', content)
        return content

    def remove_duplicate_entities_in_tree(self, tree):
        """Remove duplicate entities in a tree."""
        seen_entities = set()
        cleaned_tree = []

        for depth, entity_id, ref_type in tree:
            if entity_id not in seen_entities:
                cleaned_tree.append((depth, entity_id, ref_type))
                seen_entities.add(entity_id)

        return cleaned_tree

    def process_mechanical_design_geometric_presentation(self, root_entities):
        """Process MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION subtrees."""
        mdgpr_roots = [eid for eid in root_entities
                       if eid in self.entity_types and
                       self.entity_types[eid] == 'MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION']

        if not mdgpr_roots:
            return

        print(f"Processing {len(mdgpr_roots)} MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION subtrees...")

        for root_id in mdgpr_roots:
            # Count STYLED_ITEM and COMPLEX_ENTITY references
            styled_count, complex_count = self.count_styled_items_and_complex_entities(root_id)

            # Add section annotation
            self.output_sections.append(f"/* MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION Section - {styled_count} STYLED_ITEM, {complex_count} COMPLEX_ENTITY */")

            # Add root entity (unmodified for now; placeholders applied after renumbering)
            if root_id not in self.processed_entities:
                self.output_sections.append(self.format_entity_for_output(root_id))
                self.processed_entities.add(root_id)

            # Generate full tree
            tree = self.dfs_traverse_tree(root_id)

            # Find and unify colors
            unified_color_result = self.find_and_unify_colors(tree)
            color_replacements = {}

            if unified_color_result and len(unified_color_result) == 3:
                unified_color_id, unified_color_content, color_replacements = unified_color_result
                # Add unified color right after root
                self.output_sections.append(unified_color_content)
                self.processed_entities.add(unified_color_id)

            # Store plan for placeholder mapping to be applied post-renumbering
            plan_references = list(self.entity_references.get(root_id, []))
            plan = {
                'root_id': root_id,
                'references': plan_references,
                'styled_ids': list(ref_id for ref_id in self.entity_references.get(root_id, []) if ref_id in self.entity_types and self.entity_types[ref_id] == 'STYLED_ITEM'),
                'complex_ids': [ref_id for ref_id in self.entity_references.get(root_id, []) if ref_id in self.complex_entities],
            }
            self._mdgpr_placeholder_plans.append(plan)

            # Process STYLED_ITEM subtrees
            styled_item_refs = [ref_id for ref_id in self.entity_references.get(root_id, [])
                               if ref_id in self.entity_types and self.entity_types[ref_id] == 'STYLED_ITEM']

            for i, styled_id in enumerate(styled_item_refs, 1):
                # Explicit BEGIN tag with numeric index
                self.output_sections.append(f"/* BEGIN STYLED_ITEM {i} */")

                styled_tree = self.dfs_traverse_tree(styled_id)
                for depth, entity_id, ref_type in styled_tree:
                    if (ref_type != "CIRCULAR_REF" and
                        entity_id not in self.processed_entities and
                        (entity_id not in self.entity_types or self.entity_types[entity_id] != 'COLOUR_RGB' or entity_id not in color_replacements)):

                        content = self.format_entity_for_output(entity_id)
                        if color_replacements:
                            content = self.apply_color_replacements(content, color_replacements)
                        self.output_sections.append(content)
                        self.processed_entities.add(entity_id)

                # Explicit END tag with numeric index
                self.output_sections.append(f"/* END STYLED_ITEM {i} */")

            # Process COMPLEX_ENTITY subtrees
            complex_refs = [ref_id for ref_id in self.entity_references.get(root_id, [])
                           if ref_id in self.complex_entities]

            for i, complex_id in enumerate(complex_refs, 1):
                # Explicit BEGIN tag with numeric index for COMPLEX_ENTITY
                self.output_sections.append(f"/* BEGIN COMPLEX_ENTITY {i} */")

                complex_tree = self.dfs_traverse_tree(complex_id)
                # Remove duplicates in complex entity tree
                complex_tree = self.remove_duplicate_entities_in_tree(complex_tree)

                for depth, entity_id, ref_type in complex_tree:
                    if ref_type != "CIRCULAR_REF" and entity_id not in self.processed_entities:
                        self.output_sections.append(self.format_entity_for_output(entity_id))
                        self.processed_entities.add(entity_id)

                # Explicit END tag with numeric index
                self.output_sections.append(f"/* END COMPLEX_ENTITY {i} */")

    def prune_product_definition_branches(self, tree):
        """Prune branches from PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE and PRODUCT_DEFINITION_CONTEXT."""
        pruned_tree = []

        skip_subtree = False
        skip_depth = -1

        for depth, entity_id, ref_type in tree:
            if ref_type == "CIRCULAR_REF":
                continue

            # Reset skip if we're at a shallower or equal depth
            if skip_subtree and depth <= skip_depth:
                skip_subtree = False
                skip_depth = -1

            # If we're currently skipping, don't add this entity
            if skip_subtree:
                continue

            entity_type = self.entity_types.get(entity_id, "UNKNOWN")

            # Always include the current entity first
            pruned_tree.append((depth, entity_id, ref_type))

            # Check if this entity should have its children completely removed
            if entity_type in ['PRODUCT_DEFINITION_FORMATION_WITH_SPECIFIED_SOURCE', 'PRODUCT_DEFINITION_CONTEXT']:
                # Start skipping all children (entities at deeper levels)
                skip_subtree = True
                skip_depth = depth

        return pruned_tree

    def count_entity_references_by_type(self, entity_id):
        """Count how many entities of each type this entity references."""
        if entity_id not in self.entity_references:
            return {}

        type_counts = defaultdict(int)
        for ref_id in self.entity_references[entity_id]:
            if ref_id in self.entity_types:
                entity_type = self.entity_types[ref_id]
                type_counts[entity_type] += 1
            elif ref_id in self.complex_entities:
                type_counts['COMPLEX_ENTITY'] += 1

        return dict(type_counts)

    def should_annotate_subtree(self, entity_id, entity_type):
        """Determine if a subtree should be annotated based on references."""
        type_counts = self.count_entity_references_by_type(entity_id)

        # If any entity points to 3 or more same type of entities
        for ref_type, count in type_counts.items():
            if count >= 3:
                return True, ref_type, count

        return False, None, 0

    def are_entities_leaves(self, entity_ids):
        """Check if all entities in the list are leaves (no children)."""
        for entity_id in entity_ids:
            if entity_id in self.entity_references and self.entity_references[entity_id]:
                return False
        return True

    def process_shape_definition_representation(self, root_entities):
        """Process SHAPE_DEFINITION_REPRESENTATION subtrees."""
        sdr_roots = [eid for eid in root_entities
                     if eid in self.entity_types and
                     self.entity_types[eid] == 'SHAPE_DEFINITION_REPRESENTATION']

        if not sdr_roots:
            return

        print(f"Processing {len(sdr_roots)} SHAPE_DEFINITION_REPRESENTATION subtrees...")

        # Add section annotation
        self.output_sections.append(f"/* SHAPE_DEFINITION_REPRESENTATION Section - {len(sdr_roots)} subtrees */")

        for i, root_id in enumerate(sdr_roots, 1):
            # Add start annotation for this subtree (explicit tags)
            self.output_sections.append(f"/* BEGIN SHAPE_DEFINITION_REPRESENTATION {i} */")

            # Generate tree and prune branches
            tree = self.dfs_traverse_tree(root_id)
            pruned_tree = self.prune_product_definition_branches(tree)

            # Process the pruned tree maintaining hierarchical structure
            self.process_hierarchical_tree(pruned_tree)

            # Add end annotation for this subtree
            self.output_sections.append(f"/* END SHAPE_DEFINITION_REPRESENTATION {i} */")

    def process_hierarchical_tree(self, tree):
        """Process a tree maintaining hierarchical structure with proper indentation."""
        for depth, entity_id, ref_type in tree:
            if ref_type != "CIRCULAR_REF" and entity_id not in self.processed_entities:
                # Format with proper indentation (but no indent in final output as per requirement)
                entity_content = self.format_entity_for_output(entity_id, indent_level=0)
                self.output_sections.append(entity_content)
                self.processed_entities.add(entity_id)

    def process_remaining_entities_with_annotations(self, tree):
        """Process remaining entities with smart annotation rules."""
        entities_by_depth = defaultdict(list)

        # Group entities by depth
        for depth, entity_id, ref_type in tree:
            if ref_type != "CIRCULAR_REF" and entity_id not in self.processed_entities:
                entities_by_depth[depth].append(entity_id)

        # Process each depth level
        for depth in sorted(entities_by_depth.keys()):
            entity_ids = entities_by_depth[depth]

            for entity_id in entity_ids:
                if entity_id in self.processed_entities:
                    continue

                entity_type = self.entity_types.get(entity_id, "COMPLEX_ENTITY" if entity_id in self.complex_entities else "UNKNOWN")

                # Handle AXIS2_PLACEMENT_3D directly
                if entity_type == 'AXIS2_PLACEMENT_3D':
                    self.output_sections.append(self.format_entity_for_output(entity_id))
                    self.processed_entities.add(entity_id)
                    continue

                # Handle COMPLEX_ENTITY
                if entity_id in self.complex_entities:
                    if entity_id not in self.processed_entities:
                        # Check if this complex entity already exists in previous sections
                        # For now, apply deduplication and add
                        subtree = self.dfs_traverse_tree(entity_id)
                        cleaned_subtree = self.remove_duplicate_entities_in_tree(subtree)

                        for sub_depth, sub_entity_id, sub_ref_type in cleaned_subtree:
                            if sub_ref_type != "CIRCULAR_REF" and sub_entity_id not in self.processed_entities:
                                self.output_sections.append(self.format_entity_for_output(sub_entity_id))
                                self.processed_entities.add(sub_entity_id)
                    continue

                # Check if this entity should be annotated
                should_annotate, ref_type, ref_count = self.should_annotate_subtree(entity_id, entity_type)

                if should_annotate:
                    # Get all entities of the referenced type
                    ref_entities = [ref_id for ref_id in self.entity_references.get(entity_id, [])
                                   if (ref_id in self.entity_types and self.entity_types[ref_id] == ref_type) or
                                      (ref_type == 'COMPLEX_ENTITY' and ref_id in self.complex_entities)]

                    # Add the root entity
                    self.output_sections.append(self.format_entity_for_output(entity_id))
                    self.processed_entities.add(entity_id)

                    # Check if referenced entities are leaves
                    if self.are_entities_leaves(ref_entities):
                        # Annotate outside the entity chunk
                        self.output_sections.append(f"/* {len(ref_entities)} {ref_type} entities */")
                        for ref_id in ref_entities:
                            if ref_id not in self.processed_entities:
                                self.output_sections.append(self.format_entity_for_output(ref_id))
                                self.processed_entities.add(ref_id)
                    else:
                        # Add annotation for each entity with start and end
                        for i, ref_id in enumerate(ref_entities, 1):
                            if ref_id not in self.processed_entities:
                                self.output_sections.append(f"/* Start of {i}{'st' if i == 1 else 'nd' if i == 2 else 'rd' if i == 3 else 'th'} {ref_type} */")

                                # Process subtree of this entity
                                ref_subtree = self.dfs_traverse_tree(ref_id)
                                for sub_depth, sub_entity_id, sub_ref_type in ref_subtree:
                                    if sub_ref_type != "CIRCULAR_REF" and sub_entity_id not in self.processed_entities:
                                        self.output_sections.append(self.format_entity_for_output(sub_entity_id))
                                        self.processed_entities.add(sub_entity_id)

                                self.output_sections.append(f"/* End of {i}{'st' if i == 1 else 'nd' if i == 2 else 'rd' if i == 3 else 'th'} {ref_type} */")
                else:
                    # Regular entity processing
                    if entity_id not in self.processed_entities:
                        self.output_sections.append(self.format_entity_for_output(entity_id))
                        self.processed_entities.add(entity_id)

    def generate_restructured_step(self, output_path):
        """Generate the restructured STEP file."""
        # Final pass: renumber entities and update references to be sequential from #1
        self.output_sections = self.renumber_output_sections()

        # Apply MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION pointer placeholders
        self.output_sections = self.apply_mdgpr_placeholders(self.output_sections)

        with open(output_path, 'w') as f:
            # Write HEADER section
            f.write(self.header_section)
            f.write('\n')

            # Write restructured DATA section
            for section in self.output_sections:
                f.write(section + '\n')

            # Write closing
            f.write('ENDSEC;\n')
            f.write('END-ISO-10303-21;\n')

        print(f"Restructured STEP file written to: {output_path}")
        print(f"Total output sections: {len(self.output_sections)}")
        print(f"Total processed entities: {len(self.processed_entities)}")

    def restructure_step_file(self, input_path, output_path):
        """Main method to restructure a STEP file."""
        print(f"Restructuring STEP file: {input_path}")

        # Parse the file
        self.parse_step_file(input_path)

        # Find root entities
        root_entities = self.find_root_entities()

        # Validate root types
        unexpected_roots = self.validate_root_types(root_entities)

        # Process each root entity type in order
        self.process_product_category_relationship(root_entities)
        self.process_application_protocol_definition(root_entities)
        self.process_mechanical_design_geometric_presentation(root_entities)
        self.process_shape_definition_representation(root_entities)

        # Generate output
        self.generate_restructured_step(output_path)

        return len(self.output_sections), len(self.processed_entities)

    def renumber_output_sections(self) -> List[str]:
        """Renumber all entity IDs starting from #1 in the output and update all pointers accordingly."""
        # First pass: build mapping from old ID to new sequential ID in order of definition appearance
        id_mapping: Dict[int, int] = {}
        next_id: int = 1

        definition_pattern = re.compile(r'^\s*#(\d+)\s*=')
        for line in self.output_sections:
            match = definition_pattern.match(line)
            if match:
                old_id = int(match.group(1))
                if old_id not in id_mapping:
                    id_mapping[old_id] = next_id
                    next_id += 1

        # Second pass: apply mapping to all occurrences of #<id> tokens
        token_pattern = re.compile(r'#(\d+)\b')
        def replace_token(m):
            old = int(m.group(1))
            if old in id_mapping:
                return f"#{id_mapping[old]}"
            return m.group(0)

        renumbered_sections: List[str] = []
        for line in self.output_sections:
            renumbered_sections.append(token_pattern.sub(replace_token, line))
        # Save mapping for later placeholder processing
        self._last_id_mapping = id_mapping

        return renumbered_sections

    def apply_mdgpr_placeholders(self, sections: List[str]) -> List[str]:
        """Replace MDGPR root pointers with placeholders and insert ID mapping annotations before subtree content."""
        if not self._mdgpr_placeholder_plans:
            return sections

        updated: List[str] = []
        # Build reverse mapping new_id -> old_id for annotation clarity
        new_to_old = {new: old for old, new in self._last_id_mapping.items()}

        # Build a set for quick lookup of lines that define STYLED_ITEM or COMPLEX_ENTITY roots to annotate before their block
        mdgpr_new_roots = set()
        # Create mapping from placeholder key to new id for each plan
        plan_mappings: List[Dict[str, int]] = []
        for plan in self._mdgpr_placeholder_plans:
            new_refs = []
            for old in plan['references']:
                if old in self._last_id_mapping:
                    new_refs.append(self._last_id_mapping[old])
            # Assign placeholders in order of appearance only for the STYLED_ITEM refs first, then complex, then remaining
            placeholders: Dict[int, str] = {}
            placeholder_list: List[Tuple[int, str]] = []
            counter = 1
            # Determine ordering: keep the original order
            for old in plan['references']:
                if old in self._last_id_mapping:
                    new_id = self._last_id_mapping[old]
                    key = f"ID_{counter}"
                    placeholders[new_id] = key
                    placeholder_list.append((new_id, key))
                    mdgpr_new_roots.add(new_id)
                    counter += 1
            plan_mappings.append({'placeholders': placeholders, 'list': placeholder_list})

        # Helper regexes
        def_line = re.compile(r'^\s*#(\d+)\s*=\s*([A-Z_][A-Z0-9_]*)\s*\(')
        mdgpr_type = 'MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION'

        # Process lines, replacing the MDGPR root parameter list pointers with placeholders
        mapping_iter = iter(plan_mappings)
        current_mapping = None
        for line in sections:
            m = def_line.match(line)
            if m:
                entity_id = int(m.group(1))
                entity_type = m.group(2)
                if entity_type == mdgpr_type:
                    # Start of a MDGPR root; get next mapping
                    current_mapping = next(mapping_iter, None)
                    if current_mapping:
                        placeholders = current_mapping['placeholders']
                        # Replace the parameter list #numbers with corresponding #<ID_x> only for the immediate parameters
                        # We only replace tokens that are among placeholders
                        def repl_token(mtok):
                            nid = int(mtok.group(1))
                            if nid in placeholders:
                                return f"#{placeholders[nid]}"
                            return mtok.group(0)
                        line = re.sub(r'#(\d+)\b', repl_token, line)
                        updated.append(line)
                        continue
            updated.append(line)

        # Insert placeholder resolution annotations before each styled/complex block start
        # We search for the annotation lines that indicate start of each block and insert the mapping lines right after
        result: List[str] = []
        mapping_iter = iter(plan_mappings)
        current_mapping = None
        # Patterns to detect start annotations we already emit
        start_styled_re = re.compile(r'^/\* BEGIN STYLED_ITEM (\d+) \*/$')
        start_complex_re = re.compile(r'^/\* BEGIN COMPLEX_ENTITY (\d+) \*/$')
        # Track which placeholder keys have been emitted as we traverse; consume in order
        pending_keys: List[Tuple[int, str]] = []
        plan_index = -1
        for line in updated:
            # Advance mapping when we encounter an MDGPR definition line again
            m = def_line.match(line)
            if m and m.group(2) == mdgpr_type:
                current_mapping = next(mapping_iter, None)
                pending_keys = []
                if current_mapping:
                    pending_keys = list(current_mapping['list'])
            result.append(line)
            if start_styled_re.match(line) or start_complex_re.match(line):
                if pending_keys:
                    # Pop first placeholder mapping and insert annotation
                    new_id, key = pending_keys.pop(0)
                    old_id = new_to_old.get(new_id, None)
                    result.append(f"/* {key} = {new_id} */")

        return result


def main():
    parser = argparse.ArgumentParser(description='Restructure STEP file according to specific entity organization rules')
    parser.add_argument('step_file', help='Path to the STEP file to restructure')
    parser.add_argument('-o', '--output',
                        help='Output directory (default: ./restructured_output)')

    args = parser.parse_args()

    # Validate input file
    step_file_path = Path(args.step_file)
    if not step_file_path.exists():
        print(f"Error: STEP file not found: {step_file_path}")
        sys.exit(1)

    # Determine output path
    # Use model id (parent directory name) for output filename
    model_id = step_file_path.parent.name
    output_filename = f"{model_id}_restructured.step"

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
    else:
        # Default output directory (relative to current working directory)
        output_dir = Path('./restructured_output')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

    # Create restructurer and process file
    restructurer = StepRestructurer()

    try:
        section_count, entity_count = restructurer.restructure_step_file(step_file_path, output_path)
        print(f"\nRestructuring complete!")
        print(f"Output sections generated: {section_count}")
        print(f"Entities processed: {entity_count}")

    except Exception as e:
        print(f"Error restructuring STEP file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
