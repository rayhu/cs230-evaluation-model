#!/usr/bin/env python3
"""
Validation script for SciTSR table extraction outputs.

This script validates the extracted JSON files and compares them with ground truth.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_json_safe(json_path: Path) -> Tuple[bool, Any, str]:
    """
    Safely load a JSON file.
    
    Returns:
        Tuple of (success, data, error_message)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return True, data, ""
    except json.JSONDecodeError as e:
        return False, None, f"JSON decode error: {str(e)}"
    except Exception as e:
        return False, None, f"Error: {str(e)}"


def validate_scitsr_format(data: Dict) -> Tuple[bool, List[str]]:
    """
    Validate that data conforms to SciTSR format.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if not isinstance(data, dict):
        errors.append("Root is not a dictionary")
        return False, errors
    
    if 'cells' not in data:
        errors.append("Missing 'cells' key")
        return False, errors
    
    if not isinstance(data['cells'], list):
        errors.append("'cells' is not a list")
        return False, errors
    
    # Validate each cell
    required_keys = ['id', 'tex', 'content', 'start_row', 'end_row', 'start_col', 'end_col']
    
    for idx, cell in enumerate(data['cells']):
        if not isinstance(cell, dict):
            errors.append(f"Cell {idx} is not a dictionary")
            continue
        
        # Check required keys
        missing = [k for k in required_keys if k not in cell]
        if missing:
            errors.append(f"Cell {idx} missing keys: {missing}")
        
        # Check types
        if 'id' in cell and not isinstance(cell['id'], int):
            errors.append(f"Cell {idx}: 'id' should be int, got {type(cell['id'])}")
        
        if 'tex' in cell and not isinstance(cell['tex'], str):
            errors.append(f"Cell {idx}: 'tex' should be str, got {type(cell['tex'])}")
        
        if 'content' in cell and not isinstance(cell['content'], list):
            errors.append(f"Cell {idx}: 'content' should be list, got {type(cell['content'])}")
        
        # Check grid positions
        for key in ['start_row', 'end_row', 'start_col', 'end_col']:
            if key in cell:
                if not isinstance(cell[key], int):
                    errors.append(f"Cell {idx}: '{key}' should be int, got {type(cell[key])}")
                elif cell[key] < 0:
                    errors.append(f"Cell {idx}: '{key}' is negative: {cell[key]}")
        
        # Check span validity
        if all(k in cell for k in ['start_row', 'end_row']):
            if cell['start_row'] > cell['end_row']:
                errors.append(f"Cell {idx}: start_row > end_row ({cell['start_row']} > {cell['end_row']})")
        
        if all(k in cell for k in ['start_col', 'end_col']):
            if cell['start_col'] > cell['end_col']:
                errors.append(f"Cell {idx}: start_col > end_col ({cell['start_col']} > {cell['end_col']})")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def get_table_stats(data: Dict) -> Dict[str, Any]:
    """Get statistics about a table."""
    if 'cells' not in data:
        return {}
    
    cells = data['cells']
    
    if not cells:
        return {
            'num_cells': 0,
            'num_rows': 0,
            'num_cols': 0,
            'empty_cells': 0
        }
    
    max_row = max(cell.get('end_row', 0) for cell in cells)
    max_col = max(cell.get('end_col', 0) for cell in cells)
    empty_cells = sum(1 for cell in cells if not cell.get('content', []))
    
    total_words = sum(len(cell.get('content', [])) for cell in cells)
    
    return {
        'num_cells': len(cells),
        'num_rows': max_row + 1,
        'num_cols': max_col + 1,
        'empty_cells': empty_cells,
        'total_words': total_words,
        'avg_words_per_cell': total_words / len(cells) if cells else 0
    }


def compare_with_ground_truth(
    output_file: Path,
    gt_file: Path
) -> Dict[str, Any]:
    """
    Compare output with ground truth.
    
    Returns:
        Dictionary with comparison statistics
    """
    # Load files
    success_out, data_out, error_out = load_json_safe(output_file)
    success_gt, data_gt, error_gt = load_json_safe(gt_file)
    
    if not success_out or not success_gt:
        return {
            'error': f"Load failed - Out: {error_out}, GT: {error_gt}",
            'comparable': False
        }
    
    stats_out = get_table_stats(data_out)
    stats_gt = get_table_stats(data_gt)
    
    return {
        'comparable': True,
        'output_stats': stats_out,
        'gt_stats': stats_gt,
        'cell_count_diff': stats_out.get('num_cells', 0) - stats_gt.get('num_cells', 0),
        'row_count_diff': stats_out.get('num_rows', 0) - stats_gt.get('num_rows', 0),
        'col_count_diff': stats_out.get('num_cols', 0) - stats_gt.get('num_cols', 0),
    }


def validate_directory(
    output_dir: Path,
    gt_dir: Path = None
) -> Dict[str, Any]:
    """
    Validate all JSON files in a directory.
    
    Returns:
        Dictionary with validation summary
    """
    output_files = sorted(output_dir.glob('*.json'))
    
    # Exclude processing_stats.json if it exists
    output_files = [f for f in output_files if f.name != 'processing_stats.json']
    
    results = {
        'total_files': len(output_files),
        'valid_files': 0,
        'invalid_files': 0,
        'load_errors': 0,
        'format_errors': 0,
        'missing_in_gt': 0,
        'errors_by_file': {},
        'stats_summary': defaultdict(list)
    }
    
    for output_file in output_files:
        # Load and validate format
        success, data, error = load_json_safe(output_file)
        
        if not success:
            results['load_errors'] += 1
            results['invalid_files'] += 1
            results['errors_by_file'][output_file.name] = [error]
            continue
        
        is_valid, format_errors = validate_scitsr_format(data)
        
        if not is_valid:
            results['format_errors'] += 1
            results['invalid_files'] += 1
            results['errors_by_file'][output_file.name] = format_errors
            continue
        
        results['valid_files'] += 1
        
        # Get stats
        stats = get_table_stats(data)
        for key, value in stats.items():
            results['stats_summary'][key].append(value)
        
        # Compare with ground truth if available
        if gt_dir:
            gt_file = gt_dir / output_file.name
            if gt_file.exists():
                comparison = compare_with_ground_truth(output_file, gt_file)
                if comparison.get('comparable'):
                    for key in ['cell_count_diff', 'row_count_diff', 'col_count_diff']:
                        results['stats_summary'][key].append(comparison[key])
            else:
                results['missing_in_gt'] += 1
    
    # Calculate averages
    results['average_stats'] = {}
    for key, values in results['stats_summary'].items():
        if values:
            results['average_stats'][key] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values)
            }
    
    return results


def print_report(results: Dict[str, Any]) -> None:
    """Print validation report."""
    print("\n" + "="*70)
    print("VALIDATION REPORT")
    print("="*70)
    
    print(f"\nTotal files: {results['total_files']}")
    print(f"Valid files: {results['valid_files']} ({results['valid_files']/results['total_files']*100:.1f}%)")
    print(f"Invalid files: {results['invalid_files']} ({results['invalid_files']/results['total_files']*100:.1f}%)")
    
    if results['load_errors'] > 0:
        print(f"  - Load errors: {results['load_errors']}")
    if results['format_errors'] > 0:
        print(f"  - Format errors: {results['format_errors']}")
    if results['missing_in_gt'] > 0:
        print(f"  - Missing in ground truth: {results['missing_in_gt']}")
    
    # Print average statistics
    if results['average_stats']:
        print("\nAverage Statistics:")
        for key, stats in sorted(results['average_stats'].items()):
            print(f"  {key}:")
            print(f"    Mean: {stats['mean']:.2f}")
            print(f"    Range: [{stats['min']}, {stats['max']}]")
    
    # Print sample errors
    if results['errors_by_file']:
        print(f"\nSample Errors (showing first 5):")
        for filename, errors in list(results['errors_by_file'].items())[:5]:
            print(f"\n  {filename}:")
            for error in errors[:3]:  # Show max 3 errors per file
                print(f"    - {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate SciTSR table extraction outputs"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/SciTSR/test/json_output'),
        help='Directory containing output JSON files'
    )
    parser.add_argument(
        '--gt-dir',
        type=Path,
        default=Path('data/SciTSR/test/structure'),
        help='Directory containing ground truth JSON files'
    )
    parser.add_argument(
        '--no-gt',
        action='store_true',
        help='Do not compare with ground truth'
    )
    parser.add_argument(
        '--save-report',
        type=Path,
        help='Save detailed report to file'
    )
    
    args = parser.parse_args()
    
    if not args.output_dir.exists():
        print(f"Error: Output directory does not exist: {args.output_dir}")
        return 1
    
    gt_dir = None if args.no_gt else args.gt_dir
    
    if gt_dir and not gt_dir.exists():
        print(f"Warning: Ground truth directory does not exist: {gt_dir}")
        print("Proceeding without ground truth comparison...")
        gt_dir = None
    
    print(f"Validating files in: {args.output_dir}")
    if gt_dir:
        print(f"Comparing with ground truth in: {gt_dir}")
    
    results = validate_directory(args.output_dir, gt_dir)
    
    print_report(results)
    
    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_report, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed report saved to: {args.save_report}")
    
    print("\n" + "="*70)
    
    # Return exit code based on validation
    if results['invalid_files'] == 0:
        print("✅ All files are valid!")
        return 0
    else:
        print("⚠️  Some files have errors")
        return 1


if __name__ == '__main__':
    sys.exit(main())

