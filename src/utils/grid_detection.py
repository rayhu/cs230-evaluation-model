"""
Grid detection utilities for converting bounding boxes to logical grid positions.

This module provides functions to:
1. Detect row and column boundaries from cell bounding boxes
2. Map cells to their logical grid positions (row/col indices)
3. Handle merged cells that span multiple rows or columns
"""

import numpy as np
from typing import List, Dict, Tuple, Any


def detect_grid_structure(
    bboxes: List[Tuple[float, float, float, float]], 
    tolerance: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect row and column boundaries from cell bounding boxes.
    
    Args:
        bboxes: List of (x, y, width, height) tuples for each cell
        tolerance: Pixel tolerance for grouping similar positions (default: 5.0)
    
    Returns:
        Tuple of (row_boundaries, col_boundaries) as sorted numpy arrays
        - row_boundaries: Array of y-coordinates for row separations
        - col_boundaries: Array of x-coordinates for column separations
    """
    if not bboxes:
        return np.array([]), np.array([])
    
    # Extract y-coordinates (top and bottom) for rows
    y_coords = []
    for x, y, w, h in bboxes:
        y_coords.append(y)  # Top edge
        y_coords.append(y + h)  # Bottom edge
    
    # Extract x-coordinates (left and right) for columns
    x_coords = []
    for x, y, w, h in bboxes:
        x_coords.append(x)  # Left edge
        x_coords.append(x + w)  # Right edge
    
    # Cluster similar coordinates together
    row_boundaries = _cluster_coordinates(y_coords, tolerance)
    col_boundaries = _cluster_coordinates(x_coords, tolerance)
    
    return row_boundaries, col_boundaries


def _cluster_coordinates(coords: List[float], tolerance: float) -> np.ndarray:
    """
    Group similar coordinates within tolerance range.
    
    Args:
        coords: List of coordinate values to cluster
        tolerance: Maximum distance between coordinates in same cluster
    
    Returns:
        Sorted array of representative coordinates (mean of each cluster)
    """
    if not coords:
        return np.array([])
    
    sorted_coords = np.sort(coords)
    clusters = []
    current_cluster = [sorted_coords[0]]
    
    for coord in sorted_coords[1:]:
        if coord - current_cluster[-1] <= tolerance:
            current_cluster.append(coord)
        else:
            # Finish current cluster, start new one
            clusters.append(np.mean(current_cluster))
            current_cluster = [coord]
    
    # Add the last cluster
    if current_cluster:
        clusters.append(np.mean(current_cluster))
    
    return np.array(clusters)


def assign_cells_to_grid(
    bboxes: List[Tuple[float, float, float, float]],
    row_boundaries: np.ndarray,
    col_boundaries: np.ndarray,
    overlap_threshold: float = 0.5
) -> List[Dict[str, int]]:
    """
    Assign each cell to its grid position based on bounding boxes and grid boundaries.
    
    Args:
        bboxes: List of (x, y, width, height) tuples for each cell
        row_boundaries: Array of y-coordinates defining row separations
        col_boundaries: Array of x-coordinates defining column separations
        overlap_threshold: Minimum overlap ratio to assign cell to row/col (default: 0.5)
    
    Returns:
        List of dictionaries with keys: start_row, end_row, start_col, end_col
    """
    grid_positions = []
    
    for x, y, w, h in bboxes:
        # Calculate cell boundaries
        cell_left = x
        cell_right = x + w
        cell_top = y
        cell_bottom = y + h
        
        # Find which rows this cell spans
        start_row, end_row = _find_span(
            cell_top, cell_bottom, row_boundaries, overlap_threshold, is_vertical=True
        )
        
        # Find which columns this cell spans
        start_col, end_col = _find_span(
            cell_left, cell_right, col_boundaries, overlap_threshold, is_vertical=False
        )
        
        grid_positions.append({
            'start_row': start_row,
            'end_row': end_row,
            'start_col': start_col,
            'end_col': end_col
        })
    
    return grid_positions


def _find_span(
    start: float,
    end: float,
    boundaries: np.ndarray,
    overlap_threshold: float,
    is_vertical: bool = True
) -> Tuple[int, int]:
    """
    Find which grid cells (rows or columns) a bounding box spans.
    
    Args:
        start: Starting coordinate (top or left edge)
        end: Ending coordinate (bottom or right edge)
        boundaries: Array of grid boundaries (sorted)
        overlap_threshold: Minimum overlap ratio to include a cell
        is_vertical: True for rows (y-axis), False for columns (x-axis)
    
    Returns:
        Tuple of (start_index, end_index) for the span
    """
    if len(boundaries) < 2:
        return 0, 0
    
    cell_length = end - start
    start_idx = None
    end_idx = None
    
    # Check each grid cell for overlap
    for i in range(len(boundaries) - 1):
        grid_start = boundaries[i]
        grid_end = boundaries[i + 1]
        
        # Calculate overlap between bbox and this grid cell
        overlap_start = max(start, grid_start)
        overlap_end = min(end, grid_end)
        overlap = max(0, overlap_end - overlap_start)
        
        # Check if overlap is significant
        grid_cell_length = grid_end - grid_start
        overlap_ratio = overlap / min(cell_length, grid_cell_length) if min(cell_length, grid_cell_length) > 0 else 0
        
        if overlap_ratio >= overlap_threshold:
            if start_idx is None:
                start_idx = i
            end_idx = i
    
    # Default to 0,0 if no overlap found
    if start_idx is None:
        start_idx = 0
        end_idx = 0
    
    return start_idx, end_idx


def validate_grid_structure(
    grid_positions: List[Dict[str, int]],
    num_rows: int,
    num_cols: int
) -> List[str]:
    """
    Validate grid structure and return list of warnings/errors.
    
    Args:
        grid_positions: List of cell grid positions
        num_rows: Expected number of rows
        num_cols: Expected number of columns
    
    Returns:
        List of warning/error messages (empty if valid)
    """
    warnings = []
    
    for idx, pos in enumerate(grid_positions):
        # Check bounds
        if pos['start_row'] < 0 or pos['end_row'] >= num_rows:
            warnings.append(f"Cell {idx}: Row out of bounds ({pos['start_row']}-{pos['end_row']})")
        
        if pos['start_col'] < 0 or pos['end_col'] >= num_cols:
            warnings.append(f"Cell {idx}: Column out of bounds ({pos['start_col']}-{pos['end_col']})")
        
        # Check valid spans
        if pos['start_row'] > pos['end_row']:
            warnings.append(f"Cell {idx}: Invalid row span ({pos['start_row']} > {pos['end_row']})")
        
        if pos['start_col'] > pos['end_col']:
            warnings.append(f"Cell {idx}: Invalid column span ({pos['start_col']} > {pos['end_col']})")
    
    return warnings

