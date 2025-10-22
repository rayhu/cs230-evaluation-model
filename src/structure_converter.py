"""
Structure Converter: Transform table extraction outputs to SciTSR format.

This module converts bounding box-based table extraction results
(x, y, width, height) into the SciTSR grid-based format
(start_row, end_row, start_col, end_col).
"""

import json
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path

from utils.grid_detection import (
    detect_grid_structure,
    assign_cells_to_grid,
    validate_grid_structure
)


class SciTSRConverter:
    """
    Converter for transforming table extraction results to SciTSR format.
    
    SciTSR Format:
    {
        "cells": [
            {
                "id": int,
                "tex": str,  # LaTeX representation (optional)
                "content": [str],  # List of words/tokens in the cell
                "start_row": int,
                "end_row": int,
                "start_col": int,
                "end_col": int
            }
        ]
    }
    """
    
    def __init__(self, tolerance: float = 5.0, overlap_threshold: float = 0.5):
        """
        Initialize converter with detection parameters.
        
        Args:
            tolerance: Pixel tolerance for grouping grid boundaries (default: 5.0)
            overlap_threshold: Minimum overlap ratio for cell assignment (default: 0.5)
        """
        self.tolerance = tolerance
        self.overlap_threshold = overlap_threshold
    
    def convert(
        self,
        cells: List[Dict[str, Any]],
        image_width: Optional[int] = None,
        image_height: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert extracted table cells to SciTSR format.
        
        Args:
            cells: List of cell dictionaries with keys:
                   - bbox: (x, y, width, height) tuple
                   - text: str (cell text content)
            image_width: Optional image width for validation
            image_height: Optional image height for validation
        
        Returns:
            Dictionary with "cells" key containing SciTSR-formatted cells
        """
        if not cells:
            return {"cells": []}
        
        # Extract bounding boxes and texts
        bboxes = [cell['bbox'] for cell in cells]
        texts = [cell.get('text', '') for cell in cells]
        
        # Detect grid structure
        row_boundaries, col_boundaries = detect_grid_structure(
            bboxes, self.tolerance
        )
        
        if len(row_boundaries) < 2 or len(col_boundaries) < 2:
            # Cannot form valid grid
            return {"cells": []}
        
        # Assign cells to grid positions
        grid_positions = assign_cells_to_grid(
            bboxes, row_boundaries, col_boundaries, self.overlap_threshold
        )
        
        # Format as SciTSR cells
        scitsr_cells = []
        for idx, (grid_pos, text) in enumerate(zip(grid_positions, texts)):
            scitsr_cell = {
                "id": idx,
                "tex": "",  # LaTeX not available from extraction
                "content": self._split_text_to_words(text),
                "start_row": grid_pos['start_row'],
                "end_row": grid_pos['end_row'],
                "start_col": grid_pos['start_col'],
                "end_col": grid_pos['end_col']
            }
            scitsr_cells.append(scitsr_cell)
        
        return {"cells": scitsr_cells}
    
    def _split_text_to_words(self, text: str) -> List[str]:
        """
        Split cell text into words for content field.
        
        Args:
            text: Cell text content
        
        Returns:
            List of words/tokens
        """
        if not text or not text.strip():
            return []
        
        # Simple whitespace split
        words = text.strip().split()
        return words
    
    def convert_from_detections(
        self,
        detections: Dict[str, Any],
        ocr_results: List[Tuple[Tuple[int, int, int, int], str, float]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convert Table Transformer detections + OCR results to SciTSR format.
        
        Args:
            detections: Table Transformer detection results with bounding boxes
            ocr_results: EasyOCR results as list of (bbox, text, confidence)
                        where bbox is (x1, y1, x2, y2)
        
        Returns:
            Dictionary with "cells" key containing SciTSR-formatted cells
        """
        # Match OCR results to detected table cells
        cells = self._match_ocr_to_cells(detections, ocr_results)
        
        return self.convert(cells)
    
    def _match_ocr_to_cells(
        self,
        detections: Dict[str, Any],
        ocr_results: List[Tuple[Tuple[int, int, int, int], str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Match OCR text to detected table cells based on spatial overlap.
        
        Args:
            detections: Table cell detections with bounding boxes
            ocr_results: OCR text results with bounding boxes
        
        Returns:
            List of cells with matched text
        """
        cells = []
        
        # Extract cell bounding boxes from detections
        # Format depends on the model output structure
        cell_boxes = self._extract_cell_boxes(detections)
        
        for cell_box in cell_boxes:
            # Find all OCR results that overlap with this cell
            cell_text = self._aggregate_ocr_in_cell(cell_box, ocr_results)
            
            cells.append({
                'bbox': cell_box,
                'text': cell_text
            })
        
        return cells
    
    def _extract_cell_boxes(self, detections: Dict[str, Any]) -> List[Tuple[float, float, float, float]]:
        """
        Extract cell bounding boxes from Table Transformer detections.
        
        Args:
            detections: Detection results from Table Transformer
        
        Returns:
            List of (x, y, width, height) tuples
        """
        # This will depend on the actual structure of Table Transformer output
        # Placeholder implementation
        if 'boxes' in detections:
            boxes = detections['boxes']
            # Convert from (x1, y1, x2, y2) to (x, y, w, h) if needed
            return [(box[0], box[1], box[2]-box[0], box[3]-box[1]) for box in boxes]
        return []
    
    def _aggregate_ocr_in_cell(
        self,
        cell_box: Tuple[float, float, float, float],
        ocr_results: List[Tuple[Tuple[int, int, int, int], str, float]],
        iou_threshold: float = 0.3
    ) -> str:
        """
        Aggregate OCR text that falls within or overlaps a table cell.
        
        Args:
            cell_box: Cell bounding box (x, y, width, height)
            ocr_results: OCR results with bounding boxes and text
            iou_threshold: Minimum IoU for including OCR text
        
        Returns:
            Aggregated text from all overlapping OCR results
        """
        cell_x, cell_y, cell_w, cell_h = cell_box
        cell_x2 = cell_x + cell_w
        cell_y2 = cell_y + cell_h
        
        matched_texts = []
        
        for ocr_bbox, text, conf in ocr_results:
            # OCR bbox is typically (x1, y1, x2, y2) from EasyOCR
            # or list of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            if isinstance(ocr_bbox[0], (list, tuple)):
                # Corner points format - get bounding rectangle
                xs = [p[0] for p in ocr_bbox]
                ys = [p[1] for p in ocr_bbox]
                ocr_x1, ocr_y1 = min(xs), min(ys)
                ocr_x2, ocr_y2 = max(xs), max(ys)
            else:
                # Simple (x1, y1, x2, y2) format
                ocr_x1, ocr_y1, ocr_x2, ocr_y2 = ocr_bbox
            
            # Calculate IoU (Intersection over Union)
            intersect_x1 = max(cell_x, ocr_x1)
            intersect_y1 = max(cell_y, ocr_y1)
            intersect_x2 = min(cell_x2, ocr_x2)
            intersect_y2 = min(cell_y2, ocr_y2)
            
            if intersect_x2 > intersect_x1 and intersect_y2 > intersect_y1:
                intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
                ocr_area = (ocr_x2 - ocr_x1) * (ocr_y2 - ocr_y1)
                cell_area = cell_w * cell_h
                
                # Use IoU relative to smaller box
                iou = intersect_area / min(ocr_area, cell_area) if min(ocr_area, cell_area) > 0 else 0
                
                if iou >= iou_threshold:
                    matched_texts.append((ocr_y1, ocr_x1, text))  # Include position for sorting
        
        # Sort by position (top to bottom, left to right)
        matched_texts.sort(key=lambda x: (x[0], x[1]))
        
        # Join texts with space
        return ' '.join([text for _, _, text in matched_texts])
    
    def save_to_file(self, scitsr_data: Dict[str, Any], output_path: Path) -> None:
        """
        Save SciTSR formatted data to JSON file.
        
        Args:
            scitsr_data: SciTSR formatted data
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scitsr_data, f, indent=2, ensure_ascii=False)
    
    def validate_output(
        self,
        scitsr_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate SciTSR formatted output.
        
        Args:
            scitsr_data: SciTSR formatted data to validate
        
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        if 'cells' not in scitsr_data:
            return False, ["Missing 'cells' key"]
        
        cells = scitsr_data['cells']
        
        if not cells:
            warnings.append("No cells found in output")
            return True, warnings  # Empty is valid but warn
        
        # Find grid dimensions
        max_row = max(cell['end_row'] for cell in cells)
        max_col = max(cell['end_col'] for cell in cells)
        
        # Validate each cell
        for cell in cells:
            required_keys = ['id', 'tex', 'content', 'start_row', 'end_row', 'start_col', 'end_col']
            missing_keys = [k for k in required_keys if k not in cell]
            if missing_keys:
                warnings.append(f"Cell {cell.get('id', '?')} missing keys: {missing_keys}")
        
        # Validate grid structure
        grid_positions = [
            {
                'start_row': cell['start_row'],
                'end_row': cell['end_row'],
                'start_col': cell['start_col'],
                'end_col': cell['end_col']
            }
            for cell in cells
        ]
        
        grid_warnings = validate_grid_structure(grid_positions, max_row + 1, max_col + 1)
        warnings.extend(grid_warnings)
        
        is_valid = len([w for w in warnings if 'missing keys' in w.lower()]) == 0
        
        return is_valid, warnings

