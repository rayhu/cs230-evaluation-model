#!/usr/bin/env python3
"""
SciTSR Table Extraction Pipeline

This script extracts table structures from SciTSR dataset images using:
- Table Transformer (TATR) for structure detection
- EasyOCR for text recognition
- Custom converter for SciTSR format output

Usage:
    python scripts/extract_tables_scitsr.py --input data/SciTSR/test/img --output data/SciTSR/test/json_output
    python scripts/extract_tables_scitsr.py --single data/SciTSR/test/img/0704.1068v2.1.png
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from PIL import Image
import easyocr
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from tqdm import tqdm

from structure_converter import SciTSRConverter


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TableExtractor:
    """Table extraction pipeline using Table Transformer + EasyOCR."""
    
    def __init__(
        self,
        device: str = None,
        detection_threshold: float = 0.7,
        lang_list: List[str] = ['en']
    ):
        """
        Initialize the table extractor.
        
        Args:
            device: Device to run models on ('cuda', 'mps', or 'cpu'). If None, auto-detect.
            detection_threshold: Confidence threshold for detections
            lang_list: Languages for EasyOCR (default: English)
        """
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        self.detection_threshold = detection_threshold
        
        logger.info(f"Initializing Table Extractor on device: {device}")
        
        # Load Table Transformer model
        logger.info("Loading Table Transformer model...")
        self.model_name = "microsoft/table-transformer-structure-recognition"
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = TableTransformerForObjectDetection.from_pretrained(self.model_name)
        self.model.to(device)
        self.model.eval()
        
        # Initialize EasyOCR
        logger.info(f"Initializing EasyOCR for languages: {lang_list}")
        # EasyOCR only supports CUDA, not MPS
        use_gpu = (device == 'cuda')
        self.reader = easyocr.Reader(lang_list, gpu=use_gpu)
        
        # Initialize converter
        self.converter = SciTSRConverter(tolerance=5.0, overlap_threshold=0.5)
        
        logger.info("Initialization complete!")
    
    def extract_table(
        self,
        image_path: Path,
        visualize: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract table structure and content from an image.
        
        Args:
            image_path: Path to input image
            visualize: Whether to return visualization data
        
        Returns:
            Tuple of (scitsr_output, metadata)
        """
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            image_width, image_height = image.size
            
            # Detect table structure
            structure_detections = self._detect_structure(image)
            
            # Perform OCR
            ocr_results = self._perform_ocr(image)
            
            # Extract cells from structure
            cells = self._extract_cells(structure_detections, ocr_results, image_width, image_height)
            
            # Convert to SciTSR format
            scitsr_output = self.converter.convert(cells, image_width, image_height)
            
            # Validate output
            is_valid, warnings = self.converter.validate_output(scitsr_output)
            
            metadata = {
                'image_path': str(image_path),
                'image_size': (image_width, image_height),
                'num_cells': len(cells),
                'num_detections': len(structure_detections.get('boxes', [])),
                'num_ocr_results': len(ocr_results),
                'is_valid': is_valid,
                'warnings': warnings
            }
            
            return scitsr_output, metadata
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            raise
    
    def _detect_structure(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect table structure using Table Transformer.
        
        Args:
            image: PIL Image
        
        Returns:
            Dictionary with detection results
        """
        # Prepare image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process detections
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.detection_threshold,
            target_sizes=target_sizes
        )[0]
        
        # Convert to CPU and numpy
        detections = {
            'boxes': results['boxes'].cpu().numpy(),
            'labels': results['labels'].cpu().numpy(),
            'scores': results['scores'].cpu().numpy()
        }
        
        # Filter for table cells only
        # Label mapping: check model.config.id2label
        cell_label_ids = self._get_cell_label_ids()
        cell_mask = torch.isin(results['labels'], torch.tensor(cell_label_ids).to(self.device))
        
        if cell_mask.any():
            detections['boxes'] = results['boxes'][cell_mask].cpu().numpy()
            detections['labels'] = results['labels'][cell_mask].cpu().numpy()
            detections['scores'] = results['scores'][cell_mask].cpu().numpy()
        
        return detections
    
    def _get_cell_label_ids(self) -> List[int]:
        """
        Get label IDs for table cells from the model.
        
        Returns:
            List of label IDs corresponding to table cells
        """
        # Table Transformer structure recognition labels
        # Common labels: table, table column, table row, table column header, table projected row header, table spanning cell
        id2label = self.model.config.id2label
        
        # We want cells, columns, rows, and spanning cells
        cell_labels = ['table', 'table column', 'table row', 'table spanning cell']
        
        cell_label_ids = [
            label_id for label_id, label_name in id2label.items()
            if any(cell_label in label_name.lower() for cell_label in cell_labels)
        ]
        
        return cell_label_ids if cell_label_ids else list(range(len(id2label)))
    
    def _perform_ocr(self, image: Image.Image) -> List[Tuple[List, str, float]]:
        """
        Perform OCR on the image.
        
        Args:
            image: PIL Image
        
        Returns:
            List of (bbox, text, confidence) tuples
        """
        # EasyOCR expects numpy array or path
        import numpy as np
        image_array = np.array(image)
        
        results = self.reader.readtext(image_array)
        
        return results
    
    def _extract_cells(
        self,
        structure_detections: Dict[str, Any],
        ocr_results: List[Tuple[List, str, float]],
        image_width: int,
        image_height: int
    ) -> List[Dict[str, Any]]:
        """
        Extract cell information by matching structure detections with OCR.
        
        Args:
            structure_detections: Structure detection results
            ocr_results: OCR results
            image_width: Image width
            image_height: Image height
        
        Returns:
            List of cell dictionaries with bbox and text
        """
        cells = []
        
        boxes = structure_detections.get('boxes', [])
        
        if len(boxes) == 0:
            # No structure detected, try to infer from OCR
            logger.warning("No table structure detected, attempting to infer from OCR")
            return self._infer_cells_from_ocr(ocr_results)
        
        for box in boxes:
            # Box is in [x1, y1, x2, y2] format
            x1, y1, x2, y2 = box
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            
            # Find OCR text within this cell
            cell_text = self._get_text_in_box(box, ocr_results)
            
            cells.append({
                'bbox': (float(x), float(y), float(w), float(h)),
                'text': cell_text
            })
        
        return cells
    
    def _get_text_in_box(
        self,
        box: List[float],
        ocr_results: List[Tuple[List, str, float]],
        iou_threshold: float = 0.3
    ) -> str:
        """
        Get all OCR text within or overlapping a bounding box.
        
        Args:
            box: Bounding box [x1, y1, x2, y2]
            ocr_results: OCR results
            iou_threshold: Minimum IoU for inclusion
        
        Returns:
            Aggregated text
        """
        x1, y1, x2, y2 = box
        matched_texts = []
        
        for ocr_bbox, text, conf in ocr_results:
            # OCR bbox is list of 4 corner points
            xs = [p[0] for p in ocr_bbox]
            ys = [p[1] for p in ocr_bbox]
            ocr_x1, ocr_y1 = min(xs), min(ys)
            ocr_x2, ocr_y2 = max(xs), max(ys)
            
            # Calculate IoU
            intersect_x1 = max(x1, ocr_x1)
            intersect_y1 = max(y1, ocr_y1)
            intersect_x2 = min(x2, ocr_x2)
            intersect_y2 = min(y2, ocr_y2)
            
            if intersect_x2 > intersect_x1 and intersect_y2 > intersect_y1:
                intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
                box_area = (x2 - x1) * (y2 - y1)
                ocr_area = (ocr_x2 - ocr_x1) * (ocr_y2 - ocr_y1)
                
                iou = intersect_area / min(box_area, ocr_area) if min(box_area, ocr_area) > 0 else 0
                
                if iou >= iou_threshold:
                    matched_texts.append((ocr_y1, ocr_x1, text))
        
        # Sort by position
        matched_texts.sort(key=lambda x: (x[0], x[1]))
        
        return ' '.join([text for _, _, text in matched_texts])
    
    def _infer_cells_from_ocr(
        self,
        ocr_results: List[Tuple[List, str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Infer cell structure from OCR results when structure detection fails.
        
        Args:
            ocr_results: OCR results
        
        Returns:
            List of inferred cells
        """
        cells = []
        
        for ocr_bbox, text, conf in ocr_results:
            # Convert OCR bbox to (x, y, w, h)
            xs = [p[0] for p in ocr_bbox]
            ys = [p[1] for p in ocr_bbox]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            
            cells.append({
                'bbox': (float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                'text': text
            })
        
        return cells


def process_single_image(
    image_path: Path,
    output_path: Optional[Path] = None,
    extractor: Optional[TableExtractor] = None
) -> None:
    """
    Process a single image and optionally save output.
    
    Args:
        image_path: Path to input image
        output_path: Path to output JSON file (optional)
        extractor: TableExtractor instance (will create if None)
    """
    if extractor is None:
        extractor = TableExtractor()
    
    logger.info(f"Processing: {image_path}")
    start_time = time.time()
    
    try:
        scitsr_output, metadata = extractor.extract_table(image_path)
        
        elapsed = time.time() - start_time
        logger.info(f"Completed in {elapsed:.2f}s")
        logger.info(f"Detected {metadata['num_cells']} cells")
        
        if metadata['warnings']:
            logger.warning(f"Warnings: {metadata['warnings']}")
        
        if output_path:
            extractor.converter.save_to_file(scitsr_output, output_path)
            logger.info(f"Saved to: {output_path}")
        else:
            # Print to console
            print(json.dumps(scitsr_output, indent=2))
    
    except Exception as e:
        logger.error(f"Failed to process {image_path}: {str(e)}")
        raise


def process_batch(
    input_dir: Path,
    output_dir: Path,
    limit: Optional[int] = None,
    resume: bool = True
) -> Dict[str, Any]:
    """
    Process a batch of images from input directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory for output JSON files
        limit: Maximum number of images to process (None for all)
        resume: Skip already processed files
    
    Returns:
        Statistics dictionary
    """
    # Find all PNG images
    image_files = sorted(input_dir.glob('*.png'))
    
    if not image_files:
        logger.error(f"No PNG images found in {input_dir}")
        return {}
    
    if limit:
        image_files = image_files[:limit]
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor once
    extractor = TableExtractor()
    
    # Statistics
    stats = {
        'total': len(image_files),
        'processed': 0,
        'failed': 0,
        'skipped': 0,
        'total_time': 0,
        'errors': []
    }
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        # Determine output path
        output_filename = image_path.stem + '.json'
        output_path = output_dir / output_filename
        
        # Skip if already exists and resume=True
        if resume and output_path.exists():
            stats['skipped'] += 1
            continue
        
        try:
            start_time = time.time()
            scitsr_output, metadata = extractor.extract_table(image_path)
            elapsed = time.time() - start_time
            
            # Save output
            extractor.converter.save_to_file(scitsr_output, output_path)
            
            stats['processed'] += 1
            stats['total_time'] += elapsed
            
        except Exception as e:
            stats['failed'] += 1
            stats['errors'].append({
                'file': str(image_path),
                'error': str(e)
            })
            logger.error(f"Failed to process {image_path}: {str(e)}")
    
    # Log statistics
    logger.info("\n" + "="*60)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total files: {stats['total']}")
    logger.info(f"Processed: {stats['processed']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Skipped: {stats['skipped']}")
    if stats['processed'] > 0:
        avg_time = stats['total_time'] / stats['processed']
        logger.info(f"Average time per image: {avg_time:.2f}s")
        logger.info(f"Total processing time: {stats['total_time']/60:.2f} minutes")
    
    if stats['errors']:
        logger.warning(f"\nErrors encountered: {len(stats['errors'])}")
        for error in stats['errors'][:10]:  # Show first 10
            logger.warning(f"  {error['file']}: {error['error']}")
    
    # Save statistics
    stats_path = output_dir / 'processing_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"\nStatistics saved to: {stats_path}")
    
    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract table structures from SciTSR images"
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/SciTSR/test/img'),
        help='Input directory containing images'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/SciTSR/test/json_output'),
        help='Output directory for JSON files'
    )
    parser.add_argument(
        '--single',
        type=Path,
        help='Process a single image instead of batch'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of images to process'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Reprocess all files (do not skip existing)'
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'mps', 'cpu', 'auto'],
        default='auto',
        help='Device to use for inference (auto detects best available)'
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Single image processing
    if args.single:
        output_path = args.output / (args.single.stem + '.json') if args.output else None
        extractor = TableExtractor(device=device)
        process_single_image(args.single, output_path, extractor)
    
    # Batch processing
    else:
        process_batch(
            args.input,
            args.output,
            limit=args.limit,
            resume=not args.no_resume
        )


if __name__ == '__main__':
    main()

