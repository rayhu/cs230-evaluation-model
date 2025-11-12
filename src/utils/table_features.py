#!/usr/bin/env python3
"""
Table feature extraction utilities.

Extracts both structural and semantic features from table JSON data
to improve model performance.
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple
import re
from collections import Counter


def extract_structure_features(table_json: Any) -> np.ndarray:
    """
    Extract structural features from table JSON.
    
    These features capture table geometry, cell relationships, and structure
    without relying on text content.
    
    Args:
        table_json: Table data (dict or JSON string)
    
    Returns:
        numpy array of structural features (1D, ~30 features)
    """
    if isinstance(table_json, str):
        table_data = json.loads(table_json)
    else:
        table_data = table_json
    
    features = []
    
    # Basic dimensions
    if isinstance(table_data, list):
        # Handle list-of-lists format
        n_rows = len(table_data)
        n_cols = max(len(row) for row in table_data) if table_data else 0
        total_cells = sum(len(row) for row in table_data)
        
        features.extend([
            n_rows,
            n_cols,
            total_cells,
            n_rows * n_cols,  # Expected cells
            total_cells / max(n_rows * n_cols, 1),  # Cell density
        ])
        
        # Row/column statistics
        row_lengths = [len(row) for row in table_data]
        features.extend([
            np.mean(row_lengths) if row_lengths else 0,
            np.std(row_lengths) if len(row_lengths) > 1 else 0,
            min(row_lengths) if row_lengths else 0,
            max(row_lengths) if row_lengths else 0,
        ])
        
        # Cell content statistics (text-based)
        all_text_lengths = []
        empty_cells = 0
        for row in table_data:
            for cell in row:
                if isinstance(cell, (list, tuple)):
                    cell_text = ' '.join(str(x) for x in cell)
                else:
                    cell_text = str(cell)
                text_len = len(cell_text.strip())
                all_text_lengths.append(text_len)
                if text_len == 0:
                    empty_cells += 1
        
        features.extend([
            np.mean(all_text_lengths) if all_text_lengths else 0,
            np.std(all_text_lengths) if len(all_text_lengths) > 1 else 0,
            empty_cells,
            empty_cells / max(total_cells, 1),  # Empty cell ratio
        ])
        
    elif isinstance(table_data, dict):
        # Handle SciTSR-style format with cells
        cells = table_data.get('cells', [])
        n_rows = table_data.get('n_rows', 0)
        n_cols = table_data.get('n_cols', 0)
        
        features.extend([
            n_rows,
            n_cols,
            len(cells),
            n_rows * n_cols,
            len(cells) / max(n_rows * n_cols, 1),
        ])
        
        # Cell span statistics
        row_spans = []
        col_spans = []
        cell_areas = []
        
        for cell in cells:
            row_span = cell.get('row_span', 1)
            col_span = cell.get('col_span', 1)
            row_spans.append(row_span)
            col_spans.append(col_span)
            cell_areas.append(row_span * col_span)
        
        features.extend([
            np.mean(row_spans) if row_spans else 1,
            np.std(row_spans) if len(row_spans) > 1 else 0,
            np.mean(col_spans) if col_spans else 1,
            np.std(col_spans) if len(col_spans) > 1 else 0,
            np.mean(cell_areas) if cell_areas else 1,
            np.std(cell_areas) if len(cell_areas) > 1 else 0,
            sum(1 for s in row_spans if s > 1),  # Merged rows
            sum(1 for s in col_spans if s > 1),  # Merged cols
        ])
        
        # Content statistics
        all_text_lengths = []
        empty_cells = 0
        for cell in cells:
            content = cell.get('content', [])
            if isinstance(content, list):
                cell_text = ' '.join(str(x) for x in content)
            else:
                cell_text = str(content)
            text_len = len(cell_text.strip())
            all_text_lengths.append(text_len)
            if text_len == 0:
                empty_cells += 1
        
        features.extend([
            np.mean(all_text_lengths) if all_text_lengths else 0,
            np.std(all_text_lengths) if len(all_text_lengths) > 1 else 0,
            empty_cells,
            empty_cells / max(len(cells), 1),
        ])
    else:
        # Fallback: return zeros
        return np.zeros(30)
    
    # Pad or truncate to fixed size
    target_size = 30
    if len(features) < target_size:
        features.extend([0] * (target_size - len(features)))
    else:
        features = features[:target_size]
    
    return np.array(features, dtype=np.float32)


def extract_text_features(table_json: Any) -> Dict[str, float]:
    """
    Extract text-based statistical features.
    
    Args:
        table_json: Table data (dict or JSON string)
    
    Returns:
        Dictionary of text features
    """
    if isinstance(table_json, str):
        table_data = json.loads(table_json)
    else:
        table_data = table_json
    
    all_text = []
    
    if isinstance(table_data, list):
        for row in table_data:
            for cell in row:
                if isinstance(cell, (list, tuple)):
                    all_text.extend(str(x) for x in cell)
                else:
                    all_text.append(str(cell))
    elif isinstance(table_data, dict):
        for cell in table_data.get('cells', []):
            content = cell.get('content', [])
            if isinstance(content, list):
                all_text.extend(str(x) for x in content)
            else:
                all_text.append(str(content))
    
    # Combine all text
    full_text = ' '.join(all_text).lower()
    
    # Extract features
    words = re.findall(r'\b\w+\b', full_text)
    chars = list(full_text)
    
    features = {
        'total_chars': len(chars),
        'total_words': len(words),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'unique_words': len(set(words)) if words else 0,
        'word_diversity': len(set(words)) / max(len(words), 1),
        'numeric_count': sum(1 for w in words if w.isdigit()),
        'numeric_ratio': sum(1 for w in words if w.isdigit()) / max(len(words), 1),
        'uppercase_count': sum(1 for c in chars if c.isupper()),
        'punctuation_count': sum(1 for c in chars if c in '.,;:!?()[]{}'),
    }
    
    return features


def extract_hybrid_features(
    table_json: Any,
    sentence_transformer=None,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract hybrid features combining structure and semantic embeddings.
    
    Args:
        table_json: Table data (dict or JSON string)
        sentence_transformer: SentenceTransformer model (optional)
        normalize: Whether to normalize embeddings
    
    Returns:
        Tuple of (structure_features, embedding_features)
    """
    # Extract structure features
    struct_features = extract_structure_features(table_json)
    
    # Extract semantic embeddings if model provided
    if sentence_transformer is not None:
        if isinstance(table_json, str):
            text = table_json
        else:
            text = json.dumps(table_json)
        
        embedding = sentence_transformer.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
    else:
        embedding = np.array([])
    
    return struct_features, embedding


def extract_all_features(
    table_json: Any,
    sentence_transformer=None,
    normalize_embeddings: bool = False
) -> np.ndarray:
    """
    Extract all features (structure + text stats + embeddings) and concatenate.
    
    Args:
        table_json: Table data
        sentence_transformer: Optional SentenceTransformer model
        normalize_embeddings: Whether to normalize embeddings
    
    Returns:
        Combined feature vector
    """
    struct_features = extract_structure_features(table_json)
    text_features_dict = extract_text_features(table_json)
    text_features = np.array(list(text_features_dict.values()), dtype=np.float32)
    
    if sentence_transformer is not None:
        if isinstance(table_json, str):
            text = table_json
        else:
            text = json.dumps(table_json)
        
        embedding = sentence_transformer.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings
        )
    else:
        embedding = np.array([])
    
    # Concatenate all features
    all_features = np.concatenate([
        struct_features,
        text_features,
        embedding
    ])
    
    return all_features

