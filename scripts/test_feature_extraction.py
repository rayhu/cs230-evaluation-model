#!/usr/bin/env python3
"""
Quick test script to verify feature extraction works correctly.
"""

import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from utils.table_features import (
    extract_structure_features,
    extract_text_features,
    extract_all_features
)

# Test with sample table data
sample_table_list = [
    ["Name", "Age", "City"],
    ["John", "25", "NYC"],
    ["Jane", "30", "LA"]
]

sample_table_dict = {
    "n_rows": 3,
    "n_cols": 3,
    "cells": [
        {"r0": 0, "c0": 0, "row_span": 1, "col_span": 1, "content": ["Name"]},
        {"r0": 0, "c0": 1, "row_span": 1, "col_span": 1, "content": ["Age"]},
        {"r0": 0, "c0": 2, "row_span": 1, "col_span": 1, "content": ["City"]},
        {"r0": 1, "c0": 0, "row_span": 1, "col_span": 1, "content": ["John"]},
        {"r0": 1, "c0": 1, "row_span": 1, "col_span": 1, "content": ["25"]},
        {"r0": 1, "c0": 2, "row_span": 1, "col_span": 1, "content": ["NYC"]},
    ]
}

print("Testing feature extraction...")
print("=" * 60)

# Test structure features
print("\n1. Structure Features:")
struct_features_list = extract_structure_features(sample_table_list)
print(f"   List format: {struct_features_list.shape} features")
print(f"   Sample values: {struct_features_list[:10]}")

struct_features_dict = extract_structure_features(sample_table_dict)
print(f"   Dict format: {struct_features_dict.shape} features")
print(f"   Sample values: {struct_features_dict[:10]}")

# Test text features
print("\n2. Text Features:")
text_features_list = extract_text_features(sample_table_list)
print(f"   List format: {len(text_features_list)} features")
print(f"   Values: {text_features_list}")

text_features_dict = extract_text_features(sample_table_dict)
print(f"   Dict format: {len(text_features_dict)} features")
print(f"   Values: {text_features_dict}")

# Test hybrid features (without transformer for quick test)
print("\n3. Hybrid Features (without embeddings):")
try:
    hybrid_features = extract_all_features(
        sample_table_list,
        sentence_transformer=None,
        normalize_embeddings=False
    )
    print(f"   Shape: {hybrid_features.shape}")
    print(f"   Structure: {hybrid_features[:30].sum():.1f}")
    print(f"   Text: {hybrid_features[30:40].sum():.1f}")
    print(f"   Embeddings: {hybrid_features[40:].sum():.1f} (should be 0)")
    print("   ✅ Feature extraction works!")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Feature extraction test complete!")

