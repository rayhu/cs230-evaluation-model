# Table Structure Evaluation — Model Design Proposal

## 1. Overview

**Goal:**  
To evaluate how similar a predicted table JSON is to the ground-truth **structure**, while **ignoring cell text**.  
This ensures robustness to unseen data where the textual content inside cells is irrelevant.

**Scope:**  
- Input: Two JSON files — Ground Truth (GT) and Predicted table.  
- Output: A similarity score in [0, 1], plus detailed structural metrics.  
- Focus: Table **structure only** (rows, columns, merged cells, adjacency).

---

## 2. JSON Schema (Structure-Only)

Each table JSON should include only structural fields:

```json
{
  "n_rows": 5,
  "n_cols": 6,
  "cells": [
    {"r0":0, "c0":0, "row_span":1, "col_span":2},
    {"r0":0, "c0":2, "row_span":1, "col_span":1}
  ]
}

Derived fields:
	•	r1 = r0 + row_span
	•	c1 = c0 + col_span
	•	A cell rectangle is represented as (r0, c0, r1, c1).

⸻

3. Structural Similarity Metrics

3.1 Cell-Level IoU Matching F1 (Main Metric)
	•	Treat each cell as a rectangle (r0, c0, r1, c1).
	•	Compute IoU (Intersection over Union) between predicted and ground-truth cells.
	•	Consider a match when IoU ≥ 0.5.
	•	Use Hungarian matching on the cost matrix 1 - IoU to find optimal matches.
	•	Compute Precision, Recall, and F1.

[
F1 = \frac{2PR}{P + R}
]

⸻

3.2 Grid-Level Accuracy
	•	Rasterize both tables into an n_rows × n_cols atomic grid.
	•	Each grid cell holds an integer ID representing its merged cell membership.
	•	Compute the proportion of matching atomic cells between GT and prediction.

[
Accuracy = \frac{\text{Number of matching grid cells}}{\text{Total grid cells}}
]

⸻

3.3 TEDS-Structure (Tree Edit Distance Similarity)
	•	Convert both JSONs to structure-only trees (Table → Rows → Cells).
	•	Compute normalized tree edit distance ignoring content fields.
	•	Resulting similarity ∈ [0,1].

⸻

3.4 Final Combined Score

Weighted combination of all metrics:

[
Score = \alpha \cdot F1_{IoU} + \beta \cdot Acc_{Grid} + \gamma \cdot TEDS_{Struct}
]

Default weights:
α = 0.5, β = 0.3, γ = 0.2.

⸻

4. Algorithm Design

Step 1: Normalize JSON

def normalize(table_json):
    n_rows, n_cols = table_json["n_rows"], table_json["n_cols"]
    rects = []
    for cell in table_json["cells"]:
        r0, c0 = cell["r0"], cell["c0"]
        r1, c1 = r0 + cell["row_span"], c0 + cell["col_span"]
        rects.append((r0, c0, r1, c1))
    return n_rows, n_cols, rects


⸻

Step 2: IoU Matching and F1

from scipy.optimize import linear_sum_assignment
import numpy as np

def iou_rect(a, b):
    r0, c0 = max(a[0], b[0]), max(a[1], b[1])
    r1, c1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, r1-r0) * max(0, c1-c0)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0

def cell_f1(gt_rects, pred_rects, tau=0.5, big=1e6):
    if not gt_rects and not pred_rects: return 1,1,1
    if not gt_rects: return 0,1,0
    if not pred_rects: return 1,0,0

    C = np.zeros((len(gt_rects), len(pred_rects)))
    for i,g in enumerate(gt_rects):
        for j,p in enumerate(pred_rects):
            iou = iou_rect(g,p)
            C[i,j] = 1 - iou if iou >= tau else big

    rows, cols = linear_sum_assignment(C)
    matches = sum(1 for i,j in zip(rows, cols) if C[i,j] < big)
    P = matches / max(1, len(pred_rects))
    R = matches / max(1, len(gt_rects))
    F1 = 2*P*R / (P+R+1e-12)
    return P, R, F1


⸻

Step 3: Grid Accuracy

def rasterize(n_rows, n_cols, rects):
    grid = [[-1]*n_cols for _ in range(n_rows)]
    for idx, (r0,c0,r1,c1) in enumerate(rects):
        for r in range(r0, r1):
            for c in range(c0, c1):
                grid[r][c] = idx
    return grid

def grid_acc(gt_nrows, gt_ncols, gt_rects, pred_nrows, pred_ncols, pred_rects):
    gt_grid   = rasterize(gt_nrows, gt_ncols, gt_rects)
    pred_grid = rasterize(gt_nrows, gt_ncols, pred_rects)
    total = gt_nrows * gt_ncols
    correct = sum(1 for r in range(gt_nrows) for c in range(gt_ncols)
                  if gt_grid[r][c] == pred_grid[r][c])
    return correct / total if total else 1.0


⸻

Step 4: Combine Metrics

def structural_score(gt, pred, alpha=0.5, beta=0.3, gamma=0.2, tau=0.5):
    gnr, gnc, grects = normalize(gt)
    pnr, pnc, prects = normalize(pred)
    P, R, F1 = cell_f1(grects, prects, tau=tau)
    grid = grid_acc(gnr, gnc, grects, pnr, pnc, prects)
    teds = teds_structure(gt, pred)  # placeholder, optional
    return {
        "precision_cell": P,
        "recall_cell": R,
        "f1_cell": F1,
        "grid_acc": grid,
        "teds_struct": teds,
        "final_score": alpha*F1 + beta*grid + gamma*teds
    }


⸻

5. Optional: Learned Evaluation Model

If a machine-learned evaluation model is desired:
	1.	Input encoding:
Serialize each JSON into a path–value sequence (e.g.,
cell r0=0 c0=0 rs=1 cs=2, etc.)
or construct a structural graph (nodes = cells, edges = adjacency).
	2.	Model:
Lightweight Transformer Encoder or Graph Neural Network (GNN).
	3.	Training:
	•	Encode both GT and Pred using shared weights.
	•	Concatenate [h_gt; h_pred; |h_gt - h_pred|; h_gt * h_pred].
	•	Feed into MLP to regress the composite score above.
	•	Loss: Mean Squared Error (MSE) or pairwise ranking loss.
	4.	Usage:
	•	Fast approximate scoring for unseen JSONs.
	•	Structural sanity check for production outputs.

⸻

6. Evaluation Protocol
	•	Report metrics:
	•	F1_cell, GridAcc, TEDS, and combined Final Score.
	•	Analyze separately for:
	•	Highly merged tables
	•	Irregular headers
	•	Large/small tables
	•	Plot F1 vs IoU threshold (τ ∈ [0.3, 0.7]) for stability.
	•	Penalize invalid predictions (overlaps, out-of-bounds).

⸻

7. Deliverables

File	Description
evaluator.py	Implements normalization, matching, and scoring
metrics.json	Output JSON with all metrics
sample_gt.json	Example ground-truth table
sample_pred.json	Example prediction
(Optional) learned_scorer/	Transformer/GNN evaluation model code

Example CLI:

python evaluator.py --gt gt.json --pred pred.json \
  --alpha 0.5 --beta 0.3 --gamma 0.2 --iou_thr 0.5

Example Output:

{
  "precision_cell": 0.92,
  "recall_cell": 0.88,
  "f1_cell": 0.90,
  "grid_acc": 0.94,
  "teds_struct": 0.89,
  "final_score": 0.91
}


⸻

8. Example Scenario

Case	Description	Score
Perfect structure	All merges and rows correct	1.00
Slight merge offset	1 merged region misaligned	0.87
Wrong dimensions	Wrong row/col count	0.65
Overlaps / missing	Invalid structure	0.40


⸻

9. Summary of Recommendations

Aspect	Recommendation
Comparison focus	Structure only (ignore text)
Main metric	IoU-matched Cell F1
Auxiliary metrics	Grid Accuracy, TEDS-Structure
Matching algorithm	Hungarian matching on (1 - IoU)
Composite score	Weighted sum: α=0.5, β=0.3, γ=0.2
Optional model	Transformer/GNN regressing the composite score
Output	Final score ∈ [0,1] + detailed metrics


⸻

10. References
	•	Zhong et al., PubTables-1M: Towards Comprehensive Table Extraction, CVPR 2022.
	•	ICDAR 2019 Table Structure Recognition Competition.
	•	Li et al., Tree Edit Distance Similarity for Document Structure Evaluation, 2021.
