# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds Python modules for model training and conversion; use `main.py` for CLI entry points and keep shared helpers in `src/utils/`.
- `scripts/` contains reproducible pipelines (extraction, scoring, validation). Treat each script as a single-responsibility CLI and prefer extending these instead of notebooks.
- Data artifacts live under `data/` (ignored by Git). Keep large SciTSR assets inside `data/SciTSR/` and document any new splits in `data/README.md`.
- `notebooks/` is for exploratory analyses; once stable, port code into `src/` or `scripts/` to keep experiments reproducible.
- Store experiment configs/results inside `experiments/` and long-form docs in `docs/`.

## Build, Test, and Development Commands
- Environment bootstrap: `uv venv && source .venv/bin/activate && uv pip install -r requirements.txt` mirrors CI expectations.
- Quick setup alternative: `./setup.sh` (creates venv, installs deps, downloads core resources).
- Launch notebooks with `./start_jupyter.sh` after the venv is active to inherit dependencies.
- Run extraction: `python scripts/extract_tables_scitsr.py --input data/SciTSR/test/img --output data/SciTSR/test/json_output`.
- Score predictions: `python scripts/score_extraction.py --pred data/SciTSR/test/json_output --gt data/SciTSR/test/structure --output results/evaluation_scores.json`.
- Validate JSON payloads: `python scripts/validate_outputs.py --output-dir ... --gt-dir ... --save-report ...`.

## Coding Style & Naming Conventions
- Python code follows PEP 8 with 4-space indentation, snake_case functions, and CapWords classes. Keep modules short and composable.
- Prefer type hints for new functions and add docstrings outlining inputs/outputs.
- Entry-point scripts should expose an `if __name__ == "__main__":` block and group argparse definitions near the top.
- Use descriptive filenames such as `structure_converter.py` to reflect primary behavior; avoid numbered script names outside notebooks.

## Testing Guidelines
- No pytest harness yet; rely on data-level checks:
  - `scripts/score_extraction.py --detailed` acts as regression verification for metric logic.
  - `scripts/validate_outputs.py` doubles as an integration test for schema compliance—run it before sharing artifacts.
- Log metric deltas in `experiments/` to document coverage over SciTSR splits; flag any drop >2% F1.
- When adding new utilities, include a minimal CLI flag like `--limit` to enable deterministic smoke tests on small subsets.

## Commit & Pull Request Guidelines
- Follow the existing Git history style: imperative, descriptive summaries (e.g., “Add scoring script for assessing extraction quality…”). Keep subject lines ≤72 chars and elaborate in the body if needed.
- Each PR should include: concise description, dataset or script paths touched, validation evidence (command outputs or notebook cells), and any follow-up tasks.
- Reference related issues or experiments explicitly (`experiments/run_2024_03_15.md`) and attach screenshots for notebook UI changes when relevant.
- Ensure large data outputs remain untracked; if sharing samples, place them under `subset/` with clear naming (`subset/<split>_<count>.json`).
