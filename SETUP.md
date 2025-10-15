# CS230 Evaluation Model - Setup Guide

## Project Setup

### 1. Create Virtual Environment

A virtual environment isolates your project dependencies and prevents conflicts with system-wide Python packages.

```bash
# Navigate to the project root directory

# Create a virtual environment (requires Python 3.8 or higher)
python3 -m venv .venv
```

**What this does:**
- `python3 -m venv` - Invokes Python's built-in venv module
- `.venv` - The virtual environment directory name (already excluded in .gitignore)

---

### 2. Activate Virtual Environment

#### macOS/Linux:
```bash
source .venv/bin/activate
```

#### Windows:
```bash
.venv\Scripts\activate
```

**Verify activation:**
You should see the `(.venv)` prefix in your terminal:
```
(.venv) rayhu@Rays-MacBook cs230-evaluation-model %
```

---

### 3. Install Dependencies

```bash
# Upgrade pip (recommended)
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**This installs the following packages:**
- `datasets` - Hugging Face datasets library
- `huggingface-hub` - Access to Hugging Face Hub
- `pyarrow` - Accelerated data processing
- `pandas` - Data manipulation toolkit

---

### 4. Run the Download Script

```bash
python scripts/download_pubtables_raw.py

```
