# Uploading Dataset to Hugging Face Hub

This guide explains how to upload your dataset to Hugging Face Hub for easy sharing and use.

## 🎯 Overview

Your dataset is now set up to be uploaded to Hugging Face Hub. Once uploaded, others can use it with:

```python
from datasets import load_dataset

dataset = load_dataset("your-username/table-extraction-dataset")
```

## 📋 Prerequisites

1. **Create a Hugging Face account** (if you don't have one)
   - Go to [huggingface.co](https://huggingface.co)
   - Sign up for a free account

2. **Install dependencies** (already in your requirements.txt)
   ```bash
   pip install datasets huggingface_hub
   ```

3. **Login to Hugging Face**
   ```bash
   huggingface-cli login
   ```
   
   Enter your access token (get it from https://huggingface.co/settings/tokens)

## 🚀 Upload Process

### Step 1: Convert Dataset to Parquet Format

Convert your JSON dataset to Parquet format (which is required by modern versions of the `datasets` library):

```bash
python scripts/convert_to_parquet.py
```

This will:
- Read all JSON files from `dataset/train/` and `dataset/test/`
- Process metadata files
- Convert to Parquet format
- Save to `dataset_parquet/`

**Expected output:**
```
Processing train split...
Processing test split...
Converting to Arrow/Parquet format...
CONVERSION COMPLETE
Train examples: 11971
Test examples: 3000
```

### Step 2: Test the Dataset Locally (Optional)

Verify that the dataset loads correctly:

```bash
python test_local_dataset.py
```

Or in Python:

```python
from datasets import load_dataset

# Load from Parquet files
dataset = load_dataset("./dataset_parquet")

# Check the dataset
print(dataset)
print(f"Train: {len(dataset['train'])} samples")
print(f"Test: {len(dataset['test'])} samples")

# Inspect an example
example = dataset['train'][0]
print(f"Example ID: {example['id']}")
print(f"Similarity Score: {example['similarity_score']}")
print(f"Ground Truth Cells: {len(example['ground_truth']['cells'])}")
print(f"Generated Cells: {len(example['generated']['cells'])}")
```

### Step 3: Upload to Hugging Face Hub

Choose a repository name (e.g., `your-username/table-extraction-evaluation`):

```bash
# For public dataset
python scripts/upload_to_huggingface.py --repo-id your-username/table-extraction-evaluation

# For private dataset
python scripts/upload_to_huggingface.py --repo-id your-username/table-extraction-evaluation --private
```

**Note**: Replace `your-username` with your actual Hugging Face username.

### Step 3: Verify Upload

After uploading, you can load your dataset from anywhere:

```python
from datasets import load_dataset

# Load from Hugging Face Hub
dataset = load_dataset("your-username/table-extraction-evaluation", trust_remote_code=True)

# Use the dataset
for example in dataset['train']:
    print(f"ID: {example['id']}")
    print(f"Score: {example['similarity_score']:.3f}")
    break
```

## 📊 Dataset Structure

Your dataset has the following structure:

```
dataset/
├── train/
│   ├── generated/          # Generated table JSONs
│   ├── ground_truth/       # Ground truth table JSONs
│   └── metadata_train.jsonl  # Metadata with scores
├── test/
│   ├── generated/
│   ├── ground_truth/
│   └── metadata_test.jsonl
├── loading_script.py        # Hugging Face loading script
└── README.md               # Dataset documentation
```

## 🔧 Customization

### Modify the Conversion Script

If you need to change how data is processed, edit `scripts/convert_to_parquet.py`:

- **Data Processing**: Modify `process_split()` to change how JSON files are loaded
- **Output Format**: Modify how data is structured before converting to Parquet
- **Feature Extraction**: Add new features or modify existing ones

### Add Dataset Card

Create a detailed dataset card by editing `dataset/README.md`:

- Add dataset descriptions
- Include usage examples
- Add citation information
- Document any limitations or biases

## 🎓 After Uploading

### Share Your Dataset

Once uploaded, others can use your dataset:

```python
from datasets import load_dataset

dataset = load_dataset("your-username/table-extraction-evaluation")
train = dataset['train']
test = dataset['test']
```

### Update Documentation

Update your main README with the Hugging Face link:

```markdown
## 📦 Dataset

Our dataset is available on Hugging Face:

https://huggingface.co/datasets/your-username/table-extraction-evaluation

Load it with:
```python
from datasets import load_dataset
dataset = load_dataset("your-username/table-extraction-evaluation")
```
```

## ⚙️ Troubleshooting

### Issue: "Authentication required"

**Solution**: Run `huggingface-cli login` and enter your token.

### Issue: "No Parquet files found"

**Solution**: Run `python scripts/convert_to_parquet.py` first to convert your dataset.

### Issue: "File not found" errors during conversion

**Solution**: Check that all JSON files referenced in metadata files actually exist in the `generated/` and `ground_truth/` directories.

### Issue: "ModuleNotFoundError: No module named 'pyarrow'"

**Solution**: Make sure you're in the virtual environment and have installed dependencies:
```bash
source .venv/bin/activate
pip install pyarrow
```

### Issue: "Dataset too large"

**Solution**: Hugging Face has size limits for free accounts. Consider:
- Compressing files
- Splitting into smaller chunks
- Using Git LFS for large files

## 📝 Notes

- **Private datasets**: Only you can access them (unless you grant access)
- **Public datasets**: Anyone can access and use them
- **Versioning**: Hugging Face automatically versions your dataset
- **Community**: Public datasets can be starred, forked, and discussed

## 🎉 Next Steps

After uploading:

1. ✅ Update your project README with the Hugging Face link
2. ✅ Add dataset citation to your papers
3. ✅ Share the link with collaborators
4. ✅ Monitor usage via Hugging Face dashboard

