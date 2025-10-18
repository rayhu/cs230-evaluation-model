#!/bin/bash

#############################################
# PubTables-1M Dataset Extraction Script
# Extracts Structure and Detection datasets
#############################################

set -e  # Exit on any error

# Color output (optional, for better visibility)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_DIR="$PROJECT_ROOT/data/pubtables_raw"

echo -e "${GREEN}🔍 PubTables-1M Extraction Script${NC}"
echo "=================================="
echo "Project root: $PROJECT_ROOT"
echo "Target directory: $TARGET_DIR"
echo ""

# Check if target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo -e "${RED}❌ Error: Directory '$TARGET_DIR' does not exist${NC}"
    echo "💡 Please create it first or download data using:"
    echo "   python scripts/download_pubtables_raw_subset.py"
    exit 1
fi

# Change to target directory
cd "$TARGET_DIR" || {
    echo -e "${RED}❌ Error: Cannot change to directory '$TARGET_DIR'${NC}"
    exit 1
}

echo -e "${GREEN}✅ Working in: $(pwd)${NC}"
echo ""

# List available tar.gz files
echo "📦 Available archive files:"
ls -lh *.tar.gz 2>/dev/null || {
    echo -e "${RED}❌ Error: No .tar.gz files found in $TARGET_DIR${NC}"
    echo "💡 Please download the PubTables-1M dataset files first"
    exit 1
}
echo ""

# Confirm before extraction (optional safety measure)
read -p "⚠️  This will extract large archives. Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Extraction cancelled."
    exit 0
fi

echo -e "${GREEN}🚀 Starting extraction...${NC}"
echo ""

# Structure dataset extraction
echo "📊 Extracting Structure dataset..."
# mkdir -p PubTables-1M-Structure/{images,train,test,val,words}
# tar -xzf PubTables-1M-Structure_Filelists.tar.gz --directory PubTables-1M-Structure/
# tar -xzf PubTables-1M-Structure_Annotations_Test.tar.gz --directory PubTables-1M-Structure/test/
# tar -xzf PubTables-1M-Structure_Annotations_Train.tar.gz --directory PubTables-1M-Structure/train/
# tar -xzf PubTables-1M-Structure_Annotations_Val.tar.gz --directory PubTables-1M-Structure/val/
# tar -xzf PubTables-1M-Structure_Images_Test.tar.gz --directory PubTables-1M-Structure/images/
tar -xzf PubTables-1M-Structure_Images_Train.tar.gz --directory PubTables-1M-Structure/images/
tar -xzf PubTables-1M-Structure_Images_Val.tar.gz --directory PubTables-1M-Structure/images/
tar -xzf PubTables-1M-Structure_Table_Words.tar.gz --directory PubTables-1M-Structure/words/

echo ""
echo "🔍 Extracting Detection dataset..."
mkdir -p PubTables-1M-Detection/{images,train,test,val,words}
tar -xzf PubTables-1M-Detection_Filelists.tar.gz --directory PubTables-1M-Detection/
tar -xzf PubTables-1M-Detection_Annotations_Test.tar.gz --directory PubTables-1M-Detection/test/
tar -xzf PubTables-1M-Detection_Annotations_Train.tar.gz --directory PubTables-1M-Detection/train/
tar -xzf PubTables-1M-Detection_Annotations_Val.tar.gz --directory PubTables-1M-Detection/val/
tar -xzf PubTables-1M-Detection_Images_Test.tar.gz --directory PubTables-1M-Detection/images/
tar -xzf PubTables-1M-Detection_Images_Train_Part1.tar.gz --directory PubTables-1M-Detection/images/
tar -xzf PubTables-1M-Detection_Images_Train_Part2.tar.gz --directory PubTables-1M-Detection/images/
tar -xzf PubTables-1M-Detection_Images_Val.tar.gz  --directory PubTables-1M-Detection/images/
tar -xzf PubTables-1M-Detection_Page_Words.tar.gz --directory PubTables-1M-Detection/words/

echo ""
echo "📄 Extracting PDF annotations..."
mkdir -p PubTables-1M-PDF-Annotations/
# tar -xzf PubTables-1M-PDF_Annotations.tar.gz --directory PubTables-1M-PDF-Annotations

echo ""
echo -e "${GREEN}✅ Extraction complete!${NC}"
echo ""
echo "📂 Extracted directories:"
ls -d PubTables-1M-*/
echo ""
echo "🎉 All done!"