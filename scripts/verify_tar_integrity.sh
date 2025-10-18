#!/bin/bash

# Script to verify integrity of tar.gz files

echo "🔍 Verifying tar.gz file integrity..."
echo ""

cd /Users/rayhu/play/ai/cs230-evaluation-model/data/pubtables_raw

FAILED=0
PASSED=0

for file in *.tar.gz; do
    if [ -f "$file" ]; then
        echo -n "Testing $file ... "
        if gzip -t "$file" 2>/dev/null; then
            echo "✅ OK"
            ((PASSED++))
        else
            echo "❌ CORRUPTED"
            ((FAILED++))
            echo "   File: $file"
            ls -lh "$file"
            echo ""
        fi
    fi
done

echo ""
echo "=============================="
echo "Summary:"
echo "  ✅ Passed: $PASSED"
echo "  ❌ Failed: $FAILED"
echo "=============================="

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "⚠️  Corrupted files need to be re-downloaded!"
    exit 1
else
    echo ""
    echo "✅ All files are valid!"
    exit 0
fi

