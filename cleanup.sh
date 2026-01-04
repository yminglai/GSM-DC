#!/bin/bash
# Cleanup script for GSM-DC repository
# Removes redundant files while preserving essential functionality

echo "=========================================="
echo "GSM-DC Repository Cleanup"
echo "=========================================="
echo ""

# Function to safely remove files
safe_remove() {
    if [ -e "$1" ]; then
        echo "✓ Removing: $1"
        rm -rf "$1"
    else
        echo "✗ Not found: $1 (already removed?)"
    fi
}

echo "Removing redundant test files..."
safe_remove "test_batch.py"
safe_remove "test_ground_truth_stepwise.py"
safe_remove "test_denoise.py"

echo ""
echo "Removing git-lfs installer..."
safe_remove "git-lfs-3.2.0"

echo ""
echo "Removing problematic dataset generator..."
safe_remove "generate_dataset.py"

echo ""
echo "Removing temporary documentation..."
safe_remove "CLEANUP_PLAN.md"
safe_remove "FILES_TO_REMOVE.md"

echo ""
echo "=========================================="
echo "Cleanup Summary"
echo "=========================================="
echo ""
echo "Removed files:"
echo "  - test_batch.py"
echo "  - test_ground_truth_stepwise.py"
echo "  - test_denoise.py"
echo "  - git-lfs-3.2.0/"
echo "  - generate_dataset.py"
echo "  - CLEANUP_PLAN.md"
echo "  - FILES_TO_REMOVE.md"
echo ""
echo "Core files preserved:"
echo "  ✓ evaluate.py (NEW - consolidated evaluation script)"
echo "  ✓ test_batch_saved.py (reference implementation)"
echo "  ✓ test_saved.py (utility functions)"
echo "  ✓ data_gen/ (problem generators)"
echo "  ✓ math_gen/ (graph and problem generation)"
echo "  ✓ tools/ (validation and parsing)"
echo "  ✓ format/ (prompt formatting)"
echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Next steps:"
echo "1. Review evaluate.py and configure MODEL_PATH"
echo "2. Run: python evaluate.py"
echo "3. Delete this cleanup script: rm cleanup.sh"
