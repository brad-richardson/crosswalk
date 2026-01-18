#!/bin/bash
# Run Codex and Gemini agents on an agent labeling batch
set -e

BATCH_DIR="${1:-/workspace/matcher/agent_labels/batches/batch_2026-01-18_031013}"
CONTEXT_FILE="/workspace/matcher/docs/agent_labeling/CONTEXT.md"

if [ ! -d "$BATCH_DIR" ]; then
    echo "Batch directory not found: $BATCH_DIR"
    exit 1
fi

CANDIDATES_DIR="$BATCH_DIR/candidates"
LABELS_DIR="$BATCH_DIR/labels"

# Create labels directories
mkdir -p "$LABELS_DIR/codex"
mkdir -p "$LABELS_DIR/gemini"

# Build the prompt
CONTEXT=$(cat "$CONTEXT_FILE")

# Collect all candidate metadata
CANDIDATES=""
for candidate_dir in "$CANDIDATES_DIR"/*/; do
    if [ -f "$candidate_dir/metadata.yaml" ]; then
        candidate_name=$(basename "$candidate_dir")
        CANDIDATES="$CANDIDATES
---
### Candidate: $candidate_name

$(cat "$candidate_dir/metadata.yaml")
"
    fi
done

# Create the full prompt
PROMPT="$CONTEXT

---

# Candidates to Label

Please analyze the following candidates and provide labels. For each candidate, determine if the reference and target segments represent the same physical road.

Output your response as a CSV with columns: ref_id,target_id,label,confidence,reasoning

$CANDIDATES

---

Please provide your labels in CSV format. Remember:
- 'match' = same physical road
- 'no_match' = different roads
- 'unsure' = cannot confidently determine

Note: You don't have access to the images in this run, so base your decision on the geometric metrics:
- buffer_iou > 0.7 suggests match
- hausdorff_distance < 15m suggests match
- heading_delta < 10° suggests match
- Large offsets or heading differences suggest no_match
"

# Save prompt for reference
echo "$PROMPT" > "$BATCH_DIR/agent_prompt.md"

echo "Prompt saved to $BATCH_DIR/agent_prompt.md"
echo "Running agents..."

# Run Codex
echo ""
echo "=== Running Codex ==="
echo "$PROMPT" | codex -q --approval-mode full-auto 2>&1 | tee "$LABELS_DIR/codex/raw_output.txt"

echo ""
echo "=== Running Gemini ==="
echo "$PROMPT" | gemini 2>&1 | tee "$LABELS_DIR/gemini/raw_output.txt"

echo ""
echo "Agent outputs saved to:"
echo "  - $LABELS_DIR/codex/raw_output.txt"
echo "  - $LABELS_DIR/gemini/raw_output.txt"
