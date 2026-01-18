#!/bin/bash
# Run Claude on agent labeling batch
# Usage: ./run_claude.sh [batch_dir] [model]
#
# Models: sonnet (default), opus, haiku

# Handle Ctrl+C gracefully - kill background processes too
cleanup() {
    echo ""
    echo "Interrupted! Results so far in $OUTPUT_FILE"
    kill 0 2>/dev/null
    exit 130
}
trap cleanup INT TERM

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BATCH_DIR="${1:-$(ls -d batches/test_batch_* 2>/dev/null | tail -1)}"
MODEL="${2:-sonnet}"
CONTEXT_DOC="LABELING_INSTRUCTIONS.md"

if [[ -z "$BATCH_DIR" || ! -d "$BATCH_DIR" ]]; then
    echo "Usage: $0 <batch_dir> [model]"
    echo "Example: $0 batches/test_batch_2026-01-18_040238 sonnet"
    echo "Models: sonnet (default), opus, haiku"
    exit 1
fi

CANDIDATES_DIR="$BATCH_DIR/candidates"
OUTPUT_DIR="$BATCH_DIR/labels/claude_${MODEL}"
OUTPUT_FILE="$OUTPUT_DIR/data.csv"
LOG_FILE="$OUTPUT_DIR/run.log"
RAW_OUTPUT="$OUTPUT_DIR/raw_responses.log"

mkdir -p "$OUTPUT_DIR"
echo "ref_id,target_id,label,confidence,reasoning" > "$OUTPUT_FILE"
echo "=== Claude $MODEL Run Started: $(date) ===" > "$LOG_FILE"
echo "" > "$RAW_OUTPUT"

echo "Processing batch: $BATCH_DIR"
echo "Model: $MODEL"
echo "Context doc: $CONTEXT_DOC"
echo "Output: $OUTPUT_FILE"
echo ""

COUNT=0
FAILED=0
TOTAL=$(ls -d "$CANDIDATES_DIR"/*/ 2>/dev/null | wc -l)

for CANDIDATE_DIR in "$CANDIDATES_DIR"/*/; do
    DIR_NAME=$(basename "$CANDIDATE_DIR")
    REF_ID="${DIR_NAME%%__*}"
    TARGET_ID="${DIR_NAME##*__}"

    METADATA="${CANDIDATE_DIR}metadata.yaml"
    IMG_SAT="${CANDIDATE_DIR}satellite.png"
    IMG_GEO="${CANDIDATE_DIR}geometry.png"

    if [[ ! -f "$METADATA" ]]; then
        echo "[WARN] No metadata in $CANDIDATE_DIR" | tee -a "$LOG_FILE"
        continue
    fi

    # Extract features
    DATASET=$(grep -A1 "^candidate:" "$METADATA" | grep "dataset:" | awk '{print $2}')
    HAUSDORFF=$(grep "hausdorff_distance:" "$METADATA" | head -1 | awk '{print $2}')
    BUFFER_IOU=$(grep "buffer_iou:" "$METADATA" | awk '{print $2}')

    PROMPT="You are labeling road segment matches.

Read these files:
1. Labeling instructions: $CONTEXT_DOC
2. Satellite image: $IMG_SAT
3. Geometry image: $IMG_GEO
4. Metadata: $METADATA

Candidate: $REF_ID <-> $TARGET_ID
Dataset: $DATASET
Features: hausdorff_distance=$HAUSDORFF, buffer_iou=$BUFFER_IOU

Blue line = reference (Overture), Red dashed = target (local dataset).

Apply labeling rules:
- Sidewalk vs road = no_match
- Bike lanes: painted = match, raised/separated = no_match
- Opposite carriageways = no_match
- Centerline vs split road = match

Output ONE CSV line:
$REF_ID,$TARGET_ID,LABEL,CONFIDENCE,REASON

LABEL = match/no_match/unsure, CONFIDENCE = 0.0-1.0, REASON = brief text (no commas)"

    echo "[$(date +%H:%M:%S)] $REF_ID" >> "$LOG_FILE"
    echo -n "Processing $REF_ID... "

    # Run claude with specified model
    RAW=$(timeout 30 claude -p --model "$MODEL" "$PROMPT" 2>&1) || true

    echo "=== $REF_ID ===" >> "$RAW_OUTPUT"
    echo "$RAW" >> "$RAW_OUTPUT"

    # Parse result
    RESULT=$(echo "$RAW" | grep "^$REF_ID," | head -1) || true
    if [[ -z "$RESULT" ]]; then
        LABEL=$(echo "$RAW" | grep -oE '\b(no_match|match|unsure)\b' | head -1) || true
        if [[ -n "$LABEL" ]]; then
            RESULT="$REF_ID,$TARGET_ID,$LABEL,0.5,parsed"
        fi
    fi

    if [[ -n "$RESULT" ]]; then
        echo "$RESULT" >> "$OUTPUT_FILE"
        COUNT=$((COUNT+1))
        echo "$(echo "$RESULT" | cut -d',' -f3) [$COUNT/$TOTAL]"
    else
        FAILED=$((FAILED+1))
        echo "FAIL [$((COUNT+FAILED))/$TOTAL]" | tee -a "$LOG_FILE"
        echo "  Raw: ${RAW:0:200}" >> "$LOG_FILE"
    fi
done

echo ""
echo "=== Complete: $COUNT success, $FAILED failed ==="
echo "Output: $OUTPUT_FILE"
echo "Log: $LOG_FILE"
