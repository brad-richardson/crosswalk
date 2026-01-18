#!/bin/bash
# Unified agent labeling script
# Usage: ./run_agent.sh <agent> [batch_dir] [options]
#
# Agents: claude, codex, gemini
# Options:
#   --model <model>    Model variant (e.g., sonnet, haiku, flash)
#   --grayscale        Convert satellite images to grayscale before sending
#   --low-res          Reduce image resolution to 256x256

set -o pipefail

# Handle Ctrl+C gracefully
cleanup() {
    echo ""
    echo "Interrupted! Results so far in $OUTPUT_FILE"
    kill 0 2>/dev/null
    exit 130
}
trap cleanup INT TERM

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
AGENT="${1:-}"
shift || true

BATCH_DIR=""
MODEL=""
GRAYSCALE=false
LOW_RES=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --grayscale)
            GRAYSCALE=true
            shift
            ;;
        --low-res)
            LOW_RES=true
            shift
            ;;
        *)
            if [[ -z "$BATCH_DIR" ]]; then
                BATCH_DIR="$1"
            fi
            shift
            ;;
    esac
done

# Default batch dir
BATCH_DIR="${BATCH_DIR:-$(ls -d batches/test_batch_* 2>/dev/null | tail -1)}"

# Validate agent
case "$AGENT" in
    claude|codex|gemini)
        ;;
    *)
        echo "Usage: $0 <agent> [batch_dir] [--model <model>] [--grayscale] [--low-res]"
        echo ""
        echo "Agents:"
        echo "  claude    - Claude Code CLI (models: sonnet, opus, haiku)"
        echo "  codex     - OpenAI Codex CLI"
        echo "  gemini    - Google Gemini CLI (models: flash, pro)"
        echo ""
        echo "Options:"
        echo "  --model      Model variant to use"
        echo "  --grayscale  Convert satellite images to grayscale"
        echo "  --low-res    Reduce images to 256x256"
        echo ""
        echo "Example: $0 gemini batches/test_batch_2026-01-18 --model flash"
        exit 1
        ;;
esac

if [[ -z "$BATCH_DIR" || ! -d "$BATCH_DIR" ]]; then
    echo "Error: Batch directory not found: $BATCH_DIR"
    exit 1
fi

# Set up paths
CONTEXT_DOC="LABELING_INSTRUCTIONS.md"
CANDIDATES_DIR="$BATCH_DIR/candidates"

# Build output dir name (include image processing flags)
OUTPUT_NAME="$AGENT"
[[ -n "$MODEL" ]] && OUTPUT_NAME="${AGENT}_${MODEL}"
[[ "$GRAYSCALE" == "true" ]] && OUTPUT_NAME="${OUTPUT_NAME}_gray"
[[ "$LOW_RES" == "true" ]] && OUTPUT_NAME="${OUTPUT_NAME}_lowres"

OUTPUT_DIR="$BATCH_DIR/labels/$OUTPUT_NAME"
OUTPUT_FILE="$OUTPUT_DIR/data.csv"
LOG_FILE="$OUTPUT_DIR/run.log"
RAW_OUTPUT="$OUTPUT_DIR/raw_responses.log"
TEMP_DIR=$(mktemp -d)

mkdir -p "$OUTPUT_DIR"
echo "ref_id,target_id,label,confidence,reasoning" > "$OUTPUT_FILE"
echo "=== $AGENT Run Started: $(date) ===" > "$LOG_FILE"
echo "" > "$RAW_OUTPUT"

echo "Agent: $AGENT"
[[ -n "$MODEL" ]] && echo "Model: $MODEL"
echo "Batch: $BATCH_DIR"
echo "Output: $OUTPUT_FILE"
[[ "$GRAYSCALE" == "true" ]] && echo "Image mode: grayscale"
[[ "$LOW_RES" == "true" ]] && echo "Image resolution: 256x256"
echo ""

COUNT=0
FAILED=0
TOTAL=$(ls -d "$CANDIDATES_DIR"/*/ 2>/dev/null | wc -l)

# Prepare image if needed (uses Python/PIL)
prepare_image() {
    local src="$1"
    local dst="$2"

    if [[ "$GRAYSCALE" == "true" || "$LOW_RES" == "true" ]]; then
        python3 - "$src" "$dst" "$GRAYSCALE" "$LOW_RES" << 'PYTHON_EOF'
import sys
from PIL import Image

src, dst, grayscale, lowres = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
img = Image.open(src)
if grayscale == "true":
    img = img.convert("L")
if lowres == "true":
    img = img.resize((256, 256), Image.LANCZOS)
img.save(dst)
print(dst)
PYTHON_EOF
        return
    fi
    echo "$src"
}

# Build the prompt (same for all agents)
build_prompt() {
    local ref_id="$1"
    local target_id="$2"
    local img_sat="$3"
    local metadata="$4"

    cat <<PROMPT
Read $CONTEXT_DOC for labeling rules.
Look at $img_sat (blue=reference, red=target) and $metadata.
Do these segments represent the same physical feature?

Output ONLY one CSV line:
$ref_id,$target_id,LABEL,CONFIDENCE,REASON

Where LABEL=match/no_match/unsure, CONFIDENCE=0.0-1.0, REASON=brief text without commas
PROMPT
}

# Run agent command
run_agent() {
    local prompt="$1"

    case "$AGENT" in
        claude)
            local model_arg=""
            [[ -n "$MODEL" ]] && model_arg="--model $MODEL"
            timeout 30 claude -p $model_arg "$prompt" 2>&1
            ;;
        codex)
            local tmpout="$TEMP_DIR/codex_out.txt"
            timeout 30 codex exec -o "$tmpout" -- "$prompt" 2>&1 || true
            cat "$tmpout" 2>/dev/null
            ;;
        gemini)
            local model_arg=""
            [[ -n "$MODEL" ]] && model_arg="-m $MODEL"
            timeout --signal=KILL 30 gemini --yolo -o text $model_arg "$prompt" 2>&1
            ;;
    esac
}

# Process candidates
for CANDIDATE_DIR in "$CANDIDATES_DIR"/*/; do
    DIR_NAME=$(basename "$CANDIDATE_DIR")
    REF_ID="${DIR_NAME%%__*}"
    TARGET_ID="${DIR_NAME##*__}"

    METADATA="${CANDIDATE_DIR}metadata.yaml"
    IMG_SAT="${CANDIDATE_DIR}satellite.png"

    if [[ ! -f "$METADATA" ]]; then
        echo "[WARN] No metadata in $CANDIDATE_DIR" | tee -a "$LOG_FILE"
        continue
    fi

    # Prepare image if needed
    PROCESSED_IMG=$(prepare_image "$IMG_SAT" "$TEMP_DIR/${REF_ID}_sat.png")

    # Build prompt
    PROMPT=$(build_prompt "$REF_ID" "$TARGET_ID" "$PROCESSED_IMG" "$METADATA")

    echo "[$(date +%H:%M:%S)] $REF_ID" >> "$LOG_FILE"
    echo -n "Processing $REF_ID... "

    # Run agent
    RAW=$(run_agent "$PROMPT") || true

    echo "=== $REF_ID ===" >> "$RAW_OUTPUT"
    echo "$RAW" >> "$RAW_OUTPUT"

    # Parse result - look for CSV line with ref_id
    RESULT=$(echo "$RAW" | grep "^${REF_ID}," | head -1) || true

    # Fallback: extract label keyword
    if [[ -z "$RESULT" ]]; then
        LABEL=$(echo "$RAW" | grep -oE '\b(no_match|match|unsure)\b' | head -1) || true
        if [[ -n "$LABEL" ]]; then
            RESULT="${REF_ID},${TARGET_ID},${LABEL},0.5,parsed"
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

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "=== Complete: $COUNT success, $FAILED failed ==="
echo "Output: $OUTPUT_FILE"
echo "Log: $LOG_FILE"
