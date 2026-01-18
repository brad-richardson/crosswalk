#!/bin/bash
# Unified agent labeling script
# Usage: ./run_agent.sh <agent> [batch_dir] [options]
#
# Agents: claude, codex, gemini, ollama
# Options:
#   --model <model>    Model variant (e.g., sonnet, haiku, flash, moondream)

set -o pipefail

# Handle Ctrl+C gracefully
cleanup() {
    echo ""
    echo "Interrupted! Results so far in $OUTPUT_FILE"
    # Kill any child processes
    pkill -P $$ 2>/dev/null
    exit 130
}
trap cleanup INT TERM

# Ensure child processes get signals
set -m

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
AGENT="${1:-}"
shift || true

BATCH_DIR=""
MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
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
    claude|codex|gemini|ollama)
        ;;
    *)
        echo "Usage: $0 <agent> [batch_dir] [--model <model>]"
        echo ""
        echo "Agents:"
        echo "  claude    - Claude Code CLI (models: sonnet, opus, haiku)"
        echo "  codex     - OpenAI Codex CLI"
        echo "  gemini    - Google Gemini CLI (models: flash, pro)"
        echo "  ollama    - Local Ollama (models: llava, llava:13b)"
        echo ""
        echo "Options:"
        echo "  --model      Model variant to use"
        echo ""
        echo "Example: $0 gemini batches/test_batch_2026-01-18 --model flash"
        echo "Example: $0 ollama --model llava"
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

# Build output dir name
OUTPUT_NAME="$AGENT"
[[ -n "$MODEL" ]] && OUTPUT_NAME="${AGENT}_${MODEL}"

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
echo ""

COUNT=0
FAILED=0
TOTAL=$(ls -d "$CANDIDATES_DIR"/*/ 2>/dev/null | wc -l)

# Build the prompt - static prefix first for caching, variable content last
build_prompt() {
    local ref_id="$1"
    local target_id="$2"
    local metadata_content="$3"

    # Static prefix (cacheable across all candidates)
    cat <<'STATIC_PREFIX'
You are analyzing road segment matches. Do NOT explore files or the codebase.

TASK: Do the blue and red road segments represent the same physical road?

Images provided:
- satellite.png: satellite view (blue=reference segment, red=target segment)
- geometry.png: geometry view (same color coding)

Output format: ref_id,target_id,LABEL,CONFIDENCE,REASON
- LABEL: match, no_match, or unsure
- CONFIDENCE: 0.0 to 1.0
- REASON: brief explanation (no commas)

Output exactly ONE CSV line, nothing else.

---
STATIC_PREFIX

    # Variable suffix (changes per candidate)
    cat <<VARIABLE_SUFFIX
Candidate: $ref_id,$target_id

Metadata:
$metadata_content
VARIABLE_SUFFIX
}

# Run agent command
run_agent() {
    local prompt="$1"
    local img_sat="$2"
    local img_geo="$3"

    case "$AGENT" in
        claude)
            local model_arg=""
            [[ -n "$MODEL" ]] && model_arg="--model $MODEL"
            # Claude CLI reads images referenced in the prompt from the working directory
            # We need to run from the candidate directory so images are accessible
            local candidate_dir
            candidate_dir=$(dirname "$img_sat")
            (cd "$candidate_dir" && timeout 60 claude -p $model_arg "$prompt") 2>&1
            ;;
        codex)
            local tmpout="$TEMP_DIR/codex_out.txt"
            # Codex supports -i flag for image attachments
            timeout 60 codex exec -i "$img_sat" -i "$img_geo" -o "$tmpout" -- "$prompt" 2>&1 || true
            cat "$tmpout" 2>/dev/null
            ;;
        gemini)
            local model_arg=""
            [[ -n "$MODEL" ]] && model_arg="-m $MODEL"

            # Create isolated working directory to avoid gitignore issues
            local sandbox="$TEMP_DIR/gemini_sandbox"
            mkdir -p "$sandbox"

            # Copy images with consistent names for prompt caching
            cp "$img_sat" "$sandbox/satellite.png"
            cp "$img_geo" "$sandbox/geometry.png"

            # Write prompt to file to avoid shell escaping issues
            printf '%s' "$prompt" > "$sandbox/prompt.txt"

            # cd to sandbox directory
            local orig_dir="$PWD"
            cd "$sandbox"

            # Run Gemini with --sandbox (requires Docker)
            gemini --sandbox --yolo -o text \
                "Analyze satellite.png and geometry.png using instructions in prompt.txt" \
                prompt.txt satellite.png geometry.png 2>&1
            local exit_code=$?

            # Restore directory and cleanup
            cd "$orig_dir"
            rm -rf "$sandbox" 2>/dev/null || echo "WARNING: Failed to clean up $sandbox" >&2

            [[ $exit_code -ne 0 ]] && echo "GEMINI_ERROR: exit code $exit_code"
            ;;
        ollama)
            # Cross-platform base64 (macOS doesn't support -w0)
            local sat_base64 geo_base64
            sat_base64=$(base64 "$img_sat" | tr -d '\n')
            geo_base64=$(base64 "$img_geo" | tr -d '\n')
            local model_name="${MODEL:-llava}"

            # Build JSON payload safely with jq to handle escaping
            local json_payload
            json_payload=$(jq -n \
                --arg model "$model_name" \
                --arg prompt "$prompt" \
                --arg sat "$sat_base64" \
                --arg geo "$geo_base64" \
                '{model: $model, prompt: $prompt, images: [$sat, $geo], stream: false}')

            # Ollama vision models need more time than text-only (120s timeout)
            timeout 120 curl -s http://localhost:11434/api/generate \
                -d "$json_payload" | jq -r '.response // .error // "No response"'
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
    IMG_GEO="${CANDIDATE_DIR}geometry.png"

    if [[ ! -f "$METADATA" ]]; then
        echo "[WARN] No metadata in $CANDIDATE_DIR" | tee -a "$LOG_FILE"
        continue
    fi

    # Read metadata content
    METADATA_CONTENT=""
    if [[ -f "$METADATA" ]]; then
        METADATA_CONTENT=$(cat "$METADATA")
    fi

    # Build prompt using the function
    PROMPT=$(build_prompt "$REF_ID" "$TARGET_ID" "$METADATA_CONTENT")

    echo "[$(date +%H:%M:%S)] $REF_ID" >> "$LOG_FILE"
    echo -n "Processing $REF_ID... "

    # Run agent (pass image paths)
    RAW=$(run_agent "$PROMPT" "$IMG_SAT" "$IMG_GEO") || true

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
