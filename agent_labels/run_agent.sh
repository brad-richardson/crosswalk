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
LIMIT=""
RESUME="false"
BAIL_AFTER="2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --batch)
            BATCH_DIR="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --resume)
            RESUME="true"
            shift
            ;;
        --bail-after)
            if [[ -z "$2" || ! "$2" =~ ^[0-9]+$ ]]; then
                echo "Error: --bail-after requires a non-negative integer argument." >&2
                exit 1
            fi
            BAIL_AFTER="$2"
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
        echo "Usage: $0 <agent> [--batch <dir>] [--model <model>] [--limit <n>] [--resume] [--bail-after <n>]"
        echo ""
        echo "Agents:"
        echo "  claude    - Claude Code CLI (models: sonnet, opus, haiku)"
        echo "  codex     - OpenAI Codex CLI"
        echo "  gemini    - Google Gemini CLI (models: flash, pro)"
        echo "  ollama    - Local Ollama (models: llava, llava:13b)"
        echo ""
        echo "Options:"
        echo "  --batch        Batch directory (default: latest test_batch_*)"
        echo "  --model        Model variant to use"
        echo "  --limit        Max candidates to process"
        echo "  --resume       Continue from previous run, skip already-labeled candidates"
        echo "  --bail-after   Stop after N consecutive failures (default: 2, 0=never bail)"
        echo ""
        echo "Example: $0 gemini --batch batches/test_batch_2026-01-18 --model flash"
        echo "Example: $0 claude --model sonnet --limit 50"
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

# Handle resume mode - load existing pairs or write fresh header
EXISTING_PAIRS=""
SKIPPED=0
if [[ "$RESUME" == "true" ]] && [[ -f "$OUTPUT_FILE" ]]; then
    # Load existing ref_id,target_id pairs (skip header)
    EXISTING_PAIRS=$(tail -n +2 "$OUTPUT_FILE" | cut -d',' -f1,2)
    if [[ -z "$EXISTING_PAIRS" ]]; then
        EXISTING_COUNT=0
    else
        EXISTING_COUNT=$(printf '%s\n' "$EXISTING_PAIRS" | grep -c '^' 2>/dev/null || echo 0)
    fi
    echo "Resuming: found $EXISTING_COUNT existing labels"
else
    # Start fresh - write header
    echo "ref_id,target_id,label,confidence,reasoning" > "$OUTPUT_FILE"
fi

echo "=== $AGENT Run Started: $(date) ===" >> "$LOG_FILE"
if [[ "$RESUME" == "true" ]]; then
    # In resume mode, append to raw responses to preserve previous run data
    echo "" >> "$RAW_OUTPUT"
else
    # In fresh runs, start with a clean raw responses file
    echo "" > "$RAW_OUTPUT"
fi

echo "Agent: $AGENT"
[[ -n "$MODEL" ]] && echo "Model: $MODEL"
echo "Batch: $BATCH_DIR"
echo "Output: $OUTPUT_FILE"
echo ""

COUNT=0
FAILED=0
TOTAL_CANDIDATES=$(ls -d "$CANDIDATES_DIR"/*/ 2>/dev/null | wc -l)
# Calculate remaining candidates (excluding already labeled in resume mode)
REMAINING=$((TOTAL_CANDIDATES - EXISTING_COUNT))
[[ "$REMAINING" -lt 0 ]] && REMAINING=0
# Apply limit to remaining
TOTAL="$REMAINING"
if [[ -n "$LIMIT" ]] && [[ "$LIMIT" -lt "$TOTAL" ]]; then
    TOTAL="$LIMIT"
fi
# Track actually processed (not skipped) for limit purposes
ACTUALLY_PROCESSED=0

# Build the prompt - static prefix first for caching, variable content last
build_prompt() {
    local ref_id="$1"
    local target_id="$2"
    local metadata_content="$3"

    # Static prefix (cacheable across all candidates)
    cat <<'STATIC_PREFIX'
You are analyzing road network segment matches. Do NOT explore files or the codebase.

TASK: Do the blue and red segments represent the same physical road network segment (road, trail, sidewalk, etc)?

IMPORTANT: Segments match if they overlap on the same physical road even if one is longer than the other (subsegment matches count as match). Only mark as no_match if they are clearly different roads (parallel lanes, perpendicular streets, or completely separate locations).

Images provided:
- satellite.png: aerial view with geometry overlay (blue circles=reference, red circles=target)
- geometry.png: clean geometry view on white background (same colors and markers)

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
            # Claude CLI requires explicit "read" instruction to access images
            # Prepend instruction and run from candidate directory
            local candidate_dir
            candidate_dir=$(dirname "$img_sat")
            local full_prompt="First read satellite.png and geometry.png, then answer:

$prompt"
            (cd "$candidate_dir" && timeout 60 claude -p $model_arg "$full_prompt") 2>&1
            ;;
        codex)
            local tmpout="$TEMP_DIR/codex_out.txt"
            # Codex supports -i flag for image attachments (include both images)
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
            [[ -f "$img_sat" ]] && cp "$img_sat" "$sandbox/satellite.png"
            cp "$img_geo" "$sandbox/geometry.png"

            # Write prompt to file to avoid shell escaping issues
            printf '%s' "$prompt" > "$sandbox/prompt.txt"

            # cd to sandbox directory
            local orig_dir="$PWD"
            cd "$sandbox"

            # Run Gemini with --sandbox (requires Docker)
            # Include both images if satellite exists
            if [[ -f "satellite.png" ]]; then
                gemini --sandbox --yolo -o text \
                    "Analyze satellite.png and geometry.png using instructions in prompt.txt" \
                    prompt.txt satellite.png geometry.png 2>&1
            else
                gemini --sandbox --yolo -o text \
                    "Analyze geometry.png using instructions in prompt.txt" \
                    prompt.txt geometry.png 2>&1
            fi
            local exit_code=$?

            # Restore directory and cleanup
            cd "$orig_dir"
            rm -rf "$sandbox" 2>/dev/null || echo "WARNING: Failed to clean up $sandbox" >&2

            [[ $exit_code -ne 0 ]] && echo "GEMINI_ERROR: exit code $exit_code"
            ;;
        ollama)
            # Cross-platform base64 (macOS doesn't support -w0)
            local sat_base64=""
            local geo_base64
            [[ -f "$img_sat" ]] && sat_base64=$(base64 "$img_sat" | tr -d '\n')
            geo_base64=$(base64 "$img_geo" | tr -d '\n')
            local model_name="${MODEL:-llava}"

            # Build JSON payload safely with jq to handle escaping
            # Include both images if satellite exists
            local json_payload
            if [[ -n "$sat_base64" ]]; then
                json_payload=$(jq -n \
                    --arg model "$model_name" \
                    --arg prompt "$prompt" \
                    --arg sat "$sat_base64" \
                    --arg geo "$geo_base64" \
                    '{model: $model, prompt: $prompt, images: [$sat, $geo], stream: false}')
            else
                json_payload=$(jq -n \
                    --arg model "$model_name" \
                    --arg prompt "$prompt" \
                    --arg geo "$geo_base64" \
                    '{model: $model, prompt: $prompt, images: [$geo], stream: false}')
            fi

            # Ollama vision models need more time than text-only (120s timeout)
            timeout 120 curl -s http://localhost:11434/api/generate \
                -d "$json_payload" | jq -r '.response // .error // "No response"'
            ;;
    esac
}

# Process candidates
CONSECUTIVE_FAILS=0
LAST_ERROR=""
for CANDIDATE_DIR in "$CANDIDATES_DIR"/*/; do
    # Check limit (only count actually processed, not skipped)
    if [[ -n "$LIMIT" ]] && [[ "$ACTUALLY_PROCESSED" -ge "$LIMIT" ]]; then
        echo "Reached limit of $LIMIT candidates"
        break
    fi

    DIR_NAME=$(basename "$CANDIDATE_DIR")
    REF_ID="${DIR_NAME%%__*}"
    TARGET_ID="${DIR_NAME##*__}"

    # Skip if already labeled (resume mode)
    KEY="${REF_ID},${TARGET_ID}"
    if [[ -n "$EXISTING_PAIRS" ]] && echo "$EXISTING_PAIRS" | grep -qxF "$KEY"; then
        echo "Skipping $REF_ID (already labeled)"
        SKIPPED=$((SKIPPED+1))
        continue
    fi

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
        CONSECUTIVE_FAILS=0
        echo "$(echo "$RESULT" | cut -d',' -f3) [$COUNT/$TOTAL]"
    else
        FAILED=$((FAILED+1))
        CONSECUTIVE_FAILS=$((CONSECUTIVE_FAILS+1))
        LAST_ERROR="${RAW:0:200}"
        echo "FAIL [$((COUNT+FAILED))/$TOTAL]" | tee -a "$LOG_FILE"
        echo "  Raw: $LAST_ERROR" >> "$LOG_FILE"

        # Check for bail-out condition
        if [[ "$BAIL_AFTER" -gt 0 ]] && [[ "$CONSECUTIVE_FAILS" -ge "$BAIL_AFTER" ]]; then
            echo ""
            echo "BAILING OUT: $CONSECUTIVE_FAILS consecutive failures detected"
            echo "  Likely cause: API quota exceeded or bad configuration"
            echo "  Last error: $LAST_ERROR"
            echo "  Completed: $COUNT success, $FAILED failed, $SKIPPED skipped"
            echo "  Run with --resume to continue later"
            break
        fi
    fi

    ACTUALLY_PROCESSED=$((ACTUALLY_PROCESSED+1))
done

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "=== Complete: $COUNT success, $FAILED failed, $SKIPPED skipped ==="
echo "Output: $OUTPUT_FILE"
echo "Log: $LOG_FILE"
