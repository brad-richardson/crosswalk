# Agent Labeling - Future Work

## Recommended Workflow: Gemini First-Pass with Human Review

Based on testing (141 candidates), Gemini achieves **88% accuracy** and is recommended
for bulk pre-labeling with human review of uncertain cases.

### Workflow

1. **Run Gemini on all unlabeled candidates**
   ```bash
   ./run_agent.sh gemini --batch batches/<batch_name>
   ```

2. **Auto-accept high-confidence predictions** (confidence ≥ 0.9)
   - These are ~70-80% of predictions
   - Expected accuracy: ~95%+ for high-confidence subset

3. **Human reviews low-confidence predictions** (confidence < 0.9)
   - Focus manual effort on uncertain cases (~20-30%)
   - Also spot-check random sample of auto-accepted labels

4. **Retrain ML model** as labels accumulate

### Agent Accuracy Results (141 candidates)

| Agent | With Satellite | Geometry-Only | Notes |
|-------|---------------|---------------|-------|
| Gemini | 88.7% | 88.0% | **Recommended** - satellite adds no value |
| Claude Sonnet | 73.0% | 66.0% | Satellite helps ~7% |
| Codex (gpt-5.2) | 63.1% | - | High quota usage |

### Satellite Imagery: Skip It

Testing showed Gemini gets nearly identical accuracy with geometry-only images:
- Satellite + geometry: 88.7%
- Geometry only: 88.0%
- **Recommendation**: Use geometry-only to save fetch time and storage

The current `run_agent.sh` is configured for geometry-only mode.

## Usage

Unified script for all agents:
```bash
./run_agent.sh <agent> [--batch <dir>] [--model <model>] [--limit <n>]

# Examples:
./run_agent.sh gemini                              # Default batch
./run_agent.sh gemini --limit 50                   # First 50 candidates
./run_agent.sh claude --model sonnet --limit 100
./run_agent.sh ollama --model llava                # Local GPU mode
```

## Batch Processing (Lower Priority)

Current approach sends one candidate per API call, which burns through quotas quickly.

**Proposed change**: Batch multiple candidates into single prompts.

Example prompt structure:
```
Here is the labeling context: [LABELING_INSTRUCTIONS.md]

Process these 20 candidates and output CSV lines for each:

Candidate 1: [ref_id] <-> [target_id]
- Image: [path to image 1]
- Metadata: [key features]

Candidate 2: ...
...

Output format:
ref_id,target_id,label,confidence,reason
```

**Considerations**:
- Context window limits (images are large ~400-500KB each)
- May need to batch 5-10 candidates with images, or 20+ without images
- Could use geometry.png (smaller) instead of satellite.png for batching
- Trade-off: less per-candidate reasoning vs fewer API calls

## Model-Specific Notes

### Gemini (Recommended)
- Best accuracy (88%) with reasonable quota usage
- Runs in Docker sandbox for isolation
- ~15-20s per candidate
- Barely uses satellite imagery - geometry-only is fine

### Claude Sonnet
- Moderate accuracy (66-73%)
- Benefits from satellite imagery (+7%)
- ~10s per candidate
- Claude Haiku too small for this task

### Codex
- Lower accuracy (63%)
- Burns through API quota very quickly
- Not recommended for bulk labeling

## Local Multimodal Models (Implemented)

Run vision models locally via Ollama to avoid API costs/quotas.

### Setup

```bash
# One-time setup
./setup_ollama.sh
```

This installs Ollama and pulls vision models.

### Usage

```bash
# Requires GPU - task is too complex for CPU-only inference
./run_agent.sh ollama --model llava
```

### Available Models

| Model | Size | Quality | Speed | Notes |
|-------|------|---------|-------|-------|
| moondream | 1.8B | Poor | Fast | Too small for this task - mostly echoes prompts |
| llava | 7B | Good | Medium | Recommended minimum |
| llava:13b | 13B | Better | Slow | Better quality, needs more VRAM |

Pull additional models with `ollama pull <model>`.

**Note on small models**: Testing with moondream (1.8B) showed very poor results -
the model struggled with the spatial reasoning required and mostly echoed the
prompt template back. This task requires at least a 7B+ parameter model.
CPU-only inference is not viable due to speed constraints (~160s per candidate).

### Remote Ollama

For machines without a GPU, you can run Ollama on a remote machine with a GPU:

```bash
# On GPU machine: start Ollama with network access
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# On local machine: point to remote
export OLLAMA_HOST=http://gpu-machine:11434
./run_agent.sh ollama --model llava
```

### Alternative: llama.cpp + SYCL (Future)

For better performance on Intel Arc GPUs:
```bash
cmake -B build -DGGML_SYCL=ON
./build/bin/llava-cli -m model.gguf --image satellite.png -p "prompt"
```
- 20-50% faster than Ollama
- More setup work, vision model support is newer

## Robustness: Failure Handling and Resume (High Priority)

Current `run_agent.sh` doesn't handle API failures gracefully. When agents hit quota limits (e.g., Gemini mid-run), the batch stops but there's no easy way to resume.

### Required Features

1. **Early termination on consecutive failures**
   - If 2 API calls fail in a row (quota, rate limit, etc.), stop immediately
   - Don't waste time/quota on a doomed batch
   - Log clear error: "Stopping: 2 consecutive failures (likely quota exhausted)"

2. **Resume batch from where it left off**
   - Skip candidates that already have labels in the output CSV
   - Allow: `./run_agent.sh gemini --batch <dir> --resume`
   - Check existing `labels/<agent>/data.csv` and skip those `target_id`s

3. **Better failure logging**
   - Log each failure with timestamp and error type
   - Summary at end: "Completed: 100/124, Failed: 2, Skipped: 22 (already labeled)"

### Implementation Notes

```bash
# In run_agent.sh, track consecutive failures:
consecutive_failures=0
for candidate in candidates; do
    if ! process_candidate "$candidate"; then
        ((consecutive_failures++))
        if [ $consecutive_failures -ge 2 ]; then
            echo "ERROR: 2 consecutive failures, stopping (likely quota exhausted)"
            exit 1
        fi
    else
        consecutive_failures=0  # Reset on success
    fi
done
```

For resume, load existing labels at start:
```bash
existing_labels=$(cut -d',' -f2 "$output_csv" 2>/dev/null | tail -n +2)
# Skip if target_id already in existing_labels
```

## Other Ideas

- Parallel processing across multiple agent instances
- Caching of LABELING_INSTRUCTIONS.md in prompts
- Metadata-only mode for quick triage (skip images)
