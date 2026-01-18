# Agent Labeling - Future Work

## Usage

Unified script for all agents:
```bash
./run_agent.sh <agent> [batch_dir] [--model <model>]

# Examples:
./run_agent.sh gemini --model flash
./run_agent.sh claude --model haiku
./run_agent.sh codex
./run_agent.sh ollama --model llava  # Local GPU mode
```

## Batch Processing (Priority)

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

## Model-Specific Issues Discovered

### Gemini
- Default model can't read images - must use `-m flash`
- Slow initialization (~30s just for "Loaded cached credentials")
- Works well with flash model (~8-15s per candidate)

### Codex
- Works reliably at ~15-20s per candidate
- Quota limits hit at ~82 candidates

### Claude
- Works well but expensive for Sonnet
- Consider Haiku for bulk labeling

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

## Other Ideas

- Resume from last processed candidate (currently restarts from beginning)
- Parallel processing across multiple agent instances
- Caching of LABELING_INSTRUCTIONS.md in prompts
- Metadata-only mode for quick triage (skip images)
