# Agent Labeling - Future Work

## Usage

Unified script for all agents:
```bash
./run_agent.sh <agent> [batch_dir] [--model <model>] [--grayscale] [--low-res]

# Examples:
./run_agent.sh gemini --model flash
./run_agent.sh claude --model haiku --low-res
./run_agent.sh codex --grayscale
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

## Local Multimodal Models (No API Quota)

Run vision models locally on Intel Arc iGPU to avoid API costs/quotas.

### Recommended Models

| Model | Size | Quality | Speed (Arc iGPU) | Notes |
|-------|------|---------|------------------|-------|
| moondream2 | 1.8B | Good | ~20-40 t/s | Best for efficiency |
| LLaVA-1.6-Mistral-7B | 7B | Great | ~8-15 t/s | Good balance |
| Qwen2-VL-7B | 7B | Excellent | ~8-15 t/s | State-of-art |
| PaliGemma-3B | 3B | Good | ~15-25 t/s | Google's efficient option |

### Runtime Options

**Option 1: Ollama + IPEX** (Recommended for simplicity)
```bash
# Install Ollama with Intel GPU support
ollama run moondream

# Add to run_agent.sh as new agent type
./run_agent.sh ollama --model moondream
```
- Easy setup, API-compatible
- ~10-20 tokens/sec for 7B models

**Option 2: llama.cpp + SYCL** (Better performance)
```bash
# Compile with Intel SYCL backend
cmake -B build -DGGML_SYCL=ON
./build/bin/llava-cli -m model.gguf --image satellite.png -p "prompt"
```
- 20-50% faster than Ollama
- More setup work, vision model support is newer

### System Requirements (tested)
- Intel Core Ultra 7 265K + Arc 140V iGPU
- 62GB RAM (can run 7B+ models on CPU if needed)

### Implementation Notes
- Add `ollama` agent type to run_agent.sh
- Use Ollama's OpenAI-compatible API for easy integration
- Consider running multiple instances in parallel (no quota limits)

## Other Ideas

- Resume from last processed candidate (currently restarts from beginning)
- Parallel processing across multiple agent instances
- Caching of LABELING_INSTRUCTIONS.md in prompts
- Metadata-only mode for quick triage (skip images)
