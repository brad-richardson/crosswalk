# Agent Labeling - Future Work

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

## Other Ideas

- Resume from last processed candidate (currently restarts from beginning)
- Parallel processing across multiple agent instances
- Caching of LABELING_INSTRUCTIONS.md in prompts
- Metadata-only mode for quick triage (skip images)
