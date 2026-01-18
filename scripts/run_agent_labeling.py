#!/usr/bin/env python3
"""Multi-agent labeling driver script.

Orchestrates labeling of road segment matching candidates across multiple
AI agents (Claude, Codex, Gemini) for consensus-based validation.

Usage:
    # Label with all configured agents
    python scripts/run_agent_labeling.py agent_labels/batches/batch_2026-01-18_001

    # Label with specific agents
    python scripts/run_agent_labeling.py batch_dir --agents claude gemini

    # Dry run (show what would be done)
    python scripts/run_agent_labeling.py batch_dir --dry-run

Requirements:
    - ANTHROPIC_API_KEY environment variable for Claude
    - OPENAI_API_KEY environment variable for Codex
    - GOOGLE_API_KEY or gcloud auth for Gemini
"""

import argparse
import base64
import csv
import os
import sys
from dataclasses import dataclass
from datetime import UTC
from io import StringIO
from pathlib import Path

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class AgentConfig:
    """Configuration for an AI agent."""

    name: str
    model: str
    provider: str  # "anthropic", "openai", "google"
    max_tokens: int = 4096
    temperature: float = 0.1  # Low temperature for consistent labeling


# Agent configurations - using moderate cost models as specified
AGENT_CONFIGS = {
    "claude": AgentConfig(
        name="claude",
        model="claude-sonnet-4-5-20250514",  # Claude Sonnet 4.5
        provider="anthropic",
    ),
    "codex": AgentConfig(
        name="codex",
        model="codex-5.2-high-fast",  # Codex 5.2 high fast
        provider="openai",
    ),
    "gemini": AgentConfig(
        name="gemini",
        model="gemini-3-auto",  # Gemini 3 auto
        provider="google",
    ),
}


def load_context_prompt() -> str:
    """Load the context/instructions for agents."""
    context_path = Path(__file__).parent.parent / "docs" / "agent_labeling" / "CONTEXT.md"
    if context_path.exists():
        return context_path.read_text()
    return """You are labeling road segment matching candidates.
For each candidate, determine if the reference and target segments represent the same road.
Labels: match, no_match, unsure
Respond with CSV: ref_id,target_id,label,confidence,reasoning"""


def load_batch_manifest(batch_dir: Path) -> dict:
    """Load batch manifest."""
    manifest_path = batch_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return yaml.safe_load(manifest_path.read_text())


def load_candidate_metadata(candidate_dir: Path) -> dict:
    """Load candidate metadata."""
    metadata_path = candidate_dir / "metadata.yaml"
    if not metadata_path.exists():
        return {}
    return yaml.safe_load(metadata_path.read_text())


def encode_image(image_path: Path) -> str | None:
    """Encode image to base64 for API calls."""
    if not image_path.exists():
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_candidate_prompt(candidate_dir: Path) -> dict:
    """Build prompt content for a single candidate.

    Returns dict with 'text' and optionally 'images' keys.
    """
    metadata = load_candidate_metadata(candidate_dir)

    # Build text description
    text_parts = [
        "## Candidate to Label",
        "",
        f"**Reference ID**: {metadata.get('candidate', {}).get('ref_id', 'unknown')}",
        f"**Target ID**: {metadata.get('candidate', {}).get('target_id', 'unknown')}",
        "",
        "### Names",
        f"- Reference: {metadata.get('names', {}).get('reference', 'unnamed')}",
        f"- Target: {metadata.get('names', {}).get('target', 'unnamed')}",
        "",
        "### Classes",
        f"- Reference: {metadata.get('classes', {}).get('reference', 'unknown')}",
        f"- Target: {metadata.get('classes', {}).get('target', 'unknown')}",
        "",
        "### ML Prediction",
        f"- Decision: {metadata.get('ml_prediction', {}).get('decision', 'unknown')}",
        f"- Confidence: {metadata.get('ml_prediction', {}).get('confidence', 0):.2f}",
        "",
        "### Key Geometric Features",
    ]

    geo = metadata.get("features", {}).get("geometric", {})
    text_parts.extend(
        [
            f"- Buffer IoU: {geo.get('buffer_iou', 0):.2f}",
            f"- Overlap Ratio: {geo.get('overlap_ratio', 0):.2f}",
            f"- Hausdorff Distance: {geo.get('hausdorff_distance', 0):.1f}m",
            f"- Heading Delta: {geo.get('heading_delta', 0):.1f}°",
            f"- Length Ratio: {geo.get('length_ratio', 0):.2f}",
        ]
    )

    sem = metadata.get("features", {}).get("semantic", {})
    text_parts.extend(
        [
            "",
            "### Name Similarity",
            f"- Levenshtein: {sem.get('name_levenshtein', 0):.2f}",
            f"- Jaro-Winkler: {sem.get('name_jaro_winkler', 0):.2f}",
        ]
    )

    text_parts.extend(
        [
            "",
            "**Images are attached below. Blue line = reference, Red dashed = target.**",
            "",
            "Based on the geometric alignment and features, provide your label.",
        ]
    )

    result = {"text": "\n".join(text_parts)}

    # Load images
    geometry_path = candidate_dir / "geometry.png"
    satellite_path = candidate_dir / "satellite.png"

    images = []
    if geometry_path.exists():
        images.append(
            {"name": "geometry.png", "data": encode_image(geometry_path), "type": "image/png"}
        )
    if satellite_path.exists():
        images.append(
            {"name": "satellite.png", "data": encode_image(satellite_path), "type": "image/png"}
        )

    if images:
        result["images"] = images

    return result


def call_anthropic(
    config: AgentConfig,
    system_prompt: str,
    candidates: list[dict],
) -> list[dict]:
    """Call Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        print("Error: anthropic package not installed. Run: pip install anthropic")
        return []

    client = anthropic.Anthropic()

    # Build messages with all candidates
    messages_content = []

    for candidate in candidates:
        # Add text
        messages_content.append({"type": "text", "text": candidate["text"]})

        # Add images if present
        for img in candidate.get("images", []):
            if img["data"]:
                messages_content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img["type"],
                            "data": img["data"],
                        },
                    }
                )

    messages_content.append(
        {
            "type": "text",
            "text": "\n\nNow provide your labels for ALL candidates above as CSV (ref_id,target_id,label,confidence,reasoning). One row per candidate.",
        }
    )

    try:
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": messages_content}],
        )

        # Parse CSV response
        response_text = response.content[0].text
        return parse_csv_response(response_text)

    except Exception as e:
        print(f"Anthropic API error: {e}")
        return []


def call_openai(
    config: AgentConfig,
    system_prompt: str,
    candidates: list[dict],
) -> list[dict]:
    """Call OpenAI/Codex API."""
    try:
        import openai
    except ImportError:
        print("Error: openai package not installed. Run: pip install openai")
        return []

    client = openai.OpenAI()

    # Build messages with all candidates
    messages_content = []

    for candidate in candidates:
        # Add text
        messages_content.append({"type": "text", "text": candidate["text"]})

        # Add images if present
        for img in candidate.get("images", []):
            if img["data"]:
                messages_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img['type']};base64,{img['data']}"},
                    }
                )

    messages_content.append(
        {
            "type": "text",
            "text": "\n\nNow provide your labels for ALL candidates above as CSV (ref_id,target_id,label,confidence,reasoning). One row per candidate.",
        }
    )

    try:
        response = client.chat.completions.create(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": messages_content},
            ],
        )

        response_text = response.choices[0].message.content
        return parse_csv_response(response_text)

    except Exception as e:
        print(f"OpenAI API error: {e}")
        return []


def call_google(
    config: AgentConfig,
    system_prompt: str,
    candidates: list[dict],
) -> list[dict]:
    """Call Google Gemini API."""
    try:
        import google.generativeai as genai
    except ImportError:
        print(
            "Error: google-generativeai package not installed. Run: pip install google-generativeai"
        )
        return []

    # Configure API
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)

    model = genai.GenerativeModel(config.model)

    # Build prompt parts
    parts = [system_prompt, "\n\n"]

    for candidate in candidates:
        parts.append(candidate["text"])

        # Add images
        for img in candidate.get("images", []):
            if img["data"]:
                import io

                from PIL import Image

                img_data = base64.b64decode(img["data"])
                pil_image = Image.open(io.BytesIO(img_data))
                parts.append(pil_image)

    parts.append(
        "\n\nNow provide your labels for ALL candidates above as CSV (ref_id,target_id,label,confidence,reasoning). One row per candidate."
    )

    try:
        response = model.generate_content(
            parts,
            generation_config=genai.GenerationConfig(
                max_output_tokens=config.max_tokens,
                temperature=config.temperature,
            ),
        )

        response_text = response.text
        return parse_csv_response(response_text)

    except Exception as e:
        print(f"Google API error: {e}")
        return []


def parse_csv_response(response_text: str) -> list[dict]:
    """Parse CSV labels from agent response."""
    labels = []

    # Find CSV content (might be in code blocks)
    text = response_text

    # Try to extract from code blocks
    if "```csv" in text:
        start = text.index("```csv") + 6
        end = text.index("```", start)
        text = text[start:end]
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end]

    # Parse CSV
    try:
        reader = csv.DictReader(StringIO(text.strip()))
        for row in reader:
            if "ref_id" in row and "target_id" in row and "label" in row:
                labels.append(
                    {
                        "ref_id": row["ref_id"].strip(),
                        "target_id": row["target_id"].strip(),
                        "label": row["label"].strip().lower(),
                        "confidence": float(row.get("confidence", 1.0)),
                        "reasoning": row.get("reasoning", "").strip(),
                    }
                )
    except Exception as e:
        print(f"CSV parsing error: {e}")

        # Try line-by-line parsing as fallback
        for line in text.strip().split("\n"):
            if "," in line and not line.startswith("ref_id"):
                parts = line.split(",", 4)
                if len(parts) >= 3:
                    try:
                        labels.append(
                            {
                                "ref_id": parts[0].strip(),
                                "target_id": parts[1].strip(),
                                "label": parts[2].strip().lower(),
                                "confidence": float(parts[3]) if len(parts) > 3 else 1.0,
                                "reasoning": parts[4] if len(parts) > 4 else "",
                            }
                        )
                    except (ValueError, IndexError):
                        continue

    return labels


def run_agent(
    config: AgentConfig,
    batch_dir: Path,
    candidates_to_label: list[Path],
    batch_size: int = 5,
) -> list[dict]:
    """Run a single agent on the batch."""
    print(f"\nRunning agent: {config.name} ({config.model})")

    system_prompt = load_context_prompt()
    all_labels = []

    # Process in batches to avoid token limits
    for i in range(0, len(candidates_to_label), batch_size):
        batch = candidates_to_label[i : i + batch_size]
        print(f"  Processing candidates {i + 1}-{i + len(batch)} of {len(candidates_to_label)}")

        # Build prompts for this batch
        candidate_prompts = [build_candidate_prompt(c) for c in batch]

        # Call appropriate API
        if config.provider == "anthropic":
            labels = call_anthropic(config, system_prompt, candidate_prompts)
        elif config.provider == "openai":
            labels = call_openai(config, system_prompt, candidate_prompts)
        elif config.provider == "google":
            labels = call_google(config, system_prompt, candidate_prompts)
        else:
            print(f"  Unknown provider: {config.provider}")
            continue

        all_labels.extend(labels)
        print(f"  Got {len(labels)} labels")

    return all_labels


def save_agent_labels(batch_dir: Path, agent_id: str, labels: list[dict]):
    """Save agent labels to CSV."""
    labels_dir = batch_dir / "labels" / agent_id
    labels_dir.mkdir(parents=True, exist_ok=True)

    csv_path = labels_dir / "data.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ref_id",
                "target_id",
                "label",
                "confidence",
                "reasoning",
                "agent_id",
                "labeled_at",
            ],
        )
        writer.writeheader()

        from datetime import datetime

        now = datetime.now(UTC).isoformat()

        for label in labels:
            writer.writerow(
                {
                    **label,
                    "agent_id": agent_id,
                    "labeled_at": now,
                }
            )

    print(f"Saved {len(labels)} labels to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Run multi-agent labeling on a batch")
    parser.add_argument("batch_dir", type=Path, help="Batch directory")
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=list(AGENT_CONFIGS.keys()),
        default=list(AGENT_CONFIGS.keys()),
        help="Agents to run (default: all)",
    )
    parser.add_argument("--batch-size", type=int, default=5, help="Candidates per API call")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--limit", type=int, help="Limit number of candidates to process")

    args = parser.parse_args()

    # Validate batch directory
    if not args.batch_dir.exists():
        print(f"Error: Batch directory not found: {args.batch_dir}")
        sys.exit(1)

    # Load manifest
    try:
        manifest = load_batch_manifest(args.batch_dir)
        print(f"Batch: {manifest.get('batch_id', 'unknown')}")
        print(f"Dataset: {manifest.get('dataset', 'unknown')}")
        print(f"Total candidates: {manifest.get('total_candidates', 0)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Find candidate directories
    candidates_dir = args.batch_dir / "candidates"
    if not candidates_dir.exists():
        print(f"Error: Candidates directory not found: {candidates_dir}")
        sys.exit(1)

    candidate_dirs = sorted([d for d in candidates_dir.iterdir() if d.is_dir()])

    if args.limit:
        candidate_dirs = candidate_dirs[: args.limit]

    print(f"Candidates to process: {len(candidate_dirs)}")
    print(f"Agents: {', '.join(args.agents)}")

    if args.dry_run:
        print("\n[DRY RUN] Would process the above with specified agents.")
        return

    # Run each agent
    for agent_name in args.agents:
        config = AGENT_CONFIGS[agent_name]

        # Check for API key
        if config.provider == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                print(f"Skipping {agent_name}: ANTHROPIC_API_KEY not set")
                continue
        elif config.provider == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                print(f"Skipping {agent_name}: OPENAI_API_KEY not set")
                continue
        elif config.provider == "google":
            if not os.environ.get("GOOGLE_API_KEY"):
                print(f"Skipping {agent_name}: GOOGLE_API_KEY not set")
                continue

        labels = run_agent(config, args.batch_dir, candidate_dirs, args.batch_size)

        if labels:
            save_agent_labels(args.batch_dir, agent_name, labels)

    print("\nDone! Use 'matcher agent-consensus' to analyze results.")


if __name__ == "__main__":
    main()
