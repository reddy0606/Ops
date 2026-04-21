"""
genai/image_qa.py
────────────────────────────────────────────────────────
GENERATIVE AI — Visual Question Answering (VQA)

Powered by Anthropic Claude (claude-sonnet-4-20250514)

What it does:
  - Takes an image + a natural language question
  - Returns a detailed, intelligent answer using Claude's vision
  - Supports batch Q&A on multiple images
  - Includes a marketing analysis mode for business use

Setup:
  Create a .env file with:
    ANTHROPIC_API_KEY=your_key_here
  Or set the environment variable directly.
"""

import sys
sys.path.append("..")

import os
import base64
import json
from pathlib import Path
from PIL import Image
import io
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ─────────────────────────────────────────────────────
# Image → base64 helper
# ─────────────────────────────────────────────────────

def image_to_base64(pil_image: Image.Image, fmt: str = "JPEG") -> str:
    """Convert PIL image to base64 string for API call."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format=fmt)
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


def path_to_base64(image_path: str) -> tuple[str, str]:
    """Load image file → (base64, media_type)."""
    img = Image.open(image_path).convert("RGB")
    b64 = image_to_base64(img)
    return b64, "image/jpeg"


# ─────────────────────────────────────────────────────
# Claude Vision Q&A
# ─────────────────────────────────────────────────────

def ask_claude_about_image(
    image_b64: str,
    question: str,
    media_type: str = "image/jpeg",
    system_prompt: str = None,
) -> str:
    """
    Send an image + question to Claude and get an answer.

    Args:
        image_b64:   Base64-encoded image string
        question:    Natural language question about the image
        media_type:  MIME type (image/jpeg, image/png, image/webp)
        system_prompt: Optional system context

    Returns:
        Claude's answer as a string
    """
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found.\n"
            "Set it in .env or as environment variable:\n"
            "  export ANTHROPIC_API_KEY=your_key_here"
        )

    client = anthropic.Anthropic(api_key=api_key)

    system = system_prompt or (
        "You are an expert computer vision AI assistant. "
        "Analyse images thoroughly and answer questions clearly and concisely."
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]
    )
    return response.content[0].text


# ─────────────────────────────────────────────────────
# Ask a question about a file path
# ─────────────────────────────────────────────────────

def vqa_from_file(image_path: str, question: str) -> str:
    """
    Visual Q&A on a local image file.

    Example:
        answer = vqa_from_file("dog.jpg", "What breed is this dog?")
    """
    b64, media_type = path_to_base64(image_path)
    answer = ask_claude_about_image(b64, question, media_type)
    print(f"\n[VQA] Image    : {image_path}")
    print(f"[VQA] Question : {question}")
    print(f"[VQA] Answer   : {answer}")
    return answer


# ─────────────────────────────────────────────────────
# Batch VQA on CIFAR-10 samples
# ─────────────────────────────────────────────────────

def batch_vqa_cifar10(
    n_samples: int = 5,
    questions: list = None,
    save_json: str = "vqa_results.json"
):
    """
    Run Visual Q&A on random CIFAR-10 test images.

    Args:
        n_samples:  How many images to process
        questions:  List of questions per image (or use defaults)
        save_json:  Output file name

    Default questions asked for each image:
      1. What object is in this image?
      2. Describe the colours and textures.
      3. What is the confidence level of your detection?
    """
    default_questions = [
        "What is the main object or subject in this image? Be specific.",
        "Describe the colors, textures, and visual characteristics.",
        "On a scale of 1-10, how confident are you in your answer? Explain why.",
    ]
    questions = questions or default_questions

    print(f"\n[VQA] Batch processing {n_samples} CIFAR-10 images...")

    tf      = transforms.ToTensor()
    test_ds = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=tf
    )
    indices = torch.randperm(len(test_ds))[:n_samples]
    results = []

    for i, idx in enumerate(indices):
        img_tensor, label = test_ds[idx.item()]
        img_np  = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np).resize((128, 128), Image.LANCZOS)
        b64     = image_to_base64(pil_img)

        true_class = CIFAR10_CLASSES[label]
        print(f"\n  Image {i+1}/{n_samples} — true class: [{true_class}]")

        image_result = {
            "index": idx.item(),
            "true_class": true_class,
            "qa_pairs": []
        }

        for q in questions:
            try:
                answer = ask_claude_about_image(b64, q)
                print(f"  Q: {q}")
                print(f"  A: {answer[:120]}...")
                image_result["qa_pairs"].append({"question": q, "answer": answer})
            except Exception as e:
                print(f"  [VQA] Error: {e}")
                image_result["qa_pairs"].append({"question": q, "answer": f"Error: {e}"})

        results.append(image_result)

    # Save results
    out_path = OUTPUT_DIR / save_json
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[VQA] Results saved → {out_path}")
    return results


# ─────────────────────────────────────────────────────
# Marketing image analysis  (business use case)
# ─────────────────────────────────────────────────────

def analyze_marketing_image(image_path: str) -> dict:
    """
    Analyse a marketing/product image for business insights.

    Returns structured analysis:
      - Product description
      - Target audience
      - Sentiment & tone
      - Suggested ad copy
      - Improvement suggestions
    """
    system = """You are an expert marketing analyst and visual AI specialist.
Analyse marketing images and return your analysis ONLY as a valid JSON object
with these exact keys:
{
  "product_description": "...",
  "target_audience": "...",
  "sentiment": "positive|neutral|negative",
  "tone": "...",
  "suggested_ad_copy": "...",
  "improvements": ["...", "...", "..."],
  "score_out_of_10": 0
}"""

    b64, media_type = path_to_base64(image_path)
    question = "Analyse this marketing image and return structured JSON insights."

    raw = ask_claude_about_image(b64, question, media_type, system_prompt=system)

    try:
        clean = raw.strip().replace("```json", "").replace("```", "")
        result = json.loads(clean)
    except Exception:
        result = {"raw_response": raw}

    print(f"\n[VQA] Marketing Analysis for: {image_path}")
    print(json.dumps(result, indent=2))
    return result


# ─────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────

def run_vqa_pipeline(n_samples: int = 5):
    print("\n" + "="*55)
    print("  GENERATIVE AI — Visual Q&A (Claude Vision)")
    print("  Model  : claude-sonnet-4-20250514")
    print("="*55)

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n[VQA] ANTHROPIC_API_KEY not set.")
        print("  1. Get your key at: https://console.anthropic.com")
        print("  2. Create .env file: ANTHROPIC_API_KEY=your_key_here")
        print("  3. Re-run this module\n")
        return []

    return batch_vqa_cifar10(n_samples=n_samples)


if __name__ == "__main__":
    run_vqa_pipeline(n_samples=5)
