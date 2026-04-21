"""
genai/image_captioner.py
────────────────────────────────────────────────────────
GENERATIVE AI — Automatic Image Captioning

Models supported:
  1. BLIP  (Salesforce/blip-image-captioning-base)
     - Bootstrap Language-Image Pre-training
     - State-of-the-art image-to-text model
     - Works great on natural images

  2. ViT-GPT2  (nlpconnect/vit-gpt2-image-captioning)
     - Vision Transformer encoder + GPT-2 decoder
     - Faster, lighter alternative

Usage:
  - Pass a PIL image or file path → get a natural language caption
  - Also supports batch captioning on CIFAR-10 samples
"""

import sys
sys.path.append("..")

import torch
from PIL import Image
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
import numpy as np

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────
# Load captioning model
# ─────────────────────────────────────────────────────

def load_captioner(model_name: str = "blip"):
    """
    Load a pretrained image captioning model.

    Args:
        model_name: 'blip' or 'vit-gpt2'

    Returns:
        (model, processor, generate_fn)
    """
    print(f"[GenAI] Loading captioning model: {model_name}...")

    if model_name == "blip":
        from transformers import BlipProcessor, BlipForConditionalGeneration
        model_id  = "Salesforce/blip-image-captioning-base"
        processor = BlipProcessor.from_pretrained(model_id)
        model     = BlipForConditionalGeneration.from_pretrained(
                        model_id, torch_dtype=torch.float16
                    ).to(DEVICE)

        def generate(pil_image, prompt=None):
            if prompt:
                inputs = processor(pil_image, prompt, return_tensors="pt").to(DEVICE, torch.float16)
            else:
                inputs = processor(pil_image, return_tensors="pt").to(DEVICE, torch.float16)
            output = model.generate(**inputs, max_new_tokens=50, num_beams=4)
            return processor.decode(output[0], skip_special_tokens=True)

    elif model_name == "vit-gpt2":
        from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
        model_id  = "nlpconnect/vit-gpt2-image-captioning"
        model     = VisionEncoderDecoderModel.from_pretrained(model_id).to(DEVICE)
        processor = ViTImageProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        def generate(pil_image, prompt=None):
            pixel_values = processor(
                images=[pil_image], return_tensors="pt"
            ).pixel_values.to(DEVICE)
            output_ids = model.generate(
                pixel_values, max_length=50, num_beams=4,
                early_stopping=True
            )
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'blip' or 'vit-gpt2'")

    print(f"[GenAI] Model loaded on {DEVICE}")
    return model, processor, generate


# ─────────────────────────────────────────────────────
# Caption a single image
# ─────────────────────────────────────────────────────

def caption_image(image_path: str, model_name: str = "blip") -> str:
    """
    Generate a caption for a single image file.

    Args:
        image_path: Path to image (JPG, PNG, etc.)
        model_name: 'blip' or 'vit-gpt2'

    Returns:
        Caption string
    """
    _, _, generate = load_captioner(model_name)
    img = Image.open(image_path).convert("RGB")
    caption = generate(img)
    print(f"\n[GenAI] Image : {image_path}")
    print(f"[GenAI] Caption: {caption}")
    return caption


# ─────────────────────────────────────────────────────
# Batch caption CIFAR-10 samples
# ─────────────────────────────────────────────────────

def caption_cifar10_samples(
    n_samples: int = 10,
    model_name: str = "blip",
    save_txt: str = "captions.txt"
):
    """
    Generate captions for random CIFAR-10 test images.
    Saves results to outputs/captions.txt.

    Args:
        n_samples:  Number of images to caption
        model_name: Model to use
        save_txt:   Output filename
    """
    CIFAR10_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    print(f"\n[GenAI] Captioning {n_samples} CIFAR-10 images with {model_name}...")
    _, _, generate = load_captioner(model_name)

    tf = transforms.ToTensor()
    test_ds = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=tf
    )

    indices = torch.randperm(len(test_ds))[:n_samples]
    results = []

    for idx in indices:
        img_tensor, label = test_ds[idx.item()]
        # Convert tensor (C,H,W) → PIL image (resize for better captioning)
        img_np  = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np).resize((128, 128), Image.LANCZOS)

        caption  = generate(pil_img)
        true_cls = CIFAR10_CLASSES[label]
        result   = {"index": idx.item(), "true_class": true_cls, "caption": caption}
        results.append(result)
        print(f"  [{true_cls:>10}] → {caption}")

    # Save to file
    out_path = OUTPUT_DIR / save_txt
    with open(out_path, "w") as f:
        f.write("AI Vision Suite — Image Captions\n")
        f.write("=" * 50 + "\n\n")
        for r in results:
            f.write(f"True class : {r['true_class']}\n")
            f.write(f"Caption    : {r['caption']}\n")
            f.write("-" * 40 + "\n")

    print(f"\n[GenAI] Captions saved → {out_path}")
    return results


# ─────────────────────────────────────────────────────
# Conditional captioning  (prompt-guided)
# ─────────────────────────────────────────────────────

def conditional_caption(image_path: str, prompt: str, model_name: str = "blip") -> str:
    """
    Generate a caption guided by a text prompt.
    Only supported by BLIP.

    Example:
        conditional_caption("dog.jpg", "a photo of a")
        → "a photo of a golden retriever playing on the beach"
    """
    _, _, generate = load_captioner(model_name)
    img     = Image.open(image_path).convert("RGB")
    caption = generate(img, prompt=prompt)
    print(f"\n[GenAI] Prompt  : {prompt}")
    print(f"[GenAI] Caption : {caption}")
    return caption


# ─────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────

def run_captioning_pipeline(n_samples: int = 10, model_name: str = "blip"):
    print("\n" + "="*55)
    print("  GENERATIVE AI — Image Captioning")
    print(f"  Model  : {model_name}")
    print(f"  Device : {DEVICE}")
    print("="*55)
    results = caption_cifar10_samples(n_samples, model_name)
    return results


if __name__ == "__main__":
    run_captioning_pipeline(n_samples=10, model_name="blip")
