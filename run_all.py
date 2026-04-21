"""
run_all.py
────────────────────────────────────────────────────────
AI VISION SUITE — Master Runner

Runs the complete end-to-end pipeline:
  1. ML   — SVM + Random Forest (HOG features)
  2. DL   — Custom CNN (PyTorch)
  3. DL   — ResNet50 Transfer Learning
  4. GenAI — Image Captioning (BLIP)
  5. GenAI — Visual Q&A (Claude Vision API)
  6. Final — Model comparison chart

Usage:
  python run_all.py                     # Full pipeline
  python run_all.py --skip-transfer     # Skip (slow) transfer learning
  python run_all.py --quick             # Fast test (few epochs/samples)
  python run_all.py --module ml         # Single module only
"""

import sys
import argparse
import time
from pathlib import Path

def banner(title: str, icon: str = ""):
    width = 55
    print("\n" + "█" * width)
    print(f"  {icon}  {title}")
    print("█" * width)


def print_summary(results: dict, elapsed: float):
    print("\n" + "="*55)
    print("  FINAL RESULTS — All Models")
    print("="*55)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(acc / 5)
        print(f"  {name:<22} {acc:>6.1f}%  {bar}")
    print(f"\n  Total time: {elapsed/60:.1f} minutes")
    print("="*55)


def main():
    parser = argparse.ArgumentParser(description="AI Vision Suite")
    parser.add_argument("--module", choices=["ml","cnn","transfer","caption","vqa"],
                        help="Run a single module only")
    parser.add_argument("--skip-transfer", action="store_true",
                        help="Skip ResNet transfer learning (slow on CPU)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer epochs and samples for testing")
    args = parser.parse_args()

    # Config
    cnn_epochs     = 5  if args.quick else 30
    p1_epochs      = 3  if args.quick else 10
    p2_epochs      = 3  if args.quick else 15
    n_cap_samples  = 3  if args.quick else 10
    n_vqa_samples  = 2  if args.quick else 5

    all_results = {}
    t_start = time.time()

    print("\n")
    print("  ██████████████████████████████████████████████████")
    print("  ██                                              ██")
    print("  ██    AI VISION SUITE — Python AI Framework    ██")
    print("  ██    ML  |  Deep Learning  |  Gen AI          ██")
    print("  ██                                              ██")
    print("  ██████████████████████████████████████████████████")
    print(f"\n  Mode: {'quick' if args.quick else 'full'}")
    print(f"  Outputs will be saved to: ./outputs/\n")

    # ── Module 1: Classical ML ───────────────────────
    if not args.module or args.module == "ml":
        banner("STEP 1 — Machine Learning (SVM + Random Forest)", "ML")
        from ml.classical_classifier import run_ml_pipeline
        ml_results = run_ml_pipeline()
        all_results.update(ml_results)

    # ── Module 2: Custom CNN ─────────────────────────
    if not args.module or args.module == "cnn":
        banner("STEP 2 — Deep Learning: Custom CNN (PyTorch)", "DL")
        from dl.cnn_classifier import run_cnn_pipeline
        cnn_results = run_cnn_pipeline(epochs=cnn_epochs)
        all_results.update(cnn_results)

    # ── Module 3: Transfer Learning ──────────────────
    if not args.module or args.module == "transfer":
        if not args.skip_transfer:
            banner("STEP 3 — Deep Learning: ResNet50 Transfer Learning", "DL")
            from dl.transfer_learning import run_transfer_learning
            tl_results = run_transfer_learning(
                phase1_epochs=p1_epochs,
                phase2_epochs=p2_epochs
            )
            all_results.update(tl_results)
        else:
            print("\n[Skipped] ResNet50 transfer learning (--skip-transfer)")

    # ── Module 4: Image Captioning ───────────────────
    if not args.module or args.module == "caption":
        banner("STEP 4 — Generative AI: Image Captioning (BLIP)", "AI")
        from genai.image_captioner import run_captioning_pipeline
        run_captioning_pipeline(n_samples=n_cap_samples, model_name="blip")

    # ── Module 5: Visual Q&A ─────────────────────────
    if not args.module or args.module == "vqa":
        banner("STEP 5 — Generative AI: Visual Q&A (Claude Vision)", "AI")
        from genai.image_qa import run_vqa_pipeline
        run_vqa_pipeline(n_samples=n_vqa_samples)

    # ── Final: Comparison chart ──────────────────────
    if all_results and (not args.module):
        banner("FINAL — Model Comparison", "CHART")
        from utils.visualizer import plot_model_comparison
        plot_model_comparison(all_results, save_as="model_comparison.png")
        print_summary(all_results, time.time() - t_start)

    print("\n  All outputs saved to ./outputs/")
    print("  Run complete.\n")


if __name__ == "__main__":
    main()
