# AI Vision Suite — End-to-End Python AI System
## ML + Deep Learning + Generative AI for Image Vision

---

## Project Structure

```
ai_vision_suite/
│
├── README.md                   ← You are here
├── requirements.txt            ← All dependencies
├── run_all.py                  ← Master runner (runs everything)
│
├── ml/
│   └── classical_classifier.py ← ML: SVM / Random Forest on image features
│
├── dl/
│   ├── cnn_classifier.py       ← DL: Custom CNN with PyTorch
│   └── transfer_learning.py    ← DL: ResNet50 fine-tuning (transfer learning)
│
├── genai/
│   ├── image_captioner.py      ← Gen AI: Image-to-text captioning
│   └── image_qa.py             ← Gen AI: Visual Q&A with LLM
│
├── utils/
│   ├── data_loader.py          ← Dataset loading & augmentation
│   ├── visualizer.py           ← Plot results, confusion matrix, metrics
│   └── evaluator.py            ← Unified evaluation: accuracy, F1, AUC
│
└── outputs/                    ← Saved models, plots, reports
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full suite
```bash
python run_all.py
```

### 3. Run individual modules
```bash
python ml/classical_classifier.py       # ML only
python dl/cnn_classifier.py             # Deep Learning CNN
python dl/transfer_learning.py          # Transfer Learning ResNet
python genai/image_captioner.py         # Gen AI captioning
python genai/image_qa.py                # Visual Q&A
```

---

## What Each Module Does

| Module | Type | Task | Model |
|--------|------|------|-------|
| `classical_classifier.py` | ML | Feature extraction + classify | SVM, Random Forest |
| `cnn_classifier.py` | Deep Learning | Image classification | Custom CNN (PyTorch) |
| `transfer_learning.py` | Deep Learning | Fine-tune pretrained model | ResNet50 |
| `image_captioner.py` | Gen AI | Generate image descriptions | BLIP / ViT-GPT2 |
| `image_qa.py` | Gen AI | Answer questions about images | LLaVA / Anthropic API |

---

## Dataset
- Uses CIFAR-10 by default (auto-downloads, 10 classes, 60K images)
- Swap with your own dataset — see `utils/data_loader.py`

---

## Outputs
All outputs saved to `/outputs/`:
- `ml_results.png` — ML confusion matrix + metrics
- `cnn_training.png` — CNN loss/accuracy curves
- `transfer_results.png` — Transfer learning metrics
- `captions.txt` — Generated image captions
- `best_model.pth` — Saved CNN weights
