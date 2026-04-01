# SLM Fine-Tuning Lab

A hands-on, progressive study of Small Language Model fine-tuning techniques — from full fine-tuning to LoRA and QLoRA — with benchmarks tracking real improvement at each stage.

Built as a personal learning project and portfolio reference for AI engineering.

---

## Hardware

| Component | Spec |
|-----------|------|
| Device | MacBook Pro M4 |
| RAM | 32 GB unified memory |
| Storage | 1 TB SSD |

> All experiments are designed to run **locally** on Apple Silicon. No cloud GPU required.

---

## Learning Roadmap

| Day | Topic | Key Technique | Goal |
|-----|-------|---------------|------|
| 01 | Foundations & Environment | Tokenization, forward pass | Run first model locally |
| 02 | Full Fine-Tuning | Trainer API, hyperparameters | Establish benchmark baseline |
| 03 | Efficient Fine-Tuning | LoRA (PEFT) | Compare vs full fine-tuning |
| 04 | Quantized Fine-Tuning | QLoRA, 4-bit | Fine-tune 7B model on 32 GB RAM |
| 05 | Evaluation & Benchmarks | ROUGE, BERTScore, LM Eval | Measure real-world improvement |

---

## Benchmark Results

Results are accumulated across days in [`results/benchmarks.csv`](results/benchmarks.csv).

| Model | Technique | ROUGE-L | BERTScore | Perplexity | VRAM (GB) |
|-------|-----------|---------|-----------|------------|-----------|
| *baseline* | — | — | — | — | — |

> Table updated progressively from Day 2 onward.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/slm-finetuning-lab.git
cd slm-finetuning-lab

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

To validate your environment:

```bash
python days/day01_foundations/setup_check.py
```

---

## Project Structure

```
slm-finetuning-lab/
├── days/           # One folder per study day, self-contained
├── data/           # Datasets (descriptions + sources in data/README.md)
├── models/         # Local checkpoints (gitignored)
├── results/        # Benchmark outputs and summary CSV
└── notebooks/      # Scratch space for exploration
```

---

## Concepts Covered

- Tokenization and vocabulary
- Supervised fine-tuning (SFT) loop
- Learning rate, batch size, and epoch selection
- Overfitting detection and early stopping
- Low-Rank Adaptation (LoRA): rank, alpha, target modules
- 4-bit quantization (NF4) and QLoRA
- Evaluation metrics: ROUGE, BERTScore, perplexity
- Benchmark harness with EleutherAI LM Evaluation

---

## License

MIT