# Interference Effects in Multi-Condition ControlNet Stacking

This repository contains code, prompts, evaluation scripts, and supplementary data for the paper:

**Interference Effects in Multi-Condition ControlNet Stacking and Optimal Weight Allocation: Standardized Interference Coefficients, Fixed-A Variation, Three-Condition Expansion, Cross-Model Validation, and an Auto-Balancing Tool**

Xin Zhang, Dmitry Galkin  
Tomsk State University, Tomsk, Russia

## Contents

- `prompts/prompts_100.txt` – 100 prompts used across all conditions.
- `code/generate.py` – Core SDXL + dual ControlNet generation script.
- `code/evaluate.py` – Condition compliance metrics, SII, fixed-A decomposition, perceptual utility.
- `code/auto_balance.py` – Auto-balancing tool for ControlNet weight allocation.
- `code/utils.py` – Helper functions.
- `data/` – Supplementary CSV files for CFG generalization, seed sensitivity, and utility sensitivity.
- `appendix/` – Subjective evaluation instruction, CFG data, seed sensitivity, utility sensitivity.
- `comfyui/` – ComfyUI workflow description (export your own JSON if needed).

## Environment

- GPU: NVIDIA RTX 4070 Super 12GB
- OS: Windows 11 Pro 23H2
- Python: 3.10.11
- PyTorch: 2.1.0+cu121
- Diffusers: 0.25.0

Install dependencies:

```bash
pip install -r requirements.txt
