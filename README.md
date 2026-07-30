# Quantized Hybrid Brain-Computer Interfaces

> **Quantized Hybrid Brain-Computer Interfaces: Evaluating Quantization-Aware Training for Multi-Paradigm EEG Classification on Edge Devices**
>
> Accepted at the **IEEE International Conference on Systems, Man, and Cybernetics (SMC), 2026**

This repository contains the official implementation of **SwitchBraidNet**, a lightweight EEG classification architecture designed for quantized hybrid Brain-Computer Interface (hBCI) systems. The framework supports Quantization-Aware Training (QAT) at multiple precision levels (FP32, FP16, INT8) and evaluates both single-paradigm and hybrid BCI performance using the [OpenBMI](http://gigadb.org/dataset/100542) dataset.

---

## Highlights

- **SwitchBraidNet** — A compact architecture combining dual-scale temporal convolutions, Squeeze-and-Excitation attention, and log-variance pooling for efficient EEG decoding.
- **Quantization-Aware Training** — Train and evaluate models at 32-bit, 16-bit, and 8-bit precision for edge deployment.
- **Hybrid BCI Evaluation** — Automatically evaluates all single-paradigm and hybrid paradigm combinations (Sequential & Simultaneous) using Cartesian product confusion matrices.
- **Subject-Specific & Pooled Analysis** — Per-subject evaluation with statistical testing (Wilcoxon signed-rank tests).

---

## Repository Structure

```
├── config.py                # Experiment configuration (hyperparameters, paths)
├── train_models.py          # Training entry point
├── evaluate_models.py       # Evaluation entry point
├── analyze_results.py       # Results analysis, plotting, and LaTeX tables
├── models/
│   └── switchbraidnet.py    # SwitchBraidNet architecture
├── utils/
│   ├── data.py              # Dataset loading and preprocessing
│   ├── engine.py            # Training engine (k-fold cross-validation)
│   ├── helpers.py           # Path resolution and plotting utilities
│   ├── hybrid_eval.py       # Hybrid BCI evaluation pipeline
│   ├── metrics.py           # ITR, Kappa, and model size metrics
│   ├── optimizers.py        # Optimizer and scheduler builders
│   ├── qat.py               # QAT wrapper (PyTorch Lightning)
│   └── registry.py          # Model registry
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/HAIx-Lab/SwitchBraidNet.git
cd SwitchBraidNet

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- PyTorch Lightning ≥ 2.0
- CUDA-capable GPU (recommended) or Apple Silicon MPS

---

## Dataset Setup

This project uses the **OpenBMI** dataset ([Lee et al., 2019](https://doi.org/10.1093/gigascience/giz002)).

1. Download the dataset from [GigaDB](http://gigadb.org/dataset/100542).
2. Preprocess into `.npz` format with the following directory structure:

```
Dataset/openbmi/
├── train_sess1_MI/
│   ├── S01_01_000.npz    # Each file: data=(1, C, T), labels=int
│   └── ...
├── test_sess1_MI/
├── train_sess1_SSVEP/
├── test_sess1_SSVEP/
├── train_sess1_ERP/
├── test_sess1_ERP/
├── train_sess2_MI/
├── test_sess2_MI/
└── ...
```

3. Update the `dataset_path` in `config.py` to point to your dataset directory.

---

## Usage

### 1. Train Models

Train SwitchBraidNet across all BCI paradigms (MI, SSVEP, ERP) and precision levels (32, 16, 8-bit):

```bash
python train_models.py
```

> **Quick Test:** Set `"debug": True` in `config.py` to run with 100 samples and 2 epochs.

### 2. Evaluate Models

Run the full hybrid BCI evaluation (single + hybrid paradigms, per-subject + pooled):

```bash
python evaluate_models.py
```

Results are saved to `Results/evaluation_results.csv`.

### 3. Analyze Results

Generate LaTeX tables, plots, and statistical tests:

```bash
python analyze_results.py
```

Outputs in `Results/`:
- `latex_tables/` — LaTeX table source files
- `plots/` — Quantization robustness and stability plots (PDF)
- `statistical_results.csv` — Wilcoxon signed-rank test results
- `subject_average_summary.csv` — Cross-subject average metrics

---

## BCI Paradigms

| Paradigm | Abbreviation | Classes | Channels |
|----------|:---:|:---:|:---:|
| Motor Imagery | MI | 2 | 20 |
| Steady-State Visual Evoked Potential | SSVEP | 4 | 10 |
| Event-Related Potential | ERP | 2 | 28 |

**Hybrid modes:**
- **Sequential (Seq):** Paradigms executed one after another; latency = sum of trial times.
- **Simultaneous (Sim):** Paradigms executed in parallel; latency = max of trial times.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{siddhad2026quantized,
  title     = {Quantized Hybrid Brain-Computer Interfaces: Evaluating Quantization-Aware Training for Multi-Paradigm EEG Classification on Edge Devices},
  author    = {Siddhad, Gourav and Gupta, Anmol and Bapi, Raju S.},
  booktitle = {IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  year      = {2026}
}
```

---

## Acknowledgements

- **Dataset:** [OpenBMI](http://gigadb.org/dataset/100542) — Lee, M.-H. et al., *GigaScience*, 2019. [[DOI]](https://doi.org/10.1093/gigascience/giz002)
- **HAIx Lab:** Human-AI Interaction Laboratory, IIT Gandhinagar — [haix.iitgn.ac.in](https://haix.iitgn.ac.in)
- **Funding:** This study was supported by the IP/IITGN/CSE/YM/2324/05 grant.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
