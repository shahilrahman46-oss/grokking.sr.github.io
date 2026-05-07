# 🧠 Grokking of Modular Arithmetic
### Full Mechanistic Interpretability Suite · p = 67 · One-Layer Transformer

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/CUDA-optional-76B900?style=flat-square&logo=nvidia&logoColor=white"/>
  <img src="https://img.shields.io/badge/Colab-ready-F9AB00?style=flat-square&logo=google-colab&logoColor=white"/>
  <img src="https://img.shields.io/badge/Based%20on-Nanda%20et%20al.%202023-a78bfa?style=flat-square"/>
</p>

<p align="center">
  A reproduction and extension of <a href="https://arxiv.org/abs/2301.05217"><em>Progress measures for grokking via mechanistic interpretability</em></a> (Nanda et al., ICLR 2023) using prime <strong>p = 67</strong> as a distinct test case, with six fully-implemented analysis parts including a novel <strong>co-grokking</strong> experiment.
</p>

---

## 📥 Downloads

<p align="center">
  <a href="https://raw.githubusercontent.com/shahilrahman46-oss/grokking.sr.github.io/main/grokking_project.py">
    <img src="https://img.shields.io/badge/⬇️%20Download-grokking__project.py-0ea5e9?style=for-the-badge"/>
  </a>
  &nbsp;&nbsp;
  <a href="https://raw.githubusercontent.com/shahilrahman46-oss/grokking.sr.github.io/main/report.pdf">
    <img src="https://img.shields.io/badge/⬇️%20Download-Project%20Report%20(.pdf)-a78bfa?style=for-the-badge"/>
  </a>
</p>

---

## What is Grokking?

**Grokking** is a phenomenon where a neural network first perfectly memorises the training set, then — after a long delay with no change in training loss — suddenly generalises to unseen data. This project reproduces the effect on modular arithmetic tasks and reverse-engineers the internal algorithm the model learns.

---

## Results

### Part 1 — Grokking Curves: (a + b) mod 67

Training accuracy reaches 100% at ~epoch 500. Validation accuracy stays near 0% until **epoch ≈ 10,750**, then jumps sharply — the textbook grokking signature.

<p align="center">
  <img width="800" alt="part1_grokking" src="https://github.com/user-attachments/assets/d428ef93-6a73-400f-a886-8ec76cdecca2" />
</p>

> **Left:** Accuracy curves showing the large memorisation-to-generalisation delay.
> **Right:** Cross-entropy loss on a log scale — training loss is near 10⁻⁷ long before validation loss collapses.

---

### Part 2 — Sparse Fourier Basis (Addition & Multiplication)

A 1-D DFT of the token embedding matrix and post-ReLU MLP activations reveals that both representations concentrate power on a **sparse set of frequencies**, confirming the model implements a Fourier-rotation algorithm rather than a lookup table.

**Addition** — dominant frequencies ω ∈ {4, 6, 26, 27}:

<p align="center">
  <img width="2071" height="718" alt="part1_grokking" src="https://github.com/user-attachments/assets/c2a55bb1-aef5-4eeb-b2f3-bb4fe49d59c8" />


**Multiplication** — dominant frequencies ω ∈ {11, 21, 33} (in Z₆₆, not Z₆₇):

<p align="center">
  <img width="800" alt="part2_fourier_mul" src="https://github.com/user-attachments/assets/95572c56-c1be-4436-95d6-e54ecc40146f" />
</p>

> The shift from Z_p (addition) to Z_{p−1} (multiplication) is the model's algebraic **fingerprint** for the discrete-log isomorphism.

---

### Part 3 — Three-Phase Grokking Decomposition

Tracking four metrics jointly exposes three mechanistically distinct phases:

| Phase | What happens |
|---|---|
| **① Memorisation** | Train acc → 100%; test acc ≈ 0%; weights grow |
| **② Circuit Formation** | Fourier circuits strengthen; weight norm begins falling |
| **③ Cleanup** | Weight decay prunes memorisation; test acc jumps to 100% |

<p align="center">
  <img width="700" alt="part3_phases" src="https://github.com/user-attachments/assets/e7ddf8ac-4ad5-4cea-8499-377589cb5846" />
</p>

---

### Part 4 — Epochs-Until-Generalisation vs. Data Fraction

Models trained with fractions from 10% to 90% of all 67² pairs reveal a sharp **data threshold** for grokking:

<p align="center">
  <img width="700" alt="part4_fraction_sweep" src="https://github.com/user-attachments/assets/4be2b840-e08f-4489-bc5a-c0438e64fb51" />
</p>

| Fraction | Grokking epoch |
|---|---|
| 10–30% | ✗ No grokking within 15,000 epochs |
| 40% | ~8,100 |
| 50% | ~2,475 |
| 60% | ~1,350 |
| 70% | ~600 |
| 80% | ~525 |
| 90% | ~225 |

---

### Part 5 — Algorithm Reverse-Engineering: (a × b) mod 67

For multiplication, the model learns the **discrete-logarithm isomorphism**:

```
(Z₆₇*, ×)  ≅  (Z₆₆, +)   via   a ↦ ind_g(a)   where g = 2
```

**Learned algorithm (3 steps):**
1. **Encode** — embed token `a` as Fourier features of its discrete log
2. **Combine** — compute `log_g(a) + log_g(b) mod 66` via cosine addition formula
3. **Decode** — map back via `2^(log_g(a)+log_g(b) mod 66) mod 67`

<p align="center">
  <img width="750" alt="part5_algorithm" src="https://github.com/user-attachments/assets/1085d232-4fa2-4621-b375-035660563fb3" />
</p>

---

### Part 6 — Co-Grokking: Addition + Multiplication Simultaneously

A single model trained on **both** tasks at once shows **co-grokking**: addition groks at **ep 31,150** and multiplication at **ep 32,950** — a gap of only 1,800 epochs, suggesting shared internal Fourier representations.

<p align="center">
  <img width="800" alt="part6_cogrokking" src="https://github.com/user-attachments/assets/4b3b227f-8ac4-4b99-b183-2d75d28f8ab1" />
</p>

---

## Installation & Usage

```bash
git clone https://github.com/shahilrahman46-oss/grokking.sr.github.io
cd grokking.sr.github.io
pip install torch einops numpy matplotlib

python grokking_project.py                   # default (p=67, 25k epochs)
python grokking_project.py --p 89 --epochs 30000 --wd 0.8
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--p` | 67 | Prime modulus |
| `--epochs` | 25,000 | Training epochs |
| `--frac` | 0.30 | Training data fraction |
| `--lr` | 1e-3 | Learning rate |
| `--wd` | 0.7 | Weight decay |
| `--cogrok_epochs` | 50,000 | Epochs for co-grokking |
| `--outdir` | `grokking_output` | Output directory |

---

## Repository Structure

```
.
├── grokking_project.py       # Full analysis suite
├── report.pdf                # Project report
├── grokking_results.html     # Self-contained results page (all images embedded)
├── README.md
└── grokking_output/
    ├── part1_grokking.png
    ├── part2_fourier.png
    ├── part2_fourier_mul.png
    ├── part3_phases.png
    ├── part4_fraction_sweep.png
    ├── part5_algorithm.png
    └── part6_cogrokking.png
```

---

## Reference

```bibtex
@inproceedings{nanda2023progress,
  title     = {Progress measures for grokking via mechanistic interpretability},
  author    = {Nanda, Neel and Chan, Lawrence and Liberum, Tom and Smith, Jess and Steinhardt, Jacob},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023},
  url       = {https://arxiv.org/abs/2301.05217}
}
```

---

<p align="center"><sub>Built with PyTorch · Faithful to Nanda et al. 2023 · Extended with co-grokking & algorithm reverse-engineering</sub></p>
