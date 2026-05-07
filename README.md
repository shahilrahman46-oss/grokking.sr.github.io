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

## 📥 Download

<p align="center">
  <a href="https://github.com/shahilrahman46-oss/grokking.sr.github.io/archive/refs/heads/main.zip">
    <img src="https://img.shields.io/badge/⬇️%20Download-Source%20Code%20(.zip)-0ea5e9?style=for-the-badge"/>
  </a>
 

---

## What is Grokking?

**Grokking** is a phenomenon where a neural network first perfectly memorises the training set, then — after a long delay with no change in training loss — suddenly generalises to unseen data. This project reproduces the effect on modular arithmetic tasks and reverse-engineers the internal algorithm the model learns.

---

## Results

### Part 1 — Grokking Curves: (a + b) mod 67

Training accuracy reaches 100% at ~epoch 500. Validation accuracy stays near 0% until **epoch ≈ 10,750**, then jumps sharply — the textbook grokking signature.

<p align="center">
 <img width="2071" height="718" alt="part1_grokking" src="https://github.com/user-attachments/assets/2fb75bd9-1c53-41df-b2da-5232a0d16390" />


> **Left:** Accuracy curves showing the large memorisation-to-generalisation delay.
> **Right:** Cross-entropy loss on a log scale — training loss is near 10⁻⁷ long before validation loss collapses.

---

### Part 2 — Sparse Fourier Basis (Addition & Multiplication)

A 1-D DFT of the token embedding matrix and post-ReLU MLP activations reveals that both representations concentrate power on a **sparse set of frequencies**, confirming the model implements a Fourier-rotation algorithm rather than a lookup table.

**Addition** — dominant frequencies ω ∈ {4, 6, 26, 27}:

<p align="center">
 <img width="2071" height="755" alt="part2_fourier" src="https://github.com/user-attachments/assets/5f6e29c1-fbf6-495f-a92b-24c85ee48c46" />


**Multiplication** — dominant frequencies ω ∈ {11, 21, 33} (in Z₆₆, not Z₆₇):

<p align="center">
 <img width="2071" height="755" alt="part2_fourier_mul" src="https://github.com/user-attachments/assets/4091203c-fc4e-4b2d-8da3-573176f39ab8" />

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
 <img width="2070" height="1919" alt="part3_phases" src="https://github.com/user-attachments/assets/b8b98407-5c9b-42fb-9aac-2597d9ae83bf" />


---

### Part 4 — Epochs-Until-Generalisation vs. Data Fraction

Models trained with fractions from 10% to 90% of all 67² pairs reveal a sharp **data threshold** for grokking:

<p align="center">
 <img width="1471" height="718" alt="part4_fraction_sweep" src="https://github.com/user-attachments/assets/0001362c-43c6-4337-bf22-ab43266d1f00" />


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
<img width="1998" height="1578" alt="part5_algorithm" src="https://github.com/user-attachments/assets/48440676-c012-42aa-9bd5-1ae3c52fa9ce" />

---

### Part 6 — Co-Grokking: Addition + Multiplication Simultaneously

A single model trained on **both** tasks at once shows **co-grokking**: addition groks at **ep 31,150** and multiplication at **ep 32,950** — a gap of only 1,800 epochs, suggesting shared internal Fourier representations.

<p align="center">
<img width="2069" height="1513" alt="part6_cogrokking" src="https://github.com/user-attachments/assets/b95f55fe-e6f3-41f4-956e-b578a3fec75d" />

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
