# iPHT — Iterative Probabilistic Hough Transform

Robust discovery of filamentary structures in high-noise point clouds, with application to the cosmic web.

iPHT detects thin, faint filaments that are buried in heavy noise and are no denser than the background. Instead of relying on density contrast (which vanishes when the noise is strong), iPHT works from **directional agreement**: at each point it votes over directions, keeps points whose neighbours agree on an orientation, and removes the most uncertain points iteratively until only filaments remain.

## Quick start

Wanna use our method? Easy:

```python
from src.run_iPHT import run_iPHT

_, labeled_pred, spine = run_iPHT(radius, data)
```

Give it a radius + your (3D) points, and get back labels (`0` = noise) and the spine curves.

## Paper

This code accompanies our KDD 2026 paper:

> O. Alghamdi, M. Canducci, R. Smith, and P. Tiňo.
> *Iterative Probabilistic Hough Transform: Robust Discovery of Filamentary Structures in High-Noise Regimes.*
> Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26), 2026.
> https://doi.org/10.1145/3770855.3818955

If you use this code, please cite the paper (see [Citation](#citation) below).

## Repository contents

- `iterative_PHT/` — the iPHT method (directional voting, entropy-based filtering, segmentation, and spine fitting).
- `synthetic dataset add boundary/` — synthetic filament-in-noise datasets used in the experiments.
- `readme` — earlier notes.

## Requirements

- Python 3
- NumPy, SciPy, scikit-learn, NetworkX, Matplotlib

```bash
pip install numpy scipy scikit-learn networkx matplotlib
```

## Usage

Run iPHT on a point cloud (see the run scripts for the exact entry point):

```bash
python run_iPHT.py
```

The main parameter is the **neighbourhood radius R**; it is the only physical parameter the method requires.

## Comparison with other methods

We benchmark iPHT against Torque Clustering, KdMutual, DBSCAN, and 1-DREAM across a range of noise levels. You are welcome to compare your own method against iPHT using the datasets provided here. If you report a comparison against iPHT, please cite our paper.

## Citation

If you use iPHT, the datasets, or compare against it in your work, please cite:

```bibtex
@inproceedings{10.1145/3770855.3818955,
  author    = {Alghamdi, Othman and Canducci, Marco and Smith, Rory and Tino, Peter},
  title     = {Iterative Probabilistic Hough Transform: Robust Discovery of Filamentary Structures in High-Noise Regimes},
  year      = {2026},
  isbn      = {9798400722592},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  url       = {https://doi.org/10.1145/3770855.3818955},
  doi       = {10.1145/3770855.3818955},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  pages     = {10384--10395},
  numpages  = {12},
  keywords  = {probabilistic hough transform, filamentary structure detection, low signal-to-noise, entropy-based filtering, curve fitting, probabilistic framework, noise robustness},
  location  = {Republic of Korea},
  series    = {KDD '26}
}
```

## Contact

Othman Alghamdi — School of Computer Science, University of Birmingham.
