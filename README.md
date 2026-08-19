# Pynapse — Neural Data Engine

**Alignment, preprocessing, and peri-event extraction for calcium imaging paired with behavioral event logs.**

[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/thejoshbq/pynapse)
[![Language](https://img.shields.io/badge/python-3.8+-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Phoxel Workbench](https://img.shields.io/badge/Phoxel_Workbench-member-orange)](https://github.com/Otis-Lab-MUSC)

*Written by*: Joshua Boquiren

[![](https://img.shields.io/badge/@thejoshbq-grey?style=flat&logo=github)](https://github.com/thejoshbq)

---

## Overview

Pynapse takes the two halves of a self-administration experiment — fluorescence traces extracted from your imaging session and the behavioral event log recorded alongside it — and puts them on a single, correct timescale. It resolves frame timestamps against behavioral time (accounting for frame averaging and timestamp correction), applies the lab-standard preprocessing pipeline, and returns peri-event windows locked to whichever behavioral events you care about, ready for statistics or plotting. Sessions group into populations and projects, so the same analysis runs over one animal or an entire cohort without rewriting it.

It is the shared data layer the rest of the lab's tooling is built on: [axplorer](https://github.com/thejoshbq/axplorer) and [roigbiv](https://github.com/thejoshbq/roigbiv) consume Pynapse objects rather than reimplementing alignment or preprocessing, and Pynapse holds the canonical event code to label dictionaries used across the whole REACHER analysis stack. An optional DuckDB layer caches aligned sessions so repeat analyses skip raw-file parsing.

---

## Getting Started

Pynapse is not published on PyPI. Install from a local clone:

```bash
git clone https://github.com/thejoshbq/pynapse.git
cd pynapse
pip install -e .
```

Reference documentation lives in [`docs/`](docs/) — the database guide and the cloud-migration plan.

---

## Architecture & Dependencies

| Component | Language | Framework / Libraries |
|---|---|---|
| Data I/O & alignment | Python 3.8+ | NumPy, pandas, SciPy |
| Preprocessing pipelines | Python 3.8+ | NumPy, SciPy |
| Peri-event tensor extraction | Python 3.8+ | NumPy, pandas, scikit-learn |
| Persistence layer (optional) | Python 3.10+ | DuckDB, PyTables |
| Visualization helpers | Python 3.8+ | Matplotlib, Plotly, seaborn, narwhals |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Joshua Boquiren — [thejoshbq@proton.me](mailto:thejoshbq@proton.me)

[GitHub: thejoshbq/pynapse](https://github.com/thejoshbq/pynapse)
