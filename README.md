# Project overview

This repo contains a pipeline for working with the **TOX21** dataset: download/prepare data, run training/benchmarking, and browse results in Streamlit apps.

## What to run

- **`download.ipynb`**: downloads TOX21, converts SDF → CSV, and builds the datasets/splits used by training.
- **`training.ipynb`**: runs model training + benchmarking across targets/fingerprints and writes results to `results/`.
- **`streamlit_tox21_eda.py`**: Streamlit EDA app for the dataset (train split and chemistry/structure views).

```bash
streamlit run streamlit_tox21_eda.py
```

- **`streamlit_tox21_benchmark.py`**: Streamlit app to browse benchmark results (metrics, ROC curves, best runs, comparisons).

```bash
streamlit run streamlit_tox21_benchmark.py
```

- **`streamlit_tox21_dashboard.py`**: Combined dashboard (EDA + Benchmark).

```bash
streamlit run streamlit_tox21_dashboard.py
```
