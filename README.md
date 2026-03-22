# AlexNet Coin Classification

This repository contains the AlexNet-based coin-classification system for the `TP_CNN_monnaie` assignment. The project is meant to be run from the root with `run.py`, while the actual implementation lives in `src/`. The dataset stays in `data/`, and the written report is stored in `report/report.tex` with its compiled version in `report/report.pdf`.

First create the local venv and install the dependencies:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

Once that is done, launch the whole pipeline with:

```bash
./.venv/bin/python run.py
```

This single command checks the dataset, trains the model, evaluates it, saves the metrics, and generates a submission file unless submission generation is disabled. By default it uses the improved configuration and writes the outputs to `results`.

The run keeps the automatic outputs minimal. In normal use, `results/` contains `classification_report.json`, `validation_metrics.json`, `training_history.png`, and `submission.csv` unless `--skip-submission` is used. The older auxiliary files `results/classification_report.txt`, `results/dataset_audit.json`, `results/history.csv`, and `results/run_config.json` are not part of the pipeline anymore.

If you want to choose the mode or the output folder explicitly, you can still pass arguments. For example, the command below keeps the same pipeline but makes the destination folder explicit:

```bash
./.venv/bin/python run.py --mode improved --output-dir results
```

Invalid images are filtered before training, and the improved mode enables stronger augmentation, weighted sampling, and optional fine-tuning so the system is easier to extend for better experiments later.
