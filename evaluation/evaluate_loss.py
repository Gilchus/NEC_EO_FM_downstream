import json
import glob
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict, Any


# Currently only 
def compute_generalization_gap(
    data: Union[str, Dict[str, Any]]
) -> Tuple[float, float]:
    """
    Compute the mean and standard deviation of the generalization gap
    across folds. The generalization gap for a single fold is defined
    as:
        gap_i = val_losses_i[-1] - train_losses_i[-1]

    Parameters
    ----------
    data : str or dict
        Either a file path to a JSON file with the structure:
            {
              "test_results": {
                 ...,
                 "train_losses": [[...], [...], ..., [...]],
                 "val_losses":   [[...], [...], ..., [...]]
              }
            }
        or a dict already parsed from such a file.

    Returns
    -------
    mean_gap : float
        The mean of the per fold generalization gaps.
    std_gap : float
        The standard deviation of the per fold generalization gaps.
    """

    if isinstance(data, str):
        with open(data, 'r') as f:
            data = json.load(f)

    tr = data["test_results"]["train_losses"]
    va = data["test_results"]["val_losses"]

    if len(tr) != len(va):
        raise ValueError("Number of train and val folds must match.")

    gaps = []
    for train_loss, val_loss in zip(tr, va):
        if not train_loss or not val_loss:
            raise ValueError("Empty loss list in one of the folds.")
        gaps.append(val_loss[-1] - train_loss[-1])

    gaps = np.array(gaps)
    mean_gap = float(np.mean(gaps))
    std_gap = float(np.std(gaps, ddof=1))

    return mean_gap, std_gap

def summarize_with_context(base_dir: str):
    base = Path(base_dir)
    eval_logs = base / 'eval_logs'
    rows = []

    for model_dir in sorted(eval_logs.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            metrics_files = list(task_dir.glob('*_metrics.txt'))
            if not metrics_files:
                continue

            task = task_dir.name
            # we’ll average across all such files (usually one per fold grouping)
            all_train_means = []
            all_val_means   = []
            all_gaps        = []

            for fp in metrics_files:
                data = json.loads(fp.read_text())
                tr_losses = data['test_results']['train_losses']
                va_losses = data['test_results']['val_losses']

                # final‐epoch losses per fold
                final_tr = [fold[-1] for fold in tr_losses]
                final_va = [fold[-1] for fold in va_losses]

                mean_tr = sum(final_tr) / len(final_tr)
                mean_va = sum(final_va) / len(final_va)
                gap     = mean_va - mean_tr

                all_train_means.append(mean_tr)
                all_val_means.append(mean_va)
                all_gaps.append(gap)

            # aggregate across the file(s) if you had more than one
            mean_train_loss = sum(all_train_means) / len(all_train_means)
            mean_val_loss   = sum(all_val_means)   / len(all_val_means)
            mean_gap, std_gap = compute_generalization_gap(data)
            gap_ratio       = mean_gap / mean_val_loss if mean_val_loss != 0 else float('nan')

            rows.append({
                'model':           model,
                'task':            task,
                'mean_train_loss': mean_train_loss,
                'mean_val_loss':   mean_val_loss,
                'mean_gap':        mean_gap,
                'std_gap':         std_gap,
                'gap_ratio':       gap_ratio,
            })

    return pd.DataFrame(rows)