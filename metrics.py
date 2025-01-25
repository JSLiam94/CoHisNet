from typing import Optional, Dict, Any, Union, List
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    accuracy_score,
    cohen_kappa_score,
    classification_report,
)

def get_eval_metrics(
    targets_all: Union[List[int], np.ndarray],
    preds_all: Union[List[int], np.ndarray],
    probs_all: Optional[Union[List[float], np.ndarray]] = None,
    get_report: bool = True,
    prefix: str = "",
    roc_kwargs: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """
    Calculate evaluation metrics and return the evaluation metrics.

    Args:
        targets_all (array-like): True target values.
        preds_all (array-like): Predicted target values.
        probs_all (array-like, optional): Predicted probabilities for each class. Defaults to None.
        get_report (bool, optional): Whether to include the classification report in the results. Defaults to True.
        prefix (str, optional): Prefix to add to the result keys. Defaults to "".
        roc_kwargs (dict, optional): Additional keyword arguments for calculating ROC AUC. Defaults to {}.

    Returns:
        dict: Dictionary containing the evaluation metrics.
    """
    bacc = balanced_accuracy_score(targets_all, preds_all)
    kappa = cohen_kappa_score(targets_all, preds_all, weights="quadratic")
    acc = accuracy_score(targets_all, preds_all)
    cls_rep = classification_report(targets_all, preds_all, output_dict=True, zero_division=0)

    # 每个类别的指标
    per_class_metrics = {
        f"{prefix}class_{int(label)}_precision": cls_rep[str(label)]["precision"]
        for label in cls_rep if label.isdigit()
    }
    per_class_metrics.update({
        f"{prefix}class_{int(label)}_recall": cls_rep[str(label)]["recall"]
        for label in cls_rep if label.isdigit()
    })
    per_class_metrics.update({
        f"{prefix}class_{int(label)}_f1": cls_rep[str(label)]["f1-score"]
        for label in cls_rep if label.isdigit()
    })

    eval_metrics = {
        f"{prefix}acc": acc,
        f"{prefix}bacc": bacc,
        f"{prefix}kappa": kappa,
        f"{prefix}weighted_f1": cls_rep["weighted avg"]["f1-score"],
        **per_class_metrics,
    }

    if get_report:
        eval_metrics[f"{prefix}report"] = cls_rep

    if probs_all is not None:
        roc_auc = roc_auc_score(targets_all, probs_all, multi_class="ovr", **roc_kwargs)
        eval_metrics[f"{prefix}auroc"] = roc_auc

    return eval_metrics


def print_metrics(eval_metrics: Dict[str, Any]) -> None:
    """
    Print evaluation metrics in a formatted way.

    Args:
        eval_metrics (dict): Dictionary of evaluation metrics to print.
    """
    for k, v in eval_metrics.items():
        if "report" in k:
            continue
        print(f"Test {k}: {v:.3f}")
