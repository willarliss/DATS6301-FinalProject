from typing import Union, Tuple

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from sklearn.metrics import confusion_matrix, roc_curve, RocCurveDisplay, roc_auc_score

PlotOutputs = Union[Tuple[AxesImage, ndarray], ndarray, AxesImage]


def plot_cm(y_true: ndarray, y_pred: ndarray, *,
            figsize: Tuple[float] = (5,5),
            rounding: int = 3,
            title: str = None,
            return_cm: bool = False,
            return_fig: bool = False) -> PlotOutputs:
    """Plot confusion matrix (color gradient is fixed).
    """

    fig = plt.figure(figsize=figsize)

    cm = confusion_matrix(y_true, y_pred, normalize='true').T
    plt.imshow(cm.T, vmin=0, vmax=1)
    plt.colorbar(anchor=(0,1), fraction=0.05)

    plt.title('Confusion Matrix' if title is None else str(title))
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks([0, 1])
    plt.yticks([0, 1])

    offset = 0.02 * rounding
    plt.annotate(cm[0,0].round(rounding), (0-offset, 0), color='r')
    plt.annotate(cm[0,1].round(rounding), (0-offset, 1), color='r')
    plt.annotate(cm[1,1].round(rounding), (1-offset, 1), color='r')
    plt.annotate(cm[1,0].round(rounding), (1-offset, 0), color='r')

    if return_cm and return_fig:
        return fig, cm
    if return_cm:
        return cm
    if return_fig:
        return fig
    return None


def plot_roc(y_true: ndarray, y_score: ndarray, *,
             figsize: Tuple[float] = (6.4,4.8),
             title: str = None,
             rounding: int = 3,
             return_fig: bool = False,
             return_t: bool = False) -> PlotOutputs:
    """Plot ROC curve and show thresholds.
    """

    fig, ax = plt.subplots(figsize=figsize)
    fpr, tpr, thresh = roc_curve(y_true=y_true, y_score=y_score)
    score = roc_auc_score(y_true=y_true, y_score=y_score)
    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=score)
    thresh_idx = np.argmin(((fpr-0)**2 + (tpr-1)**2) ** 0.5)

    display.plot(ax=ax, color='b', alpha=1.)
    ax.set_title('ROC Curve' if title is None else title)
    ax.plot(fpr[thresh_idx], tpr[thresh_idx], 'r.',
            markersize=10, label=f't = {thresh[thresh_idx].round(rounding)}')
    ax.plot(0, 1, 'k.', alpha=0.5, markersize=5)
    ax.plot([0,1], [0,1], 'k--', alpha=0.5)
    ax.legend()

    if return_t and return_fig:
        return fig, thresh[thresh_idx]
    if return_t:
        return thresh[thresh_idx]
    if return_fig:
        return fig
    return None


def plot_hist(y_true: ndarray, y_like: ndarray, *,
              pdf: bool = False,
              center: float = 0.,
              scale: float = 1.,
              lb: float = None,
              ub: float = None,
              bins: int = 15,
              title: str = None,
              return_fig: bool = False,
              figsize: Tuple[float] = (6.4,4.8)) -> PlotOutputs:
    """Plot histogram of predicted log-likelihoods.
    """

    y_like = np.clip(
        a=y_like,
        a_min=-np.inf if lb is None else lb,
        a_max=np.inf if ub is None else ub,
    )

    fig = plt.figure(figsize=figsize)
    plt.hist(y_like[y_true==0], density=True, bins=bins, color='b', alpha=0.5)
    plt.hist(y_like[y_true==1], density=True, bins=bins, color='r', alpha=0.5)
    plt.title('Likelihood Histogram' if title is None else title)
    plt.xlabel('Log Likelihood')
    plt.ylabel('Density')

    if pdf:
        domain = np.linspace(y_like.min(), y_like.max(), y_like.shape[0]*2)
        range_ = 1 / (1 + np.exp((center-domain)/scale))
        plt.plot(domain, range_*(1-range_), 'k-', alpha=0.7)
        plt.axvline(center, color='k', linestyle='--', alpha=0.3)

    if return_fig:
        return fig
    return None
