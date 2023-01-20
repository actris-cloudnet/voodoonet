import os
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass

import numpy as np
from rpgpy.utils import rpg_seconds2datetime64
from torch import Tensor, concat, from_numpy
from torchmetrics.classification import BinaryConfusionMatrix

IntTuples = tuple[tuple[int, int], ...]
IntTuplesVariable = tuple[tuple[int, ...], ...]
Ints = tuple[int, ...]


@dataclass
class VoodooOptions:
    trained_model: str = (
        f"{os.path.dirname(__file__)}/trained_models/Vnet2_0-dy0.00-fnXX-cuda0.pt"
    )
    kernel_sizes: IntTuples = ((3, 3), (3, 3), (1, 3), (1, 3), (1, 3))
    pad_sizes: IntTuples = ((1, 1), (1, 1), (0, 1), (0, 1), (0, 1))
    stride_sizes: IntTuples = ((1, 2), (1, 2), (1, 2), (2, 2), (1, 2))
    num_filters: Ints = (16, 32, 64, 128, 256)
    dense_layers: Ints = (128, 128, 64)
    output_shape: int = 2
    n_channels: int = 6
    n_dbins: int = 256
    z_limits: tuple[float, float] = (-50, 20)
    device: str = "cpu"

    def dict(self) -> dict:
        return {k: str(v) for k, v in asdict(self).items()}


@dataclass
class WandbConfig:
    project: str
    name: str
    entity: str


@dataclass
class VoodooTrainingOptions:
    garbage: Ints = (0, 3, 7, 8, 9, 10)
    groups: IntTuplesVariable = ((1, 5), (2, 4, 6))
    dupe_droplets: int = 1
    learning_rate: float = 1.0e-3
    learning_rate_decay: float = 1.0e-1
    learning_rate_decay_steps: int = 1
    shuffle: bool = True
    split: float = 0.1  # -> 10% of data for validation
    wandb: WandbConfig | None = None
    epochs: int = 3
    batch_size: int = 256


def time_grid(date: str, resolution: int = 30) -> np.ndarray:
    date_components = date.split("-")
    n_time = int(24 * 60 * 60 / resolution)
    time = np.linspace(0, 24, n_time)
    return decimal_hour2unix(date_components, time)


def rpg_time2unix(time: np.ndarray) -> np.ndarray:
    radar_times = rpg_seconds2datetime64(time)
    return radar_times.astype("datetime64[s]").astype("int")


def decimal_hour2unix(date: list, time: np.ndarray) -> np.ndarray:
    unix_timestamp = np.datetime64("-".join(date)).astype("datetime64[s]").astype("int")
    return (time * 60 * 60 + unix_timestamp).astype(int)


def numpy_datetime2unix(datetime64: np.ndarray) -> np.ndarray:
    unix_timestamp = (
        (datetime64 - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
    ).astype(int)
    return unix_timestamp.astype(int)


def lin2z(array: np.ndarray | list | float) -> np.ndarray:
    return 10 * np.ma.log10(array)


def arg_nearest(array: np.ndarray, value: float | int) -> np.int64:
    i = np.searchsorted(array, value)
    return i if i <= array.shape[0] - 1 else i - 1


def reshape(data: Tensor, mask: np.ndarray) -> np.ndarray:
    new_shape = mask.shape + (data.shape[1],)
    input_reshaped = np.zeros(new_shape)
    cnt = 0
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            if mask[i, j]:
                continue
            input_reshaped[i, j, :] = data[cnt, :]
            cnt += 1
    return input_reshaped


def filter_list(rpg_lv0_files: list[str], date: list[str]) -> list[str | None]:
    regex = re.compile("".join(date))
    filtered_strings = filter(regex.search, rpg_lv0_files)
    return list(filtered_strings)


def numpy_arrays2tensor(data: list[np.ndarray]) -> Tensor:
    torch_tensors = [from_numpy(array) for array in data]
    return concat(torch_tensors, dim=0)


def keep_valid_samples(
    features: np.ndarray, target_class: np.ndarray, detect_status: np.ndarray
) -> tuple:
    valid = load_training_mask(target_class, detect_status)
    idx_valid_samples = np.where(valid)
    if len(idx_valid_samples) < 1:
        return None, None

    valid_features = features[idx_valid_samples[0], ...]
    valid_labels = target_class[idx_valid_samples[0]]

    # remove samples with low signal to noise
    mean = np.mean(valid_features, axis=(1, 2))
    idx_invalid_samples = np.argwhere(mean < 0.01)[:, 0]
    if len(idx_invalid_samples) > 0:
        valid_features = np.delete(valid_features, idx_invalid_samples, axis=0)
        valid_labels = np.delete(valid_labels, idx_invalid_samples, axis=0)

    mean = np.mean(valid_features, axis=(1, 2))
    # remove samples to high values
    idx_invalid_samples = np.argwhere(mean > 0.99)[:, 0]
    if len(idx_invalid_samples) > 0:
        valid_features = np.delete(valid_features, idx_invalid_samples, axis=0)
        valid_labels = np.delete(valid_labels, idx_invalid_samples, axis=0)

    return np.squeeze(valid_features), np.squeeze(valid_labels)


def load_training_mask(classes: np.ndarray, status: np.ndarray) -> np.ndarray:
    valid_samples = np.full(status.shape, False)
    valid_samples[status == 3] = True  # add good radar radar & lidar
    valid_samples[classes == 1] = True  # add cloud droplets only class
    valid_samples[classes == 3] = True  # add cloud droplets + drizzle/rain
    valid_samples[classes == 5] = True  # add mixed-phase
    valid_samples[classes == 6] = True  # add melting layer
    valid_samples[classes == 7] = True  # add melting layer + SCL
    valid_samples[status == 1] = False  # remove lidar only
    return valid_samples


def validation_metrics(confusion_matrix: Tensor) -> Tensor:
    TP, FP, FN, TN = confusion_matrix
    metrics = Tensor(
        [
            TP,  # true positives
            TN,  # true negatives
            FP,  # false positives
            FN,  # false negatives
            TP / max(TP + FP, 1.0e-7),  # positive predictive value (precision)
            TN / max(TN + FN, 1.0e-7),  # negative predictive value
            TP / max(TP + FN, 1.0e-7),  # true positive rate (recall)
            TN / max(TN + FP, 1.0e-7),  # true negative rate (specificity)
            (TP + TN) / max(TP + TN + FP + FN, 1.0e-7),  # acccuracy
            2 * TP / max(2 * TP + FP + FN, 1.0e-7),  # f1-score
        ]
    )
    return metrics


def metrics_to_dict(metrics: Tensor) -> dict:
    m_dict = {
        "true_positives": metrics[0],
        "true_negatives": metrics[1],
        "false_positives": metrics[2],
        "false_negatives": metrics[3],
        "ppv": metrics[4],
        "npv": metrics[5],
        "tpr": metrics[6],
        "tnr": metrics[7],
        "acc": metrics[8],
        "f1": metrics[9],
    }
    return m_dict


def calc_cm(pred_labels: Tensor, true_labels: Tensor) -> Tensor:
    """Returns confusion matrix entries in the following order: TP, FP, FN,
    TN."""
    cm = BinaryConfusionMatrix()
    cm.to(pred_labels.device)
    return cm(pred_labels, true_labels).flatten()


def get_metrics(met: np.ndarray) -> OrderedDict:
    TP, TN, FP, FN = met
    metric = OrderedDict(
        {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "precision": TP / max(TP + FP, 1.0e-7),
            "npv": TN / max(TN + FN, 1.0e-7),
            "recall": TP / max(TP + FN, 1.0e-7),
            "specificity": TN / max(TN + FP, 1.0e-7),
            "accuracy": (TP + TN) / max(TP + TN + FP + FN, 1.0e-7),
            "F1-score": 2 * TP / max(2 * TP + FP + FN, 1.0e-7),
        }
    )
    return metric
