import logging
import os.path
import random
from tempfile import NamedTemporaryFile

import netCDF4
import numpy as np
import requests
import torch
from requests.adapters import HTTPAdapter, Retry
from rpgpy import RPGFileError, read_rpg
from rpgpy.header import read_rpg_header
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from torch import Tensor

from voodoonet import utils
from voodoonet.utils import VoodooOptions, VoodooTrainingOptions

from .torch_model import VoodooNet


def train(
    training_data: str,
    trained_model: str,
    training_options: VoodooTrainingOptions = VoodooTrainingOptions(),
    model_options: VoodooOptions = VoodooOptions(),
) -> None:
    """Train a new Voodoo model."""
    x_train, y_train, x_test, y_test = load_training_data(
        training_data, training_options=training_options
    )
    model = VoodooNet(
        x_train.shape, options=model_options, training_options=training_options
    )
    model.optimize(
        x_train,
        y_train,
        x_test,
        y_test,
        epochs=training_options.epochs,
        batch_size=training_options.batch_size,
    )
    model.save(path=trained_model, aux=model.options.dict())


def infer(
    rpg_lv0_files: list,
    target_time: np.ndarray | None = None,
    options: VoodooOptions = VoodooOptions(),
    training_options: VoodooTrainingOptions = VoodooTrainingOptions(),
) -> np.ndarray:
    """Use existing Voodoo model to infer measurement data."""
    voodoo_droplet = VoodooDroplet(target_time, options, training_options)
    valid_files = _get_files_with_common_height(rpg_lv0_files)
    for filename in valid_files:
        voodoo_droplet.calc_prob(filename)
    return voodoo_droplet.prob_liquid


def _get_files_with_common_height(files: list) -> list:
    valid_files = []
    for file in files:
        try:
            valid_files.append((file, read_rpg_header(file)[0]["RAltN"]))
        except (RPGFileError, IndexError):
            continue
    n_alts = [n_alt for _, n_alt in valid_files]
    most_common = max(set(n_alts), key=n_alts.count)
    return [file for file, n_alt in valid_files if n_alt == most_common]


def generate_training_data(
    rpg_lv0_files: list,
    classification_files: list,
    output_filename: str,
    options: VoodooOptions = VoodooOptions(),
    training_options: VoodooTrainingOptions = VoodooTrainingOptions(),
) -> None:
    """Generate Voodoo training dataset."""
    voodoo_droplet = VoodooDroplet(None, options, training_options)
    features, labels = voodoo_droplet.compile_dataset(
        rpg_lv0_files, classification_files
    )
    _save_training_data(features, labels, output_filename)


def generate_training_data_for_cloudnet(
    site: str,
    output_filename: str,
    options: VoodooOptions = VoodooOptions(),
    training_options: VoodooTrainingOptions = VoodooTrainingOptions(),
    n_days: int | None = None,
    tempfile_prefix: str | None = None,
) -> None:
    """Generate training dataset directly using Cloudnet API.

    Experimental.
    """
    url = "https://cloudnet.fmi.fi/api"
    classification_metadata = requests.get(
        f"{url}/files",
        {"site": site, "product": "classification"},
        timeout=60,
    ).json()
    try:
        classification_dates = [
            row["measurementDate"] for row in classification_metadata
        ]
    except TypeError:
        logging.error(f"Invalid site '{site}'.")
        return
    if not classification_dates:
        logging.error(f"No classification files found for site '{site}'.")
        return
    rpg_metadata = requests.get(
        f"{url}/raw-files",
        {
            "site": site,
            "instrument": "rpg-fmcw-94",
            "dateFrom": min(classification_dates),
            "dateTo": max(classification_dates),
        },
        timeout=60,
    ).json()
    rpg_metadata = [
        row
        for row in rpg_metadata
        if row["filename"].endswith(".LV0")
        and row["measurementDate"] in classification_dates
    ]
    rpg_dates = list(set(row["measurementDate"] for row in rpg_metadata))
    classification_metadata = [
        row for row in classification_metadata if row["measurementDate"] in rpg_dates
    ]
    if n_days is not None and len(classification_metadata) > n_days:
        classification_metadata = random.sample(classification_metadata, n_days)
        classification_dates = [
            row["measurementDate"] for row in classification_metadata
        ]
        rpg_metadata = [
            row
            for row in rpg_metadata
            if row["measurementDate"] in classification_dates
        ]
    if not classification_metadata:
        logging.error(
            f"No matching classification / RPG Level 0 files found for site '{site}'."
        )
        return
    voodoo_droplet = VoodooDroplet(None, options, training_options)
    features, labels = voodoo_droplet.compile_dataset_using_api(
        rpg_metadata, classification_metadata, tempfile_prefix=tempfile_prefix
    )
    _save_training_data(features, labels, output_filename)


def load_training_data(
    filename: str,
    training_options: VoodooTrainingOptions = VoodooTrainingOptions(),
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    data = torch.load(filename)

    x, y = data["features"], data["labels"]
    x = torch.unsqueeze(x, dim=1)
    x = torch.transpose(x, 3, 2)

    if training_options.garbage is not None:
        for i in training_options.garbage:
            y[y == i] = 999
        x = x[y < 999]
        y = y[y < 999]

    if training_options.dupe_droplets > 0:
        y_list = [(y == i).clone().detach() for i in training_options.groups[0]]
        y_tmp = torch.stack(y_list, dim=0)
        y_tmp = torch.sum(y_tmp, dim=0)
        idx_droplet = torch.argwhere(y_tmp)[:, 0]
        x = torch.cat(
            [
                x,
                torch.cat(
                    [x[idx_droplet] for _ in range(training_options.dupe_droplets)],
                    dim=0,
                ),
            ]
        )
        y = torch.cat(
            [
                y,
                torch.cat(
                    [y[idx_droplet] for _ in range(training_options.dupe_droplets)]
                ),
            ]
        )

    if training_options.shuffle:
        perm = torch.randperm(len(y))
        x, y = x[perm], y[perm]

    # drop some percentage from the data
    if 0 < training_options.split < 1:
        idx_split = int(x.shape[0] * training_options.split)
        x_train, y_train = x[idx_split:, ...], y[idx_split:]
        x_test, y_test = x[:idx_split, ...], y[:idx_split]
    else:
        raise ValueError("Provide split between 0 and 1!")

    tmp1 = torch.clone(y_train)
    tmp2 = torch.clone(y_test)
    for i, val in enumerate(training_options.groups):
        for class_no in val:
            tmp1[y_train == class_no] = i
            tmp2[y_test == class_no] = i

    y_train = tmp1
    y_test = tmp2

    del tmp1, tmp2, x, y

    y_train = torch.nn.functional.one_hot(
        y_train.to(torch.int64), num_classes=len(training_options.groups)
    ).float()
    y_test = torch.nn.functional.one_hot(
        y_test.to(torch.int64), num_classes=len(training_options.groups)
    ).float()

    return x_train.float(), y_train, x_test.float(), y_test


class VoodooDroplet:
    def __init__(
        self,
        target_time: np.ndarray | None,
        options: VoodooOptions,
        training_options: VoodooTrainingOptions,
    ):
        self.target_time = target_time
        self.options = options
        self.training_options = training_options
        self.prob_liquid: np.ndarray = np.array([])
        self._feature_list: list = []
        self._label_list: list = []
        self._model: VoodooNet | None = None

    def calc_prob(self, filename: str) -> None:
        spectra_norm, non_zero_mask, time_ind = self._extract_features(filename)
        if len(time_ind) > 0 and non_zero_mask.shape[1] == self.prob_liquid.shape[1]:
            prediction = self._predict(spectra_norm)
            if prediction.shape != (0,):
                prob = utils.reshape(prediction, ~non_zero_mask)
                prob = gaussian_filter(prob, sigma=1)
                self.prob_liquid[time_ind, :] = prob[:, :, 0]

    def compile_dataset(
        self, rpg_files: list[str], target_class_files: list[str]
    ) -> tuple[Tensor, Tensor]:
        for classification_file in target_class_files:
            logging.info(f"Categorize file: {os.path.basename(classification_file)}")
            with netCDF4.Dataset(classification_file) as nc:
                target_classification = nc.variables["target_classification"][:]
                detection_status = nc.variables["detection_status"][:]
                year, month, day = nc.year, nc.month, nc.day
                self.target_time = utils.decimal_hour2unix(
                    [year, month, day], nc.variables["time"][:]
                )
            rpg_files_of_day = utils.filter_list(rpg_files, [year[2:], month, day])

            if (n_files := len(rpg_files_of_day)) > 0:
                logging.info(f"Processing {n_files} RPG files...")

            for filename in rpg_files_of_day:
                logging.debug(filename)
                assert isinstance(filename, str)
                features, non_zero_mask, time_ind = self._extract_features(filename)
                try:
                    self._append_features(
                        time_ind,
                        target_classification,
                        detection_status,
                        non_zero_mask,
                        features,
                    )
                except ValueError:
                    continue
        return self._convert_features()

    def compile_dataset_using_api(
        self,
        rpg_metadata: list[dict],
        classification_metadata: list[dict],
        tempfile_prefix: str | None = None,
    ) -> tuple[Tensor, Tensor]:
        session = requests.Session()
        retries = Retry(total=10, backoff_factor=0.2)
        session.mount("https://", HTTPAdapter(max_retries=retries))

        for classification_meta in classification_metadata:
            logging.info(f"Categorize file: {classification_meta['filename']}")
            res = session.get(classification_meta["downloadUrl"])
            with NamedTemporaryFile(prefix=tempfile_prefix) as temp_file:
                with open(temp_file.name, "wb") as f:
                    f.write(res.content)
                with netCDF4.Dataset(temp_file.name) as nc:
                    target_classification = nc.variables["target_classification"][:]
                    detection_status = nc.variables["detection_status"][:]
                    self.target_time = utils.decimal_hour2unix(
                        [nc.year, nc.month, nc.day], nc.variables["time"][:]
                    )
            rpg_files_of_day = [
                row
                for row in rpg_metadata
                if row["measurementDate"] == classification_meta["measurementDate"]
            ]
            if (n_files := len(rpg_files_of_day)) > 0:
                logging.info(f"Processing {n_files} RPG files...")

            for rpg_meta in rpg_files_of_day:
                res = session.get(rpg_meta["downloadUrl"])
                with NamedTemporaryFile(prefix=tempfile_prefix) as temp_file:
                    with open(temp_file.name, "wb") as f:
                        f.write(res.content)
                        (
                            features,
                            non_zero_mask,
                            time_ind,
                        ) = self._extract_features(temp_file.name)
                    try:
                        self._append_features(
                            time_ind,
                            target_classification,
                            detection_status,
                            non_zero_mask,
                            features,
                        )
                    except ValueError:
                        continue
        return self._convert_features()

    def _append_features(
        self,
        time_ind: np.ndarray,
        target_classification: np.ndarray,
        detection_status: np.ndarray,
        non_zero_mask: np.ndarray,
        features: np.ndarray,
    ) -> None:
        if len(time_ind) == 0:
            raise ValueError
        classes = target_classification[time_ind, :]
        status = detection_status[time_ind, :]
        ind = np.where(non_zero_mask)
        features, labels = utils.keep_valid_samples(features, classes[ind], status[ind])
        try:
            if len(labels) == 0:
                raise ValueError
        except TypeError as exc:
            raise ValueError from exc
        assert features.ndim == 3
        assert len(labels) == features.shape[0]
        self._feature_list.append(features)
        self._label_list.append(labels)

    def _convert_features(self) -> tuple[Tensor, Tensor]:
        if len(self._feature_list) > 0 and len(self._label_list) > 0:
            features_tensor = utils.numpy_arrays2tensor(self._feature_list)
            label_tensor = utils.numpy_arrays2tensor(self._label_list)
            return features_tensor, label_tensor
        logging.error("No valid classification / RPG Level 0 files.")
        return Tensor([]), Tensor([])

    def _extract_features(
        self, filename: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        empty = (np.array([]), np.array([]), np.array([]))
        try:
            header, data = read_rpg(filename)
        except (IndexError, RPGFileError):
            logging.error(f"Error reading RPG file {filename}")
            return empty
        self._init_arrays(header, data)
        assert self.target_time is not None
        radar_time = utils.rpg_time2unix(data["Time"])
        time_ind = np.where(
            (self.target_time > min(radar_time)) & (self.target_time < max(radar_time))
        )[0]
        if len(time_ind) == 0:
            return empty
        tot_spec = data["TotSpec"]
        n_time = tot_spec.shape[0]
        feature_indices = self._feature_indices(
            radar_time, self.target_time[time_ind], n_time
        )
        # Only pixels with signal in any of the neighbouring profiles are candidates
        has_signal = tot_spec.max(axis=2) > 0
        candidates = has_signal[feature_indices].any(axis=1)
        ind_time, ind_range = np.nonzero(candidates)
        spectra, mask = _sample_spectra(
            tot_spec,
            data["SLv"],
            header,
            feature_indices[ind_time],
            ind_range,
            self.options.n_dbins,
        )
        valid = mask.any(axis=(1, 2))
        non_zero_mask = np.zeros(candidates.shape, dtype=bool)
        non_zero_mask[ind_time[valid], ind_range[valid]] = True
        spectra = np.transpose(spectra[valid], (0, 2, 1))
        spectra_norm = self._normalize_spectra(spectra)
        return spectra_norm, non_zero_mask, time_ind

    def _feature_indices(
        self, time_orig: np.ndarray, time_new: np.ndarray, n_time: int
    ) -> np.ndarray:
        """Indices of the neighbouring radar profiles for each target time."""
        mid = self.options.n_channels // 2
        ind_time = np.minimum(np.searchsorted(time_orig, time_new), n_time - 1)
        indices = ind_time[:, None] + np.arange(-mid, mid)
        return np.clip(indices, 0, n_time - 1)

    def _normalize_spectra(self, spectra: np.ndarray) -> np.ndarray:
        """Convert spectra to dBZ and normalize between 0 and 1."""
        z_min, z_max = self.options.z_limits
        valid = spectra > 0
        # log10 in float32 followed by float64 math matches the trained model input
        log_spectra = np.log10(spectra, where=valid, out=np.zeros_like(spectra))
        spectra_z = 10 * log_spectra.astype(np.float64)
        data_normalized = (spectra_z - z_min) / (z_max - z_min)
        np.clip(data_normalized, 0.0, 1.0, out=data_normalized)
        data_normalized[~valid] = 1.0
        return data_normalized

    def _init_arrays(self, header: dict, data: dict) -> None:
        """Init target time and liquid probability arrays."""
        if self.target_time is None:
            timestamp = utils.rpg_seconds2datetime64(data["Time"][0])
            date = str(timestamp.astype("datetime64[D]"))
            self.target_time = utils.time_grid(date)
        if self.prob_liquid.shape == (0,):
            self.prob_liquid = np.zeros((len(self.target_time), len(header["RAlts"])))

    def _predict(self, data: np.ndarray) -> Tensor:
        tensor = torch.Tensor(data)
        tensor = torch.unsqueeze(tensor, dim=1)
        tensor = torch.transpose(tensor, 3, 2)
        if self._model is None:
            self._model = VoodooNet(tensor.shape, self.options, self.training_options)
            self._model.load_state_dict(
                torch.load(
                    self.options.trained_model, map_location=self.options.device
                )["state_dict"]
            )
        prediction = self._model.predict(tensor, batch_size=256).to("cpu")
        return prediction


def _sample_spectra(
    tot_spec: np.ndarray,
    sensitivity: np.ndarray,
    header: dict,
    ind_time: np.ndarray,
    ind_range: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract spectra of the given pixels resampled to n_bins velocity bins.

    Fill values (<= 0) are replaced with the sensitivity limit.
    Returns the spectra and the signal mask, both with shape
    (n_samples, n_channels, n_bins).
    """
    n_samples, n_channels = ind_time.shape
    spectra = np.zeros((n_samples, n_channels, n_bins), dtype=np.float32)
    mask = np.zeros((n_samples, n_channels, n_bins), dtype=bool)
    chirp_limits = np.append(header["RngOffs"], tot_spec.shape[1])
    for chirp, (ia, ib) in enumerate(zip(chirp_limits[:-1], chirp_limits[1:])):
        in_chirp = (ind_range >= ia) & (ind_range < ib)
        if not in_chirp.any():
            continue
        ind_bin, in_bounds = _nearest_bin_indices(
            header["velocity_vectors"][chirp], n_bins
        )
        t_ind = ind_time[in_chirp]
        r_ind = ind_range[in_chirp]
        spec = tot_spec[t_ind[:, :, None], r_ind[:, None, None], ind_bin[None, None, :]]
        signal = spec > 0
        spec = np.where(signal, spec, sensitivity[t_ind, r_ind[:, None]][:, :, None])
        spec[:, :, ~in_bounds] = -999.0
        signal[:, :, ~in_bounds] = False
        spectra[in_chirp] = spec
        mask[in_chirp] = signal
    return spectra, mask


def _nearest_bin_indices(
    velocity: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Map n_bins evenly spaced velocity bins to the nearest original bins."""
    ia, ib = int(np.argmin(velocity)), int(np.argmax(velocity)) + 1
    velocity = velocity[ia:ib]
    f = interp1d(
        velocity,
        np.arange(ia, ib),
        kind="nearest",
        bounds_error=False,
        fill_value=-1,
    )
    ind = f(np.linspace(velocity[0], velocity[-1], n_bins))
    in_bounds = ind >= 0
    return np.where(in_bounds, ind, 0).astype(int), in_bounds


def _save_training_data(
    features: Tensor,
    labels: Tensor,
    file_name: str,
) -> None:
    torch.save({"features": features, "labels": labels}, file_name)
