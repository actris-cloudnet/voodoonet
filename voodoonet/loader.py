import logging
import os.path
import random
from tempfile import NamedTemporaryFile

import netCDF4
import numpy as np
import requests
import torch
from requests.adapters import HTTPAdapter, Retry
from rpgpy import read_rpg
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
    for filename in rpg_lv0_files:
        voodoo_droplet.calc_prob(filename)
    return voodoo_droplet.prob_liquid


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

    return x_train, y_train, x_test, y_test


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

    def calc_prob(self, filename: str) -> None:
        spectra_norm, non_zero_mask, time_ind = self._extract_features(filename)
        if len(time_ind) > 0:
            prediction = self._predict(spectra_norm)
            if prediction.shape == (0,):
                prob = np.zeros(non_zero_mask.shape)
            else:
                prob = utils.reshape(prediction, ~non_zero_mask)
                prob = gaussian_filter(prob, sigma=1)
                prob = prob[:, :, 0]
            self.prob_liquid[time_ind, :] = prob

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
        try:
            header, data = read_rpg(filename)
        except IndexError:
            logging.error(f"Error reading RPG file {filename}")
            return np.array([]), np.array([]), np.array([])
        self._init_arrays(header, data)
        assert self.target_time is not None
        radar_time = utils.rpg_time2unix(data["Time"])
        time_ind = np.where(
            (self.target_time > min(radar_time)) & (self.target_time < max(radar_time))
        )
        if len(time_ind) == 0:
            return np.array([]), np.array([]), np.array([])
        non_zero_mask = data["TotSpec"] > 0.0
        spectra = _replace_fill_value(data["TotSpec"], data["SLv"])
        spectra = _interpolate_to_256(spectra, header)
        non_zero_mask = _interpolate_to_256(non_zero_mask, header) >= 0.5
        interp_var, interp_mask = self._hyperspectral_image(
            radar_time,
            spectra,
            non_zero_mask,
            self.target_time[time_ind],
        )
        non_zero_mask = (interp_mask.any(axis=3)).any(axis=2)
        ind = np.where(non_zero_mask)
        spectra = interp_var[ind[0], ind[1], :, :]
        spectra = utils.lin2z(spectra)
        spectra_norm = self._normalize_spectra(spectra)
        return spectra_norm, non_zero_mask, time_ind[0]

    def _normalize_spectra(self, spectra: np.ndarray) -> np.ndarray:
        """Normalize spectra between 0 and 1."""
        z_min, z_max = self.options.z_limits
        data_normalized = (spectra - z_min) / (z_max - z_min)
        data_normalized[data_normalized < 0.0] = 0.0
        data_normalized[data_normalized > 1.0] = 1.0
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
        voodoo_net = VoodooNet(tensor.shape, self.options, self.training_options)
        voodoo_net.load_state_dict(
            torch.load(self.options.trained_model, map_location=self.options.device)[
                "state_dict"
            ]
        )
        prediction = voodoo_net.predict(tensor, batch_size=256).to("cpu")
        return prediction

    def _hyperspectral_image(
        self,
        time_orig: np.ndarray,
        spec_vh: np.ndarray,
        mask: np.ndarray,
        time_new: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_time_new = len(time_new)
        n_time, n_range, n_vel = spec_vh.shape
        mid = self.options.n_channels // 2

        shape = (n_time_new, self.options.n_channels, n_range, n_vel)
        ip_var = np.full(shape, fill_value=-999.0, dtype=np.float32)
        ip_msk = np.full(shape, fill_value=True)
        feature_indices = np.zeros((n_time_new, self.options.n_channels), dtype=int)

        for ind in range(n_time_new):
            ind_time = utils.arg_nearest(time_orig, time_new[ind])
            feature_indices[ind, :] = np.array(range(ind_time - mid, ind_time + mid))
        feature_indices[feature_indices < 0] = 0
        feature_indices[feature_indices >= n_time] = n_time - 1

        for idx_time, idx_features in enumerate(feature_indices):
            ip_var[idx_time, :, :, :] = spec_vh[idx_features, :, :]
            ip_msk[idx_time, :, :, :] = mask[idx_features, :, :]

        ip_var = np.transpose(ip_var, axes=[0, 2, 3, 1])
        ip_msk = np.transpose(ip_msk, axes=[0, 2, 3, 1])

        return ip_var, ip_msk


def _replace_fill_value(data: np.ndarray, new_fill: np.ndarray) -> np.ndarray:
    fill_3d = np.broadcast_to(new_fill[..., None], new_fill.shape + (data.shape[2],))
    data[data <= 0] = fill_3d[data <= 0]
    return data


def _interpolate_to_256(rpg_data: np.ndarray, rpg_header: dict) -> np.ndarray:
    n_bins = 256
    n_time, n_range, _ = rpg_data.shape
    spec_new = np.zeros((n_time, n_range, n_bins))
    chirp_limits = np.append(rpg_header["RngOffs"], n_range)
    for ind, (ia, ib) in enumerate(zip(chirp_limits[:-1], chirp_limits[1:])):
        spec = rpg_data[:, ia:ib, :]
        if rpg_header["SpecN"][ind] == n_bins and max(spec.shape) == n_bins:
            spec_new[:, ia:ib, :] = spec
        else:
            old = rpg_header["velocity_vectors"][ind]
            iaa, ibb = int(np.argmin(old)), int(np.argmax(old)) + 1
            old = old[iaa:ibb]
            f = interp1d(
                old,
                spec[:, :, iaa:ibb],
                axis=2,
                bounds_error=False,
                fill_value=-999.0,
                kind="nearest",
            )
            spec_new[:, ia:ib, :] = f(np.linspace(old[0], old[-1], n_bins))

    return spec_new


def _save_training_data(
    features: Tensor,
    labels: Tensor,
    file_name: str,
) -> None:
    torch.save({"features": features, "labels": labels}, file_name)
