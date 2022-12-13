import numpy as np
import torch
import xarray as xr
from rpgpy import read_rpg
from scipy.ndimage import gaussian_filter
from torch import Tensor, save
from tqdm.auto import tqdm  # remove when implementation is done

from voodoonet import utils
from voodoonet.utils import VoodooOptions

from .torch_model import VoodooNet


def run(
    rpg_lv0_files: list,
    target_time: np.ndarray | None = None,
    options: VoodooOptions = VoodooOptions(),
) -> np.ndarray:
    voodoo_droplet = VoodooDroplet(target_time, options)
    for filename in rpg_lv0_files:
        voodoo_droplet.calc_prob(filename)
    return voodoo_droplet.prob_liquid


def generate_trainingdata(
    rpg_lv0_files: list,
    classification_files: list,
    options: VoodooOptions = VoodooOptions(),
) -> tuple[Tensor, Tensor]:
    voodoo_droplet = VoodooDroplet(None, options)
    voodoo_droplet.compile_dataset(rpg_lv0_files, classification_files)
    return voodoo_droplet.features, voodoo_droplet.labels


def save_trainingdata(
    features: Tensor,
    labels: Tensor,
    file_name: str,
) -> None:
    save({"features": features, "labels": labels}, file_name)


class VoodooDroplet:
    def __init__(self, target_time: np.ndarray | None, options: VoodooOptions):
        self.target_time = target_time
        self.options = options
        self.prob_liquid = np.array([])
        self.features = Tensor([])
        self.labels = Tensor([])

    def calc_prob(self, filename: str) -> None:
        spectra_norm, non_zero_mask, time_ind = self.extract_features(filename)
        if len(time_ind) > 0:
            prediction = self._predict(spectra_norm)
            prob = utils.reshape(prediction, ~non_zero_mask)
            prob = gaussian_filter(prob, sigma=1)
            prob = prob[:, :, 0]
            self.prob_liquid[time_ind, :] = prob

    def extract_features(self, filename: str) -> tuple:
        header, data = read_rpg(filename)
        self._init_arrays(header, data)
        assert self.target_time is not None
        radar_time = utils.rpg_time2unix(data["Time"])
        time_ind = np.where(
            (self.target_time > min(radar_time)) & (self.target_time < max(radar_time))
        )
        if len(time_ind[0]) == 0:
            return np.array([]), np.array([]), np.array([])
        non_zero_mask = data["TotSpec"] > 0.0
        spectra = _replace_fill_value(data["TotSpec"], data["SLv"])
        spectra = utils.interpolate_to_256(spectra, header)
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
        return spectra_norm, non_zero_mask, time_ind

    def compile_dataset(self, rpg_lv0_files: list[str], target_class_files: list[str]) -> None:

        feature_list = []
        label_list = []

        for class_file in sorted(target_class_files):
            xr_dataset = xr.open_mfdataset(class_file)
            target_classification = xr_dataset["target_classification"]
            detection_status = xr_dataset["detection_status"]
            self.target_time = utils.numpy_datetime2unix(target_classification.time.values)
            year, month, day = xr_dataset.year, xr_dataset.month, xr_dataset.day

            daily_rpg_lv0_files = utils.filter_list(rpg_lv0_files, [year[2:], month, day])

            for filename in tqdm(daily_rpg_lv0_files):
                assert isinstance(filename, str)
                features, non_zero_mask, time_ind = self.extract_features(filename)

                classes = target_classification.values[time_ind[0], :]
                status = detection_status.values[time_ind[0], :]

                ind = np.where(non_zero_mask)
                features, labels = utils.keep_valid_samples(
                    features, classes[ind[0], ind[1]], status[ind[0], ind[1]]
                )

                feature_list.append(features)
                label_list.append(labels)

        self.features = utils.numpy_arrays2tensor(feature_list)
        self.labels = utils.numpy_arrays2tensor(label_list)

    def _init_arrays(self, header: dict, data: dict) -> None:
        """Init target time and liquid probability arrays."""
        if self.target_time is None:
            timestamp = utils.rpg_seconds2datetime64(data["Time"][0])
            date = str(timestamp.astype("datetime64[D]"))
            self.target_time = utils.time_grid(date)
        if self.prob_liquid.shape == (0,):
            self.prob_liquid = np.zeros((len(self.target_time), len(header["RAlts"])))

    def _normalize_spectra(self, spectra: np.ndarray) -> np.ndarray:
        """Normalize spectra between 0 and 1."""
        z_min, z_max = self.options.z_limits
        data_normalized = (spectra - z_min) / (z_max - z_min)
        data_normalized[data_normalized < 0.0] = 0.0
        data_normalized[data_normalized > 1.0] = 1.0
        return data_normalized

    def _predict(self, data: np.ndarray) -> Tensor:
        tensor = torch.Tensor(data)
        tensor = torch.unsqueeze(tensor, dim=1)
        tensor = torch.transpose(tensor, 3, 2)
        voodoo_net = VoodooNet(tensor.shape, self.options)
        voodoo_net.load_state_dict(
            torch.load(self.options.trained_model, map_location=self.options.device)["state_dict"]
        )
        prediction = voodoo_net.predict(tensor, batch_size=256).to(self.options.device)
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

        shape = (n_time_new, n_range, n_vel, self.options.n_channels)
        ip_var = np.full(shape, fill_value=-999.0, dtype=np.float32)
        ip_msk = np.full(shape, fill_value=True)

        for ind_vel in range(n_vel):
            for ind_time in range(n_time_new):
                ind_time_nearest = utils.arg_nearest(time_orig, time_new[ind_time])
                for i, itmp in enumerate(range(-mid, mid, 1)):
                    iTdiff = itmp if ind_time_nearest + itmp < n_time else 0
                    ip_var[ind_time, :, ind_vel, i] = spec_vh[ind_time_nearest + iTdiff, :, ind_vel]
                    ip_msk[ind_time, :, ind_vel, i] = mask[ind_time_nearest + iTdiff, :, ind_vel]
        return ip_var, ip_msk


def _replace_fill_value(data: np.ndarray, new_fill: np.ndarray) -> np.ndarray:
    """
    Replaces the fill value of a spectrum container by their time
    and range specific mean noise level.
    Args:
        data: 3D spectrum array (time, range, velocity)
        new_fill: 2D new fill values for 3rd dimension (velocity)

    Return:
        var (numpy.array) : spectrum with mean noise
    """

    n_ts, n_rg, _ = data.shape
    masked = np.all(data <= 0.0, axis=2)

    for iT in range(n_ts):
        for iR in range(n_rg):
            if masked[iT, iR]:
                data[iT, iR, :] = new_fill[iT, iR]
            else:
                data[iT, iR, data[iT, iR, :] <= 0.0] = new_fill[iT, iR]
    return data


def load_trainingdata(
    filename: str,
    garbage: list[int],
    dupe_droplets: int,
    groups: list[list[int]],
    shuffle: bool = True,
    split: float = 0.1,  # -> 10% of data for validation
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    data = torch.load(filename)

    X, y = data["features"], data["labels"]
    X = torch.unsqueeze(X, dim=1)
    X = torch.transpose(X, 3, 2)

    if garbage is not None:
        for i in garbage:
            y[y == i] = 999
        X = X[y < 999]
        y = y[y < 999]

    if dupe_droplets > 0:
        # lookup indices for cloud dorplet bearing classes
        idx_CD = torch.argwhere(
            torch.sum(torch.stack([torch.tensor(y == i) for i in groups[0]], dim=0), dim=0)
        )[:, 0]
        X = torch.cat([X, torch.cat([X[idx_CD] for _ in range(dupe_droplets)], dim=0)])
        y = torch.cat([y, torch.cat([y[idx_CD] for _ in range(dupe_droplets)])])

    if shuffle:
        perm = torch.randperm(len(y))
        X, y = X[perm], y[perm]

    # drop some percentage from the data
    if 0 < split < 1:
        idx_split = int(X.shape[0] * split)
        X_train, y_train = X[idx_split:, ...], y[idx_split:]
        X_test, y_test = X[:idx_split, ...], y[:idx_split]
    else:
        raise ValueError("Provide split between 0 and 1!")

    tmp1 = torch.clone(y_train)
    tmp2 = torch.clone(y_test)
    for i, val in enumerate(groups):  # i from 0, ..., ngroups-1
        for jclass in val:
            tmp1[y_train == jclass] = i
            tmp2[y_test == jclass] = i

    y_train = tmp1
    y_test = tmp2

    del tmp1, tmp2, X, y

    y_train = torch.nn.functional.one_hot(y_train.to(torch.int64), num_classes=len(groups)).float()
    y_test = torch.nn.functional.one_hot(y_test.to(torch.int64), num_classes=len(groups)).float()

    return X_train, y_train, X_test, y_test
