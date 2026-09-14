import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.interpolate import interp1d

from voodoonet import loader, utils
from voodoonet.utils import VoodooOptions, VoodooTrainingOptions

N_BINS = 256


def _reference_resample(spectra: np.ndarray, header: dict) -> np.ndarray:
    """Nearest-neighbour resampling of the full spectra cube to N_BINS."""
    n_time, n_range, _ = spectra.shape
    out = np.zeros((n_time, n_range, N_BINS))
    limits = np.append(header["RngOffs"], n_range)
    for chirp, (ia, ib) in enumerate(zip(limits[:-1], limits[1:])):
        velocity = header["velocity_vectors"][chirp]
        iaa, ibb = int(np.argmin(velocity)), int(np.argmax(velocity)) + 1
        velocity = velocity[iaa:ibb]
        f = interp1d(
            velocity,
            spectra[:, ia:ib, iaa:ibb],
            axis=2,
            bounds_error=False,
            fill_value=-999.0,
            kind="nearest",
        )
        out[:, ia:ib, :] = f(np.linspace(velocity[0], velocity[-1], N_BINS))
    return out


@pytest.fixture
def rpg_data() -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(1)
    n_time, n_range, n_vel = 20, 30, 512
    spectra = rng.random((n_time, n_range, n_vel), dtype=np.float32)
    spectra[spectra < 0.9] = 0.0
    spectra[:, 10:15, :] = 0.0
    sensitivity = rng.random((n_time, n_range), dtype=np.float32) + 0.1
    velocity_full = np.linspace(-5, 5, n_vel)
    velocity_half = np.zeros(n_vel)
    velocity_half[:256] = np.linspace(-3, 3, 256)
    header = {
        "RngOffs": np.array([0, 12]),
        "SpecN": np.array([512, 256]),
        "velocity_vectors": [velocity_full, velocity_half],
    }
    return spectra, sensitivity, header


def test_nearest_bin_indices() -> None:
    velocity = np.zeros(512)
    velocity[:256] = np.linspace(-3, 3, 256)
    ind, in_bounds = loader._nearest_bin_indices(velocity, N_BINS)
    assert np.array_equal(ind, np.arange(256))
    assert in_bounds.all()
    ind, in_bounds = loader._nearest_bin_indices(np.linspace(-5, 5, 512), N_BINS)
    assert ind.min() == 0
    assert ind.max() == 511
    assert np.all(np.diff(ind) > 0)


def test_sample_spectra_matches_full_resampling(rpg_data: tuple) -> None:
    spectra, sensitivity, header = rpg_data
    n_time, n_range, _ = spectra.shape
    mask_ref = _reference_resample(spectra > 0, header) >= 0.5
    filled = np.where(spectra > 0, spectra, sensitivity[:, :, None])
    spectra_ref = _reference_resample(filled, header)
    ind_time = np.array([[0, 1, 2], [5, 5, 6], [19, 19, 19], [3, 4, 5]])
    ind_range = np.array([0, 11, 12, 29])
    sampled, mask = loader._sample_spectra(
        spectra, sensitivity, header, ind_time, ind_range, N_BINS
    )
    assert sampled.shape == (4, 3, N_BINS)
    for i, (t_ind, r_ind) in enumerate(zip(ind_time, ind_range)):
        assert np.array_equal(sampled[i], spectra_ref[t_ind, r_ind])
        assert np.array_equal(mask[i], mask_ref[t_ind, r_ind])


def test_feature_indices() -> None:
    droplet = loader.VoodooDroplet(None, VoodooOptions(), VoodooTrainingOptions())
    time_orig = np.arange(0, 100, 10)
    time_new = np.array([-5, 12, 41, 95, 200])
    indices = droplet._feature_indices(time_orig, time_new, len(time_orig))
    assert indices.shape == (5, 6)
    assert np.array_equal(indices[0], [0, 0, 0, 0, 1, 2])
    assert np.array_equal(indices[1], [0, 0, 1, 2, 3, 4])
    assert np.array_equal(indices[2], [2, 3, 4, 5, 6, 7])
    assert np.array_equal(indices[4], [6, 7, 8, 9, 9, 9])


def test_normalize_spectra() -> None:
    droplet = loader.VoodooDroplet(None, VoodooOptions(), VoodooTrainingOptions())
    spectra = np.array([[1e-5, 1.0, 100.0, 1e30, 0.0]], dtype=np.float32)
    normalized = droplet._normalize_spectra(spectra)
    assert normalized.dtype == np.float64
    assert np.allclose(normalized, [[0.0, 50 / 70, 1.0, 1.0, 1.0]])


def test_reshape() -> None:
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mask = np.array([[True, False, True], [False, False, True]])
    out = utils.reshape(data, mask)
    assert out.shape == (2, 3, 2)
    assert np.array_equal(out[0, 1], [1.0, 2.0])
    assert np.array_equal(out[1, 0], [3.0, 4.0])
    assert np.array_equal(out[1, 1], [5.0, 6.0])
    assert np.all(out[mask] == 0)


class _Meta:
    def __init__(self, filename: str, date: str):
        self.filename = filename
        self.measurement_date = datetime.date.fromisoformat(date)


class _FakeClient:
    def __init__(self) -> None:
        self.downloaded: list[str] = []

    def files(self, **kwargs: object) -> list:
        return [
            _Meta(f"{d}_classification.nc", d) for d in ("2021-01-10", "2021-01-11")
        ]

    def raw_files(self, **kwargs: object) -> list:
        return [
            _Meta("210110_000000_P05_ZEN.LV0", "2021-01-10"),
            _Meta("210110_010000_P05_ZEN.LV0", "2021-01-10"),
            _Meta("210112_000000_P05_ZEN.LV0", "2021-01-12"),
        ]

    def download(self, metadata: list, **kwargs: object) -> list[Path]:
        paths = [Path(kwargs["output_directory"], m.filename) for m in metadata]  # type: ignore[arg-type]
        self.downloaded += [p.name for p in paths]
        return paths


def test_generate_training_data_for_cloudnet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _FakeClient()
    days: list[tuple[list[str], str]] = []
    monkeypatch.setattr(loader, "APIClient", lambda: client)
    monkeypatch.setattr(
        loader.VoodooDroplet,
        "compile_day",
        lambda self, rpg, cls: days.append((rpg, cls)),
    )
    monkeypatch.setattr(
        loader.VoodooDroplet,
        "convert_features",
        lambda self: (torch.Tensor([1]), torch.Tensor([0])),
    )
    output = tmp_path / "train.pt"
    loader.generate_training_data_for_cloudnet(
        "leipzig-lim", str(output), download_dir=str(tmp_path)
    )
    # only 2021-01-10 has both classification and LV0 files
    assert len(days) == 1
    rpg, cls = days[0]
    assert cls.endswith("2021-01-10_classification.nc")
    assert [Path(p).name for p in rpg] == [
        "210110_000000_P05_ZEN.LV0",
        "210110_010000_P05_ZEN.LV0",
    ]
    assert sorted(client.downloaded) == sorted(
        [Path(cls).name, *[Path(p).name for p in rpg]]
    )
    assert output.exists()
