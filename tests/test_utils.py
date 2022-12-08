import numpy as np
import pytest

from voodoonet import utils


@pytest.mark.parametrize(
    "data_in, value, result",
    [
        (np.array([1, 2, 3]), 2.0, 1),
        (np.array([1, 2, 3]), -100.0, 0),
        (np.array([1, 2, 3]), 1.5, 1),
        (np.array([1, 2, 3]), 100.0, 2),
    ],
)
def test_arg(data_in: np.ndarray, value: float, result: int) -> None:
    assert utils.arg_nearest(data_in, value) == result
