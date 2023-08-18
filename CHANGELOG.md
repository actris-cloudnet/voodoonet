# Changelog

## 0.1.4 – 2023-08-18

- Fix URL names
- Harmonize workflows with other repositories

## 0.1.3 – 2023-04-27

- Add `batch_size` to training options
- Add model options to train argument and move tensors to right device
- Add tempfile prefix option
- Add dataset iterator
- Fix type hints and add py.typed

## 0.1.2 – 2023-01-11

- Send predictions to cpu for voodoo post-processing
- Make file handling more robust
- Use human-readable `pylint` problem names
- Update `checkout` and `setup-python` versions
- Improve API response parsing

## 0.1.1 – 2022-12-19

- Add `n_days` parameter
- Use `Retry` strategy
- Bug fixes

## 0.1.0 – 2022-12-19

- Optimize time-consuming functions
- Move `wandb` config and `epochs` to `VoodooTrainingOptions`
- Organize / rename functions
- Use netCDF4 instead of `xarray` and `dask`
- Use logging
- Update README.md
- Add CITATION.cff

## 0.0.4 – 2022-12-13

- Fix interpolation

## 0.0.3 – 2022-12-13

- Avoid error when no time indices

## 0.0.2 – 2022-12-13

- Add trained model file to PyPI package

## 0.0.1 – 2020-12-12

Initial release
