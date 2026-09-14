[![VoodooNet CI](https://github.com/actris-cloudnet/voodoonet/actions/workflows/test.yml/badge.svg)](https://github.com/actris-cloudnet/voodoonet/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/voodoonet.svg)](https://badge.fury.io/py/voodoonet)
[![DOI](https://zenodo.org/badge/575846028.svg)](https://zenodo.org/badge/latestdoi/575846028)

# VoodooNet

Predicting liquid droplets in mixed-phase clouds beyond lidar attenuation using artificial neural nets and Doppler cloud radar spectra

<div align="center">
  <a href="https://github.com/actris-cloudnet/voodoonet">
    <img src="https://raw.githubusercontent.com/actris-cloudnet/voodoonet/main/voodoonet/img/voodoo_logo.png" alt="VOODOO logo" width="630" height="270">
  </a>
</div>

VOODOO is a machine learning approach based convolutional neural networks (CNN) to relate Doppler spectra morphologies to the presence of (supercooled) liquid cloud droplets in mixed-phase clouds.

## Installation

### Prerequisites

VoodooNet requires Python 3.10 or newer.

Before installing VoodooNet, install PyTorch [according to your infrastructure](https://pytorch.org/get-started/locally/). Otherwise pip installs the default PyTorch build, which on Linux includes CUDA libraries and is several gigabytes. For example on a Linux machine without GPU you might run:

```sh
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

### From PyPI

```sh
pip3 install voodoonet
```

To log training runs with [Weights & Biases](https://wandb.ai/), install the `train` extra:

```sh
pip3 install voodoonet[train]
```

### Locally for development

```sh
pip3 install -e .[dev]
```

## Citing

If you wish to acknowledge VoodooNet in your publication, please cite:

> Schimmel et al. (2022). Identifying cloud droplets beyond lidar attenuation from vertically pointing cloud radar observations using artificial neural networks. _Atmos. Meas. Tech._, _15_(18), 5343–5366. <https://doi.org/10.5194/amt-15-5343-2022>

## Usage

### Make predictions using the default model and settings

```python
import glob
import voodoonet

rpg_files = glob.glob('/path/to/rpg/files/*.LV0')
probability_liquid = voodoonet.infer(rpg_files)
```

You can for example plot the resulting liquid probability:

```python
import matplotlib.pyplot as plt

plt.pcolor(probability_liquid.T)
plt.show()
```

![](https://raw.githubusercontent.com/actris-cloudnet/voodoonet/main/voodoonet/img/voodoo_plot.png)

### Generate a training data set

Download some RPG-FMCW-94 raw files and corresponding classification files from the [Cloudnet data portal](https://cloudnet.fmi.fi/) using [cloudnet-api-client](https://pypi.org/project/cloudnet-api-client/), which is installed with voodoonet. For example, for [Leipzig LIM](https://cloudnet.fmi.fi/site/leipzig-lim) on 2021-01-10:

```python
import voodoonet
from cloudnet_api_client import APIClient

client = APIClient()
rpg_meta = client.raw_files(
    site_id="leipzig-lim",
    instrument_id="rpg-fmcw-94",
    filename_suffix=".LV0",
    date="2021-01-10",
)
classification_meta = client.files(
    site_id="leipzig-lim",
    product_id="classification",
    date="2021-01-10",
)
rpg_files = client.download(rpg_meta, "data/")
classification_files = client.download(classification_meta, "data/")
voodoonet.generate_training_data(rpg_files, classification_files, 'training-data-set.pt')
```

Alternatively, just use N random days from a site. Files are downloaded into `download_dir` (default `cloudnet-data`) and reused on subsequent runs:

```python
import voodoonet
voodoonet.generate_training_data_for_cloudnet('leipzig-lim', 'training-data-set.pt', n_days=5)
```

### Train a VoodooNet model

```python
import voodoonet

pre_computed_training_data_set = 'training-data-set.pt'
voodoonet.train(pre_computed_training_data_set, 'trained-model.pt')
```

### Make predictions using the new model

```python
import glob
import voodoonet
from voodoonet.utils import VoodooOptions

rpg_files = glob.glob('/path/to/rpg/files/*.LV0')
options = VoodooOptions(trained_model='trained-model.pt')
probability_liquid = voodoonet.infer(rpg_files, options=options)
```
