[![VoodooNet CI](https://github.com/actris-cloudnet/voodoonet/actions/workflows/test.yml/badge.svg)](https://github.com/actris-cloudnet/voodoonet/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/voodoonet.svg)](https://badge.fury.io/py/voodoonet)

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

VoodooNet requires Python 3.10.

Before installing VoodooNet, install PyTorch [according to your infrastructure](https://pytorch.org/get-started/locally/). For example on a Linux machine without GPU you might run:

```sh
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

### From PyPI

```sh
pip3 install voodoonet
```

## Usage

### Make predictions using the default model and settings

```python
import glob
import voodoonet

rpg_files = glob.glob('/path/to/rpg/files/*.LV0')
probability_liquid = voodoonet.infer(rpg_files)
```

### Generate a training data set

```python
import glob
import voodoonet

rpg_files = glob.glob('/path/to/rpg/files/*.LV0')
classification_files = glob.glob('/path/to/classification/files/*.nc')
voodoonet.generate_training_data(rpg_files, classification_files, 'training-data-set.pt')
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
from voodoonet import VoodooOptions

rpg_files = glob.glob('/path/to/rpg/files/*.LV0')
options = VoodooOptions(trained_model='new_model.pt')
probability_liquid = voodoonet.infer(rpg_files, options=options)
```
