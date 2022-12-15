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

### Make predictions using default settings

```python
import voodoonet
import glob

rpg_files = glob.glob('/path/to/rpg/files/*.LV0')
probability_liquid = voodoonet.run(rpg_files)
```

### Generate the training dataset

```python
import voodoonet
import glob

rpg_files = glob.glob('/path/to/rpg/files/*.LV0')
classification_files = glob.glob('/path/to/classification/files/*.nc')

features, labels = voodoonet.generate_trainingdata(rpg_files, classification_files)
voodoonet.save_trainingdata(features, labels, '/path/to/trainingset/data.pt')
```

### Train a VoodooNet model

```python
import voodoonet
from voodoonet.torch_model import VoodooNet
from voodoonet.utils import VoodooOptions, VoodooTrainingOptions

X_train, y_train, X_test, y_test = voodoonet.loader.load_trainingdata(
    '/path/to/trainingset/data.pt',
    options=VoodooTrainingOptions()
)

# Load the model and train.
voodoo_model = VoodooNet(
    X_train.shape,
    options=VoodooOptions(),
    training_options=VoodooTrainingOptions()
)
voodoo_model.print_nparams()
voodoo_model.optimize(X_train, y_train, X_test, y_test, epochs=5)

# Save model and statistics.
voodoo_model.save(path='/path/to/voodoonet/model.pt', aux=voodoo_model.options.dict())
```

### Make predictions using a new model

```python
import voodoonet
from voodoonet.utils import VoodooOptions
import glob

rpg_files = glob.glob('/path/to/rpg/files/*.LV0')

options = VoodooOptions(trained_model='/path/to/new/model.pt')
probability_liquid = voodoonet.run(rpg_files, options=options)
```
