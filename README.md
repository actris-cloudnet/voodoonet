# VoodooNet
Predicting liquid droplets in mixed-phase clouds beyond lidar attenuation using artificial neural nets and Doppler cloud radar spectra


<div align="center">
  <a href="https://github.com/remsens-lim/Voodoo">
    <img src="voodoonet/img/voodoo_logo.png" alt="Logo" width="720" height="280">
  </a>
</div>

VOODOO is a machine learning approach based convolutional neural networks (CNN) to relate Doppler spectra morphologies to the presence of (supercooled) liquid cloud droplets in mixed-phase clouds.

## Prerequisites

### Python

VoodooNet requires Python3.10.

### Torch

Before installing `VoodooNet`, install `torch` [according to your infrastructure](https://pytorch.org/get-started/locally/). For example,
```sh
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

## Basic usage

### Make predictions
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
    '/path/to/trainingset/data.pt', options=VoodooTrainingOptions()
)

# load the model and train
voodoo_model = VoodooNet(X_train.shape, options=VoodooOptions(), training_options=VoodooTrainingOptions())
voodoo_model.print_nparams()

voodoo_model.optimize(X_train, y_train, X_test, y_test, epochs=5)

# save model and statistics
voodoo_model.save(path='/path/to/voodoonet/model.pt', aux=voodoo_model.options.dict())
```
