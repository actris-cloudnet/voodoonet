# VoodooNet
Predicting liquid droplets in mixed-phase clouds beyond lidar attenuation using artificial neural nets and Doppler cloud radar spectra


<div align="center">
  <a href="https://github.com/remsens-lim/Voodoo">
    <img src="voodoonet/img/voodoo_logo.png" alt="Logo" width="520" height="280">
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
```python
import voodoonet
import glob

rpg_files = glob.glob('/path/to/rpg/files/*.LV0')
probability_liquid = voodoonet.run(rpg_files)
```
