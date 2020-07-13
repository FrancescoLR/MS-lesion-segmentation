# Multiple sclerosis cortical and WM lesion segmentation at 3T MRI: a deep learning method based on FLAIR and MP2RAGE

This is the code repository of the Neuroimage: Clinical [paper](https://doi.org/10.1016/j.nicl.2020.102335).

## Overview
This software provide a multiple sclerosis cortical and white matter lesion segmentation. The input required are co-registered FLAIR and MP2RAGE volumes of the subject acquired at 3T. The code depends on the [NiftyNet framework](https://niftynet.readthedocs.io/) and it is based on [Tensorflow](https://www.tensorflow.org/). A GPU is required to train the network with new data, whereas a LINUX machine is sufficient for inference.

## Install
To install the software clone the repository and pull the niftynet code in the submodule with `git submodule update --init`.
Next, install the required dependencies: <br />
`cd NiftyNet/` <br />
`pip install -r requirements.txt` <br />

To run the code please follow the instructions on the [NiftyNet website](https://niftynet.readthedocs.io/en/dev/). A configuration file and a trained model are provided in separate folders in the root directory.


## License
This software is released under the version 2.0 of the Apache License. Please read the license terms before using the software. A copy of this license is present in the root directory.

## Cite
If you use this code please cite the follwing references:

- Gibson, Eli, et al. "NiftyNet: a deep-learning platform for medical imaging." Computer methods and programs in biomedicine 158 (2018): 113-122. [doi](https://doi.org/10.1016/j.cmpb.2018.01.025)
- La Rosa, Francesco, et al. "Multiple sclerosis cortical and WM lesion segmentation at 3T MRI: a deep learning method based on FLAIR and MP2RAGE." NeuroImage: Clinical (2020): 102335. [doi](https://doi.org/10.1016/j.nicl.2020.102335)
- La Rosa, Francesco, Abdulkadir. Ahmed, Thiran, Jean-Philippe, Granziera, Cristina, & Bach Cuadra, Merixtell. (2020, July 7). Software: Multiple sclerosis cortical and WM lesion segmentation at 3T MRI: a deep learning method based on FLAIR and MP2RAGE (Version v1.0). [doi](https://doi.org/10.5281/zenodo.3932736)


