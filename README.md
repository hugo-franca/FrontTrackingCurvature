# A machine learning strategy for computing interface curvature in Front-Tracking methods
 This repository contains the example code mentioned in the paper published at the Journal of Computational Physics. The paper can be accessed on this link: [LINK](https://doi.org/10.1016/j.jcp.2021.110860).
 
 If you have any issues running the code, you can contact us at: [franca.hugo1@gmail.com](mailto:franca.hugo1@gmail.com).

## Requirements to run these examples
 In order to run these examples, you will need an installation of Python. The examples were last tested on **Python version 3.8.5**, but they will likely work on other versions reasonably near this one as well.

 Below, we list the five python modules you will need to install in order to run the examples. Below each module, we list the python command you can use to install it, and also the version of that module that we last tested the examples with. Once again, they will probably work with other versions as well.

 1. #### Module Numpy
	- Intallation command: **python -m pip install numpy**
	- Last tested with version: **1.21.4**
 2. #### Module Matplotlib:
 	- Intallation command: **python -m pip install matplotlib**
 	- Last tested with version: **3.5.0**
 3. #### Module Pandas: 
 	- Intallation command: **python -m pip install pandas**
 	- Last tested with version: **1.3.4**
 4. #### Module Tensorflow:
 	- Intallation command: **python -m pip install tensorflow**
 	- Last tested with version: **2.7.0**
 5. #### Module scipy: 
 	- Intallation command: **python -m pip install scipy**
 	- Last tested with version: **1.7.2**
 
## Running the examples
 These examples are composed of four python scripts. You should run them in the following order.
 
 1. #### Script: _generate_circles_dataset.py_
 	This script will generate the training dataset described in the paper, which is created using circles of different radii. The dataset will be printed to a csv file in the current folder.
 2. #### Script: _training.py_
 	This script will read the dataset file that was generated on the previous step and then perform the network training. This may take a few hours and, at the end, a new folder and some files will be created to store the trained network weights.
 3. #### Script: _testing_circles.py_
 	This script will read the trained network weights printed on step 2 and test the network on the dataset from step 1. The results will be printed to a PNG image on the same folder.
 4. #### Script: _testing_sinWave.py_
 	This script will initially create a new dataset using a Front-Tracking interface with the shape of a sine wave (as described in the paper). The trained network weights from step 2 will be read and the network is tested on this new dataset. The results will be printed to a PNG image on the same folder.
