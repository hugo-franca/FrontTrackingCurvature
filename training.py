import sys;
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import os

from NeuralNet import *;

# Spacing between particles. In the paper, we test h=0.0004 and h=0.0001
h = 0.0004

# Defines the number of inputs for the network. In the paper, we test N=3, N=5 and N=7
N = 5

# Reading the training dataset from the csv file.
# This needs to have been previously created by running the script generate_circles_dataset.py using the same parameters as above
dataset = pd.read_csv('dataset_circles_N%d-h%g.csv' % (N, h), na_values = "?", sep=";", skipinitialspace=True);

# Separating 80% for training and 20% for test
dataset_train = dataset.sample(frac=0.8, random_state=0);
dataset_test = dataset.drop(dataset_train.index);

# Separating the targets into a different variable
targets_train = dataset_train.pop('curv');
targets_test = dataset_test.pop('curv');

# Defining the network sctructure (number of layers, nodes, etc)
# Note: the class "Layer" is implemented in the file NeuralNet.py
numberInputs = len(dataset_train.keys());
layers = [];
layers.append( Layer(numberNodes=200, activation='relu', numberInputs=numberInputs) );
layers.append( Layer(numberNodes=200, activation='relu') );
layers.append( Layer(numberNodes=200, activation='relu') );
layers.append( Layer(numberNodes=1) );


# Creating a folder where the output files will be created with the trained network info
folder_out = 'model_network_N%d_h%g' % (N, h)
if not os.path.exists(folder_out):
	os.mkdir(folder_out)
folder_out = folder_out + '/model'

# Creating the network object and printing some info about it
# Note: the "NeuralNet" class is defined in the file NeuralNet.py
net = NeuralNet(list_layers=layers)
net.model.summary()

# Training the network. The trained network will be written in files at the folder specified above
# These files will by read by other scripts for tests
net.train(dataset_train, targets_train, 3000, folder_out=folder_out)

