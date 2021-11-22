import sys;
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

from NeuralNet import *;

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "15"


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

# Creating the network object and loading the trained weights from a file
# This file waspreviously  created by the script "training.py"
net = NeuralNet(list_layers=layers, model_file='model_network_N%d_h%g/model' % (N, h));
net.model.summary()

# Creating the first plot: using the training data (80%)
predictions = net.test(dataset_train);
plot_lim = [-500, 500]
markersize = 3
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
ax1.set_title('Train data')
ax1.set_xlabel('True Values');
ax1.set_ylabel('Predictions');
ax1.axis('scaled')
ax1.set_xlim(plot_lim)
ax1.set_ylim(plot_lim)
curv_target = np.array(targets_train.to_list())/h
curv_prediction = np.array(predictions)/h
ax1.plot(curv_target, curv_prediction, 'o', markersize=markersize)
ax1.plot(plot_lim, plot_lim)


# Creating the second plot: using the test data (20%)
predictions = net.test(dataset_test);
ax2.set_title('Test data')
ax2.set_xlabel('True Values');
ax2.set_ylabel('Predictions');
ax2.axis('scaled')
ax2.set_xlim(plot_lim)
ax2.set_ylim(plot_lim)
curv_target = np.array(targets_test.to_list())/h
curv_prediction = np.array(predictions)/h
ax2.plot(curv_target, curv_prediction, 'o', markersize=markersize);
ax2.plot(plot_lim, plot_lim)

# Printing the image to a png file
plt.savefig('test_circles-N%d_h%g.png' % (N, h));
