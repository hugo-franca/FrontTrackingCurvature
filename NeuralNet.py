import sys;
import numpy as np;
import pandas as pd;
from tensorflow.keras.models import Sequential;
from tensorflow.keras.layers import Dense;
from tensorflow.keras import optimizers;
from tensorflow.keras.callbacks import Callback;
from tensorflow.keras.models import load_model;
from tensorflow.keras.optimizers import SGD
import tensorflow as tf;

# This class represents (in a simplified way) one layer of our neural network
class Layer:

	# Number of nodes for this layer, activation function (linear by default)
	# Optionally, if it's the first layer also the number of inputs
	def __init__(self, numberNodes, activation=None, numberInputs=None):
		self.numberNodes = numberNodes;
		self.activation = activation;
		self.numberInputs = numberInputs;

# This class represents the neural network
class NeuralNet:

	# Initialize a new untrained network with a list of layers
	# and, optionally, also read the already-trained weights from a file
	def __init__(self, model_file=None, list_layers=None):

		# Creating a sequential Keras model
		self.model = Sequential();

		# Adding the Keras layers to this model
		for layer in list_layers:
			if( layer.numberInputs ):
				self.model.add( Dense(layer.numberNodes, activation=layer.activation, input_shape=[layer.numberInputs]) );
				self.numberInputs = layer.numberInputs
			else:
				self.model.add( Dense(layer.numberNodes, activation=layer.activation) );

		# Optimizer to use in the training later on
		optimizer = SGD(learning_rate=0.1);

		# Calling the Keras function to create the network model internally (still needs to be trained)
		self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error']);

		# If a file was given by the user, we read the model weights from this file
		if( model_file ):
			self.model.load_weights(model_file + '.chk');
			self.stats_norm = pd.read_pickle(model_file + '.pkl');
			return;

	# Function to normalize the dataset before training or testing
	def norm(self, x):
		train_stat = self.stats_norm;
		return (x - train_stat['mean']) / train_stat['std'];

	# Function that performs the network training
	# If a folder is given, the trained weights will be written to a file in this folder
	def train(self, dataset_train, targets_train, epochs, folder_out=None):

		# Normalizing the training dataset
		self.stats_norm = dataset_train.describe();
		self.stats_norm = self.stats_norm.transpose();
		normed_train_data = self.norm(dataset_train);

		# If a folder was given, create Keras callback that prints the model to a file
		# Otherwise, just use a simple callback that prints some things to the terminal (PrintDot is defined at the end of this script)
		if( folder_out ):
			self.stats_norm.to_pickle(folder_out + '.pkl');
			cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=(folder_out + '.chk'), save_weights_only=True, verbose=1)
		else:
			cp_callback = PrintDot()
		
		# Calling the keras function that actually performs the training
		# This instruction may take hours to run
		self.model.fit(
		    normed_train_data, targets_train,
		    batch_size = 512,
		    epochs=epochs, validation_split = 10.0/80.0, verbose=0,
		    callbacks=[cp_callback, PrintDot()]);
	   
	# Function that evaluates this network for a given set of input
	# Use this when the network has already been trained and you just want to evaluate it
	def test(self, test_data):

		# Normalizing the data
		normed_test_data = self.norm(test_data);

		# Calling the Keras function to evaluate the model
		return self.model.predict(normed_test_data).flatten();

# Callback to print some training info to the terminal every epoch
class PrintDot(Callback):
    def on_epoch_end(self, epoch, logs):
        if( epoch%1==0 ):
            print('Epoch: %s' % (epoch));
            print(logs);
            sys.stdout.flush();