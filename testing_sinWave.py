import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys

from NeuralNet import *

# Bissection algorithm. Will be used later on to generate points with uniform distribution on a sine wave
def bissection(function, a, b):

	parar = 1e-10
	while( abs(a-b)>parar ):
		fa = function(a)
		fb = function(b)

		if( fa*fb > 0 ):
			return float('nan')
		elif( fa==0 ):
			return a
		elif( fb==0 ):
			return b
		else:
			meio = 0.5*(a+b)
			if( function(meio)*fa < 0 ):
				b = meio
			else:
				a = meio

	return 0.5*(a + b)

# This method below creates the Front-Tracking particles over a sine wave
# The particles will have uniform spacing
# This method can actually be used for any function "f", not just a sine wave
def InitializeParticlesSinWave(xMin, xMax, f, f_derivative, numberSegments):
   
	points = np.zeros([numberSegments+1, 2]);

	# Calculating the total length of this function's curve
	f_length = lambda x:np.sqrt( 1 + f_derivative(x)**2 )
	totalLength_curve = scipy.integrate.quad(f_length, xMin, xMax)

	# Calculating the spacing between particles (totalLength/numberSegments)
	h = totalLength_curve[0]/numberSegments

	# The first particle has x=xMin, then we loop finding the other points until x=xMax
	x1 = xMin;
	i = 0;
	while( x1<xMax ):
		y1 = f(x1)
		points[i, 0] = x1
		points[i, 1] = y1

		# We find x2 such that the distance between || x2 - x1 ||^2 == h^2
		f_dist = lambda x:( (x-x1)**2 + (f(x) - y1)**2 - h**2 )
		x2 = bissection(f_dist, x1, x1 + h)

		x1 = x2
		i = i + 1

	points[i, 0] = xMax
	points[i, 1] = f(xMax)

	# flipping so its counter-clockwise
	points = np.flip(points, 0)

	return [points, h];

# Given a front-tracking interface and a given points p_i: this function will
# return the 4N inputs that will go into the neural network for this points p_i
# Remember that the inputs are the slope of tangential and normal vectors to each line segment around p_i
# There are N segmens to one side of p_i (each has a tangential and normal vector, so 2N inputs)
# There are N segments to the other side of p_i (more 2N inputs)
# So... 4N inputs in total will be returned by this function
def DetermineNetworkInputs(points, index_point, N, invert):
	slopes = [];

	# Initial index will be N segments behind the center index_point (p_i)
	index = index_point - N;

	# Then we loop over 2N segments.
	# For each segment, we add the tangential and normal angle slope to the list
	for i in range( 2*N ):

		# The two consecutive points that form this segment. 
		# Theres a mod operator here, just to handle the case when the index is at the end of the points list
		index1 = index % len(points);
		index2 = (index + 1) % len(points);
		x1 = points[index1][0];
		y1 = points[index1][1];
		x2 = points[index2][0];
		y2 = points[index2][1];
		index += 1;

		# v is this segment tangential vector and n is the normal
		vx = x2 - x1
		vy = y2 - y1
		nx = vy
		ny = -vx
		if invert:
			nx = -nx
			ny = -ny

		# Calculate the normal and tangential slope angle for this segment and add them to the list
		slope_normal = VectorAngle(nx, ny);
		slope_tangential = VectorAngle(vx, vy);
		slopes.append(slope_normal);
		slopes.append(slope_tangential);

	return slopes;

# Returns the slope angle for a given vector
def VectorAngle(vx, vy):
	norm = np.sqrt(vx*vx + vy*vy);

	if( vx>=0 and vy>=0 ): # First quadrant
		return np.arcsin(vy/norm);
	elif( vx<0 and vy>=0 ):	# Second quadrant
		return np.pi - np.arcsin(vy/norm);
	elif( vx>=0 and vy<0 ): # Third quadrant
		return np.arcsin(vy/norm);
	elif( vx<0 and vy<0 ): # Fourth quadrant
		return - np.pi - np.arcsin(vy/norm);
	else:
		print('Unexpected case in the function VectorAngle. Something weird happened.\n');
		return 1e+10; #Algo esquisito

# This function will create the dataset and print it to a csv file
def CreateDataset(numberSegments, N, csvFileName):

	# Opening the csv file and printing the header
	file = open(csvFileName, 'w');
	file.write('x;');
	for i in range(4*N):
		file.write('column%s;' % (i+1));
	file.write('curv\n');

	# Domain boundaries
	xMin = -np.pi;
	xMax = np.pi;

	# Sin wave and its derivative
	f = lambda x:np.sin(x)
	f_derivative = lambda x:np.cos(x)

	# Creating the Front-Tracking particles
	[points, h] = InitializeParticlesSinWave(xMin, xMax, f, f_derivative, numberSegments)

	# Looping over each particle and printing one entry in the dataset file
	for p in range(len(points)):

		# Ignoring the edge points
		if( p<=N or (p+N)>=len(points) ):
			continue;

		# Calculating the inputs for this point
		slopes = DetermineNetworkInputs(points, p, N, False);

		# We will also print the exact curvature in the file, just for comparison later on
		x = points[p, 0];
		curvature = np.sin(x)*( (1.0 + np.cos(x)*np.cos(x))**(-3.0/2.0) );
		curvature = h*curvature;

		# Printing everything to the file
		file.write('%s;' % (x));
		for i in slopes:
			file.write('%s;' % i);
		file.write('%s\n' % (curvature));


	file.close();

	return h


# Spacing used previously when training the model. This is NOT the spacing for the sine wave particle
# This variable will literally only be used to find the folder where the trained model is located
# Remember that the folder has this number in its name
h_model = 0.0004

# Defines the number of inputs for the network. In the paper, we test N=3, N=5 and N=7
N = 5

# Number of line segments to discretize the sine wave
numberSegments_sineWave = 200

# Output file to print the sine wave dataset
file_dataset = 'dataset_sinWave_N%s.csv' % (N)

# Creating the dataset and printing to a file
h_sin = CreateDataset(numberSegments_sineWave, N, file_dataset)

# Reading the dataset using the pandas format
dataset = pd.read_csv(file_dataset, na_values = "?", sep=";", skipinitialspace=True)
vector_x = np.array( dataset.pop('x') )
vector_curv = np.array( dataset.pop('curv') )

# Defining the network sctructure (number of layers, nodes, etc)
# Note: the class "Layer" is implemented in the file NeuralNet.py
numberInputs = len(dataset.keys());
layers = [];
layers.append( Layer(numberNodes=200, activation='relu', numberInputs=numberInputs) );
layers.append( Layer(numberNodes=200, activation='relu') );
layers.append( Layer(numberNodes=200, activation='relu') );
layers.append( Layer(numberNodes=1) );

# Creating the network object and loading the trained weights from a file
# This file was previously  created by the script "training.py"
net = NeuralNet(list_layers=layers, model_file='model_network_N%d_h%g/model' % (N, h_model));
net.model.summary()

# Using the network to predict the curvature
predictions = np.array( net.test(dataset) )

# Bringing results back to original dimension
vector_curv /= h_sin
predictions /= h_sin

# Initializing plots
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = "15"
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

# First plot: curvature versus x-coordinate
ax1.set_xlabel('x');
ax1.set_ylabel('Curvature');
ax1.plot(vector_x, predictions, 'o', fillstyle='none')
ax1.plot(vector_x, vector_curv)
ax1.legend(['Machine learning', 'Analytical'])

# Second plot: scatter plot exact curvature versus predicted curvature
ax2.set_xlabel('True Values');
ax2.set_ylabel('Predictions');
ax2.plot(vector_curv, predictions, 'o', fillstyle='none')
ax2.plot([-1.05, 1.05], [-1.05, 1.05])

# Saving the plot to a file
plt.savefig('test_sinWave_N%d.png' % (N))
print('\n\nFinished succesfully. A png with the results has been created.\n')