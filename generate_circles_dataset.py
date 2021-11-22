import numpy as np;
import matplotlib.pyplot as plt;
import time;
import sys;

# Initialize the marker particles that describe a Front-Tracking ellipse
def InitializeParticlesEllipse(xCenter, yCenter, radiusX, radiusY, angle, numberSegments):
	deltaAngle = 2*(np.pi)/numberSegments;
   
	points = np.zeros([numberSegments, 2]);
	
	for i in range(numberSegments):
		points[i, 0] = xCenter + radiusX*np.cos(angle);
		points[i, 1] = yCenter + radiusY*np.sin(angle);
		angle += deltaAngle;

	return points;

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


# Spacing between particles. In the paper, we test h=0.0004 and h=0.0001
h = 0.0004

# Defines the number of inputs for the network. In the paper, we test N=3, N=5 and N=7
N = 5

# The range for circle radiuses to be used in the complete dataset
# Also the total number of circles to be used in the complete dataset
initial_radius = 0.00225
final_radius = 0.475
numberCircles = 65

# We generate a list of 65 radiuses using a exponential distribution (for more circles with small radius)
t = np.linspace( 0.0, 1.0, numberCircles )
f = lambda t: initial_radius*np.exp( np.log(final_radius/initial_radius)*t ) 
list_radiuses = f(t)

# Opening a CSV file where we will write our complete dataset. Also writing the file header
# The file has 4N + 1 columns (the +1 is the exact curvature for each entry)
file = open('dataset_circles_N%d-h%g.csv' % (N, h), 'w');
for i in range(4*N):
	file.write('column%s;' % (i+1));
file.write('curv\n');


# Looping over each circle radius...
radius_index = 1;
for radius in list_radiuses:

	# Calculate how many line segments this circle will have
	perimeter = 2*np.pi*radius;
	numberSegments = int( np.floor( perimeter/h ) );

	# Creates the list of front-tracking marker points for this circle
	# Also a secondary list in reverse order
	points = InitializeParticlesEllipse(0.0, 0.0, radius, radius, 0.0, numberSegments);
	points_reverse = np.flip(points, 0)

	# The exact curvature for this circle
	curvature = h*(1.0/radius);

	# If the number of points is larger than 1000, i randomly choose 1000 of them to enter the dataset
	# This avoids the dataset getting enormously large (we tested using every point as well, doesnt improve much)
	selected_points = np.arange(0, numberSegments, 1)
	select_quantity = 1000
	if( numberSegments>select_quantity ):
		selected_points = np.random.choice(selected_points, select_quantity, replace=False)
	else:
		select_quantity = numberSegments

	# Loop over each point p_i and add an entry to the dataset
	for p in selected_points:

		# Slopes for the counterclockwise front-tracking markers list. Normals points outside of circle.
		slopes_out= DetermineNetworkInputs(points, p, N, False);

		# Slopes for the counterclockwise front-tracking markers list. Normals points inside of circle.
		slopes_in = DetermineNetworkInputs(points, p, N, True);

		# Slopes for the clockwise front-tracking markers list. Normals points outside of circle.
		slopes_clockwise_out = DetermineNetworkInputs(points_reverse, p, N, True);

		# Slopes for the clockwise front-tracking markers list. Normals points inside of circle.
		slopes_clockwise_in = DetermineNetworkInputs(points_reverse, p, N, False);

		# Printing everything in the csv file
		for i in slopes_out:
			file.write('%s;' % i);
		file.write('%s\n' % (curvature));

		for i in slopes_in:
			file.write('%s;' % i);
		file.write('%s\n' % (-curvature));

		for i in slopes_clockwise_out:
			file.write('%s;' % i);
		file.write('%s\n' % (curvature));

		for i in slopes_clockwise_in:
			file.write('%s;' % i);
		file.write('%s\n' % (-curvature));


	print('Completed circle %s/%s - %d segments\n' % (radius_index, list_radiuses.shape[0], select_quantity));
	sys.stdout.flush();
	radius_index +=  1;


file.close();