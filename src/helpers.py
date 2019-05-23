import math
import numpy

# Call the chosen sigmoidal function
def sigmoid_function(x, choice):
	switcher = {
		0: sig0,
		1: sig1,
		2: sig2,
		3: sig3
	}
	func = switcher.get(choice, lambda:"Invalid choice")
	return func(x)

# Function which does nothing with input, mainly for simplicity in autoencoder creation
def sig0(x):
	return abs(x)

# Logistic function with output in the domain: (0; 1)
def sig1(x):
	try:
		ret = 1/(1+math.exp(-x))
	except OverflowError:
		if x < 0:
			ret = 0
		else:
			ret = 1
	return ret

# Logistic function with output in the domain: (-1; 1)
def sig2(x):
	try:
		ret =  (2/(1+math.exp(-x)))-1
	except OverflowError:
		if x < 0:
			ret = -1
		else:
			ret = 1
	return ret

# Hyperbolic tangent with output in the domain: (-1; 1)
def sig3(x):
	try:
		ret =  math.tanh(x)
	except OverflowError:
		if x < 0:
			ret = -1
		else:
			ret = 1
	return ret


# # Calculate the weighted sum
# def weighted_sum(values, weights):
# 	v = numpy.array(values)
# 	w = numpy.array(weights)
# 	return sum(numpy.multiply(v, w))


# Normalize 1d data
def normalize_1d(data):
	d = numpy.array(data)
	return (d - min(data))/(max(data) - min(data))


# Normalize 2d data array by column
def normalize_2d(data):
	d = numpy.array(data)
	for i in range(d.shape[1]):
		d[:, i] = normalize_1d(d[:, i])
	return d
	
