import math
import numpy

# Call the chosen sigmoidal function
def sigmoid_function(x, choice):
	switcher = {
		1: sig1,
		2: sig2,
		3: sig3
	}
	func = switcher.get(choice, lambda:"Invalid choice")
	return func(x)

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


# Calculate the weighted sum
def weighted_sum(values, weights):
	v = numpy.array(values)
	w = numpy.array(weights)
	return sum(numpy.multiply(v, w))


# Normalize data values
def normalize_data(data):
	d = numpy.array(data)
	return (d - min(data))/(max(data) - min(data))

		

# # Divide data into input and expected result lists
# def parse_data(data, nr_inputs, nr_outputs):
# 	inputData = []
# 	expectedResult = []
# 	for line in data:
# 		inputData.append(line[:nr_inputs])
# 		expectedResult.append(line[-nr_outputs:])

# 	return [inputData, expectedResult]

