import helpers

class NN_Input:
	def __init__(self, values=[]):
		# Input values
		self.values = values

class NN_Output:
	def __init__(self, values=[]):
		# Output values
		self.values = values

	def add_output(self, newOutput):
		self.values.append(newOutput)

class Input_Value:
	def __init__(self):
		# Value of this input, named 'y' so it can be easily used in other function which also handle neurons 
		self.y = 0

class Synapse:
	def __init__(self, s_in=0, s_out=0, s_weight=0):
		# Input neuron
		self.input = s_in
		# Output neuron
		self.output = s_out
		# Weight
		self.weight = s_weight

	def update_weight(self, eta):
		d_n = self.output.delta
		y_n = self.output.y
		y_k = self.input.y
		delta_w = -eta * d_n * (1-y_n) * y_n * y_k
		self.weight -= delta_w


class Neuron:
	def __init__(self, n_in=[], n_out=[], n_S=0, n_delta=0, n_y=0):
		# Input synapses
		self.inputs = n_in
		# Output synapses
		self.outputs = n_out
		# Weighted sum
		self.S = n_S
		# Delta parameter
		self.delta = n_delta
		# Output value
		self.y = n_y

	def add_input(self, newInput):
		self.inputs.append(newInput)
	
	def add_output(self, newOutput):
		self.outputs.append(newOutput)

	def calculate_sum(self):
		tmpSum = 0
		for i in range(len(self.inputs)):
			y_k = self.inputs[i].input.y
			w = self.inputs[i].weight
			tmpSum += y_k * w
		self.S = tmpSum

	def update_y(self, choice):
		self.y = helpers.sigmoid_function(self.S, choice)

	def calculate_delta(self):
		tmpSum = 0
		for i in range(len(self.outputs)):
			d_n = self.outputs[i].output.delta
			y_n = self.outputs[i].output.y
			w = self.outputs[i].weight
			tmpSum += d_n * w * (1-y_n) * y_n
		self.delta = tmpSum


class Layer:
	def __init__(self, nr_neurons=0, neurons=[]):
		# Neurons
		self.neurons = neurons
		for i in range(nr_neurons):
			self.add_neuron(Neuron(n_in=[], n_out=[]))

	def add_neuron(self, newNeuron):
		self.neurons.append(newNeuron)

	def calculate_output_values(self, choice):
		# Calculate the output values y for each neuron in the layer
		for n in range(len(self.neurons)):
			self.neurons[n].calculate_sum()
			self.neurons[n].update_y(choice)

	def update_weights(self, eta):
		# Update weights for all input synapses of the neurons in the layer
		for n in range(len(self.neurons)):
			for s in range(len(self.neurons[n].inputs)):
				self.neurons[n].inputs[s].update_weight(eta)

	def calculate_deltas(self):
		# Calculate delta value for each neuron in the layer
		for n in range(len(self.neurons)):
			self.neurons[n].calculate_delta()


class Network:
	def __init__(self, layers=[]):
		# Create the desired amount of layers
		self.layers = layers
		# for i in range(nr_layers):
		# 	self.add_layer(Layer())

	def add_layer(self, newLayer):
		self.layers.append(newLayer)

	def input_propagation(self, choice):
		for l in range(len(self.layers)):
			self.layers[l].calculate_output_values(choice)

	def error_propagation(self, expected, eta):
		# The neurons in the output layer have their deltas calculated differently
		for n in range(len(self.layers[-1].neurons)):
			self.layers[-1].neurons[n].delta = expected[n] - self.layers[-1].neurons[n].y
		# Update weights going into output layer
		self.layers[-1].update_weights(eta)
		# Error propagation for the rest of the network
		for l in range(len(self.layers)-2, -1, -1):
			self.layers[l].calculate_deltas()
			self.layers[l].update_weights(eta)

	def get_input_list(self):
		# Getting a list of the network's input objects
		InputList = []
		for i in range(len(self.layers[0].neurons[0].inputs)):
			InputList.append(self.layers[0].neurons[0].inputs[i].input)
		return InputList

	def debug(self):
		for l in range(len(self.layers)):
			print("Layer {0}".format(l))
			for n in range(len(self.layers[l].neurons)):
				neuron = self.layers[l].neurons[n]
				print("Neuron {0}: S={1}, y={2}, delta={3}".format(n, neuron.S, neuron.y, neuron.delta))
				for s in range(len(neuron.inputs)):
					print("Synpase {0}: Weight={1}".format(s, neuron.inputs[s].weight))