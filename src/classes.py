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


class Synapse:
	def __init__(self, s_in=0, s_out=0, s_weight=0):
		# Input neuron
		self.input = s_in
		# Output neuron
		self.output = s_out
		# Weight
		self.weight = s_weight


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


class Layer:
	def __init__(self, nr_neurons=0, neurons=[]):
		# Neurons
		self.neurons = neurons
		for i in range(nr_neurons):
			print("Adding neuron {0}".format(i))
			self.add_neuron(Neuron(n_in=[], n_out=[]))

	def add_neuron(self, newNeuron):
		self.neurons.append(newNeuron)


class Network:
	def __init__(self, layers=[]):
		# Create the desired amount of layers
		self.layers = layers
		# for i in range(nr_layers):
		# 	self.add_layer(Layer())

	def add_layer(self, newLayer):
		self.layers.append(newLayer)