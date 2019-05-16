import fileHandling
import xlsParser
import helpers
import classes
import random
import numpy as np

def initialize_network(data_filename):
	weight_range_l = -0.1
	weight_range_h = 0.1
	sig_choice = 1
	eta = 0.1

	#[tmpIn, tmpExp] = fileHandling.readData(data_filename)
	#categories = np.unique(tmpExp)
	[tmpIn, tmpExp, categories] = xlsParser.read_xls(data_filename)
	nr_categories = len(categories)
	trInput = classes.NN_Input(values=np.array(tmpIn))
	trExpected = classes.NN_Output(values=np.array(tmpExp))
	trOutput = classes.NN_Output(values=[])

	# Create the network (list of layers which have a list of neurons)
	NeuralNetwork = classes.Network(layers=[])
	nr_layers = int(input("-> Enter desired number of layers: "))
	for i in range(nr_layers):
		print("Layer {0}".format(i))
		tmp_number = int(input("-> Enter desired number of neurons in layer {0}: ".format(i)))
		# print("Desired number of neurons = {0}".format(tmp_number))
		NeuralNetwork.add_layer(classes.Layer(nr_neurons=tmp_number, neurons=[]))

	# The output layer (as many neurons as there are possible outputs)
	NeuralNetwork.add_layer(classes.Layer(nr_neurons=nr_categories, neurons=[]))	

	# Add connections into the network
	for i in range(len(NeuralNetwork.layers[0].neurons)):	# Amount of neurons in the layer
		for j in range(trInput.values.shape[1]):	# Amount of inputs going into the network
			NeuralNetwork.layers[0].neurons[i].add_input(classes.Synapse(s_out=NeuralNetwork.layers[0].neurons[i], s_weight=random.uniform(weight_range_l, weight_range_h)))

	# Add connections out of the network
	for i in range(len(NeuralNetwork.layers[nr_layers-1].neurons)):		# Amount of neurons in the layer
		for j in range(nr_categories):	# Amount of output categories to classify into
			# Neuron <i> of the last NN layer has an output synapse <j> whose input is that neuron and output is neuron <j> of the output layer
			NeuralNetwork.layers[nr_layers-1].neurons[i].add_output(classes.Synapse(s_in=NeuralNetwork.layers[nr_layers-1].neurons[i], s_out=NeuralNetwork.layers[nr_layers].neurons[j], s_weight=random.uniform(weight_range_l, weight_range_h)))
			# Neuron <j> of the output layer has an input synapse which is the output synpase <j> of the neuron <i> in the last NN layer
			NeuralNetwork.layers[nr_layers].neurons[j].add_input(NeuralNetwork.layers[nr_layers-1].neurons[i].outputs[j])

	# Add connections within the network
	for l in range(nr_layers-1):
		for i in range(len(NeuralNetwork.layers[l].neurons)):
			for j in range(len(NeuralNetwork.layers[l+1].neurons)):
				NeuralNetwork.layers[l].neurons[i].add_output(classes.Synapse(s_in=NeuralNetwork.layers[l].neurons[i], s_out=NeuralNetwork.layers[l+1].neurons[j], s_weight=random.uniform(weight_range_l, weight_range_h)))
				NeuralNetwork.layers[l+1].neurons[j].add_input(NeuralNetwork.layers[l].neurons[i].outputs[j])

	# Training phase
	for n in range(trInput.values.shape[0]): # For each line of input data
		# Input propagation phase
		# Layer 0 (Possibly simplified by making additional inputs neurons with just a y value)
		for i in range(len(NeuralNetwork.layers[0].neurons)):
			# Calculate the weighted sum for neuron <i>
			NeuralNetwork.layers[0].neurons[i].S = 0
			for j in range(len(NeuralNetwork.layers[0].neurons[i].inputs)):
				NeuralNetwork.layers[0].neurons[i].S += trInput.values[n, j] * NeuralNetwork.layers[0].neurons[i].inputs[j].weight
			NeuralNetwork.layers[0].neurons[i].y = helpers.sigmoid_function(NeuralNetwork.layers[0].neurons[i].S, choice=sig_choice)
		# Following layers
		for l in range(nr_layers):
			for i in range(len(NeuralNetwork.layers[l+1].neurons)):	# For hidden layer <l+1>
				# Calculate the weighted sum for neuron <i>
				NeuralNetwork.layers[l+1].neurons[i].S = 0
				for j in range(len(NeuralNetwork.layers[l+1].neurons[i].inputs)):
					NeuralNetwork.layers[l+1].neurons[i].S += NeuralNetwork.layers[l].neurons[j].y * NeuralNetwork.layers[l+1].neurons[i].inputs[j].weight
				NeuralNetwork.layers[l+1].neurons[i].y = helpers.sigmoid_function(NeuralNetwork.layers[l+1].neurons[i].S, choice=sig_choice)

		# Error propagation phase
		# Output layer
		nnIn = []
		nnExp = []
		nnOut = []
		for i in range(len(NeuralNetwork.layers[nr_layers].neurons)):
			# Calculate the error for neuron <i>
			NeuralNetwork.layers[nr_layers].neurons[i].delta = trExpected.values[n, i] - NeuralNetwork.layers[nr_layers].neurons[i].y
			nnExp.append(trExpected.values[n, i])
			nnOut.append(NeuralNetwork.layers[nr_layers].neurons[i].y)
		# Printing results for loop <n>
		nnIn.append(trInput.values[n])
		#nnIn = ["%.2f" % float(elem) for elem in nnIn]
		#nnOut = ["%.2f" % float(elem) for elem in nnOut]
		print("Training phase {0}: Input {1}, Expected {2}, Output {3}".format(n, nnIn, nnExp, nnOut))
		trOutput.values.append(nnOut)	# Adding results to output table
		# Lower layers
		for l in range(nr_layers-1, -1, -1):
			for i in range(len(NeuralNetwork.layers[l].neurons)):
				# Calculate the error for neuron <i>
				NeuralNetwork.layers[l].neurons[i].delta = 0
				for j in range(len(NeuralNetwork.layers[l].neurons[i].outputs)):	# For each of neuron <i> synapses <j>
					delta_n = NeuralNetwork.layers[l+1].neurons[j].delta
					y_n = NeuralNetwork.layers[l+1].neurons[j].y
					y_k = NeuralNetwork.layers[l].neurons[i].y
					w_correction = -eta * delta_n * (1-y_n) * y_n * y_k						# Calculating correction of the synapse weight
					NeuralNetwork.layers[l].neurons[i].outputs[j].weight += w_correction	# Applying it
					NeuralNetwork.layers[l].neurons[i].delta += delta_n * NeuralNetwork.layers[l].neurons[i].outputs[j].weight * (1-y_n) * y_n

	return NeuralNetwork