import fileHandling
import xlsParser
import helpers
import classes
import random
import numpy as np

def initialize_network(data_filename, weight_range_l, weight_range_h, sig_choice, eta, training_loops):

	# Read data
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
		#print("Layer {0}".format(i))
		tmp_number = int(input("-> Enter desired number of neurons in layer {0}: ".format(i)))
		NeuralNetwork.add_layer(classes.Layer(nr_neurons=tmp_number, neurons=[]))

	# The output layer (as many neurons as there are possible outputs)
	NeuralNetwork.add_layer(classes.Layer(nr_neurons=nr_categories, neurons=[]))	

	# Add connections into the network
	InputList = []
	for i in range(trInput.values.shape[1]):
		InputList.append(classes.Input_Value())
	for i in range(len(NeuralNetwork.layers[0].neurons)):	# Amount of neurons in the layer
		for j in range(trInput.values.shape[1]):	# Amount of inputs going into the network
			NeuralNetwork.layers[0].neurons[i].add_input(classes.Synapse(s_out=NeuralNetwork.layers[0].neurons[i], s_weight=random.uniform(weight_range_l, weight_range_h)))
			NeuralNetwork.layers[0].neurons[i].inputs[j].input = InputList[j]

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
	for m in range(training_loops):
		for n in range(trInput.values.shape[0]): # For each line of input data
			# Input propagation phase
			for i in range(trInput.values.shape[1]): 	# Number of inputs
				InputList[i].y = trInput.values[n, i]			# Set input values
			NeuralNetwork.input_propagation(choice=1)

			# Error propagation phase
			NeuralNetwork.error_propagation(trExpected.values[n, :], eta)

			# Printing results for phase <n>
			nnIn = []
			nnExp = []
			nnOut = []
			for i in range(len(NeuralNetwork.layers[nr_layers].neurons)):
				nnExp.append(trExpected.values[n, i])
				nnOut.append(NeuralNetwork.layers[nr_layers].neurons[i].y)
			# Printing results for loop <n>
			nnIn.append(trInput.values[n])
			#nnIn = ["%.2f" % float(elem) for elem in nnIn]
			nnOut = ["%.2f" % float(elem) for elem in nnOut]
			# print("Training phase {0}: Input {1}, Expected {2}, Output {3}".format(n, nnIn, nnExp, nnOut))
			trOutput.values.append(nnOut)	# Adding results to output table

			# NeuralNetwork.debug() # Print out all values in the network (i.e. synapse weights)

	return NeuralNetwork


def run_network(data_filename, NeuralNetwork, sig_choice):

	# Read data
	[tmpIn, tmpExp, categories] = xlsParser.read_xls(data_filename)
	nr_categories = len(categories)
	netInput = classes.NN_Input(values=np.array(tmpIn))
	netExpected = classes.NN_Output(values=np.array(tmpExp))
	netOutput = classes.NN_Output(values=[])

	# Getting a list of the network's input objects
	InputList = NeuralNetwork.get_input_list()

	# Pushing the data through the network
	for n in range(netInput.values.shape[0]): # For each line of input data
		for i in range(netInput.values.shape[1]): 	# Number of inputs
			InputList[i].y = netInput.values[n, i]			# Set input values
		NeuralNetwork.input_propagation(choice=1)

		# Printing results for phase <n>
		nnIn = []
		nnExp = []
		nnOut = []
		for i in range(len(NeuralNetwork.layers[-1].neurons)):
			nnExp.append(netExpected.values[n, i])
			nnOut.append(NeuralNetwork.layers[-1].neurons[i].y)
		# Printing results for loop <n>
		nnIn.append(netInput.values[n])
		#nnIn = ["%.2f" % float(elem) for elem in nnIn]
		nnOut = ["%.2f" % float(elem) for elem in nnOut]
		#print("Results for row {0}: Input {1}, Expected {2}, Output {3}".format(n, nnIn, nnExp, nnOut))
		netOutput.values.append(nnOut)	# Adding results to output table
	netOutput.values = np.array(netOutput.values)

	number_correct = 0
	for n in range(netExpected.values.shape[0]):
		if np.argmax(netOutput.values[n, :]) == np.argmax(netExpected.values[n, :]):
			number_correct += 1
	percent_correct = number_correct / netExpected.values.shape[0] * 100
	print("Percentage of correct classification: {0}%".format(percent_correct))