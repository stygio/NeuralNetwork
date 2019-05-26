import fileHandling
import xlsParser
import helpers
import classes
import random
import numpy as np


def initialize_mlp(data_filename, weight_range_l, weight_range_h, sig_choice, eta, training_loops):

	# Read data
	#[tmpIn, tmpExp] = fileHandling.readData(data_filename)
	#categories = np.unique(tmpExp)
	[tmpIn, tmpExp, categories] = xlsParser.read_xls(data_filename)
	nr_categories = len(categories)
	trInput = classes.NN_Input(values=np.array(helpers.normalize_2d(tmpIn)))
	trExpected = classes.NN_Output(values=np.array(tmpExp))
	trOutput = classes.NN_Output(values=[])

	# Create the network (list of layers which have a list of neurons)
	NeuralNetwork = classes.Network(layers=[])
	nr_layers = int(input("-> Enter desired number of layers: "))
	for i in range(nr_layers):
		tmp_number = int(input("Enter desired number of neurons in layer {0}: ".format(i)))
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
			NeuralNetwork.input_propagation(choice=sig_choice)

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



def create_autoencoder(data, weight_range_l, weight_range_h, sig_choice, eta, autoencoder_loops):

	dataArray = classes.NN_Input(values=np.array(data))
	dataArray.values = helpers.normalize_2d(dataArray.values)
	data_length = dataArray.values.shape[1]

	# Create the network (list of layers which have a list of neurons)
	AutoEncoder = classes.Network(layers=[])
	nr_layers = int(input("-> AutoEncoder - Enter desired number of layers: "))
	for i in range(nr_layers):
		tmp_number = int(input("Enter desired number of neurons in layer {0}: ".format(i)))
		AutoEncoder.add_layer(classes.Layer(nr_neurons=tmp_number, neurons=[]))

	# The output layer (as many neurons as there are possible outputs)
	AutoEncoder.add_layer(classes.Layer(nr_neurons=data_length, neurons=[]))	

	# Add connections into the network
	InputList = []
	for i in range(data_length):
		InputList.append(classes.Input_Value())
	for i in range(len(AutoEncoder.layers[0].neurons)):	# Amount of neurons in the layer
		for j in range(data_length):	# Amount of inputs
			AutoEncoder.layers[0].neurons[i].add_input(classes.Synapse(s_out=AutoEncoder.layers[0].neurons[i], s_weight=random.uniform(weight_range_l, weight_range_h)))
			AutoEncoder.layers[0].neurons[i].inputs[j].input = InputList[j]

	# Add connections out of the network
	for i in range(len(AutoEncoder.layers[nr_layers-1].neurons)):		# Amount of neurons in the layer
		for j in range(data_length):	# Amount of outputs
			# Neuron <i> of the last AE layer has an output synapse <j> whose input is that neuron and output is neuron <j> of the output layer
			AutoEncoder.layers[nr_layers-1].neurons[i].add_output(classes.Synapse(s_in=AutoEncoder.layers[nr_layers-1].neurons[i], s_out=AutoEncoder.layers[nr_layers].neurons[j], s_weight=random.uniform(weight_range_l, weight_range_h)))
			# Neuron <j> of the output layer has an input synapse which is the output synpase <j> of the neuron <i> in the last NN layer
			AutoEncoder.layers[nr_layers].neurons[j].add_input(AutoEncoder.layers[nr_layers-1].neurons[i].outputs[j])

	# Add connections within the network
	for l in range(nr_layers-1):
		for i in range(len(AutoEncoder.layers[l].neurons)):
			for j in range(len(AutoEncoder.layers[l+1].neurons)):
				AutoEncoder.layers[l].neurons[i].add_output(classes.Synapse(s_in=AutoEncoder.layers[l].neurons[i], s_out=AutoEncoder.layers[l+1].neurons[j], s_weight=random.uniform(weight_range_l, weight_range_h)))
				AutoEncoder.layers[l+1].neurons[j].add_input(AutoEncoder.layers[l].neurons[i].outputs[j])

	# Training phase
	for m in range(autoencoder_loops):
		np.random.shuffle(dataArray.values)
		learning_rate = eta
		for n in range(dataArray.values.shape[0]): # For each line of input data
			# Input propagation phase
			for i in range(data_length): 	# Number of inputs
				InputList[i].y = dataArray.values[n, i]			# Set input values
			AutoEncoder.input_propagation(choice=sig_choice)

			# for i in range(data_length):	# Number of outputs
			# 	AutoEncoder.layers[nr_layers].neurons[i].y = AutoEncoder.layers[nr_layers].neurons[i].S

			# Error propagation phase
			learning_rate = 1*learning_rate
			AutoEncoder.error_propagation(dataArray.values[n, :], learning_rate)

			# Printing results for last phase
			if m == autoencoder_loops-1:
				aeOut = []
				for i in range(len(AutoEncoder.layers[nr_layers].neurons)):
					aeOut.append(AutoEncoder.layers[nr_layers].neurons[i].y)
				# Printing results for loop <n>
				aeData = ["%.2f" % float(elem) for elem in dataArray.values[n]]
				aeOut = ["%.2f" % float(elem) for elem in aeOut]
				#print("Training phase {0}: Data {1}, Output {2}".format(n, aeData, aeOut))

			#AutoEncoder.debug() # Print out all values in the network (i.e. synapse weights)

	# Extracting the encoder portion
	for i in range(nr_layers-1):
		if len(AutoEncoder.layers[i].neurons) < len(AutoEncoder.layers[i+1].neurons):
			encoderLayers = []
			for j in range(i+1):
				encoderLayers.append(AutoEncoder.layers[j])
			for j in range(len(encoderLayers[-1].neurons)):
				encoderLayers[-1].neurons[j].outputs = []
			Encoder = classes.Network(layers=encoderLayers)
			break

	return Encoder



def initialize_autoencoder_mlp(data_filename, weight_range_l, weight_range_h, sig_choice, eta, ae_loops, training_loops):

	# Read data
	#[tmpIn, tmpExp] = fileHandling.readData(data_filename)
	#categories = np.unique(tmpExp)
	[tmpIn, tmpExp, categories] = xlsParser.read_xls(data_filename)
	nr_categories = len(categories)
	trInput = classes.NN_Input(values=np.array(helpers.normalize_2d(tmpIn)))
	trExpected = classes.NN_Output(values=np.array(tmpExp))
	trOutput = classes.NN_Output(values=[])

	# Create the autoencoder
	Encoder = create_autoencoder(trInput.values, weight_range_l, weight_range_h, sig_choice, eta, ae_loops)

	# Create the network (list of layers which have a list of neurons)
	NeuralNetwork = classes.Network(layers=[])
	nr_layers = int(input("-> Neural Network - Enter desired number of layers: "))
	layer0_raw_neurons = 0
	layer0_encoder_neurons = 0
	for i in range(nr_layers):
		# Layer 0 has a part connected to raw inputs and a part connected to the existing encoder
		if i == 0:
			layer0_raw_neurons 	   = int(input("Enter desired number of neurons in layer {0} connected to the inputs: ".format(i)))
			layer0_encoder_neurons = int(input("Enter desired number of neurons in layer {0} connected to the encoder outputs: ".format(i)))
			NeuralNetwork.add_layer(classes.Layer(nr_neurons=layer0_raw_neurons+layer0_encoder_neurons, neurons=[]))
		else:
			tmp_number = int(input("Enter desired number of neurons in layer {0}: ".format(i)))
			NeuralNetwork.add_layer(classes.Layer(nr_neurons=tmp_number, neurons=[]))

	# The output layer (as many neurons as there are possible outputs)
	NeuralNetwork.add_layer(classes.Layer(nr_neurons=nr_categories, neurons=[]))	

	# Add connections into the network
	InputList = Encoder.get_input_list() # The encoder already initialized network Input objects
	# Connections for layer 0 raw neurons
	for i in range(layer0_raw_neurons):	# Amount of raw neurons in the layer
		for j in range(len(InputList)):	# Amount of inputs going into the network
			# Adding the connections, should be fine because raw neurons are first in layer 0 (the lower numbers)
			NeuralNetwork.layers[0].neurons[i].add_input(classes.Synapse(s_out=NeuralNetwork.layers[0].neurons[i], s_weight=random.uniform(weight_range_l, weight_range_h)))
			NeuralNetwork.layers[0].neurons[i].inputs[j].input = InputList[j]
	# Connections for layer 0 neurons connected to the encoder
	for i in range(layer0_raw_neurons, layer0_raw_neurons + layer0_encoder_neurons):
		for j in range(len(Encoder.layers[-1].neurons)): # Amount of encoder output neurons
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
			Encoder.input_propagation(choice=sig_choice)
			NeuralNetwork.input_propagation(choice=sig_choice)

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
			nnIn = ["%.2f" % float(elem) for elem in trInput.values[n]]
			nnOut = ["%.2f" % float(elem) for elem in nnOut]
			#print("Training phase {0}: Input {1}, Expected {2}, Output {3}".format(n, nnIn, nnExp, nnOut))
			trOutput.values.append(nnOut)	# Adding results to output table

			# NeuralNetwork.debug() # Print out all values in the network (i.e. synapse weights)

	return NeuralNetwork



def run_mlp(data_filename, NeuralNetwork, sig_choice):

	# Read data
	[tmpIn, tmpExp, categories] = xlsParser.read_xls(data_filename)
	nr_categories = len(categories)
	netInput = classes.NN_Input(values=np.array(helpers.normalize_2d(tmpIn)))
	netExpected = classes.NN_Output(values=np.array(tmpExp))
	netOutput = classes.NN_Output(values=[])

	# Getting a list of the network's input objects
	InputList = NeuralNetwork.get_input_list()

	# Pushing the data through the network
	for n in range(netInput.values.shape[0]): # For each line of input data
		for i in range(netInput.values.shape[1]): 	# Number of inputs
			InputList[i].y = netInput.values[n, i]			# Set input values
		NeuralNetwork.input_propagation(choice=sig_choice)

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
	print("~> Percentage of correct classification: {0}%".format(percent_correct))