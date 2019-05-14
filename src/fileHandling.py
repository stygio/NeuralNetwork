# Needs to be changed to an array for speed and slicing

def readData(filename):
	f = open(filename, "r")

	if f.mode == 'r':
		lines = f.readlines()
		inData = []
		outData = []
		for line in lines:
			line = line.strip()
			line = line.split("|")
			inputs = line[0].strip()
			inputs = inputs.split(" ")
			outputs = line[1].strip()
			outputs = outputs.split(" ")
			inTmp = []
			outTmp = []

			if isinstance(inputs, str):
				inTmp = float(inputs)
			else:
				for i in range(len(inputs)):
					inTmp.append(float(inputs[i]))

			if isinstance(outputs, str):
				outTmp = float(outputs)
			else:
				for i in range(len(outputs)):
					outTmp.append(float(outputs[i]))

			inData.append(inTmp)
			outData.append(outTmp)

		return [inData, outData]
