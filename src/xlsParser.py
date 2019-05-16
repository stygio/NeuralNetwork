import pandas as pd
import numpy as np

def read_xls(filename):
	raw_data = pd.read_excel(filename)
	colNames = raw_data.columns

	data = np.array(raw_data)
	[numR, numC] = np.shape(data)
	categories = np.unique(data[:,-1])
	inputs = data[:,0:-1].astype(float)
	outputs = []
	for i in data[:,-1]:
		row = []
		for j in categories:
			if i == j:
				row.append(1)
			else:
				row.append(0)
		outputs.append(row)
	outputs = np.array(outputs)
	categories = categories.astype(str)
	return [inputs, outputs, categories]