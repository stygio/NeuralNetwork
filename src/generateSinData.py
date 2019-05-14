#!/usr/bin/python3

import random
import math

f = open("sinus_dataset.txt", "w+")
f2 = open("more_columns.txt", "w+")

for i in range(500):
		x = random.uniform(-2*math.pi, 2*math.pi)
		y = int(math.sin(x) >= 0)
		if y == 0:
			y1 = 1
			y2 = 0
		else:
			y1 = 0
			y2 = 1
		f.write("{0} | {1} {2}\n".format(x, y1, y2))
		f2.write("{0} {1} {2} | {3} {4}\n".format(x, x, x, y1, y2))

f.close()
f2.close()