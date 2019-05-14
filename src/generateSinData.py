#!/usr/bin/python3

import random
import math

f = open("sinus_dataset.txt", "w+")
f2 = open("more_columns.txt", "w+")

for i in range(10):
		x = random.uniform(-2*math.pi, 2*math.pi)
		y = int(math.sin(x) >= 0)
		f.write("{0} | {1}\n".format(x, y))
		f2.write("{0} {1} {2} | {3}\n".format(x, x, x, y))

f.close()
f2.close()