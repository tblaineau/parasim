import parallax_simulator_pipeline.parameter_generator as pmg
import numpy as np
import pandas as pd
import logging

#TODO : Basic tests

"""logging.basicConfig(level=logging.INFO)

np.random.seed(85140)

pmg.generate_xvts('test_xvts', 100000)

try:
	xvts = np.load('test_xvts.npy')
except FileNotFoundError:
	logging.error("File xvts not found")

mlg = pmg.MicrolensingGenerator('test_xvts.npy', seed=85140, tmin=48928, tmax=48928+365.25, max_blend=0.7)

pms = mlg.generate_parameters(nb_parameters=100000)
dtypes = dict(names=list(pms.keys()), formats=['f8']*len(pms.keys()))
t = np.zeros((1, len(pms['u0'])), dtype=dtypes)
for key in pms.keys():
	t[key] = pms[key]
pms = t
del t
print(pms)
"""

#pmg.generate_parameters_file()

a = np.load('blended_parameters.npy')
print(a['index'])
