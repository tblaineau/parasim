import parallax_simulator_pipeline.parameter_generator as pmg
import parallax_simulator_pipeline.similarity_estimator as sme
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

global_seed = 1995281
masses = [0.1, 1, 10, 30, 100, 300]
nb_parameters = int(100000*0.3)
np.random.seed(1995281)
#"/Users/tristanblaineau/Documents/Work/Python/merger/merger/test/xvt_thick_disk.npy"
mlg = pmg.MicrolensingGenerator(xvt_file=10000000, seed=85140, tmin=48928, tmax=48928+365.25, min_blend=0.7, max_blend=1.0)
pms = []
for mass in masses:
	pms.append(pmg.dict_of_lists_to_numpy_structured_array(mlg.generate_parameters(mass=mass, nb_parameters=nb_parameters)))
pms = np.concatenate(pms)
pms = pd.DataFrame(pms).to_records()
np.save('blended_parameters_comp', pms)