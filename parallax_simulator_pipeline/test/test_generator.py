import parallax_simulator_pipeline.parameter_generator as pmg
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

np.random.seed(85140)

pmg.generate_xvts('test_xvts', 100000)

try:
	xvts = np.load('test_xvts')
except FileNotFoundError:
	logging.error("File xvts not found")

