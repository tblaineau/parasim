import numba as nb
import numpy as np

@nb.njit
def metropolis_hastings(func, g, nb_samples, x0, *args):
	"""
	Metropolis-Hasting algorithm to pick random value following the joint probability distribution func

	Parameters
	----------
	func : function
		 Joint probability distribution
	g : function
		Randomizer. Choose it wisely to converge quickly and have a smooth distribution
	nb_samples : int
		Number of points to return. Need to be large so that the output distribution is smooth
	x0 : array-like
		Initial point
	args :
		arguments to pass to *func*


	Returns
	-------
	np.array
		Array containing all the points
	"""
	burnin = 1000
	samples = np.empty((nb_samples+burnin, , len(x0)))
	current_x = x0
	accepted=0
	rds = np.random.uniform(0., 1., nb_samples+burnin)			# We generate the rs beforehand, for SPEEEED
	for idx in range(nb_samples+burnin):
		proposed_x = g(current_x)
		tmp = func(current_x, *args)
		if tmp!=0:
			threshold = min(1., func(proposed_x, *args) / tmp)
		else:
			threshold = 1
		if rds[idx] < threshold:
			current_x = proposed_x
			accepted+=1
		samples[idx] = current_x
	print(accepted, accepted/nb_samples)
	# We crop the hundred first to avoid outliers from x0
	return samples[burnin:]
