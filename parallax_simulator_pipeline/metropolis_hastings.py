import numba as nb
import numpy as np

@nb.njit
def randomizer_gauss(x):
	"""x and vt randomizer"""
	return np.array([np.random.normal(loc=x[0], scale=0.1), np.random.normal(loc=x[1], scale=300)])


@nb.njit
def hc_randomizer_thickdisk_LMC(x):
	""" x and vr, vtheta, vz randomizer"""
	scales = [0.1, 17., 16., 15.]
	return np.array([np.random.normal(loc=x[0], scale=scales[0]),
					 np.random.normal(loc=x[1], scale=scales[1]),
					 np.random.normal(loc=x[2], scale=scales[2]),
					 np.random.normal(loc=x[3], scale=scales[3])])


@nb.njit
def hc_randomizer_halo_LMC(x):
	""" x and vr, vtheta, vz randomizer"""
	scales = [0.2, 200., 200., 200.]
	return np.array([np.random.normal(loc=x[0], scale=scales[0]),
					 np.random.normal(loc=x[1], scale=scales[1]),
					 np.random.normal(loc=x[2], scale=scales[2]),
					 np.random.normal(loc=x[3], scale=scales[3])])


@nb.njit
def hc_randomizer_GC_halo(x):
	#x, vr, vtheta, vz, vr_s, vtheta_s, vz_s, r_s, l_s, b_s  + pop
	scales = [0.05, 20., 20., 20., 5., 5., 5., 1000., np.pi/500., np.pi/500.]
	return np.array([np.random.normal(loc=x[i], scale=scales[i]) for i in range(len(scales))]+[np.random.uniform(0, 1)])


@nb.njit
def hc_randomizer_GC_thin(x):
	#x, vr, vtheta, vz, vr_s, vtheta_s, vz_s, r_s, l_s, b_s  + pop
	scales = [0.01, 20., 20., 20., 5., 5., 5., 100., np.pi/50., np.pi/50.]
	return np.array([np.random.normal(loc=x[i], scale=scales[i]) for i in range(len(scales))])


@nb.njit
def metropolis_hastings(func, g, nb_samples, x0, burnin=1000, *args):
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
	samples = np.empty((nb_samples+burnin, len(x0)))
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
