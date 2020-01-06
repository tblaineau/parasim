import numpy as np
import pandas as pd
import scipy.optimize
import scipy.integrate
from iminuit import Minuit
import numba as nb
import time
import logging

from parallax_simulator_pipeline.parameter_generator import microlens_parallax, microlens_simple
from scipy.signal import find_peaks


# UTILITIES #


@nb.njit
def simple_difference(t, u0, t0, tE, pu0, pt0, ptE, pdu, ptheta, blend):
	""" simple difference between parallax and simple."""
	return microlens_parallax(t, 19, blend, pu0, pt0, ptE, pdu, ptheta) - microlens_simple(t, 19., 0., u0, t0, tE, 0., 0.)


@nb.njit
def squared_difference(t, u0, t0, tE, pu0, pt0, ptE, pdu, ptheta, blend):
	"""njitted squared difference between parallax and simple. (so not an absolute difference)"""
	t = np.array([t])
	return (simple_difference(t, u0, t0, tE, pu0, pt0, ptE, pdu, ptheta, blend)) ** 2


# ESTIMATORS #


def integral_curvefit(params, epsabs=1e-8):
	if abs(params["tE"]) < 608.75:
		a = params['t0'] - 3652.5
		b = params['t0'] + 3652.5
	else:
		a = params['t0'] - 6 * abs(params['tE'])
		b = params['t0'] + 6 * abs(params['tE'])

	def minuit_wrap(u0, t0, tE):
		tE = np.power(10, tE)
		quadargs = (u0, t0, tE, params['u0'], params['t0'], params['tE'], params['delta_u'], params['theta'], params['blend'])
		val = scipy.integrate.quad(squared_difference, a, b, args=quadargs, epsabs=epsabs)[0]
		return val

	def de_wrap(x):
		u0, t0, tE = x
		tE = np.power(10, tE)
		quadargs = (u0, t0, tE, params['u0'], params['t0'], params['tE'], params['delta_u'], params['theta'], params['blend'])
		val = scipy.integrate.quad(squared_difference, a, b, args=quadargs, epsabs=epsabs)[0]
		return val

	m = Minuit(minuit_wrap,
			   u0=params['u0'],
			   t0=params['t0'],
			   tE=np.log10(np.abs(params['tE'])),
			   error_u0=0.1,
			   error_t0=10,
			   error_tE=0.01,
			   limit_u0=(0, 3),
			   limit_t0=(params['t0'] - 400, params['t0'] + 400),
			   limit_tE=(0, 5),
			   errordef=1,
			   print_level=0
			   )
	m.migrad()
	errs = dict(m.errors)

	if errs["t0"] >= np.power(10, abs(params["tE"])):
		de_bounds = [(0, 3), (a, b), (0, 5)]
		res = scipy.optimize.differential_evolution(de_wrap, de_bounds, strategy='best1bin', popsize=40)
		resx = dict(zip(["u0", "t0", "tE"], [res.x[0], res.x[1], np.power(10, res.x[2])]))
		return [res.fun, resx]
	else:
		resx = dict(m.values)
		resx['tE'] = np.power(10, resx['tE'])
		return [m.get_fmin().fval, resx]


def count_peaks(params, min_prominence=0., base_mag=19.):
	"""
	Compute the number of peaks in the parallax event curve between t0-2*tE and t0+2*tE

	Parameters
	----------
	params : dict
		Dictionary containing the parameters to compute the parallax curve
	min_prominence : float
		Minimum prominence of a peak to be taken into account, in magnitude
	base_mag : float
		Base magnitude of the unlensed source

	Returns
	-------
	int
		Number of peaks detected
	"""
	t = np.arange(params['t0']-2*np.abs(params['tE'])-1000, params['t0']+2*np.abs(params['tE'])+1000, 1)
	cpara = microlens_parallax(t, **params)
	peaks, infos = find_peaks(base_mag-cpara, prominence=min_prominence)
	if len(peaks):
		return [len(peaks), infos["prominences"]]
	else:
		return 0


def minmax_distance(params, time_sampling=0.5, pop_size=40):
	"""Compute distance by minimizing the maximum difference between parallax curve and no-parallax curve
	by changing u0, t0 and tE values."""
	t = np.arange(params['t0'] - 400, params['t0'] + 400, time_sampling)

	def fitter_minmax(g):
		u0, t0, tE = g
		return (simple_difference(t, u0, t0, tE, params['u0'], params['t0'], params['tE'], params['delta_u'], params['theta'], params['blend']) ** 2).max()

	bounds = [(0, 3), (params['t0'] - 400, params['t0'] + 400), (min(1., abs(params['tE']*0.1)), abs(params['tE'])*1.5)]
	init_pop = np.array([np.random.uniform(bounds[0][0], bounds[0][1], pop_size),
				np.random.uniform(bounds[1][0], bounds[1][1], pop_size),
				np.random.uniform(bounds[2][0], bounds[2][1], pop_size),
				]).T

	init_pop[0] = [params['u0'], params['t0'], params['tE']]

	res = scipy.optimize.differential_evolution(fitter_minmax, bounds=bounds, disp=False,
												mutation=(0.5, 1.0), strategy='currenttobest1bin', recombination=0.9,
												init=init_pop)
	return [np.sqrt(res.fun), res.x]


def compute_distances(output_name, distance, parameter_list, nb_samples=None, start=None, end=None, **distance_args):
	"""
	Compute distance between parallax and no-parallax curves using the *distance* function.

	Parameters
	----------

	output_name : str
		Name of the pandas pickle file where is stocked a Dataframe containing the parameters with the associated distance
	distance : function
		Function used to compute the distance between parallax curve and no-parallax curve, with same parameters
	parameter_list : list
		List of lens event parameters
	nb_samples : int
		Compute the distance for the **nb_samples** first parameters sets. If nb_samples, start and end are None, compute distance for all parameters.
	start : int
		If nb_samples is None, the index of parameter_list from which to computing distance
	end : int
		If nb_samples is None, the index of parameter_list where to stop computing distance
	**distance_args : distance arguments
		Arguments to pass to distance function.
	"""
	if nb_samples is None:
		if start is not None and end is not None:
			parameter_list = parameter_list[start:end]
		elif (start is not None and end is None) or (start is None and end is not None):
			logging.error('Start and end should both be initialized if nb_samples is None.')
			return None
	else:
		parameter_list = parameter_list[:nb_samples]
	df = pd.DataFrame.from_records(parameter_list)

	st1 = time.time()

	ds = []
	i = 0
	for params in parameter_list:
		i += 1
		n_params = {key: params[key] for key in ['u0', 't0', 'tE', 'delta_u', 'theta']}
		n_params['blend'] = params['blend_red_M']
		n_params['mag'] = 19.
		ds.append(distance(n_params, **distance_args))
		if i % 100 == 0:
			logging.debug(i)

	logging.info(f'{len(parameter_list)} distances computed in {time.time()-st1:.2f} seconds.')

	df = df.assign(distance=ds)
	df.to_pickle(output_name)


#


def max_parallax(params):
	""" Find maximum value of parallax light curve"""
	def minus_parallax(t):
		t = np.array([t])
		return microlens_parallax(t, 19, params['blend'], params['u0'], params['t0'], params['tE'], params['delta_u'], params['theta'])

	m = Minuit(minus_parallax,
			   t = params["t0"],
			   error_t = 180,
			   errordef = 1,
			   print_level=0)
	m.migrad()
	return [m.get_fmin().fval, dict(m.values)['t']]

#logging.basicConfig(level=logging.DEBUG)

# st = time.time()
# pms = np.load('params1M_0.npy', allow_pickle=True)[:1000]
# print(len(pms))
# end = time.time()
# logging.info(f'{len(pms)} parameters loaded in {end-st:.2f} seconds.')
#
# df = pd.DataFrame.from_records(pms)

# df2 = pd.read_pickle('scipyminmax.pkl')['idx']
#
# df = df.merge(df2, on='idx', suffixes=('',''))
# print(len(df))

# print(df.idx.sort_values())
# pms = df.to_records()
#
#
# compute_distances('trash.pkl', integral_curvefit, pms, nb_samples=1000)

"""
logging.debug('Loading xvt_samples')
all_xvts = np.load('../test/xvt_thick_disk.npy')
logging.debug('Done')
logging.debug('Shuffling')
np.random.shuffle(all_xvts)
logging.debug('Done')
logging.debug('Generating parameters sets')
pms = generate_parameter_file('parameters_u02f_TD', all_xvts[:100000], [0.1, 1, 10, 30, 100, 300])

pms = np.load('parameters_u02f_TD.npy', allow_pickle=True)
print(len(pms))
for idx, a in enumerate(np.split(np.array(pms), 20)):
	np.save('parameters_u02f_TD_'+str(idx), a, allow_pickle=True)"""
