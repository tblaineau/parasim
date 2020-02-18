import numpy as np
import astropy.units as units
import astropy.constants as constants
import numba as nb
import logging
import pandas as pd

from parallax_simulator_pipeline.metropolis_hastings import metropolis_hastings, hc_randomizer_thickdisk_LMC

COLOR_FILTERS = {
	'red_E':{'mag':'red_E', 'err': 'rederr_E'},
	'red_M':{'mag':'red_M', 'err': 'rederr_M'},
	'blue_E':{'mag':'blue_E', 'err': 'blueerr_E'},
	'blue_M':{'mag':'blue_M', 'err': 'blueerr_M'}
}

a = 5000
rho_0 = 0.0079
d_sol = 8500
vrot_sol = 239 #km/s
l_lmc, b_lmc = 280.4652/180.*np.pi, -32.8884/180.*np.pi
r_lmc = 8500
l_gc, b_gc = 0., 0.
r_gc = 8000
r_earth = (150*1e6*units.km).to(units.pc).value
t_obs = ((52697 - 48928) << units.d).to(units.s).value

# Thick Disk parameters
sigma_r = 56.1
sigma_theta = 46.1
sigma_z = 35.1	# speed dispersion of deflector particular speed (in heliospherical galactic coordinates)
sigma = 35		# column density of the disk
H = 1000.		# height scale
R = 3500.		# radial length scale


pc_to_km = (units.pc.to(units.km))
kms_to_pcd = (units.km/units.s).to(units.pc/units.d)

cosb_lmc = np.cos(b_lmc)
cosl_lmc = np.cos(l_lmc)
A = d_sol ** 2 + a ** 2
B = d_sol * cosb_lmc * cosl_lmc
r_0 = np.sqrt(4*constants.G/(constants.c**2)*r_lmc*units.pc).decompose([units.Msun, units.pc]).value

epsilon = (90. - 66.56070833)*np.pi/180.
delta_lmc = -69.756111 * np.pi/180.
alpha_lmc = 80.89417 * np.pi/180.

delta_gc = -(29+(0.+28.1/60.)/60.) /180 *np.pi
alpha_gc = (15.+(45.+40.04/60.)/60.)/24. * 2*np.pi

def compute_i():
	""" Compute the i vector coordinates, in heliopsheric galactic coordinates.
	i is the vector of the projected plane referential."""

	rot1qc = np.array([
		[1, 0, 0],
		[0, np.cos(epsilon), np.sin(epsilon)],
		[0, -np.sin(epsilon), np.cos(epsilon)]
	])

	def eq_to_ec(v):
		return rot1qc @ np.array(v)

	def ec_to_eq(v):
		return rot1qc.T @ np.array(v)

	rotYlmc = np.array([
		[np.cos(delta_lmc), 0, np.sin(delta_lmc)],
		[0, 1, 0],
		[-np.sin(delta_lmc), 0, np.cos(delta_lmc)]
	])
	rotZlmc = np.array([
		[np.cos(alpha_lmc), np.sin(alpha_lmc), 0],
		[-np.sin(alpha_lmc), np.cos(alpha_lmc), 0],
		[0, 0, 1]
	])
	def eq_to_lmc(v):
		return rotYlmc @ rotZlmc @ v

	K = eq_to_lmc(ec_to_eq([0, 0, 1]))
	i = np.cross(K, np.array([1, 0, 0]))
	i = i/np.linalg.norm(i)
	return i


def compute_thetas(vr, vtheta, vz, x):
	v = project_from_gala(vr, vtheta, vz, x)
	v = np.array([np.zeros(len(v[0])), *v])
	i = compute_i()
	thetas = np.arctan2(i[1]*v[2]-i[2]*v[1], i[1]*v[1]+i[2]*v[2])
	return thetas

@nb.njit
def r(mass):
	R_0 = r_0*np.sqrt(mass)
	return r_earth/R_0


@nb.njit
def R_E(x, mass):
	return r_0*np.sqrt(mass*x*(1-x))


def cartgal_to_heliosphgal(vx, vy, vz):
	"""Transform cartesian galactocentric coordinates to heliocentric galactic coordinates
	cartesian galactocentric defined by :
		origin in galactic center
		x toward the sun
		y in the rotation direction of the Sun (anti-trigo in regard to the galactic North Pole)
		z toward the galactic North Pole

	heliocentric galactic defined by:
		origin at the Sun
		x toward the galactic center
		z toward the galactic north pole
		y to make the referential direct
	"""
	v = np.array([vx, vy, vz])
	rot1 = np.array([
		[np.cos(l_lmc), np.sin(l_lmc), 0],
		[-np.sin(l_lmc), np.cos(l_lmc), 0],
		[0, 0, 1]
	])

	rot2 = np.array([
		[np.cos(b_lmc), 0, np.sin(b_lmc)],
		[0, 1, 0],
		[-np.sin(b_lmc), 0, np.cos(b_lmc)]
	])

	# print(v)
	# print(rot1 @ v)
	# print(rot2 @ rot1 @ v)
	return rot2 @ rot1 @ v


v_lmc = cartgal_to_heliosphgal(-57, -100, 20)
# Compute LMC speed vector in heliospherical galactic coordinates
v_sun = cartgal_to_heliosphgal(11.1, 12.24 + vrot_sol, 7.25)
# Compute speed vector of the Sun in heliospherical galactic coordinates (particular speed + global rotation speed)


@nb.njit
def vrot(r):
	"""Absolute global rotation speed in the Milky Way
	Parameters
	----------
	r : float
		distance to GC

	Returns
	-------
	float
		Absolute global rotation speed
	"""
	return vrot_sol*(1.00767*(r/d_sol)**0.0394 + 0.00712)


@nb.njit
def project_from_gala(vr, vtheta, vz, x):
	"""Project speed vector located on the LoS toward the LMC, at x, in heliospherical galactic coordinates.
	Only returns components orthogonal to the LoS

	Parameters
	----------
	vr, vtheta, vz : float
		speed vector coordinates in heliospherical galactic coordinates
	x : float
		distance ratio to LMC

	Returns
	-------
	vgala_theta, vgala_phi : (float, float)
			theta and phi components of projected speed vector orthogonal to LoS, in heliospherical galactic coordinates
	"""
	r = np.sqrt((x * r_lmc * np.cos(b_lmc) * np.cos(l_lmc) - d_sol) ** 2 + (x * r_lmc * np.cos(b_lmc) * np.sin(l_lmc)) ** 2)
	sin_theta = (x * r_lmc * np.sin(l_lmc) * np.cos(b_lmc))
	cos_theta = (x * r_lmc * np.cos(b_lmc) * np.cos(l_lmc) - d_sol)
	theta = np.arctan2(sin_theta, cos_theta)
	cosa = np.cos(theta - l_lmc)
	sina = np.sin(theta - l_lmc)

	vhelio_r = vr * cosa - vtheta * sina
	vhelio_theta = vr * sina + vtheta * cosa
	vhelio_z = vz

	# vgala_r = np.cos(b_lmc) * vhelio_r + np.sin(b_lmc) * vhelio_z
	vgala_theta = vhelio_theta
	vgala_phi = - np.sin(b_lmc) * vhelio_r + np.cos(b_lmc) * vhelio_z

	return vgala_theta, vgala_phi


@nb.njit
def vt_from_vs(vr, vtheta, vz, x):
	"""Transform speed vector located on the LoS toward the LMC, at x, in heliospherical galactic coordinates,
	then project it on the plane orthogonal to the LoS and returns the norm

	Parameters
	----------
	vr, vtheta, vz : float
		speed vector coordinates in heliospherical galactic coordinates
	x : float
		distance ratio to LMC

	Returns
	-------
	vt : float
		norm of speed vector orthogonally projected to LoS
	"""
	r = np.sqrt((x * r_lmc * np.cos(b_lmc) * np.cos(l_lmc) - d_sol) ** 2 + (x * r_lmc * np.cos(b_lmc) * np.sin(l_lmc)) ** 2)
	sin_theta = (x * r_lmc * np.sin(l_lmc) * np.cos(b_lmc))
	cos_theta = (x * r_lmc * np.cos(b_lmc) * np.cos(l_lmc) - d_sol)
	theta = np.arctan2(sin_theta, cos_theta)
	cosa = np.cos(theta - l_lmc)
	sina = np.sin(theta - l_lmc)
	vtheta1 = vtheta - vrot(r)		# Add the global rotation speed to the deflector speed

	# logging.debug(vr, vtheta, vz, r)

	vhelio_r = vr * cosa - vtheta1 * sina
	vhelio_theta = vr * sina + vtheta1 * cosa
	vhelio_z = vz

	# logging.debug(vhelio_r, vhelio_theta, vhelio_z)

	# vgala_r = np.cos(b_lmc) * vhelio_r + np.sin(b_lmc) * vhelio_z
	vgala_theta = vhelio_theta
	vgala_phi = - np.sin(b_lmc) * vhelio_r + np.cos(b_lmc) * vhelio_z

	# logging.debug(vgala_r, vgala_theta, vgala_phi, theta*180/np.pi)

	vt = np.sqrt((vgala_theta - v_sun[1] * (1 - x) - v_lmc[1] * x) ** 2 + (vgala_phi - v_sun[2] * (1 - x) - v_lmc[2] * x) ** 2)
	# vt = np.sqrt((vgala_theta - v_sun[1]*(1-x))**2 + (vgala_phi - v_sun[2]*(1-x))**2)		# Do not take the LMC speed into account
	# vt= np.sqrt(vgala_theta**2 + vgala_phi**2)											# Only take the speed of the deflector into account
	return vt  # , vgala_r, vgala_theta, vgala_phi


@nb.njit
def rho_halo(x):
	return rho_0*A/((x*r_lmc)**2-2*x*r_lmc*B+A)


@nb.njit
def f_vt(v_T, v0=220):
	return (2*v_T/(v0**2))*np.exp(-v_T**2/(v0**2))


@nb.njit
def p_xvt(x, v_T, mass):
	return rho_halo(x)/mass*r_lmc*(2*r_0*np.sqrt(mass*x*(1-x))*t_obs*v_T)


@nb.njit
def delta_u_from_x(x, mass):
	return r(mass)*np.sqrt((1-x)/x)


@nb.njit
def tE_from_xvt(x, vt, mass):
	return r_0 * np.sqrt(mass*x*(1-x)) / (vt*kms_to_pcd)


@nb.njit
def rho_disk(r, z, sigma, H, R):
	"""Disk dark matter density"""
	return sigma/2/H * np.exp(-(r-d_sol)/R) * np.exp(-np.abs(z)/H)


@nb.njit
def gaussian(x, mu, sigma):
	"""Gaussian"""
	return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2/2/sigma**2)


@nb.njit
def p_vt_disk(vr, vtheta, vz, sig_r, sig_theta, sig_z):
	"""Particular speed vector probability distribution in disk"""
	return gaussian(vr, 0, sig_r)*gaussian(vtheta, 0, sig_theta)*gaussian(vz, 0, sig_z)


@nb.njit
def pdf_xvs_disk(vec, params):
	"""Disk geometry probabilty density function

	Parameters
	----------
	vec : np.array([float, float, float, float)]
		distance ratio to the LMC and deflector speed vector coordinates (in heliospherical galactic coordinates)
	sig_r, sig_theta, sig_z : float
		speed dispersion of deflector particular speed (in heliospherical galactic coordinates)
	sigma : float
		column density of the disk
	H : float
		height scale
	R : float
		radial length scale

	Returns
	-------
	float
		pdf of (x, vr, vtheta, vz) for a disk, toward LMC

	"""
	x, vr, vtheta, vz = vec
	sig_r, sig_theta, sig_z, sigma, H, R = params
	if x<0 or x>1:
		return 0		# x should be in [0, 1]
	z_sol = 26
	z = np.sqrt((x*r_lmc*np.sin(b_lmc))**2 + z_sol**2)
	r = np.sqrt((x*r_lmc*np.cos(b_lmc)*np.cos(l_lmc) - d_sol)**2 + (x*r_lmc*np.cos(b_lmc)*np.sin(l_lmc))**2)
	return np.sqrt(x*(1-x)) * p_vt_disk(vr, vtheta, vz, sig_r, sig_theta, sig_z) * rho_disk(r, z, sigma, H, R) * np.abs(vt_from_vs(vr, vtheta, vz, x))


class MicrolensingGenerator:
	"""
	Class to generate microlensing paramters

	Parameters
	----------
	xvt_file : str or int
		If a str : Path to file containing x - v_T pairs generated through the Hasting-Metropolis algorithm
		If int, number of xvt pairs to pool
	seed : int
		Seed used for numpy.seed
	tmin : float
		lower limit of t_0
	tmax : float
		upper limits of t_0
	max_blend : float
		maximum authorized blend, if max_blend=0, no blending
	"""
	def __init__(self, xvt_file=None, seed=None, tmin=48928., tmax=52697., u_max=2.,  max_blend=0., min_blend=0.):
		self.seed = seed
		self.xvt_file = xvt_file

		self.tmin = tmin
		self.tmax = tmax
		self.u_max = u_max
		self.max_blend = max_blend
		self.min_blend = min_blend
		self.blending = bool(max_blend)
		self.blend_pdf = None
		self.generate_mass = False

		if self.seed:
			np.random.seed(self.seed)

		if self.xvt_file:
			if isinstance(self.xvt_file, int):
				logging.info(f"Generating {self.xvt_file} x-vt pairs... ")
				self.xs, vr, vtheta, vz = metropolis_hastings(pdf_xvs_disk, hc_randomizer_thickdisk_LMC, 1000000, np.array([0.9, 10., 10., 10.]), (sigma_r, sigma_theta, sigma_z, sigma, H, R)).T
				self.vts = vt_from_vs(vr, vtheta, vz, self.xs)
				self.thetas = compute_thetas(vr, vtheta, vz, self.xs)
			else:
				logging.error(f"Currently, xvts must be the number of set to generate : {self.xvt_file}")

	def generate_parameters(self, mass=30., seed=None, nb_parameters=1):
		"""
		Generate a set of microlensing parameters, including parallax and blending using S-model and fixed mass

		Parameters
		----------
		seed : str
			Seed used for parameter generation (EROS id)
		mass : float
			mass for which generate parameters (\implies \delta_u, t_E)
		nb_parameters : int
			number of parameters set to generate
		Returns
		-------
		dict
			Dictionnary of lists containing the parameters set
		"""
		if isinstance(seed, str):
			seed = int(seed.replace('lm0', '').replace('k', '0').replace('l', '1').replace('m', '2').replace('n', '3'))
			np.random.seed(seed)
		elif seed:
			np.random.seed(seed)
		if self.generate_mass:
			mass = np.random.uniform(0, 200, size=nb_parameters)
		else:
			mass = np.array([mass]*nb_parameters)
		u0 = np.random.uniform(-self.u_max, self.u_max, size=nb_parameters)
		indices = np.random.randint(0, len(self.xs), size=nb_parameters)
		x = self.xs[indices]
		vt = self.vts[indices]
		theta = self.thetas[indices]
		delta_u = delta_u_from_x(x, mass=mass)
		tE = tE_from_xvt(x, vt, mass=mass)
		t0 = np.random.uniform(self.tmin, self.tmax, size=nb_parameters)
		params = {
			'u0': u0,
			't0': t0,
			'tE': tE,
			'delta_u': delta_u,
			'theta': theta,
			'mass': mass,
			'x': x,
			'vt': vt,
		}

		for key in COLOR_FILTERS.keys():
			if self.blending:
				params['blend_'+key] = np.random.uniform(self.min_blend, self.max_blend, size=nb_parameters)
			else:
				params['blend_'+key] = [0] * nb_parameters
		return params


# We define parallax parameters.
PERIOD_EARTH = 365.2422
alphaS = 80.8941667*np.pi/180.
deltaS = -69.7561111*np.pi/180.
epsilon = (90. - 66.56070833)*np.pi/180.		# source in LMC
t_origin = 51442 								# (21 septembre 1999) #58747 #(21 septembre 2019)

sin_beta = np.cos(epsilon)*np.sin(deltaS) - np.sin(epsilon)*np.cos(deltaS)*np.sin(alphaS)
beta = np.arcsin(sin_beta) 						# ok because beta is in -pi/2; pi/2
if abs(beta) == np.pi/2:
	lambda_star = 0
else:
	lambda_star = np.sign((np.sin(epsilon)*np.sin(deltaS)+np.cos(epsilon)*np.sin(alphaS)*np.cos(deltaS))/np.cos(beta)) * np.arccos(np.cos(deltaS)*np.cos(alphaS)/np.cos(beta))


@nb.njit
def microlens_parallax(t, mag, blend, u0, t0, tE, delta_u, theta):
	tau = (t-t0)/tE
	phi = 2*np.pi * (t-t_origin)/PERIOD_EARTH - lambda_star
	t1 = u0**2 + tau**2
	t2 = delta_u**2 * (np.sin(phi)**2 + np.cos(phi)**2*sin_beta**2)
	t3 = -2*delta_u*u0 * (np.sin(phi)*np.sin(theta) + np.cos(phi)*np.cos(theta)*sin_beta)
	t4 = 2*tau*delta_u * (np.sin(phi)*np.cos(theta) - np.cos(phi)*np.sin(theta)*sin_beta)
	u = np.sqrt(t1+t2+t3+t4)
	parallax = (u**2+2)/(u*np.sqrt(u**2+4))
	return - 2.5*np.log10(blend + (1-blend)* parallax) + mag


@nb.njit
def microlens_simple(t, mag, blend, u0, t0, tE, delta_u=0, theta=0):
	u = np.sqrt(u0*u0 + ((t-t0)**2)/tE/tE)
	amp = (u**2+2)/(u*np.sqrt(u**2+4))
	return - 2.5*np.log10(blend + (1-blend)* amp) + mag


def dict_of_lists_to_numpy_structured_array(pms):
	""" Function to convert a dictionary of list of float to a structured array """
	dtypes = dict(names=list(pms.keys()), formats=['f8']*len(pms.keys()))
	t = np.zeros(len(pms['u0']), dtype=dtypes)
	for key in pms.keys():
		t[key] = pms[key]
	pms = t
	del t
	return pms


def generate_parameters_file(savename, global_seed=1995281, masses=[0.1, 1, 10, 30, 100, 300], nb_parameters=1000):
	np.random.seed(global_seed)
	mlg = MicrolensingGenerator(100000000, tmin=48928, tmax=48928+365.25, max_blend=0., u_max=2.)
	pms = []
	for mass in masses:
		pms.append(dict_of_lists_to_numpy_structured_array(mlg.generate_parameters(mass=mass, nb_parameters=nb_parameters)))
	pms = np.concatenate(pms)
	pms = pd.DataFrame(pms).to_records()
	np.save(savename, pms)