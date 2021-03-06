import numpy as np
import astropy.units as units
import astropy.constants as constants
import numba as nb
import logging
import pandas as pd
import matplotlib.pyplot as plt

core_radius = 5000
rho_0 = 0.0079
d_sol = 8500
vrot_sol = 239  # +/- 7 km/s
l_lmc, b_lmc = 280.4652/180.*np.pi, -32.8884/180.*np.pi
r_lmc = 55000
l_gc, b_gc = 0., 0.
#r_gc = 8000
r_earth = (150*1e6*units.km).to(units.pc).value
t_obs = ((52697 - 48928) << units.d).to(units.s).value

# Thick Disk parameters
sigma_r_thick = 56.1
sigma_theta_thick = 46.1
sigma_z_thick = 35.1	# speed dispersion of deflector particular speed (in heliospherical galactic coordinates)
sigma_thick = 35		# column density of the disk
H_thick = 1000.		# height scale
R_thick = 3500.		# radial length scale
# Thin Disk parameters
sigma_r_thin = 27.4
sigma_theta_thin = 20.8
sigma_z_thin = 16.3  # speed dispersion of deflector particular speed (in heliospherical galactic coordinates)
sigma_thin = 50.
H_thin = 325.
R_thin = 3500.

sigma_h = 120  # km/s  halo speed dispersion


# Bar geometry parameters
PHI = 13*np.pi/180.
a = 1490.
b = 580.
c = 400.
M_B = 1.7e10

sigma_bar = 110.  # speed dispersion in the bar
omega_bar = 39.  # km/s/kpc global roation speed
v_sun = np.array([11.1, 12.24 + vrot_sol, 7.25])  # sun global+peculiar speed


pc_to_km = (units.pc.to(units.km))
kms_to_pcd = (units.km/units.s).to(units.pc/units.d)

cosb_lmc = np.cos(b_lmc)
cosl_lmc = np.cos(l_lmc)
A = d_sol ** 2 + core_radius ** 2
B = d_sol * cosb_lmc * cosl_lmc
r_0 = np.sqrt(4*constants.G/(constants.c**2)*r_lmc*units.pc).decompose([units.Msun, units.pc]).value

epsilon = (90. - 66.56070833)*np.pi/180.
delta_lmc = -69.756111 * np.pi/180.
alpha_lmc = 80.89417 * np.pi/180.
delta_gc = -(29+(0.+28.1/60.)/60.) /180 *np.pi
alpha_gc = (15.+(45.+40.04/60.)/60.)/24. * 2*np.pi
# r_s = 8500
# l_s = 0.
# b_s = 0.


@nb.njit
def gaussian(x, mu, sigma):
	"""Gaussian"""
	return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2/2/sigma**2)


@nb.njit
def p_v_bar(vr_s, vtheta_s, vz_s, sigma_bar=sigma_bar):
	"""Compute probability of source velocity vector in the bar"""
	return gaussian(vr_s, 0, sigma_bar)*gaussian(vtheta_s, 0, sigma_bar)*gaussian(vz_s, 0, sigma_bar)


@nb.njit
def vrot(d):
	"""Absolute global rotation speed in the Milky Way
	Parameters
	----------
	d : float
		distance to GC

	Returns
	-------
	float
		Absolute global rotation speed
	"""
	return vrot_sol*(1.00767*(d/d_sol)**0.0394 + 0.00712)


@nb.njit
def deflector_speed(x, r_s, l_s, b_s, vr, vtheta, vz, global_speed):
	"""
	Compute deflector speed in sun spherical coordinates toward the source.

	Parameters
	----------
	x : float
		Ratio between distance to source and distance to deflector r_D/r_S
	r_s : float
		(parsec) Distance from Sun to source
	l_s : float
		(radians) Galactic longitude of the source
	b_s : float
		(radians) Galactic latitude of the source
	vr, vtheta, vz : float, float, float
		(km/s) Particular speed of the deflector, in galactocentric cylindrical coordinates
	global_speed : function
		(njit) Function returning the theta global rotation speed of the deflector (in gal. cyl.) (null_global_speed for halo, disk_global_speed for thick disk)

	Returns
	-------
	deflector speed in sun spherical coordinates toward the source
	"""
	r = np.sqrt((x * r_s * np.cos(b_s) * np.cos(l_s) - d_sol) ** 2 + (x * r_s * np.cos(b_s) * np.sin(l_s)) ** 2)
	sin_theta = (x * r_s * np.sin(l_s) * np.cos(b_s))
	cos_theta = (x * r_s * np.cos(b_s) * np.cos(l_s) - d_sol)
	theta = np.arctan2(sin_theta, cos_theta)
	cosa = np.cos(theta - l_s)
	sina = np.sin(theta - l_s)

	vr1 = vr
	vtheta1 = vtheta + global_speed(r)
	vz1 = vz

	vhelio_r = vr1 * cosa - vtheta1 * sina
	vhelio_theta = vr1 * sina + vtheta1 * cosa
	vhelio_z = vz1

	vgala_r = np.cos(b_s) * vhelio_r + np.sin(b_s) * vhelio_z
	vgala_theta = vhelio_theta
	vgala_phi = - np.sin(b_s) * vhelio_r + np.cos(b_s) * vhelio_z

	return vgala_r, vgala_theta, vgala_phi


@nb.njit
def bar_global_speed(d):
	"""
	Global speed of an object in the bar.

	Parameters
	----------
	d : float
		(parsec) Distance from GC

	Returns
	-------
	Theta component of the rotation speed (in gal. cyl.)
	"""
	return - (d / 1000. * omega_bar)


@nb.njit
def disk_global_speed(d):
	"""Global speed of an object in the bar.

	Parameters
	----------
	d : float
		(parsec) Distance from GC

	Returns
	-------
	Theta component of the rotation speed (in gal. cyl.)
	"""
	return - vrot(d)


@nb.njit
def null_global_speed(d):
	"""
	Empty function when no global roation speed
	Parameters
	----------
	d : float

	Returns
	-------
	float
	"""
	return 0.


@nb.njit
def source_speed(r_s, l_s, b_s, vr_s, vtheta_s, vz_s, global_speed):
	"""
	Compute source speed in sun spherical coordinates toward the source.
	Parameters
	----------
	r_s : float
		(parsec) Distance from Sun to source
	l_s : float
		(radians) Galactic longitude of the source
	b_s : float
		(radians) Galactic latitude of the source
	vr_s, vtheta_s, vz_s : float, float, float
		(km/s) Particular speed of the source, in galactocentric cylindrical coordinates
	global_speed : function
		(njit) Function returning the theta global rotation speed of the source (in gal. cyl.) (bar_global_speed for bar, disk_global_speed for thin disk)

	Returns
	-------
	Source speed in sun spherical coordinates toward the source
	"""
	r = np.sqrt((r_s * np.cos(b_s) * np.cos(l_s) - d_sol) ** 2 + (r_s * np.cos(b_s) * np.sin(l_s)) ** 2)
	sin_theta = (r_s * np.sin(l_s) * np.cos(b_s))
	cos_theta = (r_s * np.cos(b_s) * np.cos(l_s) - d_sol)
	theta = np.arctan2(sin_theta, cos_theta)
	cosa = np.cos(theta - l_s)
	sina = np.sin(theta - l_s)

	vr1 = vr_s
	vtheta1 = vtheta_s + global_speed(r)
	vz1 = vz_s

	vhelio_r = vr1 * cosa - vtheta1 * sina
	vhelio_theta = vr1 * sina + vtheta1 * cosa
	vhelio_z = vz1

	vgala_r = np.cos(b_s) * vhelio_r + np.sin(b_s) * vhelio_z
	vgala_theta = vhelio_theta
	vgala_phi = - np.sin(b_s) * vhelio_r + np.cos(b_s) * vhelio_z

	return vgala_r, vgala_theta, vgala_phi


@nb.njit
def sun_speed(l_s, b_s):
	"""
	Compute the Sun speed vector coordinates in the helio. sph. coordinates, pointing toward the source.

	Parameters
	----------
	l_s : float
		(radians) Galactic longitude of the source
	b_s : float
		(radians) Galactic latitude of the source


	Returns
	-------

	"""
	cosa = -np.cos(l_s)
	sina = np.sin(l_s)

	vr1 = v_sun[0]
	vtheta1 = v_sun[1]
	vz1 = v_sun[2]

	# Add the global rotation speed to the deflector speed

	# logging.debug(vr, vtheta, vz, r)

	vhelio_r = vr1 * cosa - vtheta1 * sina
	vhelio_theta = vr1 * sina + vtheta1 * cosa
	vhelio_z = vz1

	# logging.debug(vhelio_r, vhelio_theta, vhelio_z)

	vgala_r = np.cos(b_s) * vhelio_r + np.sin(b_s) * vhelio_z
	vgala_theta = vhelio_theta
	vgala_phi = - np.sin(b_s) * vhelio_r + np.cos(b_s) * vhelio_z

	# logging.debug(vgala_r, vgala_theta, vgala_phi, theta*180/np.pi)
	return vgala_r, vgala_theta, vgala_phi


@nb.njit
def v_T_from(x, r_s, l_s, b_s, vr, vtheta, vz, vr_s, vtheta_s, vz_s, source_global_speed, deflector_global_speed):
	"""
	Compute effective defelctor transverse speed modulus in the projected plane orthogonal to the line of sight, at the deflector position,
	taking into accont the defelctor speed, the Sun speed and the source speed.
	Parameters
	----------
	x : float
		Ratio between distance to source and distance to deflector r_D/r_S
	r_s : float
		(parsec) Distance from Sun to source
	l_s : float
		(radians) Galactic longitude of the source
	b_s : float
		(radians) Galactic latitude of the source
	vr, vtheta, vz : float, float, float
		(km/s) Particular speed of the deflector, in galactocentric cylindrical coordinates
	vr_s, vtheta_s, vz_s : float, float, float
		(km/s) Particular speed of the source, in galactocentric cylindrical coordinates
	source_global_speed : function
		(njit) Function returning the theta global rotation speed of the source (in gal. cyl.) (bar_global_speed for bar, disk_global_speed for thin disk)
	deflector_global_speed : function
		(njit) Function returning the theta global rotation speed of the deflector (in gal. cyl.) (null_global_speed for halo, disk_global_speed for thick disk)

	Returns
	-------
	float
		Deflector transverse speed modulus
	"""
	_, tvsun_t, tvsun_z = sun_speed(x, r_s, l_s, b_s)
	_, tvsource_t, tvsource_z = source_speed(r_s, l_s, b_s, vr_s, vtheta_s, vz_s, source_global_speed)
	_, tdef_t, tdef_z = deflector_speed(x, r_s, l_s, b_s, vr, vtheta, vz, deflector_global_speed)
	vt = np.sqrt(
		(tdef_t - tvsun_t * (1 - x) - tvsource_t * x) ** 2 + (tdef_z - tvsun_z * (1 - x) - tvsource_z * x) ** 2)
	return vt


@nb.njit
def project_from(x, r_s, l_s, b_s, vr, vtheta, vz, vr_s, vtheta_s, vz_s, source_global_speed, deflector_global_speed):
	"""
	Compute deflector effective velocity vector in the projected plane orthogonal to the line of sight, at the deflector position,
	taking into accont the defelctor speed, the Sun speed and the source speed.
	Same parameters as v_T_from
	"""
	tvsun_r, tvsun_t, tvsun_z = sun_speed(x, r_s, l_s, b_s)
	tvsource_r, tvsource_t, tvsource_z = source_speed(r_s, l_s, b_s, vr_s, vtheta_s, vz_s, source_global_speed)
	tdef_r, tdef_t, tdef_z = deflector_speed(x, r_s, l_s, b_s, vr, vtheta, vz, deflector_global_speed)
	return (tdef_r - tvsun_r * (1 - x) - tvsource_r * x), (tdef_t - tvsun_t * (1 - x) - tvsource_t * x), (tdef_z - tvsun_z * (1 - x) - tvsource_z * x)


@nb.njit
def bar_matter_density(r_s, l_s, b_s):
	"""
	Compute matter density in M_sol/pc^-3 at l_s, b_s galactic coordinates, distance r_s
	Parameters
	----------
	r_s : float
		(parsec) Distance from Sun to source
	l_s : float
		(radians) Galactic longitude of the source
	b_s : float
		(radians) Galactic latitude of the source

	Returns
	-------
	Matter density in M_sol/pc^-3

	"""
	d1 = -r_s*np.cos(l_s)*np.cos(b_s) + d_sol
	d2 = r_s*np.sin(l_s)*np.cos(b_s)
	phi = np.arctan2(d2, d1) - PHI
	if phi == -PHI:
		r_B = d_sol - r_s
	else:
		r_B = np.sqrt(d1**2+d2**2)
	X = r_B*np.cos(phi)
	Y = -r_B*np.sin(phi)
	Z = r_s*np.sin(b_s)
	r2 = np.sqrt(((X/a)**2+(Y/b)**2)**2 + (Z/c)**4)
	return M_B/(6.57*np.pi*a*b*c)*np.exp(-r2/2.)


@nb.njit
def rho_halo_d(d):
	"""
	Matter density in M_sol/pc^-3 at distance d from Galactic Center

	Parameters
	----------
	d : float
		(parsec) Distance from the Galactic Center.
	"""
	return rho_0*(core_radius**2+d_sol**2)/(d**2+core_radius**2)


@nb.njit
def p_v_halo(vr, vtheta, vz):
	"""Particular speed vector probability distribution in halo"""
	v = np.sqrt(vr**2 + vtheta**2 + vz**2)
	return 4*np.pi*v**2 * np.power(2*np.pi*sigma_h**2, -3./2.) * np.exp(-v**2 /(2*sigma_h**2))


@nb.njit
def disk_matter_density(d, z, sigma, H, R):
	"""
	Disk dark matter density in M_sol/pc^-3
	Parameters
	----------
	d : float
		(parsec) Distance from the Galactic Center.
	z : float
		(parsec) Height from galactic plane
	sigma : float
		(M_sol/pc^-2) Disk column density parameter
	H : float
		(parsec) Characteristic height of the disk
	R : float
		(parsec) Characteristic radius of the disk

	Returns
	-------

	"""
	return sigma/2/H * np.exp(-(d-d_sol)/R) * np.exp(-np.abs(z)/H)


@nb.njit
def p_v_disk(vr, vtheta, vz, sig_r, sig_theta, sig_z):
	"""Particular speed vector probability distribution in disk"""
	return gaussian(vr, 0, sig_r)*gaussian(vtheta, 0, sig_theta)*gaussian(vz, 0, sig_z)


angular_limits = 10


@nb.njit
def pdf_xvs_GC_halo(vec, params):
	"""Disk geometry probabilty density function, sources toward GC
	"""
	x, vr, vtheta, vz, vr_s, vtheta_s, vz_s, r_s, l_s, b_s, pop = vec
	sigma_bar, sigma_thin, H_thin, R_thin, sig_r_thin, sig_theta_thin, sig_z_thin = params
	if x < 0 or x > 1 or l_s < -angular_limits * np.pi / 180 or l_s > angular_limits * np.pi / 180 or b_s < -angular_limits * np.pi / 180 or b_s > angular_limits * np.pi / 180 or r_s < 0 or r_s > 50000:
		return 0.  # x should be in [0, 1]
	z_sol = 26
	z = np.sqrt((x * r_s * np.sin(b_s)) ** 2 + z_sol ** 2)
	r = np.sqrt((x * r_s * np.cos(b_s) * np.cos(l_s) - d_sol) ** 2 + (x * r_s * np.cos(b_s) * np.sin(l_s)) ** 2)
	if pop > 0.5:
		# bar source
		return (np.sqrt(x * (1 - x))
				* p_v_halo(vr, vtheta, vz)
				* rho_halo_d(np.sqrt(r ** 2 + z ** 2))
				* np.abs(v_T_from(x, r_s, l_s, b_s, vr, vtheta, vz, vr_s, vtheta_s, vz_s, bar_global_speed, null_global_speed))
				* bar_matter_density(r_s, l_s, b_s)
				* p_v_bar(vr_s, vtheta_s, vz_s, sigma_bar))
	else:
		# thin disk source
		return (np.sqrt(x * (1 - x))
				* p_v_halo(vr, vtheta, vz)
				* rho_halo_d(np.sqrt(r ** 2 + z ** 2))
				* np.abs(v_T_from(x, r_s, l_s, b_s, vr, vtheta, vz, vr_s, vtheta_s, vz_s, disk_global_speed, null_global_speed))
				* disk_matter_density(r, z, sigma_thin, H_thin, R_thin)
				* p_v_disk(vr_s, vtheta_s, vz_s, sig_r_thin, sig_theta_thin, sig_z_thin))
