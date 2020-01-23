import parallax_simulator_pipeline.pg_disk as pgd
import numpy as np
import matplotlib.pyplot as plt
import time

print(pgd.compute_i())

"""
st1 = time.time()
s_thick = pgd.metropolis_hastings(pgd.pdf_xvs_disk, pgd.randomize_gauss_total_hardcoded, 10000, np.array([0.9, 10., 10., 10.]), (56.1, 46.1, 35.1, 35, 1000., 35000.))
print(time.time()-st1)

print(s_thick)
thetas = pgd.compute_thetas(*s_thick[:, 1:].T, s_thick[:, 0])
print(thetas)

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='polar')
i = pgd.compute_i()
ax.set_theta_zero_location("E", offset=np.arctan2(i[2], i[1])*180./np.pi)
ax.hist(thetas, bins=100)
ax.grid(True)
x = np.linspace(0, 1, 10)
plt.show()"""


"""mlg = pgd.MicrolensingGenerator(xvt_file=10000000, seed=1234567, tmin=48928., tmax=52697., u_max=2.,  max_blend=0., min_blend=0.)
pms = mlg.generate_parameters(mass=10., seed=1234567, nb_parameters=100000)
print(pms)

np.save("temp.npy", pms)"""

pms = np.load("temp.npy", allow_pickle=True)[()]

print(pms['mass'][0])

plt.hist(pms['tE'], bins=100, range=(0, 250))
plt.xlabel(r'$t_E$')
plt.show()

plt.hist(pms['delta_u'], bins=100, range=(0, 0.2))
plt.xlabel(r"$\pi_E$")
plt.show()

plt.hist(pms['x'], bins=300, range=(0, 1))
plt.xlabel(r"x")
plt.show()

plt.hist(pms['vt'], bins=100, range=(0, 250))
plt.xlabel(r"$v_R$")
plt.show()

plt.hist(pms['u0'], bins=100)
plt.xlabel(r"$u_O$")
plt.show()

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='polar')
i = pgd.compute_i()
ax.set_theta_zero_location("E", offset=np.arctan2(i[2], i[1])*180./np.pi)
ax.hist(pms["theta"], bins=100)
ax.grid(True)
x = np.linspace(0, 1, 10)
plt.show()