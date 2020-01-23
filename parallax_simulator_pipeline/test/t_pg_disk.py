import parallax_simulator_pipeline.pg_disk as pgd
import numpy as np
import time

print(pgd.compute_i())


st1 = time.time()
s_thick = pgd.metropolis_hastings(pgd.pdf_xvs_disk, pgd.randomize_gauss_total_hardcoded, 10000, np.array([0.9, 10., 10., 10.]), (56.1, 46.1, 35.1, 35, 1000., 35000.))
print(time.time()-st1)

print(s_thick)
thetas = pgd.compute_thetas(*s_thick[:, 1:].T, s_thick[:, 0])
print(thetas)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='polar')
i = pgd.compute_i()
ax.set_theta_zero_location("E", offset=np.arctan2(i[2], i[1])*180./np.pi)
ax.hist(thetas, bins=100)
ax.grid(True)
x = np.linspace(0, 1, 10)
plt.show()


mlg = pgd.MicrolensingGenerator(xvt_file=10000000, seed=1234567, tmin=48928., tmax=52697., u_max=2.,  max_blend=0., min_blend=0.)
pms = mlg.generate_parameters(mass=10., seed=1234567, nb_parameters=1000)
print(pms)