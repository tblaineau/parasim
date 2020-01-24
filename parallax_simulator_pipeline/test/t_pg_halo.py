import parallax_simulator_pipeline.pg_halo as pgh
import numpy as np
import matplotlib.pyplot as plt
import time

st1 = time.time()
s_halo = pgh.metropolis_hastings(pgh.pdf_xvs_halo, pgh.randomize_gauss_halo_hardcoded, 1000000, np.array([0.2, 100., 100., 100.]))
print(time.time()-st1)

thetas = pgh.compute_thetas(*s_halo[:, 1:].T, s_halo[:, 0])

plt.hist(s_halo[:,0], bins=100)
plt.show()

plt.hist(s_halo[:,1:], bins=100, histtype='step')
plt.show()

vts = pgh.vt_from_vs(*s_halo[:,1:].T, s_halo[:,0])
plt.hist(vts, bins=100)
plt.show()

dus = pgh.delta_u_from_x(s_halo[:,0], 10.)
tEs = pgh.tE_from_xvt(s_halo[:,0], vts, 10.)

plt.hist2d(dus, tEs, bins=300, range=((0, 0.1), (0, 1000)))
plt.show()

fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='polar')
i = pgh.compute_i()
ax.set_theta_zero_location("E", offset=np.arctan2(i[2], i[1])*180./np.pi)
xs = np.linspace(0, 1, 10)
cm1 = plt.get_cmap("Blues")
for ci in range(len(xs)-1):
	ax.hist(thetas[(s_halo[:,0]>xs[ci]) & (s_halo[:,0]<xs[ci+1])], bins=100, density=True, histtype='step', color=cm1(xs[ci+1]))
theta_lmc = pgh.angle(i, pgh.v_lmc)
#ax.plot([0, theta_lmc], [0, 10000])
#ax.plot([0, pgh.angle(i, pgh.v_sun)], [0, 10000])
ax.grid(True)
x = np.linspace(0, 1, 10)
plt.show()
