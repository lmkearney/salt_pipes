import numpy as np
import matplotlib.pyplot as plt

sand = (0.9, 0.85, 0.0)
mud = (0.66, 0.41, 0.21)

data = np.loadtxt('./Levant_data/Oceanus_fold.csv', delimiter=',', skiprows=1)
x = data[:,0]-data[:,0][0]
sand_horizon = data[:,1]-data[:,1][0]
diff_sand = np.zeros(len(x))
for i in range(1,len(x)):
    diff_sand[i] = np.sqrt( (x[i]-x[i-1])**2 + (sand_horizon[i]-sand_horizon[i-1])**2 )

length_sand = 0.
length_now = x[-1]-x[0]
for i in range(len(x)):
    length_sand += diff_sand[i]

strain = -(length_now-length_sand)/length_sand

dx = length_sand-length_now

shift = 0.065
plt.figure(figsize=(7.5,2))
plt.plot([x[0]+shift, x[-1] + dx + shift], [0.36, 0.36], c='darkgrey', label='initial', lw=1.25)
plt.plot(x+dx/2.+shift, sand_horizon+0.05, label='present-day', c='royalblue', lw=1.25)

plt.scatter([x[0]+shift, x[-1]+dx+shift], [0.36, 0.36], c='darkgrey', s=20., marker='|')
plt.scatter([x[0]+dx/2.+shift, x[-1]+dx/2.+shift], [sand_horizon[0]+0.05, sand_horizon[-1]+0.05], c='royalblue', s=20., marker='|')

plt.xlim([0, 8])
plt.ylim([0., 1])
plt.xlabel('horizontal distance (km)')
plt.ylabel('vertical distance (km)')
plt.legend(frameon=False, ncol=2, bbox_to_anchor=(1.0, 1.4))
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()
