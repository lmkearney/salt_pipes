import numpy as np
import matplotlib.pyplot as plt

Oppo_data = np.loadtxt('./Levant_data/Oppo_data.csv', delimiter=',', dtype='str', skiprows=1)
trails = list(Oppo_data.T)

pipedict = {}
for i in range(len(trails)):
    trails[i] = np.array(trails[i][trails[i] != '']).astype(np.float)
    pipedict[i+1] = trails[i]
    
Oceanus_data = np.loadtxt('./Levant_data/Oceanus_data.csv', delimiter=',', dtype='str', skiprows=1)
pipedict[0] = Oceanus_data.astype('float')
