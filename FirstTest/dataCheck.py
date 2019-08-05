import numpy as np


data = np.load('unifiedPOMCP_Test.npy'); 
for key in data[0].keys():
	print(key,len(data[0][key])); 