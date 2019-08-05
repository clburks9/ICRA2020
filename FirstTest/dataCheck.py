import numpy as np
import matplotlib.pyplot as plt 

data = np.load('../data/dataUnified_E1_Test.npy',allow_pickle=True,encoding='latin1').item(); 
#print(data['Data'][0]['Beliefs'])

bels = data['Data'][0]['Beliefs']; 
#print(len(bels));
a = np.array(bels); 
x = [i for i in range(0,len(a))]; 
plt.plot(x,a[:,0,0],c='b'); 
plt.plot(x,a[:,0,1],c='r'); 
plt.show();


qs = data['Data'][0]['TreeInfo'][0]['Execution Time'];
print(qs)
plt.plot(qs); 
plt.show();
