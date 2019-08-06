import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm

# data = np.load('../data/dataUnified_E1_Test.npy',allow_pickle=True,encoding='latin1').item(); 
# #print(data['Data'][0]['Beliefs'])

# bels = data['Data'][0]['Beliefs']; 
# #print(len(bels));
# a = np.array(bels); 
# x = [i for i in range(0,len(a))]; 
# plt.plot(x,a[:,0,0],c='b'); 
# plt.plot(x,a[:,0,1],c='r'); 
# plt.show();


# qs = data['Data'][0]['TreeInfo'][0]['Execution Time'];
# print(qs)
# plt.plot(qs); 
# plt.show();

def E1DirectionalNormalPlots(save = False):

	names = ['Dipole','Unified','Vectored']
	colors = ['r','b','g']; 

	fig,axarr = plt.subplots(3,sharex=True); 
	for n in names:
		fileData = np.load('/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/Data/data{}_E1_A.npy'.format(n),allow_pickle=True,encoding='latin1').item(); 

		rewards = np.array([fileData['Data'][i]['Rewards'] for i in range(0,len(fileData['Data']))]);

		sumRew = [sum(rewards[i]) for i in range(0,len(rewards))]
		print(n); 
		print(np.mean(sumRew),np.std(sumRew)); 
		print(""); 

		axarr[names.index(n)].hist(sumRew,bins=10,density=True,color=colors[names.index(n)]);
		mu,std = norm.fit(sumRew); 
		axarr[names.index(n)].set_xlim([100,800]); 
		xmin,xmax = axarr[names.index(n)].get_xlim(); 
		x = np.linspace(xmin,xmax,100); 
		p = norm.pdf(x,mu,std); 
		axarr[names.index(n)].plot(x,p,'k',linewidth=2); 
		axarr[names.index(n)].axvline(np.mean(sumRew),c='k',linestyle='--'); 
		
		axarr[names.index(n)].set_ylabel(n); 
	axarr[2].set_xlabel('Final Reward'); 
	#axarr[1].set_ylabel('Frequency'); 
	if(save):
		plt.savefig("/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/img/dataE1Norms.png")
	else:
		plt.show();


def E1DirectionalFirstCatchPlots(save=False):

	names = ['Dipole','Unified','Vectored']
	colors = ['r','b','g']; 

	fig,axarr = plt.subplots(3,sharex=True); 
	for n in names:
		fileData = np.load('/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/Data/data{}_E1_A.npy'.format(n),allow_pickle=True,encoding='latin1').item(); 

		rewards = np.array([fileData['Data'][i]['Rewards'] for i in range(0,len(fileData['Data']))]);

		firstReward = []; 
		for r in rewards:
			for i in range(0,len(r)):
				if(r[i] != 0):
					firstReward.append(i); 
					break; 
		axarr[names.index(n)].hist(firstReward,color=colors[names.index(n)]);
		axarr[names.index(n)].axvline(np.mean(firstReward),c='k',linestyle='--'); 
		print(n); 
		print(np.mean(firstReward),np.std(firstReward)); 
		print("");

	plt.show();

if __name__ == '__main__':
	#E1DirectionalNormalPlots(save=False);
	E1DirectionalFirstCatchPlots()