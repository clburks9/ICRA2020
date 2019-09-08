
import numpy as np; 
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind as ttest
from scipy.stats import ttest_rel as tRel


def checkData():
	print("Remember to cull and replace bad runs where they started on top of each other")


	#golfRun = ['dataGolf_E1_B.npy','dataGolfHumanDirs_E1_B.npy','dataGolf_E1_Perfect.npy']; 
	golfRun = ['dataGolf_E1_D.npy','dataGolfHumanDirs_E1_D.npy','dataGolf_E1_Perfect-RunD.npy']; 
	

	allCatchNames = []; 
	allQueryNames = []; 
	allExceptions = [0,0,0]; 
	for name in golfRun:
		#fileData = np.load('/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/Data/data{}_E1_{}.npy'.format(n,run),allow_pickle=True,encoding='latin1').item(); 

		fileData = np.load('/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/Data/{}'.format(name),allow_pickle=True,encoding='latin1').item(); 

		meta = fileData['Meta']; 
		data = fileData['Data']; 
		#print(meta);

		allCatch = []; 
		allQueries = []; 


		for run in data:
			rew = run['Rewards']; 
			info = np.array(run['TreeInfo']); 
			#print(info[0])
			for i in range(0,len(info)):
				allQueries.append(info[i]['Tree Queries']); 

			#allQueries.extend(info[:]['Tree Queries']); 
			for i in range(0,len(rew)):	
				if(rew[i]==1):
					if(i<3):
						allExceptions[golfRun.index(name)] += 1; 
						break; 
					allCatch.append(i);
					break; 
				# if(i==99):
				# 	allCatch.append(100);


		print(name,100*(len(allCatch)/max(1,100-allExceptions[golfRun.index(name)])),np.mean(allCatch),np.std(allCatch));
		allCatchNames.append(np.array(allCatch)); 
		allQueryNames.append(np.array(allQueries)); 
		# print(data[3]['States'][0]); 


	allCatchNames = np.array(allCatchNames);
	allQueryNames = np.array(allQueryNames)

	allStats = np.zeros(shape=(3,3)); 
	for i in range(0,3):
		for j in range(0,i):
			t,p = ttest(allCatchNames[i],allCatchNames[j],nan_policy='omit'); 
			allStats[i,j] = p; 
	np.set_printoptions(precision=6,suppress=True); 
	print(allStats)

	# for i in range(0,len(golfRun)):
	# 	print(golfRun[i],np.mean(allQueryNames[i]),np.std(allQueryNames[i])); 
	#print(allQueryNames[1].tolist())


def estimateGraph():
	golfRun = ['dataGolf_E1_D.npy','dataGolfHumanDirs_E1_D.npy','dataGolf_E1_Perfect-RunD.npy']; 
	
	allErrors = np.zeros(shape=(3,100,101));  
	allSD = np.zeros(shape=(3,100,101)); 


	for name in golfRun:
		#fileData = np.load('/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/Data/data{}_E1_{}.npy'.format(n,run),allow_pickle=True,encoding='latin1').item(); 

		fileData = np.load('/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/Data/{}'.format(name),allow_pickle=True,encoding='latin1').item(); 

		meta = fileData['Meta']; 
		data = fileData['Data']; 

		error = []; 

		for run in data:
			states = run['States']; 
			bel = run['Beliefs'] 
			#print(bel[0])
			for i in range(0,len(states)):
				#error.append(np.mean(np.square(np.array(states[i])[2:3]-np.array(bel[i][0][2:3]))))
				error = np.sqrt(np.mean((np.array(states[i])[2:4] - np.array(bel[i][0]))**2))
				#print(len(error)); 

				#a = 1/0; 
				allErrors[golfRun.index(name),data.index(run),i] += error; 
				allSD[golfRun.index(name),data.index(run),i] += np.sqrt(bel[i][1][0]**2 + bel[i][1][1]**2); 
	meanErrors = np.zeros(shape=(3,101));
	meanSD = np.zeros(shape=(3,101));  
	for n in range(0,3):
		for i in range(0,100):
			for j in range(0,101):
				meanErrors[n,j] += allErrors[n,i,j]/100
				meanSD[n,j] += allSD[n,i,j]/100; 

	x = [i for i in range(0,101)]; 
	plt.plot(x,meanErrors[0],c='r',label='Non-Human',linewidth=3); 
	#plt.fill_between(x,meanErrors[0]+meanSD[0],meanErrors[0]-meanSD[0],color='r',alpha=0.25)
	plt.plot(x,meanErrors[1],c='g',label='Human',linewidth=3); 
	#plt.fill_between(x,meanErrors[1]+meanSD[1],meanErrors[1]-meanSD[1],color='g',alpha=0.25)
	plt.plot(x,meanErrors[2],c='m',label='Non-Human Prior Knowledge',linewidth=3)
	#plt.fill_between(x,meanErrors[2]+meanSD[2],meanErrors[2]-meanSD[2],color='b',alpha=0.25)
	#plt.ylim([0,500])
	plt.axvline(18.64,c='g',linestyle='--')
	plt.axvline(31.86,c='r',linestyle='--')
	plt.axvline(28.49,c='m',linestyle='--')

	plt.legend(); 
	plt.title("Average RMSE"); 
	plt.xlabel("Timestep"); 
	plt.ylabel("Error (m)"); 
	plt.show()



def heatMap():
	golfRun = ['dataGolf_E1_D.npy','dataGolfHumanDirs_E1_D.npy','dataGolf_E1_Perfect-RunD.npy']; 

	heatMaps = np.zeros(shape=(3,83,83)); 

	for name in golfRun:
		#fileData = np.load('/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/Data/data{}_E1_{}.npy'.format(n,run),allow_pickle=True,encoding='latin1').item(); 

		fileData = np.load('/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/Data/{}'.format(name),allow_pickle=True,encoding='latin1').item(); 

		meta = fileData['Meta']; 
		data = fileData['Data']; 

		error = []; 

		for run in data:
			states = run['States']; 
			#print(states[0]); 
			#print(bel[0])
			for i in range(0,len(states)):
				heatMaps[golfRun.index(name),int(states[i][0]/10),int(states[i][1]/10)] += 1; 

	#print(sum(sum(heatMaps[0])))
	print(np.amax(heatMaps[0]))

	fig,axarr = plt.subplots(1,3); 
	for i in range(0,3):
		axarr[i].imshow(heatMaps[i],origin='lower',vmax=24,vmin=0); 
	plt.show(); 

def questions():
	

if __name__ == '__main__':
	
	print("Graphs/Charts/Figures:")
	print("1. Table of results, first capture times")
	print("2. Estimation Error")
	print("3. Question Analysis"); 



	#checkData(); 
	#estimateGraph(); 
	#heatMap(); 
	questions(); 