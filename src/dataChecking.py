
import numpy as np; 
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind as ttest
from scipy.stats import ttest_rel as tRel
import matplotlib.colors as mcolors
import sys
sys.path.append("../GridExperiments/common"); 
from roadNode import RoadNode

def checkData():
	print("Remember to cull and replace bad runs where they started on top of each other")


	#golfRun = ['dataGolf_E1_B.npy','dataGolfHumanDirs_E1_B.npy','dataGolf_E1_Perfect.npy']; 
	#golfRun = ['dataGolf_E1_E.npy','dataGolfHumanFactored_E1_E.npy','dataGolf_E1_Perfect-RunE.npy']; 
	golfRun = ['dataGridMode_E2_A.npy','dataGridMQuest_E2_A.npy']; 

	allCatchNames = []; 
	allQueryNames = []; 
	allExceptions = [0,0,0,0]; 
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
					if(i<4):
						allExceptions[golfRun.index(name)] += 1; 
						#print(i)
						break; 
					allCatch.append(i);
					break; 
				# if(i==99):
				# 	allCatch.append(100);

		
		#print(len(allCatch)/26)
		print(name,100*(len(allCatch)/max(1,100-allExceptions[golfRun.index(name)])),np.mean(allCatch),np.std(allCatch));
		allCatchNames.append(np.array(allCatch)); 
		allQueryNames.append(np.array(allQueries)); 
		# print(data[3]['States'][0]); 


	allCatchNames = np.array(allCatchNames);
	allQueryNames = np.array(allQueryNames)

	allStats = np.zeros(shape=(2,2)); 
	for i in range(0,2):
		for j in range(0,i):
			t,p = ttest(allCatchNames[i],allCatchNames[j],nan_policy='omit'); 
			allStats[i,j] = p; 
	np.set_printoptions(precision=6,suppress=True); 
	print(allStats)

	# for i in range(0,len(golfRun)):
	# 	print(golfRun[i],np.mean(allQueryNames[i]),np.std(allQueryNames[i])); 
	#print(allQueryNames[1].tolist())


def estimateGraph():
	#golfRun = ['dataGolf_E1_D.npy','dataGolfHumanDirs_E1_D.npy','dataGolf_E1_Perfect-RunD.npy']; 
	golfRun = ['dataGolf_E1_E.npy','dataGolfHumanFactored_E1_E.npy','dataGolf_E1_Perfect-RunE.npy']; 
	#golfRun = ['dataGolf_E1_E.npy','dataGolfHumanDirs_E1_E.npy','dataGolf_E1_Perfect-RunE.npy']; 
	
	allErrors = np.zeros(shape=(3,100,101));  
	allSD = np.zeros(shape=(3,100,101)); 

	print("You're looking at the estimation plots. Keep in mind, the capture radius here is 75 meters, so measurements can't actually get better than that for the most part")

	for name in golfRun:
		#fileData = np.load('/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/Data/data{}_E1_{}.npy'.format(n,run),allow_pickle=True,encoding='latin1').item(); 

		fileData = np.load('/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/Data/{}'.format(name),allow_pickle=True,encoding='latin1').item(); 

		meta = fileData['Meta']; 
		data = fileData['Data']; 

		error = []; 

		for run in data:
			states = run['States']; 
			bel = run['Beliefs'] 
			rew = run['Rewards']

			#print(bel[0])

			for i in range(0,len(states)):
				#error.append(np.mean(np.square(np.array(states[i])[2:3]-np.array(bel[i][0][2:3]))))
				error = np.sqrt(np.mean((np.array(states[i])[2:4] - np.array(bel[i][0]))**2))
				#print(len(error)); 

				#a = 1/0; 
				allErrors[golfRun.index(name),data.index(run),i] += error; 
				allSD[golfRun.index(name),data.index(run),i] += np.sqrt(bel[i][1][0]**2 + bel[i][1][1]**2); 
				
				# if(i<100 and rew[i] == 1):
				# 	break; 
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
	plt.axvline(25.47,c='g',linestyle='--')
	plt.axvline(34.16,c='r',linestyle='--')
	plt.axvline(31.22,c='m',linestyle='--')
	#plt.xlim([0,50])

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
	#golfRun = ['dataGolf_E1_D.npy','dataGolfHumanDirs_E1_D.npy','dataGolf_E1_Perfect-RunD.npy']; 
	golfRun = ['dataGolf_E1_E.npy','dataGolfHumanFactored_E1_E.npy','dataGolf_E1_Perfect-RunE.npy']; 

	heatMaps = np.zeros(shape=(3,83,83)); 

	for name in golfRun:
		#fileData = np.load('/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/Data/data{}_E1_{}.npy'.format(n,run),allow_pickle=True,encoding='latin1').item(); 

		fileData = np.load('/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/Data/{}'.format(name),allow_pickle=True,encoding='latin1').item(); 

		meta = fileData['Meta']; 
		data = fileData['Data']; 

		error = []; 

		totalActions = 0; 
		totalQuestions = 0; 

		for run in data:
			acts = run['Actions']; 
			for a in acts:
				if(a>=4):
					totalQuestions += 1; 
				totalActions += 1; 

		print(name, totalActions, totalQuestions, totalQuestions/totalActions)


def resAcc(scen='E',save=False):

	accs = [.3,.5,.7,.9,.95,.99]; 
	ress = [.3,.5,.7,.9,.95,.99]; 

	allRatios = np.zeros(shape=(6,6)); 

	for ac in accs:
		for re in ress:
			fileData = np.load('/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/Data/dataGolfScan_{}_{}_{}.npy'.format(ac,re,scen),allow_pickle=True,encoding='latin1').item(); 
			data = fileData['Data']; 


			totalActions = 0; 
			totalQuestions = 0; 

			for run in data:
				acts = run['Actions']; 
				for a in acts:
					if(a>=4):
						totalQuestions += 1; 
					totalActions += 1; 

			#print(ac,re, totalActions, totalQuestions, totalQuestions/totalActions)
			allRatios[accs.index(ac),ress.index(re)] = totalQuestions/totalActions; 

	plt.figure();
	plt.imshow(allRatios,origin='lower',cmap='inferno'); 
	plt.xlabel("Accuracy"); 
	plt.ylabel("Responsivity");
	plt.xticks([0,1,2,3,4,5],[.3,.5,.7,.9,.95,.99])
	plt.yticks([0,1,2,3,4,5],[.3,.5,.7,.9,.95,.99])
	plt.colorbar();
	plt.title("Percentage of Questions asked by Robot given Different Humans")
	#plt.show()
	if(save):
		plt.savefig('/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/img/BigAccRes_{}.png'.format(scen))
	else:
		plt.show();


	############
	#This block for the bar graphs
	############
	# fig,axarr = plt.subplots(2,2,sharex=True,sharey=True); 
	# axarr[0][0].imshow(allRatios,origin='lower'); 
	# axarr[0][0].set_xlabel("Accuracy"); 
	# #locs = axarr[0][0].get_xticks();
	# axarr[0][0].set_xticklabels([0,.3,.5,.7,.9,.95,.99])
	# #axarr[0][0].set_xticklabels(accs); 
	# axarr[0][0].set_ylabel("Responsivity"); 
	# axarr[0][0].set_yticklabels([0,.3,.5,.7,.9,.95,.99]); 

	# #print(np.transpose(np.sum(allRatios,axis=1)))

	# tmp = np.sum(allRatios,axis=1); 
	# tmp -= min(tmp); 
	# tmp /= sum(tmp); 

	# clist = [(0, "darkblue"),(0.1,'blue'),(0.15,'teal'), (0.2, "green"), (.4, "yellow"),(1,"yellow")]
	# rvb = mcolors.LinearSegmentedColormap.from_list("", clist)



	# axarr[0][1].barh([0,1,2,3,4,5],np.sum(allRatios,axis=1),edgecolor='k',color=rvb(tmp))

	# tmp = np.sum(allRatios,axis=0); 
	# tmp -= min(tmp); 
	# tmp /= sum(tmp); 
	# print(tmp);

	
	# clist = [(0, "darkblue"),(0.15,'blue'),(0.2,'teal'), (0.22, "green"), (.25, "yellow"),(1,"yellow")]
	# rvb = mcolors.LinearSegmentedColormap.from_list("", clist)
	# axarr[1][0].bar([0,1,2,3,4,5],np.sum(allRatios,axis=0),edgecolor='k',color=rvb(tmp))
	
	# axarr[1][0].set_xlabel("Accuracy"); 
	# fig.suptitle("Percentage of Questions asked by Robot given different Humans"); 
	

	# plt.show();


	allErrors = np.zeros(shape=(6,6,50,51));  
	#allSD = np.zeros(shape=(36,100,101)); 

	accs = [.3,.5,.7,.9,.95,.99]; 
	ress = [.3,.5,.7,.9,.95,.99]; 


	for ac in accs:
		for re in ress:
			fileData = np.load('/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/Data/dataGolfScan_{}_{}_{}.npy'.format(ac,re,scen),allow_pickle=True,encoding='latin1').item(); 
			meta = fileData['Meta']; 
			data = fileData['Data']; 

			error = []; 

			for run in data:
				states = run['States']; 
				bel = run['Beliefs'] 
				rew = run['Rewards']

				#print(bel[0])

				for i in range(0,len(states)):
					#error.append(np.mean(np.square(np.array(states[i])[2:3]-np.array(bel[i][0][2:3]))))
					error = np.sqrt(np.mean((np.array(states[i])[2:4] - np.array(bel[i][0]))**2))
					#print(len(error)); 

					#a = 1/0; 
					allErrors[accs.index(ac),ress.index(re),data.index(run),i] += error; 
					#allSD[golfRun.index(name),data.index(run),i] += np.sqrt(bel[i][1][0]**2 + bel[i][1][1]**2); 
					
					# if(i<100 and rew[i] == 1):
					# 	break; 

	meanErrors = np.zeros(shape=(6,6,51));
	#meanSD = np.zeros(shape=(6,6,51));  
	for n in range(0,6):
		for k in range(0,6):
			for i in range(0,50):
				for j in range(0,51):
					meanErrors[n,k,j] += allErrors[n,k,i,j]/50
	
	markers = ['o','v','P','X','d','*']
	x = [i for i in range(0,51)]; 
	fig,axarr = plt.subplots(1,2,sharey=True); 
	for i in range(0,6):
		for j in range(0,6):
			#axarr[0].plot(x,meanErrors[i,j],c=[1-i/5,i/5,0],linewidth=1,marker = markers[j],markersize=7,markerfacecolor='black'); 
			axarr[0].plot(x,meanErrors[i,j],c=[1-i/5,i/5,0],linewidth=2); 
			axarr[1].plot(x,meanErrors[i,j],c=[1-j/5,j/5,0],linewidth=2); 
	axarr[0].set_title("Error According To Accuracy"); 
	axarr[1].set_title("Error According To Responsivity"); 
	axarr[0].set_xlabel("Timestep"); 
	axarr[1].set_xlabel("Timestep"); 
	axarr[0].set_ylabel("Average RMSE")
	if(save):
		plt.savefig('/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/img/ErrorAccRes_{}.png'.format(scen))
	else:
		plt.show();




	accs = [.3,.5,.7,.9,.95,.99]; 
	ress = [.3,.5,.7,.9,.95,.99]; 

	allCatchAverages = np.zeros(shape=(6,6)); 
	
	
	for ac in accs:
		for re in ress:
			fileData = np.load('/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/Data/dataGolfScan_{}_{}_{}.npy'.format(ac,re,scen),allow_pickle=True,encoding='latin1').item(); 
			meta = fileData['Meta']; 
			data = fileData['Data']; 

			#print(meta);

			#allCatch = []; 


			for run in data:
				rew = run['Rewards']; 
				

				for i in range(0,len(rew)):	
					if(rew[i]==1):
						allCatchAverages[accs.index(ac),ress.index(re)] += i/50; 
						break; 
					if(i==49):
						allCatchAverages[accs.index(ac),ress.index(re)] += 1; 


	print(allCatchAverages[np.where(allCatchAverages>34.16)].shape); 

	plt.figure(); 
	plt.imshow(allCatchAverages,origin='lower'); 
	plt.xlabel("Accuracy"); 
	plt.ylabel("Responsivity");
	plt.xticks([0,1,2,3,4,5],[.3,.5,.7,.9,.95,.99])
	plt.yticks([0,1,2,3,4,5],[.3,.5,.7,.9,.95,.99])
	plt.colorbar();
	plt.title("Average Time to Capture for Different Humans")
	

	if(save):
		plt.savefig('/mnt/c/Users/clbur/OneDrive/Work Docs/Projects/HARPS 2019/img/TTCAccRes_{}.png'.format(scen))
	else:
		plt.show();


if __name__ == '__main__':
	
	print("Graphs/Charts/Figures:")
	print("1. Table of results, first capture times")
	print("2. Estimation Error")
	print("3. Question Analysis"); 



	checkData(); 
	#estimateGraph(); 
	#heatMap(); 
	#questions(); 

	#resAcc(scen='E',save=True); 