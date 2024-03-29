import warnings
warnings.simplefilter("ignore")

import sys
sys.path.append('../../src'); 
from treeNode import Node
import numpy as np; 
#from testProblemSpec import *;
from gridMQuestSpec import *; 
sys.path.append('../common'); 
from roadNode import *;

import matplotlib.pyplot as plt

import cProfile
import time; 
#from gaussianMixtures import GM,Gaussian
#from softmaxModels import Softmax;
from copy import deepcopy 
import random
import time


class POMCP:

	def __init__(self):
	 	pass;


	def simulate(self,s,h,depth):
		
		#check if node is in tree
		#if not, add nodes for each action
		if(depth <= 0):
			return 0; 

		h.data.append(s); 

		if(not h.hasChildren()):
			for a in range(0,numActs):
				#self.addActionNode(h,a); 
				h.addChildID(a); 


		#find best action acccording to c
		act = np.argmax([ha.Q + c*np.sqrt(np.log(h.N)/ha.N) for ha in h]); 

		#generate s,o,r
		sprime = generate_s(s,act); 
		o = generate_o(sprime,act); 
		r = generate_r(s,act); 


		#if o not in ha.children
			#add it and estimate value
		#else recurse 
		if(o not in h[act].getChildrenIDs()):
			h[act].addChildID(o); 
			return estimate_value(s,h[act]); 
			#return rollout(s,depth); 
		
		if(isTerminal(s,act)):
			return r

		q = r + gamma*self.simulate(sprime,h[act].getChildByID(o),depth-1); 
		
		#update node values
		h.N += 1; 
		h[act].N += 1; 
		h[act].Q += (q-h[act].Q)/h[act].N; 
		

		return q; 


	def resampleSet(h,sSet):
		if(len(sSet)>=maxTreeQueries):
			return sSet; 

		while(len(sSet) < maxTreeQueries):
			ind = np.random.randint(0,len(sSet)); 
			tmp = deepcopy(sSet[ind]);
			#tmp[2] += np.random.normal(0,.005);  
			#tmp[3] += np.random.normal(0,.005);
			tmp[2] += (tmp[5].loc[0]-tmp[4].loc[0])*np.random.random()*.25 # + np.random.normal(0,dev); 
			tmp[3] += (tmp[5].loc[1]-tmp[4].loc[1])*np.random.random()*.25

			sSet.append(tmp); 
		return sSet

	def search(self,b,h,depth,inform = False):
		#Note: You can do more proper analytical updates if you sample during runtime
		#but it's much faster if you pay the sampling price beforehand. 
		#TLDR: You need to change this before actually using
		#print("Check your sampling before using this in production")

		#sSet = b.sample(maxTreeQueries); 
		sSet = b; 
		count = 0; 


		startTime = time.clock(); 

		while(time.clock()-startTime < maxTime and count < maxTreeQueries):
			#print(time.clock()-startTime)
			s = sSet[count]; 
			#s = b.sample(1)[0]
			count += 1; 
			#s = [-2,0]; 
			#s = b.sample(1)[0]; 
			self.simulate(s,h,depth); 
		if(inform):
			info = {"Execution Time":0,"Tree Queries":0,"Tree Size":0}
			info['Execution Time'] = time.clock()-startTime; 
			info['Tree Queries'] = count; 
			#info['Tree Size'] = len(h.traverse()); 
			return np.argmax([a.Q for a in h]),info; 
			#print([a.Q for a in h])
		else:
			return np.argmax([a.Q for a in h]); 



def propogateAndMeasure(sSet,act,o):
	
	sSetPrime = []; 

	for s in sSet:
		sSetPrime.append(generate_s(s,act)); 

	origLen = len(sSetPrime); 

	s = np.array(sSetPrime); 
	
	weights = [0 for i in range(0,len(s))]; 
	for i in range(0,len(s)):
		weights[i] = obs_weight(s[i],act,o); 
		

	weights /= np.sum(weights); 

	csum = np.cumsum(weights); 
	csum[-1] = 1; 

	indexes = np.searchsorted(csum,np.random.random(len(s))); 
	s[:] = s[indexes]; 
	

	#print(s)

	return s


def correctAgentLoc(trueS,sSet):
	for s in sSet:
		s[0] = trueS[0]; 
		s[1] = trueS[1]; 

def runSims(sims = 10,steps = 10,verbosity = 2,simIdent = 'Test'):

	#set up data collection
	dataPackage = {'Meta':{'NumActs':numActs,'maxDepth':maxDepth,'c':c,'maxTreeQueries':maxTreeQueries,'maxTime':maxTime,'gamma':gamma,'numObs':numObs,'problemName':problemName,'agentSpeed':agentSpeed,'targetMaxSpeed':targetMaxSpeed,'targetNoise':targetNoise,'accuracy':accuracy,'availability':availability,'leaveRoadChance':leaveRoadChance},'Data':[]}
	for i in range(0,sims):
		dataPackage['Data'].append({'ModeBels':[],'Beliefs':[],'States':[],'Actions':[],'Observations':[],'Rewards':[],'TreeInfo':[]});

	if(verbosity >= 1):
		print("Starting Data Collection Run: {}".format(simIdent)); 
		print("Running {} simulations of {} steps each".format(sims,steps))
	
	allFirstCatches = []; 

	np.random.seed(23423421); 

	#run individual sims
	for count in range(0,sims):
		if(verbosity >= 2):
			print("Simulation: {} of {}".format(count+1,sims)); 
		


		#Make Problem
		h = Node(); 
		solver = POMCP(); 

		#Initialize Belief and State
		network = readInNetwork('../common/flyovertonNetwork.yaml')
		setNetworkNodes(network); 
		target,curs,goals = populatePoints(network,maxTreeQueries); 
		pickInd = np.random.randint(0,len(target)); 
		trueS = [np.random.random()*8,np.random.random()*8,target[pickInd][0],target[pickInd][1],curs[pickInd],goals[pickInd],0]; 

		sSet = []; 
		for i in range(0,len(target)):
			sSet.append([trueS[0],trueS[1],target[i][0],target[i][1],curs[i],goals[i],0]); 

		#For storage purposes, only the mean and sd of the belief are kept
		#dataPackage['Data'][count]['Beliefs'].append(sSet);
		mean = [sum([sSet[i][2] for i in range(0,len(sSet))])/len(sSet),sum([sSet[i][3] for i in range(0,len(sSet))])/len(sSet)]; 
		
		tmpBel = np.array(sSet); 
		mean = [np.mean(tmpBel[:,2]),np.mean(tmpBel[:,3])];
		sd =  [np.std(tmpBel[:,2]),np.std(tmpBel[:,3])];

		dataPackage['Data'][count]['Beliefs'].append([mean,sd]); 
		dataPackage['Data'][count]['States'].append(trueS); 
		if(verbosity >= 4):
			fig,ax1 = plt.subplots(); 
		

		for step in range(0,steps):
			if(verbosity>=3):
				print("Step: {}".format(step));  
			if(verbosity >=4):
				fig,ax1 = displayNetworkMap('../common/flyovertonNetwork.yaml',fig,ax1,False,redraw=True);


			act,info = solver.search(sSet,h,depth = min(maxDepth,steps-step+1),inform=True);

			#print(info)

			trueS = generate_s(trueS,act); 
			r = max(0,generate_r(trueS,act));
			o = generate_o(trueS,act); 

			#print(act,o);

			if(sum(dataPackage['Data'][count]['Rewards']) == 0 and r!=0):
				allFirstCatches.append(step); 

			# if(verbosity>=3):
			# 	if(act > 3):
			# 		actMap = {4:'Is the Target on the Road?',5:'Is the Target off-road?'}
			# 		print("Action: {}".format(actMap[act]));
			# 		print("Observation: {}".format(o)); 
			# 		print("");

			tmpHAct = h.getChildByID(act); 
			tmpHObs = tmpHAct.getChildByID(o); 

			# if(tmpHObs != -1 and len(tmpHObs.data) > 0):
			# 	h = tmpHObs; 
			# 	#sSet = solver.resampleNode(h); 
			# else:
			# 	h = tmpHAct[0]; 
			# 	#print("Error: Child Node Not Found!!!"); 

			h = Node(); 
			sSet = propogateAndMeasure(sSet,act,o); 
			correctAgentLoc(trueS,sSet); 
			
			tmpBel = np.array(sSet); 
			#print(len(tmpBel)); 
			mean = [np.mean(tmpBel[:,2]),np.mean(tmpBel[:,3])];
			sd =  [np.std(tmpBel[:,2]),np.std(tmpBel[:,3])];

			modeBels = [len(np.where(tmpBel[:,6] == 0)[0]), len(np.where(tmpBel[:,6] == 1)[0])]; 


			################################################
			if(verbosity >= 4):
				ax2 = fig.add_subplot(111,label='belief'); 
				sp=[tmpBel[:,2],tmpBel[:,3]];
				
				#ax2.hist2d(sp[0],sp[1],bins=40,range=[[-.2,8.2],[-.2,8.2]],cmin=1,cmap='Reds',zorder=2);
				ax2.scatter(sp[0],sp[1],c='r',zorder=2,marker='*',s=2);
				#ax2.scatter(sp[0],sp[1],c='k',zorder=2);
				ax2.set_xlim([-0.2,8.2]); 
				ax2.set_ylim([-0.2,8.2]);

				ax2.scatter(trueS[0],trueS[1],c=[0,0,1],zorder = 3); 
				ax2.scatter(trueS[2],trueS[3],c=[1,0,0],zorder = 3,edgecolor='black'); 
				#ax2.arrow(trueS[0],trueS[1],trueS[0]-trueS[0],trueS[1]-trueS[1],edgecolor=[0,0,1],head_width = 0.25,facecolor =[0,0,.5],zorder=3); 
				#ax2.arrow(trueS[2],trueS[3],trueS[2]-trueS[2],trueS[3]-trueS[3],edgecolor=[1,0,0],head_width = 0.25,facecolor = [.5,0,0],zorder=3); 
				
				ax1.set_xlim([-0.2,8.2]); 
				ax1.set_ylim([-0.2,8.2]); 
				plt.axis('off')
				ax1.axis('off')
				ax2.axis('off')
				plt.axis('off')
				#plt.colorbar() 
				plt.pause(0.01);
				ax2.remove();
			################################################

			dataPackage['Data'][count]['Beliefs'].append([mean,sd]); 
			dataPackage['Data'][count]['States'].append(trueS); 
			dataPackage['Data'][count]['Actions'].append(act); 
			dataPackage['Data'][count]['Observations'].append(o); 
			dataPackage['Data'][count]['Rewards'].append(r); 
			dataPackage['Data'][count]['TreeInfo'].append(info);
			dataPackage['Data'][count]['ModeBels'].append(modeBels); 
			
			# if(isTerminal(trueS,act)):
			# 	#print(trueS); 
			# 	if(verbosity >= 2):
			# 		print("Captured after: {} steps".format(step)); 
			# 	break; 


		print("Capture Time: {}".format(allFirstCatches[-1])); 
		print("Average Capture Time: {}".format(np.mean(allFirstCatches))); 
		print(""); 
		np.save('../../data/dataGridMQuest_E2_{}'.format(simIdent),dataPackage)

	#save all data



if __name__ == '__main__':

	if(len(sys.argv) > 1):
		runSims(int(sys.argv[1]),int(sys.argv[2]),verbosity = int(sys.argv[4]),simIdent=sys.argv[3]); 
	else:
		runSims(1,100,verbosity=4); 
 

	#simForward(100); 



	


