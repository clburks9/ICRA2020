import warnings
warnings.simplefilter("ignore")

import sys
sys.path.append('../src'); 
from treeNode import Node
import numpy as np; 
#from testProblemSpec import *;
from unifiedCountingSpec import *; 
import matplotlib.pyplot as plt

import cProfile
import time; 
#from gaussianMixtures import GM,Gaussian
#from softmaxModels import Softmax;
from copy import deepcopy 
import random

class POMCP:

	def __init__(self):
	 	pass;


	def simulate(self,s,h,depth):
		
		#check if node is in tree
		#if not, add nodes for each action

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

	def resampleNode(self,h):

		b = h.data; 
		if(len(h.data) == 0):
			print("Error: ResampleNode, Empty Data Node!!!!")
			raise Exception; 





		if(len(h.data) >= maxTreeQueries):
			return h.data; 

		while(len(b) < maxTreeQueries):
			#flip a coin and see if you random scatter
			coin = np.random.random(); 
			if(coin < 0.001):
				ind = np.random.randint(0,len(b));
				tmp = deepcopy(b[ind]); 
				tmp[2] = np.random.random()*10; 
				tmp[3] = np.random.random()*10; 
				tmp[4] = np.random.choice([-1,1]); 
			else:
				ind = np.random.randint(0,len(b)); 
				tmp = deepcopy(b[ind]);
				tmp[2] += np.random.normal(0,.125);  
				tmp[3] += np.random.normal(0,.125);

			b.append(tmp); 
		return b; 


	def search(self,b,h,inform = False):
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
			self.simulate(s,h,maxDepth); 
		if(inform):
			info = {"Execution Time":0,"Tree Queries":0,"Tree Size":0}
			info['Execution Time'] = time.clock()-startTime; 
			info['Tree Queries'] = count; 
			info['Tree Size'] = len(h.traverse()); 
			return np.argmax([a.Q for a in h]),info; 
			#print([a.Q for a in h])
		else:
			return np.argmax([a.Q for a in h]); 




def simForward(steps = 10):
	#Make problem
	h = Node(); 
	solver = POMCP(); 

	#make belief
	#b = GM(); 
	#b.addG(Gaussian([2,2,8,8],np.identity(4),1)); 
	#sSet = b.sample(maxTreeQueries); 

	# sSet = (np.random.rand(maxTreeQueries,4)*10).tolist(); 

	# for i in range(0,len(sSet)):
	# 	sSet[i].append(np.random.choice([0,1,2])); 
	# 	sSet[i].append(np.random.choice([-1,1])); 

	# trueS = (np.random.rand(4)*10).tolist()
	# trueS.append(np.random.choice([0,1,2]));
	# trueS.append(np.random.choice([-1,1])); 

	trueX = np.random.random()*10; 
	trueY = np.random.random()*10; 
	sSet = []; 
	for i in range(0,maxTreeQueries):
		sSet.append([trueX,trueY,np.random.random()*10,np.random.random()*10,np.random.choice([0,1,2]),np.random.choice([-1,1])]); 

	trueS = sSet[np.random.choice([0,len(sSet)-1])]; 


	fig,ax = plt.subplots(); 
	plotFudge = 20; 
	allPrevs = np.zeros(shape=(steps,6)); 
	allRewards = []; 
	allMeans = np.zeros(shape = (steps,2)); 
	allVars = np.zeros(shape = (steps,2)); 
	#get action
	for step in range(0,steps):
		allPrevs[step] = np.array(trueS); 
		act = solver.search(sSet,h,False);

		r = generate_r(trueS,act);  
		trueS = generate_s(trueS,act); 
		o = generate_o(trueS,act); 

		allRewards.append(r); 

		

		tmpHAct = h.getChildByID(act); 
		tmpHObs = tmpHAct.getChildByID(o); 
		tmpBel = np.array(h.data); 
		allMeans[step] = [np.mean(tmpBel[:,2]),np.mean(tmpBel[:,3])];
		allVars[step] =  [np.std(tmpBel[:,2]),np.std(tmpBel[:,3])]; 

		if(tmpHObs != -1 and len(tmpHObs.data) > 0):
			h = tmpHObs; 
			sSet = solver.resampleNode(h); 
		else:
			#h = np.random.choice(h.children);
			# print(h); 
			# print("State: {}".format(trueS));
			# print("Action: {}".format(act)); 
			# print("Observation: {}".format(o)); 
			# raise("Error: Child Node not Found!!!")
			h = tmpHAct[0]; 
			print("Error: Child Node Not Found!!!"); 
			
		
		print("Step: {} of {}".format(step+1,steps))
		print("State: {}".format(trueS));
		print("Action: {}".format(act)); 
		print("Observation: {}".format(o)); 
		print("Distance: {0:.2f}".format(dist(trueS)))
		print("Ac Reward: {}".format(sum(allRewards))); 
		print("Belief Mean: {0:.2f},{0:.2f}".format(np.mean(tmpBel[:,2]),np.mean(tmpBel[:,3]))); 
		print("Belief Length: {}".format(len(tmpBel)))
		print(""); 
		#print(info)

		#ax.scatter(trueS[0],trueS[1],c=[0,0,1,((step+plotFudge)/(steps+plotFudge))]); 
		#ax.scatter(trueS[2],trueS[3],c=[0,1,0,((step+plotFudge)/(steps+plotFudge))]); 
		ax.scatter(tmpBel[:,2],tmpBel[:,3],c=[1,0,0,0.25],marker='*',s=2)
		ax.scatter(allPrevs[step][0],allPrevs[step][1],c=[0,0,1]); 
		ax.scatter(allPrevs[step][2],allPrevs[step][3],c=[0,1,0]); 
		ax.arrow(allPrevs[step][0],allPrevs[step][1],trueS[0]-allPrevs[step][0],trueS[1]-allPrevs[step][1],edgecolor=[0,0,1],head_width = 0.25,facecolor =[0,0,.5]); 
		ax.arrow(allPrevs[step][2],allPrevs[step][3],trueS[2]-allPrevs[step][2],trueS[3]-allPrevs[step][3],edgecolor=[0,1,0],head_width = 0.25,facecolor = [0,.5,0]); 
		
		allC = np.zeros(shape=(step,4)); 
		allC[:,2] = 1; 
		for i in range(0,len(allC)):
			allC[i,3] = .6*(i/len(allC)) 

		ax.scatter(allPrevs[:,0],allPrevs[:,1],c=allC)
		 
		
		plt.xlim([-0.5,10.5]); 
		plt.ylim([-0.5,10.5]); 
		plt.pause(0.001)

		plt.cla();

	plt.clf(); 


	print("Final Accumlated Reward after {} steps: {}".format(steps,sum(allRewards))); 
	fig,axarr = plt.subplots(2); 
	x = range(0,steps); 

	axarr[0].plot(allMeans[:,0],c='g');
	axarr[0].plot(allMeans[:,0] + 2*allVars[:,0],c='g',linestyle='--')
	axarr[0].plot(allMeans[:,0] - 2*allVars[:,0],c='g',linestyle='--')
	axarr[0].plot(allPrevs[:,2],c='k',linestyle='--')
	axarr[0].fill_between(x,allMeans[:,0] - 2*allVars[:,0],allMeans[:,0] + 2*allVars[:,0],alpha=0.25,color='g')
	axarr[0].set_ylim([-0.5,10.5]); 
	axarr[0].set_ylabel('North Estimate')

	axarr[1].plot(allMeans[:,1],c='g');
	axarr[1].plot(allMeans[:,1] + 2*allVars[:,1],c='g',linestyle='--')
	axarr[1].plot(allMeans[:,1] - 2*allVars[:,1],c='g',linestyle='--')
	axarr[1].plot(allPrevs[:,3],c='k',linestyle='--')
	axarr[1].fill_between(x,allMeans[:,1] - 2*allVars[:,1],allMeans[:,1] + 2*allVars[:,1],alpha=0.25,color='g')
	axarr[1].set_ylim([-0.5,10.5]); 
	axarr[1].set_ylabel('East Estimate')
	fig.suptitle("Estimates with 2 sigma bounds at reward: {}".format(sum(allRewards))); 





	plt.show()


def runSims(sims = 10,steps = 10,verbosity = 1,simIdent = 'Test'):

	#set up data collection
	dataPackage = {'Meta':{'NumActs':numActs,'maxDepth':maxDepth,'c':c,'maxTreeQueries':maxTreeQueries,'maxTime':maxTime,'gamma':gamma,'numObs':numObs,'problemName':problemName},'Data':[]}
	for i in range(0,sims):
		dataPackage['Data'].append({'Beliefs':[],'States':[],'Actions':[],'Observations':[],'Rewards':[]}); 


	print("Starting Data Collection Run: {}".format(simIdent)); 

	#run individual sims
	for count in range(0,sims):
		print("Simulation: {} of {}".format(count+1,sims)); 
		
		#Make Problem
		h = Node(); 
		solver = POMCP(); 

		#Initialize Belief and State
		trueX = np.random.random()*10; 
		trueY = np.random.random()*10; 
		sSet = []; 
		for i in range(0,maxTreeQueries):
			sSet.append([trueX,trueY,np.random.random()*10,np.random.random()*10,np.random.choice([0,1,2]),np.random.choice([-1,1])]); 

		trueS = sSet[np.random.choice([0,len(sSet)-1])];

		dataPackage['Data'][count]['Beliefs'].append(sSet); 
		dataPackage['Data'][count]['States'].append(trueS); 

		for step in range(0,steps):
			act = solver.search(sSet,h,False);
			trueS = generate_s(trueS,act); 
			r = generate_r(trueS,act);
			o = generate_o(trueS,act); 

			tmpHAct = h.getChildByID(act); 
			tmpHObs = tmpHAct.getChildByID(o); 

			if(tmpHObs != -1 and len(tmpHObs.data) > 0):
				h = tmpHObs; 
				sSet = solver.resampleNode(h); 
			else:
				h = tmpHAct[0]; 
				#print("Error: Child Node Not Found!!!"); 

			dataPackage['Data'][count]['Beliefs'].append(sSet); 
			dataPackage['Data'][count]['States'].append(trueS); 
			dataPackage['Data'][count]['Actions'].append(act); 
			dataPackage['Data'][count]['Observations'].append(o); 
			dataPackage['Data'][count]['Rewards'].append(r); 

		print("Accumlated Reward: {}".format(sum(dataPackage['Data'][count]['Rewards'])));
		print(""); 
		np.save('../data/dataUnified_E1_{}'.format(simIdent),dataPackage)

	#save all data



if __name__ == '__main__':

	if(len(sys.argv) > 1):
		runSims(100,100,simIdent=sys.argv[1]); 

	#simForward(100); 



	


