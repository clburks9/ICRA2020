import warnings
warnings.simplefilter("ignore")
from treeNode import Node
import numpy as np; 
#from testProblemSpec import *;
from testProblemSwitch4D import *; 
import matplotlib.pyplot as plt

import cProfile
import time; 
from gaussianMixtures import GM,Gaussian
from softmaxModels import Softmax;
from copy import deepcopy 
import random

class POMCP:

	def __init__(self):
	 	pass;


	def simulate(self,s,h,depth,m = [.125,.125,.125,.125,.125,.125]):
		
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
		sprime = generate_s(s,act,m=m); 
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

		q = r + gamma*self.simulate(sprime,h[act].getChildByID(o),depth-1,m=m); 
		
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


	def search(self,b,h,inform = False,m=[.25,.25,.25,.25,.25,.25]):
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
			self.simulate(s,h,maxDepth,m=m); 
		if(inform):
			info = {"Execution Time":0,"Tree Queries":0,"Tree Size":0}
			info['Execution Time'] = time.clock()-startTime; 
			info['Tree Queries'] = count; 
			info['Tree Size'] = len(h.traverse()); 
			return np.argmax([a.Q for a in h]),info; 
			#print([a.Q for a in h])
		else:
			return np.argmax([a.Q for a in h]); 


def examineObsDist(h):

	allNodes = h.gatherAllNodes(); 

	data = {'left':[],'right':[],'near':[]}; 

	#print(allNodes);

	for no in allNodes:
		if(no.id in data.keys()):
			data[no.id].extend(no.data); 

	#for key in data.keys():
		#print(key,len(data[key])); 

	le = np.array(data['left']).T; 
	re = np.array(data['right']).T; 
	ne = np.array(data['near']).T; 

	plt.scatter(le[0],le[1],c='r'); 
	plt.scatter(re[0],re[1],c='b'); 
	plt.scatter(ne[0],ne[1],c='g'); 
	plt.xlim([-5,5]); 
	plt.ylim([-5,5])
	plt.show();


def simForward(steps = 10):
	#Make problem
	h = Node(); 
	solver = POMCP(); 

	#make belief
	b = GM(); 
	b.addG(Gaussian([2,2,8,8],np.identity(4),1)); 
	sSet = b.sample(maxTreeQueries); 
	for i in range(0,len(sSet)):
		sSet[i].append(np.random.choice([-1,1])); 

	trueS = b.sample(1)[0]
	trueS.append(np.random.choice([-1,1])); 


	allModes = np.zeros(shape=(steps,6)); 
	
	# allModes[:20,5] = 1; 
	# allModes[20:50,4] = 1; 
	# allModes[50:,5] = 1; 
	allModes[0:10,0] = 1; 
	allModes[10:20,1] = 1; 
	allModes[20:30,2] = 1; 
	allModes[30:40,3] = 1; 
	allModes[40:50,4] = 1; 
	allModes[50:60,5] = 1; 
	allModes[60:70,0] = 1; 
	allModes[70:80,1] = 1; 
	allModes[80:90,2] = 1; 
	allModes[90:100,3] = 1; 

	fig,ax = plt.subplots(); 
	plotFudge = 20; 
	allPrevs = np.zeros(shape=(steps,5)); 
	allRewards = []; 
	allMeans = np.zeros(shape = (steps,2)); 
	allVars = np.zeros(shape = (steps,2)); 
	#get action
	for step in range(0,steps):
		allPrevs[step] = np.array(trueS); 
		act = solver.search(sSet,h,False,m=allModes[step]);
		#act = solver.search(sSet,h,False,m=[1/6,1/6,1/6,1/6,1/6,1/6]);
		#act = solver.search(sSet,h,False,m=[0,1,0,0,0,0]);

		r = generate_r(trueS,act);  
		trueS = generate_s(trueS,act,allModes[step]); 
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




if __name__ == '__main__':

	# np.random.seed(0)
	# random.seed(0); 
	#print("Note to self, estimate hangs on vertical vs horizontal tests")

	simForward(100); 
	# h = Node(); 
	
	# solver = POMCP(); 

	# b = GM(); 
	# b.addG(Gaussian([2,2,4,2],np.identity(4),1)); 
	# #b.addG(Gaussian([-4,0],[[0.5,0],[0,0.5]],0.5)); 
	# #b.addG(Gaussian([4,0],[[1,0],[0,1]],0.5)); 
	# #b.addG(Gaussian(0,2,0.5)); 
	# #b.addG(Gaussian(0,1,0.5)); 

	# sSet = b.sample(maxTreeQueries); 

	# act,info = solver.search(sSet,h,True); 
	# print("Action: {}".format(act)); 
	# print(info)
	#print("Tree Sims: {}".format(solver.treeSims)); 
	#print("Tree Size: {}".format(len(h.traverse()))); 

	#examineObsDist(h); 
	#print(h.children);



	


