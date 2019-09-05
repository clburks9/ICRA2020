import warnings
warnings.simplefilter("ignore")

import sys
sys.path.append('../../src'); 
from treeNode import Node
import numpy as np; 
#from testProblemSpec import *;
from golfHumanSpec import *; 
import matplotlib.pyplot as plt

import cProfile
import time; 
#from gaussianMixtures import GM,Gaussian
#from softmaxModels import Softmax;
from copy import deepcopy 
import random
import time
from PIL import Image
from matplotlib.patches import Circle

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
		sprime = generate_s(s,act,truth=False); 
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
		sSetPrime.append(generate_s(s,act,truth=False)); 

	origLen = len(sSetPrime); 

	s = np.array(sSetPrime); 
	#sm = Softmax(); 
	#sm.buildOrientedRecModel([sSetPrime[0][0],sSetPrime[0][1]],0,1,1,steepness=7);

	#measurements = ['Near','West','South','North','East']
	#weights = [sm.pointEvalND(measurements.index(o),[s[i][2],s[i][3]]) for i in range(0,len(s))]; 
	weights = [0 for i in range(0,len(s))]; 
	upWeight = .99; 
	downWeight = .01; 
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


def runSims(sims = 10,steps = 10,verbosity = 1,simIdent = 'Test',vis=False):

	global numActs;

	#set up data collection
	dataPackage = {'Meta':{'NumActs':numActs,'maxDepth':maxDepth,'c':c,'maxTreeQueries':maxTreeQueries,'maxTime':maxTime,'gamma':gamma,'numObs':numObs,'problemName':problemName,'agentSpeed':agentSpeed,'targetMaxSpeed':targetMaxSpeed,'targetNoise':targetNoise,'accuracy':accuracy,'availability':availability},'Data':[]}
	for i in range(0,sims):
		dataPackage['Data'].append({'Beliefs':[],'States':[],'Actions':[],'Observations':[],'Rewards':[],'TreeInfo':[]});


	print("Starting Data Collection Run: {}".format(simIdent)); 

	#set up transition modes
	initialize(); 

	potentialSketches = [[200,125],[675,280],[620,680],[375,650],[90,575],[490,730],[330,220],[530,560],[680,400],[170,710],[230,500],[500,410],[530,160],[780,50],[400,400]]; 
	sketchProbs = np.ones(shape=(len(potentialSketches))); 
	sketchProbs /= sum(sketchProbs); 


	#run individual sims
	for count in range(0,sims):
		#np.random.seed(count+120243123); 
		print("Simulation: {} of {}".format(count+1,sims)); 
		
		#Make Problem
		h = Node(); 
		solver = POMCP(); 
		if(vis and count==0):
			fig,ax = plt.subplots(); 
			img = Image.open('../../img/bigGolfOverhead.png'); 

		#Initialize Belief and State
		trueX = np.random.random()*bounds[0]; 
		trueY = np.random.random()*bounds[1]; 
		sSet = []; 
		for i in range(0,maxTreeQueries):
			sSet.append([trueX,trueY,np.random.random()*bounds[0],np.random.random()*bounds[1],np.random.random()*targetMaxSpeed-targetMaxSpeed/2,np.random.random()*targetMaxSpeed-targetMaxSpeed/2]); 

		trueS = sSet[np.random.choice([0,len(sSet)-1])];

		#For storage purposes, only the mean and sd of the belief are kept
		#dataPackage['Data'][count]['Beliefs'].append(sSet);
		mean = [sum([sSet[i][2] for i in range(0,len(sSet))])/len(sSet),sum([sSet[i][3] for i in range(0,len(sSet))])/len(sSet)]; 
		
		tmpBel = np.array(sSet); 
		mean = [np.mean(tmpBel[:,2]),np.mean(tmpBel[:,3])];
		sd =  [np.std(tmpBel[:,2]),np.std(tmpBel[:,3])];
		dataPackage['Data'][count]['Beliefs'].append([mean,sd]); 
		dataPackage['Data'][count]['States'].append(trueS); 
		




		for step in range(0,steps): 

			if(step%20 == 1):
				tmp = np.random.choice([i for i in range(0,len(potentialSketches))],p=sketchProbs); 
				addSketch(potentialSketches[tmp]);
				numActs+=5; 
				sketchProbs[tmp] = 0; 
				sketchProbs /= sum(sketchProbs); 


			act,info = solver.search(sSet,h,depth = min(maxDepth,steps-step+1),inform=True);
			trueS = generate_s(trueS,act,truth=True); 
			r = max(0,generate_r(trueS,act));
			o = generate_o(trueS,act); 

			#print(act,numActs,len(allSketches));
			dirs = ['near','east','west','north','south']
			#print(dirs[(act-4)%5],(act-4)//5,o); 
			#print(trueS[0],trueS[1]); 
			#print("");

			tmpHAct = h.getChildByID(act); 
			tmpHObs = tmpHAct.getChildByID(o); 

			# if(tmpHObs != -1 and len(tmpHObs.data) > 0):
			# 	h = tmpHObs; 
			# 	sSet = solver.resampleNode(h); 
			# else:
			# 	h = tmpHAct[0]; 
			# 	print("Error: Child Node Not Found!!!"); 


			h = Node(); 
			sSet = propogateAndMeasure(sSet,act,o); 
			correctAgentLoc(trueS,sSet); 

			tmpBel = np.array(sSet); 
			#print(h.data); 
			#print([np.mean(tmpBel[:,0]),np.mean(tmpBel[:,1])])
			#print(""); 
			mean = [np.mean(tmpBel[:,2]),np.mean(tmpBel[:,3])];
			sd =  [np.std(tmpBel[:,2]),np.std(tmpBel[:,3])];

			#modeBels = [len(np.where(tmpBel[:,4] == 0)[0]), len(np.where(tmpBel[:,4] == 1)[0]),len(np.where(tmpBel[:,4] == 2)[0])]; 


			dataPackage['Data'][count]['Beliefs'].append([mean,sd]); 
			dataPackage['Data'][count]['States'].append(trueS); 
			dataPackage['Data'][count]['Actions'].append(act); 
			dataPackage['Data'][count]['Observations'].append(o); 
			dataPackage['Data'][count]['Rewards'].append(r); 
			dataPackage['Data'][count]['TreeInfo'].append(info);
			#dataPackage['Data'][count]['ModeBels'].append(modeBels); 

			if(vis and count == 0):
				ax.imshow(np.flip(img,0),origin='lower',zorder=1); 
				ax.scatter(tmpBel[:,2],tmpBel[:,3],c=[1,0,0,0.25],marker='*',s=2,zorder=2)
				ax.scatter(trueS[0],trueS[1],c=[0,0,1],zorder=3,edgecolor='black'); 
				ax.scatter(trueS[2],trueS[3],c=[1,0,0],zorder=3,edgecolor='black'); 
				for p in allSketches:
					ax.add_patch(Circle((p[0],p[1]),75,edgecolor='k',facecolor=[0,0,0,0],zorder=3)); 
					ax.text(p[0],p[1],allSketches.index(p)); 
				#ax.arrow(trueS[0],trueS[1],trueS[0]-trueS[0],trueS[1]-trueS[1],edgecolor=[0,0,1],head_width = 0.25,facecolor =[0,0,.5]); 
				#ax.arrow(trueS[2],trueS[3],trueS[2]-trueS[2],trueS[3]-trueS[3],edgecolor=[0,1,0],head_width = 0.25,facecolor = [0,.5,0]); 
				ax.set_title("Step: {}".format(step))
				ax.axis('equal')
				plt.xlim([0,bounds[0]]); 
				plt.ylim([0,bounds[1]]); 
				#ax.axis('off')
				plt.pause(0.001)

				plt.cla();

		print("Accumlated Reward: {}".format(sum(dataPackage['Data'][count]['Rewards'])));
		print("Average Final Reward: {}".format(sum([sum(dataPackage['Data'][i]['Rewards']) for i in range(0,count+1)])/(count+1)));
		print(""); 
		np.save('../../data/dataGolfHuman_E1_{}'.format(simIdent),dataPackage)

	#save all data



if __name__ == '__main__':

	#np.random.seed(2)

	if(len(sys.argv) > 1):
		runSims(int(sys.argv[1]),int(sys.argv[2]),simIdent=sys.argv[3]); 
	else:
		runSims(1,100,vis=True); 




	


