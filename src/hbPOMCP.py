import warnings
warnings.simplefilter("ignore")
from treeNode import Node
import numpy as np; 
#from testProblemSpec import *;
from testProblem2D import *; 
import matplotlib.pyplot as plt

import cProfile
import time; 
from gaussianMixtures import GM,Gaussian
from softmaxModels import Softmax;

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



	def search(self,b,h,inform = False):
		#Note: You can do more proper analytical updates if you sample during runtime
		#but it's much faster if you pay the sampling price beforehand. 
		#TLDR: You need to change this before actually using
		print("Check your sampling before using this in production")

		sSet = b.sample(10000); 

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


if __name__ == '__main__':
	h = Node(); 
	
	solver = POMCP(); 

	b = GM(); 
	b.addG(Gaussian([-4,0],[[0.5,0],[0,0.5]],0.5)); 
	b.addG(Gaussian([4,0],[[1,0],[0,1]],0.5)); 
	#b.addG(Gaussian(0,2,0.5)); 
	#b.addG(Gaussian(0,1,0.5)); 

	act,info = solver.search(b,h,True); 
	print("Action: {}".format(act)); 
	print(info)
	#print("Tree Sims: {}".format(solver.treeSims)); 
	#print("Tree Size: {}".format(len(h.traverse()))); 

	examineObsDist(h); 
	#print(h.children);


	


