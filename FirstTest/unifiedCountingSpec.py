import numpy as np
from copy import deepcopy
import sys
import matplotlib.pyplot as plt
sys.path.append("../src"); 
from softmaxModels import Softmax


numActs= 5;
numObs = 5;  
gamma = .9; 
maxTime = 0.5;
maxDepth = 25;
c=1;
maxTreeQueries = 1000; 
problemName = '2DModalTrackingSimple'


#Alright, let's do a 2D search problem
#2 Robots, each moving in 2D
#The cop robot moves freely, while the 
#robber moves vertically or horizontally depending 
#on the mode in which it is currently moving
#bounded on [0,10],[0,10]

#target must be faster than the cop

#not actually a moving target, just a moving goal,
#where robot has noise
#standard left, right, up, down, near observations

def generate_s(s,a):
	

	#States:
	#0,1: Agent
	#2,3: Target
	#4: mode
	#5: direction

	#modes: 
	#0: horizontal
	#1: vertical
	#2: stationary

	sprime = deepcopy(s); 

	#Extract Mode info
	useM = sprime[4]
	direct = sprime[5];

	#Move according to mode
	if(useM == 0):
		sprime[2] += direct + np.random.normal(0,0.125)
		sprime[3] += np.random.normal(0,0.125)
	elif(useM == 1):
		sprime[2] += np.random.normal(0,0.125)
		sprime[3] += direct + np.random.normal(0,0.125)
	


	
	#Handle Turning Around
	if(sprime[2] < 0):
		sprime[2] = 0; 
		sprime[5] = -sprime[5]; 
	elif(sprime[2] > 10):
		sprime[2] = 10; 
		sprime[5] = -sprime[5]; 
	else:
		if(sprime[3] < 0):
			sprime[3] = 0; 
			sprime[5] = -sprime[5]; 
		elif(sprime[3] > 10):
			sprime[3] = 10; 
			sprime[5] = -sprime[5]; 



	#Transition Mode
	pmm = [[.90,.05,.05],[.05,.90,.05],[.05,.05,.9]]

	#print(sprime)
	sprime[4] = np.random.choice([0,1,2],p=pmm[int(sprime[4])])


	#actions
	#0: left
	#1: right
	#2: up
	#3: down

	#Move the Agent
	agentSpeed = 1; 
	if(a == 0):
		sprime[0] -= agentSpeed
	elif(a==1):
		sprime[0] += agentSpeed
	elif(a == 2):
		sprime[1] += agentSpeed
	elif(a == 3):
		sprime[1] -= agentSpeed

	sprime[0] = min(10,max(0,sprime[0]));
	sprime[1] = min(10,max(0,sprime[1]));
	

	return sprime; 


def dist(s):
	return np.sqrt((s[0]-s[2])**2 + (s[1]-s[3])**2); 

def generate_r(s,a):
	if(dist(s) < 1):
		return 10; 
	else:
		return 0; 

	#return max(100,1/dist(s)); 

	#return 20-dist(s)


def generate_o(s,a):


	##flip coin for noise
	coin = np.random.random(); 
	if(coin < 0.02):
		return np.random.choice(['Near','East','West','North','South'])


	if(dist(s) < 1):
		return 'Near'; 

	di = [s[2]-s[0],s[3]-s[1]]; 
	if(abs(di[0]) > abs(di[1])):
		if(di[0] > 0):
			return 'East'
		else:
			return 'West'
	else:
		if(di[1] > 0):
			return 'North'
		else:
			return 'South'

	#Softmax Sampling
	# useS = deepcopy(s); 

	# words = ['Near','North','West','South','East']

	# model = Softmax(); 
	# model.buildOrientedRecModel([useS[0],useS[1]],0,2,2,steepness = 10);

	# testValues = np.zeros(5);  
	# for i in range(0,len(words)):
	# 	testValues[i] = model.pointEvalND(i,[useS[2],useS[3]]); 
	# testValues /= sum(testValues); 
	# #print(testValues); 
	# return words[np.random.choice([0,1,2,3,4],p=testValues)]; 

def estimate_value(s,h):
	#how far can you get in the depth left
	
	return 1/dist(s)

def rollout(s,depth): 

	if(depth <= 0):
		return 0; 
	else:
		#random action
		a = np.random.randint(0,numActs)
		sprime = generate_s(s,a,0); 
		r = generate_r(s,a); 
		return r + gamma*rollout(sprime,a)


def isTerminal(s,act):
	return False; 





if __name__ == '__main__':
	
	colors = {'Near':'b','North':'r','West':'g','South':'m','East':'k'}; 

	for i in range(0,500):
		s = [0,0,np.random.random()*10,np.random.random()*10,1,1]; 
		o = generate_o(s,0); 
		plt.scatter(s[2],s[3],c=colors[o]); 

	plt.ylim([0,10]); 
	plt.xlim([0,10]); 
	plt.show(); 