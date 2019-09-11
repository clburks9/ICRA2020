import numpy as np
from copy import deepcopy
import sys
import matplotlib.pyplot as plt
sys.path.append("../src"); 
from softmaxModels import Softmax


numActs= 4;
numObs = 3;  
gamma = .95; 
maxTime = 2;
maxDepth = 15;
c=1;
maxTreeQueries = 2000; 
agentSpeed = 1; 
problemName = 'ModeGrid'
leaveRoadChance = 0.05; 
targetMaxSpeed = 0.25; 
targetNoise = 0.15; 
network = None;

bounds = [8,8]; 
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

def setNetworkNodes(net):
	global network
	network = net; 


def generate_s(s,a):

	speed = targetMaxSpeed; 
	dev = targetNoise; 

	sprime = deepcopy(s); 

	#p = sprime[2]; 
	#print(sprime[4])
	c = sprime[4].loc; 
	g = sprime[5].loc; 
	mode = sprime[6]; 


	
	#if mode is 0, check for mode transition
	if(sprime[6] == 0):
		coin = np.random.random(); 
		if(coin < leaveRoadChance):
			sprime[6] = 1; 

			#if mode transitions to 1, pick random new goal
			newGoal = np.random.choice(network); 
			g = newGoal.loc;
			sprime[5] = newGoal; 



	if(sprime[6] == 0):
		#move it one step along the distnace between cur and goal
		if(c[0] > g[0]):
			sprime[2] -= speed + np.random.normal(0,dev); 
		elif(c[0] < g[0]):
			sprime[2] += speed + np.random.normal(0,dev);

		if(c[1] > g[1]):
			sprime[3] -= speed + np.random.normal(0,dev);
		elif(c[1] < g[1]):
			sprime[3] += speed + np.random.normal(0,dev);
	elif(sprime[6] == 1):

		#move along vector to goal
		#sprime[2] += (g[0]-c[0])*speed/2 + np.random.normal(0,dev); 
		#sprime[3] += (g[1]-c[1])*speed/2 + np.random.normal(0,dev); 

		sprime[2] += (speed/2)*(g[0]-c[0])/distance(c,g) + np.random.normal(0,dev); 
		sprime[3] += (speed/2)*(g[1]-c[1])/distance(c,g) + np.random.normal(0,dev); 


	#if point has reached goal choose new goal
	#note: You'll want to make this not perfect equivalence
	#if(s[2] == g[0] and s[3] == g[1]):
	if(distance([sprime[2],sprime[3]],c) > distance(c,g)):
		#print("Goal Reached!!!");
		l = [i for i in range(0,len(sprime[5].neighbors))]; 
		if(len(l) > 1):
			if(sprime[6] == 0):
				l.remove(sprime[5].neighbors.index(sprime[4])); 
		sprime[4] = sprime[5]; 
		sprime[2] = sprime[4].loc[0]; 
		sprime[3] = sprime[4].loc[1]; 
		tmp = np.random.choice(l); 
		sprime[5] = sprime[4].neighbors[tmp]; 
		sprime[6] = 0; 



	#actions
	#0: left
	#1: right
	#2: up
	#3: down

	#Move the Agent
	if(a == 0):
		sprime[0] -= agentSpeed
	elif(a==1):
		sprime[0] += agentSpeed
	elif(a == 2):
		sprime[1] += agentSpeed
	elif(a == 3):
		sprime[1] -= agentSpeed

	sprime[0] = min(bounds[0],max(0,sprime[0]));
	sprime[1] = min(bounds[1],max(0,sprime[1]));
	

	return sprime; 




def dist(s):
	return np.sqrt((s[0]-s[2])**2 + (s[1]-s[3])**2); 

def distance(a,b):
	return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def generate_r(s,a):


	if(dist(s) < 0.5):
		return 1; 
	else:
		return 0; 

def generate_o(s,a):




	coin = np.random.random(); 
	flipped = .02; 

	if(dist(s) > 1):
		if(coin>flipped):
			return 'Far';
		else:
			return 'Near' 
	elif(dist(s) > .5 and dist(s)<1):
		if(coin>flipped):
			return 'Near'
		else:
			return np.random.choice(['Far','Caught']);
	elif(dist(s) < .5):
		if(coin > flipped):
			return 'Caught'
		else:
			return 'Near'



def estimate_value(s,h):
	#how far can you get in the depth left
	
	return min(100,1/dist(s))

def rollout(s,depth): 

	if(depth <= 0):
		return 0; 
	else:
		#greedy action
		a = 0; 
		di = [s[2]-s[0],s[3]-s[1]]; 
		if(np.sqrt(di[0]**2 + di[1]**2) < 1):
			a=4
		if(abs(di[0]) > abs(di[1])):
			if(di[0] > 0):
				a=1; 
			else:
				a=0;
		else:
			if(di[1] > 0):
				a=2;
			else:
				a=3

		sprime = generate_s(s,a); 
		r = generate_r(s,a); 
		return r + gamma*rollout(sprime,depth-1)


def isTerminal(s,act):
	# if(dist(s) < 0.5):
	# 	return True; 
	# else:
	# 	return False 
	return False



def obs_weight(s,a,o):
	upWeight = 0.98; 
	downWeight = 0.02; 

	if(dist(s) > 1):
		if(o=='Far'):
			return upWeight;
		else:
			return downWeight 
	elif(dist(s) > .5 and dist(s)<1):
		if(o=='Near'):
			return upWeight; 
		else:
			return downWeight; 
	else:
		if(o=='Caught'):
			return upWeight
		else:
			return downWeight





if __name__ == '__main__':
	
	colors = {'Near':'b','North':'r','West':'g','South':'m','East':'k'}; 

	for i in range(0,500):
		s = [0,0,np.random.random()*10,np.random.random()*10,1,1]; 
		o = generate_o(s,0); 
		plt.scatter(s[2],s[3],c=colors[o]); 

	plt.ylim([0,10]); 
	plt.xlim([0,10]); 
	plt.show(); 