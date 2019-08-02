import numpy as np
from copy import deepcopy

numActs= 5;
numObs = 5;  
gamma = .9; 
maxTime = 0.5; 
maxDepth = 15;
c=10;
maxTreeQueries = 10000; 


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

def generate_s(s,a,m = [.25,.25,.25,.25,.25,.25]):
	
	#modes: 
	#0: horizontal
	#1: vertical
	#2: pos-diagonal
	#3: neg-diagonal
	#4: stationary
	#5: circle around point (5,5)

	sprime = deepcopy(s); 

	# coin = np.random.random(); 
	# direct = 1; 
	# if(coin < 0.5): 
	# 	direct = -1; 
	useM = np.random.choice(a=[0,1,2,3,4,5],p=m); 

	direct = sprime[4];

	if(useM == 0):
		sprime[2] += direct + np.random.normal(0,0.125)
		sprime[3] += np.random.normal(0,0.125)
	elif(useM == 1):
		sprime[2] += np.random.normal(0,0.125)
		sprime[3] += direct + np.random.normal(0,0.125)
	elif(useM == 2):
		sprime[2] += direct + np.random.normal(0,0.125)
		sprime[3] += direct + np.random.normal(0,0.125)
	elif(useM == 3):
		sprime[2] += direct + np.random.normal(0,0.125)
		sprime[3] -= direct + np.random.normal(0,0.125)
	elif(useM == 4):
		pass; 
	elif(useM == 5):
		#move along the angle from current angle
		#so find the angle to point 5,5
		#if direct == 1, clockwise
		tmpS = [sprime[2]-5,sprime[3]-5]; 
		r = np.sqrt(tmpS[0]**2 + tmpS[1]**2); 
		theta = np.arctan2(tmpS[1],tmpS[0]) + .25*direct; 
		#tmpS = [r*np.cos(theta),r*np.sin(theta)];
		sprime[2] = r*np.cos(theta) + 5; 
		sprime[3] = r*np.sin(theta) + 5; 


	
	#sprime[2] = min(10,max(0,sprime[2]));
	#sprime[3] = min(10,max(0,sprime[3]));

	if(sprime[2] < 0):
		sprime[2] = 0; 
		sprime[4] = -sprime[4]; 
	elif(sprime[2] > 10):
		sprime[2] = 10; 
		sprime[4] = -sprime[4]; 
	else:
		if(sprime[3] < 0):
			sprime[3] = 0; 
			sprime[4] = -sprime[4]; 
		elif(sprime[3] > 10):
			sprime[3] = 10; 
			sprime[4] = -sprime[4]; 






	#actions
	#0: left
	#1: right
	#2: up
	#3: down

	# if(a == 0):
	# 	sprime[0] -= np.random.normal(1,.25); 
	# elif(a==1):
	# 	sprime[0] += np.random.normal(1,.25); 
	# elif(a == 2):
	# 	sprime[1] += np.random.normal(1,.25); 
	# elif(a == 3):
	# 	sprime[1] -= np.random.normal(1,.25); 

	if(a == 0):
		sprime[0] -= 1
	elif(a==1):
		sprime[0] += 1
	elif(a == 2):
		sprime[1] += 1
	elif(a == 3):
		sprime[1] -= 1

	sprime[1] = min(10,max(0,sprime[1]));
	sprime[0] = min(10,max(0,sprime[0]));

	return sprime; 


def dist(s):
	return np.sqrt((s[0]-s[2])**2 + (s[1]-s[3])**2); 

def generate_r(s,a):
	if(dist(s) < 2):
		return 10; 
	else:
		return 0; 

	#return max(100,1/dist(s)); 

	#return 20-dist(s)


def generate_o(s,a):
	# if(s[0] - s[1] < -1):
	# 	return 'left'; 
	# elif(s[0] - s[1] > 1):
	# 	return 'right'; 
	# else:
	# 	return 'near'; 


	#flip coin for noise
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

