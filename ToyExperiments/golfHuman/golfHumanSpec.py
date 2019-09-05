import numpy as np
from copy import deepcopy
import sys
import matplotlib.pyplot as plt
sys.path.append("../src"); 
from PIL import Image


numActs= 4;
numObs = 2;  
gamma = .95; 
maxTime = 1;
maxDepth = 25;
c=1;
maxTreeQueries = 10000; 
problemName = 'GolfHuman'
agentSpeed = 50; 
targetMaxSpeed = 25; 
targetNoise = 10; 
allSketches = []; 
availability = 0.95; 
accuracy = .95;


bounds = [828-1,828-1]; 

speedMap = None; 
useMap = None; 

def initialize():
	speedImg = Image.open('../../img/bigGolfSpeed.png'); 
	speedImg = np.asarray(speedImg,dtype=np.int32); 

	global speedMap
	global useMap
	speedMap = ((speedImg[:,:,1] - .5*speedImg[:,:,0])/255 + 1.1); 
	
	speedMap = np.flip(speedMap,0); 
	speedMap = np.transpose(speedMap)
	useMap = np.ones(shape=speedMap.shape); 
	# plt.imshow(speedMap,origin='lower'); 
	# plt.show();


def addSketch(p):
	global allSketches; 
	global numActs; 
	global useMap
	global speedMap

	numActs += 5; 
	allSketches.append(p); 

	for i in range(-150+int(p[0]),150+int(p[0])):
		if(i>=0 and i<bounds[0]):
			for j in range(-150+int(p[0]),150+int(p[0])):
				if(j>=0 and j<bounds[1]):
					useMap[i,j] = speedMap[i,j]; 

#new one
def generate_s(s,a,truth=True):

	global useMap
	global speedMap

	#states:
	#0,1: agent x,y
	#2,3: target x,y
	#4,5: target xdot,ydot
	sprime = deepcopy(s); 


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




	#Target NCV Model
	if(truth):
		modifier = speedMap[int(s[2]),int(s[3])]; 
	else:
		modifier = useMap[int(s[2]),int(s[3])]; 
	F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]]); 
	
	#Maybe tag avoid? 
	#ok, new take on tag avoid. Do the NCV if far, run if near
	#
	# if(dist(sprime) < 200):
	# 	sprime[2] += modifier*min(targetMaxSpeed,max(-targetMaxSpeed,sprime[2]-sprime[0])); 
	# 	sprime[3] += modifier*min(targetMaxSpeed,max(-targetMaxSpeed,sprime[3]-sprime[0])); 
	
	sprime[2] += sprime[4]*modifier; 
	sprime[3] += sprime[5]*modifier; 




	sprime[4] += np.random.normal(0,targetNoise);
	sprime[5] += np.random.normal(0,targetNoise); 

	sprime[4] = min(targetMaxSpeed,max(-targetMaxSpeed,sprime[4]));  
	sprime[5] = min(targetMaxSpeed,max(-targetMaxSpeed,sprime[5]));  

	if(sprime[2] < 0):
		sprime[4] = -sprime[4]; 
	elif(sprime[2] > bounds[0]):
		sprime[4] = -sprime[4]; 
	else:
		if(sprime[3] < 0):
			sprime[5] = -sprime[5]; 
		elif(sprime[3] > bounds[1]):
			sprime[5] = -sprime[5]; 

	sprime[2] = min(bounds[0],max(0,sprime[2]));
	sprime[3] = min(bounds[1],max(0,sprime[3]));

	return sprime; 



def dist(s):
	return np.sqrt((s[0]-s[2])**2 + (s[1]-s[3])**2); 

def generate_r(s,a):

	if(a<4):
		if(dist(s) < 75):
			return 1; 
		else:
			return 0;
	else:
		return 0; 

	#return max(100,1/dist(s)); 

	#return 20-dist(s)


def generate_o(s,a):



	
	# if(coin < 0.01):
	# 	return np.random.choice(['Caught','Near','Far'])



	##Non-response rate
	coin = np.random.random(); 
	flipped = .02; 

	if(dist(s) > 75 and a<4):
		if(coin>flipped):
			return 'Far';
		else:
			return 'Near' 
	elif(dist(s) < 75):
		if(coin > flipped):
			return 'Near'
		else:
			return 'Far'
	else:
		coin = np.random.random(); 
		if(coin < 1-availability):
			return 'None';
		atmp = a-4; 
		#near,east,west,north,south for each
		#atmp%5 = dir
		#atmp//5 = sketch
		sk = atmp//5; 
		p = allSketches[sk]; 
		dirs = atmp%5; 

		coin = np.random.random(); 
		flipped = 1-accuracy; 

		toRet = ''; 

		di = [s[2]-p[0],s[3]-p[1]]; 
		if(np.sqrt(di[0]**2 + di[1]**2) < 75):
			if(dirs==0):
				toRet='Yes' if coin > flipped else 'No'
			else:
				toRet= 'No' if coin > flipped else 'Yes'
		if(abs(di[0]) > abs(di[1])):
			if(di[0] > 0):
				#east 
				if(dirs==1):
					toRet= 'Yes' if coin > flipped else 'No'
				else:
					toRet= 'No' if coin > flipped else 'Yes'
			else:
				#west
				if(dirs==2):
					toRet= 'Yes' if coin > flipped else 'No'
				else:
					toRet= 'No' if coin > flipped else 'Yes'

		else:
			if(di[1] > 0):
				if(dirs==3):
					toRet= 'Yes' if coin > flipped else 'No'
				else:
					toRet= 'No' if coin > flipped else 'Yes'
			else:
				if(dirs==4):
					toRet= 'Yes' if coin > flipped else 'No'
				else:
					toRet= 'No' if coin > flipped else 'Yes'

		return toRet


def estimate_value(s,h):
	#how far can you get in the depth left
	
	return min(10,1/dist(s));


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
	return False; 


def distance(a,b):
	return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def dist(s):
	return np.sqrt((s[0]-s[2])**2 + (s[1]-s[3])**2); 

def obs_weight(s,a,o):



	upWeight = 0.98; 
	downWeight = 0.02; 

	if(dist(s) > 75 and a<4):
		if(o=='Far'):
			return upWeight;
		else:
			return downWeight 
	elif(dist(s)<75):
		if(o=='Near'):
			return upWeight
		else:
			return downWeight
	else:
		if(o=='None'):
			return 1-availability; 

		upWeight = accuracy; 
		downWeight = 1-accuracy; 

		atmp = a-4; 
		#near,east,west,north,south for each
		#atmp%5 = dir
		#atmp//5 = sketch
		sk = atmp//5; 
		p = allSketches[sk]; 
		dirs = atmp%5; 


		toRet = 0; 


		di = [s[2]-p[0],s[3]-p[1]]; 
		if(np.sqrt(di[0]**2 + di[1]**2) < 75):
			if(dirs==0):
				toRet=upWeight if o=='Yes' else downWeight; 
			else:
				toRet=upWeight if o=='No' else downWeight; 
		if(abs(di[0]) > abs(di[1])):
			if(di[0] > 0):
				#east 
				if(dirs==1):
					toRet=upWeight if o=='Yes' else downWeight; 
				else:
					toRet=upWeight if o=='No' else downWeight; 
			else:
				#west
				if(dirs==2):
					toRet=upWeight if o=='Yes' else downWeight; 
				else:
					toRet=upWeight if o=='No' else downWeight; 

		else:
			if(di[1] > 0):
				if(dirs==3):
					toRet=upWeight if o=='Yes' else downWeight; 
				else:
					toRet=upWeight if o=='No' else downWeight; 
			else:
				if(dirs==4):
					toRet=upWeight if o=='Yes' else downWeight; 
				else:
					toRet=upWeight if o=='No' else downWeight; 

		return toRet; 




if __name__ == '__main__':
	
	colors = {'Near':'b','North':'r','West':'g','South':'m','East':'k'}; 

	for i in range(0,500):
		s = [0,0,np.random.random()*10,np.random.random()*10,1,1]; 
		o = generate_o(s,0); 
		plt.scatter(s[2],s[3],c=colors[o]); 

	plt.ylim([0,10]); 
	plt.xlim([0,10]); 
	plt.show(); 