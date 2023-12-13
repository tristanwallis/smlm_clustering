# -*- coding: utf-8 -*-
'''
SYNTHETIC_DATA_GENERATOR_CLI
COMMAND LINE (CLI) VERSION FOR THE GENERATION OF SYNTHETIC TRXYT FILES
GENERATE SYNTHETIC TRXYT FILES USED FOR NANOSCALE SPATIOTEMPORAL INDEXING CLUSTERING (NASTIC AND SEGNASTIC) OR 3D DBSCAN (BOOSH).

Design and coding: Tristan Wallis
Queensland Brain Institute
The University of Queensland
Fred Meunier: f.meunier@uq.edu.au

REQUIRED:
Python 3.8 or greater
python -m pip install numpy scipy scikit-learn

OUTPUT:
[1] .trxyt files - used as input files for NASTIC/segNASTIC/BOOSH
File naming format: synthetic_YYYYMMDD-HHMMSS_file#_.trxyt
Space separated: Trajectory X(um) Y(um) T(sec)  
No headers

example: 

1 9.0117 39.86 0.02
1 8.9603 39.837 0.04
1 9.093 39.958 0.06
1 9.0645 39.975 0.08
2 9.1191 39.932 0.1
2 8.9266 39.915 0.12
etc

[2] metrics.tsv files - contain the parameters that were selected by the user, and the metrics that were generated from these parameters which are then used to generate the .trxyt file.
File naming format: synthetic_YYYYMMDD-HHMMSS_file#_metrics.tsv

example:

PARAMETERS:
==========
ACQUISITION TIME (s): 320
FRAME TIME (s): 0.02
SEED NUMBER: 80
MIN TRAJ AROUND SEED: 4
MAX TRAJ AROUND SEED: 12
MIN TRAJ STEPS: 8
MAX TRAJ STEPS: 30
X SIZE (um): 10
Y SIZE (um): 10
RADIUS (um): 0.1
MAX STEPLENGTH (um): 0.1
UNCLUSTERED BACKGROUND TRAJ: 1000
UNCLUST STEPLENGTH MULTIPLIER: 2
HOTSPOT PROBABILITY: 0.2
MAX CLUSTERS PER HOTSPOT: 3
CLUSTER TRAJ ORBIT: True

GENERATED METRICS:
=================
TOTAL TRAJECTORIES: 1895
CLUSTERED TRAJECTORIES: 895
UNCLUSTERED TRAJECTORIES: 1000
TOTAL CLUSTERS: 110
SINGLETON CLUSTERS: 80
AVERAGE TRAJECTORIES PER CLUSTER: 8.136363636363637 +/- 0.2358306132306391
AVERAGE CLUSTER RADIUS: 0.08004795415240253 +/- 0.0009018811959358476

Generated .trxyt files [1] and corresponding metrics.tsv files [2] are saved within the same directory as the  synthetic_data_generator_gui.py script, in a folder that is created with the naming format: synthetic_data_output_YYYYMMDD-HHMMSS

USAGE:
Parameters need to be changed in the #VARS section below. 

CHECK FOR UPDATES:
https://github.com/tristanwallis/smlm_clustering/releases
'''

lastchanged = "20231208"

# LOAD MODULES
print ("Loading modules...")
import os
import numpy as np
import random
import datetime
from scipy.spatial import ConvexHull
import math

# VARS
acquisition_time = 320 # pretend length of acquisition
frame_time = 0.02 # sec 
seed_num = 80 # number of seed points, where each point will be a single cluster
min_traj_num = 4 # min number of trajectories around each seed (default 4)
max_traj_num = 12 # max number of trajectories around each seed (default 16)
min_traj_length = 8 # min number of trajectory steps (default 8)
max_traj_length = 30 # max number of trajectory steps (default 30)
x_size = 10 # pretend microns
y_size = 10 # pretend microns
radius = 0.1 # radius around each seed to make trajectories (um)
steplength = 0.1 # maximum step length within trajectory (um)
noise = 1000 # number of unclustered trajectories 
unconst = 2 # steplength multiplier of unclustered trajectories
hotspotprobability = 0.2 # chance of a given seed point generating multiple spatially overlapping but temporally distinct clusters
hotspotmax = 3 # maximum number of temporal clusters at a given hotspot
orbit = True # clustered trajectories orbit their spawn point rather than random walking


stamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now()) # datestamp
clusterdict = {} # dictionary holding clusters

# Initial directory
cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd)
initialdir = cwd

os.system('cls' if os.name == 'nt' else 'clear')
print ("SYNTHETIC CLI - Tristan Wallis {}\n-----------------------------------------------------".format(lastchanged))

# FUNCTIONS
		
# CONVEX HULL OF EXTERNAL POINTS, AND THEN INTERNAL POINTS
def double_hull(points):
	# Get the hull of the original points
	all_points = np.array(points)
	ext_hull = ConvexHull(all_points)
	ext_area = ext_hull.volume
	vertices = ext_hull.vertices
	vertices = np.append(vertices,vertices[0])
	ext_x = [all_points[vertex][0] for vertex in vertices]
	ext_y = [all_points[vertex][1] for vertex in vertices]
	ext_points = np.array(all_points[ext_hull.vertices])
		
	# Get the hull of the points inside the hull
	int_points = np.array([x for x in all_points if x not in ext_points])
	try:
		int_hull = ConvexHull(int_points)
		int_area = int_hull.volume
		vertices = int_hull.vertices
		vertices = np.append(vertices,vertices[0])
		int_x = [int_points[vertex][0] for vertex in vertices]
		int_y = [int_points[vertex][1] for vertex in vertices]
	except:
		int_x,int_y,int_area = ext_x,ext_y,ext_area
	return ext_x,ext_y,ext_area,int_x,int_y,int_area

# SEEDS
print ("Generating spatiotemporal cluster seeds...")
seeds = []
for i in range(seed_num):
	x = random.random()*x_size # random x
	y = random.random()*y_size # random y
	t = random.random()*acquisition_time # random t
	seeds.append([x,y,t])

# TRAJECTORIES
print ("Generating trajectories around cluster seeds...")
clustercounter = 0
clusttrajcounter = 0
trajectories = []
for seed in seeds:
	# Original cluster at each seed point
	clustercounter+=1
	x_seed,y_seed,t_seed = seed[0],seed[1],seed[2] # x, y and t for this cluster
	trajnum = random.randint(min_traj_num,max_traj_num) # number of trajectories for this cluster
	clusterdict[clustercounter]={"trajectories":[]} # empty trajectory dictionary for this cluster
	
	#print (x_seed,y_seed,t_seed,trajnum)
	for j in range(trajnum):
		x_orig = x_seed + 0.5*(random.uniform(-radius,radius)) # starting x point for trajectory 
		y_orig = y_seed + 0.5*(random.uniform(-radius,radius))# starting y point for trajectory
		t = round((t_seed + (random.random()*10))/2,2)*2 # starting time, within 10 sec of the cluster t 
		traj_length = random.randint(min_traj_length,max_traj_length) # steps for this trajectory
		traj = []
		x = x_orig
		y = y_orig
		for i in range(traj_length):
			if orbit:
				# Random walk constrained around spawn point
				x = x_orig  + 0.5*(random.uniform(-steplength,steplength))
				y = y_orig  + 0.5*(random.uniform(-steplength,steplength))
			else:	
				# Random walk unconstrained, can wander from spawn point
				x += 0.5*(random.uniform(-steplength,steplength))	
				y += 0.5*(random.uniform(-steplength,steplength))	
			t += frame_time
			traj.append([x,y,t])
			
		trajectories.append(traj)
		clusttrajcounter +=1
		clusterdict[clustercounter]["trajectories"].append(traj)
		
	# Spatially overlapping, temporally distinct clusters at each seed point 	
	if random.random() < hotspotprobability:
		for k in range(0,random.randint(1,hotspotmax-1)):
			clustercounter+=1
			clusterdict[clustercounter]={"trajectories":[]}
			x_seed = seed[0]+ random.uniform(-0.25,0.25)*radius # hotspot cluster x
			y_seed = seed[1]+ random.uniform(-0.25,0.25)*radius # hotspot cluster y
			t_seed = random.random()*acquisition_time
			trajnum = random.randint(min_traj_num,max_traj_num)
			for tr in range(trajnum):
				x_orig = x_seed + 0.5*(random.uniform(-radius,radius))
				y_orig = y_seed + 0.5*(random.uniform(-radius,radius))
				t = round((t_seed + (random.random()*10))/2,2)*2
				traj_length = random.randint(min_traj_length,max_traj_length)
				traj = []
				x = x_orig
				y = y_orig
				for i in range(traj_length):
					if orbit:
						# Random walk constrained around spawn point
						x = x_orig  + 0.5*(random.uniform(-steplength,steplength))
						y = y_orig  + 0.5*(random.uniform(-steplength,steplength))
					else:
						# Random walk unconstrained, can wander from spawn point
						x += 0.5*(random.uniform(-steplength,steplength))	
						y += 0.5*(random.uniform(-steplength,steplength))
					
					t += frame_time
					traj.append([x,y,t])
				trajectories.append(traj)
				clusttrajcounter +=1				
				clusterdict[clustercounter]["trajectories"].append(traj)
		
# Noise		
print ("Generating unclustered trajectories with higher mobility...")
for i in range(noise):
	x_orig = random.random()*x_size
	y_orig = random.random()*y_size	
	t = round((random.random()*acquisition_time)/2,2)*2
	traj_length = random.randint(min_traj_length,max_traj_length)
	traj=[]
	x = x_orig
	y = y_orig
	for i in range(traj_length):
		# Random walk unconstrained, can wander from spawn point
		x += 0.5*unconst*(random.uniform(-steplength,steplength))	
		y += 0.5*unconst*(random.uniform(-steplength,steplength))	
		t += frame_time	
		traj.append([x,y,t])
	trajectories.append(traj)		

# Metrics
traj_nums = []
radii= []
for num in clusterdict:
	cluster_trajectories = clusterdict[num]["trajectories"]
	clusterdict[num]["traj_num"]=len(cluster_trajectories) # number of trajectories in each cluster
	clusterpoints = [point[:2]  for traj in cluster_trajectories for point in traj] # all x,y points for trajectories in cluster 
	ext_x,ext_y,ext_area,int_x,int_y,int_area = double_hull(clusterpoints) # internal and external convex hull of cluster points 
	clusterdict[num]["area"] = int_area # internal hull area as cluster area (um2)
	clusterdict[num]["radius"] = math.sqrt(int_area/math.pi) # radius of cluster internal hull (um)
	traj_nums.append(clusterdict[num]["traj_num"])
	radii.append(clusterdict[num]["radius"])
	#print ("{},{},{}".format(num,clusterdict[num]["traj_num"],clusterdict[num]["radius"]))

print ("Total traj:",clusttrajcounter + noise)
print ("Clustered traj:",clusttrajcounter)
print ("Total clusters:", clustercounter)
print ("Avg traj per cluster:", np.average(traj_nums))
print ("Avg cluster radius:", np.average(radii))

# Output
print ("Writing TRXYT...")
with open("synthetic_{}.trxyt".format(stamp),"w") as outfile: 
	for tr,traj in enumerate(trajectories,start=1):
		for seg in traj:
			x,y,t = seg
			outline = "{} {} {} {}\n".format(tr,x,y,t)
			outfile.write(outline)
print ("Writing metrics...")
with open("synthetic_{}_metrics.tsv".format(stamp),"w") as outfile: 
	outfile.write("PARAMETERS:\n==========\n")
	outfile.write("ACQUISITION TIME (s): {}\n".format(acquisition_time))	
	outfile.write("FRAME TIME (s): {}\n".format(frame_time))	
	outfile.write("SEED NUMBER: {}\n".format(seed_num))	
	outfile.write("MIN TRAJ AROUND SEED: {}\n".format(min_traj_num))	
	outfile.write("MAX TRAJ AROUND SEED: {}\n".format(max_traj_num))	
	outfile.write("MIN TRAJ STEPS: {}\n".format(min_traj_length))	
	outfile.write("MAX TRAJ STEPS: {}\n".format(max_traj_length))	
	outfile.write("X SIZE (um): {}\n".format(x_size ))	
	outfile.write("Y SIZE (um): {}\n".format(y_size))	
	outfile.write("RADIUS (um): {}\n".format(radius))	
	outfile.write("MAX STEPLENGTH (um): {}\n".format(steplength))	
	outfile.write("UNCLUSTERED BACKGROUND TRAJ: {}\n".format(noise))	
	outfile.write("UNCLUST STEPLENGTH MULTIPLIER: {}\n".format(unconst))	
	outfile.write("HOTSPOT PROBABILITY: {}\n".format(hotspotprobability))	
	outfile.write("MAX CLUSTERS PER HOTSPOT: {}\n".format(hotspotmax))	
	outfile.write("CLUSTER TRAJ ORBIT: {}\n".format(orbit))	

	outfile.write("\nGENERATED METRICS:\n===============\n")
	outfile.write("TOTAL TRAJECTORIES: {}\n".format(clusttrajcounter + noise))
	outfile.write("CLUSTERED TRAJECTORIES: {}\n".format(clusttrajcounter))	
	outfile.write("UNCLUSTERED TRAJECTORIES: {}\n".format(noise))	
	outfile.write("TOTAL CLUSTERS: {}\n".format(clustercounter))
	outfile.write("SINGLETON CLUSTERS: {}\n".format(seed_num))
	outfile.write("AVERAGE TRAJECTORIES PER CLUSTER: {} +/- {}\n".format(np.average(traj_nums),np.std(traj_nums)/math.sqrt(len(traj_nums))))	
	outfile.write("AVERAGE CLUSTER RADIUS: {} +/- {}\n".format(np.average(radii),np.std(radii)/math.sqrt(len(radii))))	
	
cont = input("Done. Return to exit.")