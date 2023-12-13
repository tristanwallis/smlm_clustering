# -*- coding: utf-8 -*-
'''
SYNTHETIC_DATA_GENERATOR_GUI 
PYSIMPLEGUI BASED GUI TO GENERATE SYNTHETIC TRXYT FILES USED FOR NANOSCALE SPATIOTEMPORAL INDEXING CLUSTERING (NASTIC AND SEGNASTIC) OR 3D DBSCAN (BOOSH). 

Design and coding: Tristan Wallis and Alex McCann
Queensland Brain Institute
The University of Queensland
Fred Meunier: f.meunier@uq.edu.au

REQUIRED:
Python 3.8 or greater
python -m pip install colorama numpy scipy pysimplegui scikit-learn

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

NOTES: 
This script has been tested and will run as intended on Windows 7/10/11, with minor interface anomalies on Linux, and possible tk GUI performance issues on MacOS.
Feedback, suggestions and improvements are welcome. Sanctimonious critiques on the pythonic inelegance of the coding are not.

CHECK FOR UPDATES:
https://github.com/tristanwallis/smlm_clustering/releases
'''

last_changed = "20231211"

# MAIN PROG AND FUNCTIONS
if __name__ == "__main__":
	# LOAD MODULES
	print ("\nLoading modules...")
	import PySimpleGUI as sg
	import os
	from colorama import init as colorama_init
	from colorama import Fore
	from colorama import Style

	sg.set_options(dpi_awareness=True) # turns on DPI awareness (Windows only)
	sg.theme('DARKGREY11')
	colorama_init()
	os.system('cls' if os.name == 'nt' else 'clear')
	
	print(f'{Fore.GREEN}============================================================{Style.RESET_ALL}')
	print(f'{Fore.GREEN}SYNTHETIC {last_changed} initialising...{Style.RESET_ALL}')
	print(f'{Fore.GREEN}============================================================{Style.RESET_ALL}')

	# POPUP WINDOW
	popup = sg.Window("Initialising...",[[sg.T("SYNTHETIC DATA GENERATOR initialising\nLots of modules...",font=("Arial bold",18))]],finalize=True,no_titlebar = True,alpha_channel=0.9)
	
	import time
	import numpy as np
	import random
	import datetime
	from scipy.spatial import ConvexHull
	from sklearn.cluster import DBSCAN
	from sklearn import manifold, datasets, decomposition, ensemble, random_projection
	import math
	import webbrowser
	import warnings
	
	warnings.filterwarnings("ignore")
	
	# SIMPLE CONVEX HULL AROUND SPLASH CLUSTERS
	def hull(points):
		points = np.array(points)
		hull = ConvexHull(points)
		hullarea = hull.volume
		vertices = hull.vertices
		vertices = np.append(vertices,vertices[0])
		hullpoints = np.array(points[hull.vertices])
		return hullpoints,hullarea
		
	# DBSCAN
	def dbscan(points,epsilon,minpts):
		db = DBSCAN(eps=epsilon, min_samples=minpts).fit(points)
		core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
		core_samples_mask[db.core_sample_indices_] = True
		labels = db.labels_ # which sample belongs to which cluster
		clusterlist = list(set(labels)) # list of clusters
		return labels,clusterlist
			
	# CREATE AND DISPLAY EACH SPLASH MOLECULE
	def initialise_particles(graph):
		colors = ["#0f0","#010"]
		particles=100
		particle_size=1
		xmin =-180
		xmax=180
		ymin=-100
		ymax=100
		obj_list = []
		fill,line=colors
		for i in range(particles):
			startpos = [random.randint(xmin,xmax),random.randint(ymin,ymax)]
			obj = graph.draw_circle(startpos,particle_size,fill_color=fill,line_color=line,line_width=0)
			xm = random.uniform(-1, 1)
			ym = random.uniform(-1, 1)
			obj_list.append([obj,startpos,xm,ym])
		return obj_list	
		
	# CREATE ANIMATED SPLASH
	def create_splash():
		stepsize=4
		slowdown = 15
		xmin =-250
		xmax=250
		ymin=-170
		ymax=170
		epsilon=10
		minpts=3
		timeout=50
		clusters = []
		cluster_update = 250 
		cluster_color = "#900"
		canvas="#000"
		ct = 0
		sg.theme('DARKGREY11')
		graph = sg.Graph((xmax-xmin,ymax-ymin),graph_bottom_left = (xmin,ymin),graph_top_right = (xmax,ymax),background_color=canvas,key="-GRAPH-",pad=(0,0))
		layout = [
			[graph],
			[sg.Button("OK",key="-OK-")]
		]
		splash = sg.Window("Cluster Sim",layout, no_titlebar = True,finalize=True,alpha_channel=0.9,grab_anywhere=True,element_justification="c", keep_on_top = True)
		obj_list=initialise_particles(graph)
		graph.DrawText("S Y N T H E T I C v{}".format(last_changed),(0,130),color="white",font=("Any",16),text_location="center")
		graph.DrawText("Design and coding: Tristan Wallis and Alex McCann",(0,50),color="white",font=("Any",10),text_location="center")
		graph.DrawText("Queensland Brain Institute",(0,-20),color="white",font=("Any",10),text_location="center")	
		graph.DrawText("The University of Queensland",(0,-50),color="white",font=("Any",10),text_location="center")	
		graph.DrawText("Fred Meunier f.meunier@uq.edu.au",(0,-80),color="white",font=("Any",10),text_location="center")	
		graph.DrawText("PySimpleGUI: https://pypi.org/project/PySimpleGUI/",(0,-120),color="white",font=("Any",10),text_location="center")	
		while True:	
			# READ AND UPDATE VALUES
			event, values = splash.read(timeout=timeout) 
			ct += timeout
			# Exit	
			if event in (sg.WIN_CLOSED, '-OK-'): 
				break
			# UPDATE EACH PARTICLE
			dists = [] # distances travelled by all particles
			# Dbscan to check for interacting points
			allpoints = [i[1] for i in obj_list]
			labels,clusterlist = dbscan(allpoints,epsilon,minpts)
			# Alter particle movement		
			for num,obj in enumerate(obj_list):
				dot,pos,xm,ym = obj 
				splash["-GRAPH-"].move_figure(dot,xm,ym)
				pos[0]+=xm
				pos[1]+=ym
				# Closed universe
				if pos[0] > xmax:
					pos[0] = xmin
					splash["-GRAPH-"].RelocateFigure(dot,pos[0],pos[1])
				if pos[0] < xmin:
					pos[0] = xmax
					splash["-GRAPH-"].RelocateFigure(dot,pos[0],pos[1])			
				if pos[1] > ymax:
					pos[1] = ymin
					splash["-GRAPH-"].RelocateFigure(dot,pos[0],pos[1])	
				if pos[1] < ymin:
					pos[1] = ymax	
					splash["-GRAPH-"].RelocateFigure(dot,pos[0],pos[1])	
				# Lower speed in a cluster		
				if labels[num] > -1:
					obj[2] = random.uniform(-(slowdown/100)*stepsize, (slowdown/100)*stepsize)
					obj[3] = random.uniform(-(slowdown/100)*stepsize, (slowdown/100)*stepsize)
				# Randomly change direction and speed
				else:	
					obj[2] = random.uniform(-stepsize, stepsize)
					obj[3] = random.uniform(-stepsize, stepsize)
			# Draw borders around clusters
			if ct > cluster_update:
				ct = 0
				if len(clusters) > 0:
					for cluster in clusters:
						splash["-GRAPH-"].delete_figure(cluster)
					clusters = []
				allpoints = [i[1] for i in obj_list]
				labels,clusterlist = dbscan(allpoints,epsilon*1.5,minpts)	
				clusterdict = {i:[] for i in clusterlist}
				clust_traj = [i for i in labels if i > -1]
				clust_radii = []	
				for num,obj in enumerate(obj_list):
					clusterdict[labels[num]].append(obj[1])
				for clust in clusterdict:
					if clust > -1:
						clusterpoints = clusterdict[clust]
						try:
							hullpoints,hullarea = hull(clusterpoints)
							cluster = splash["-GRAPH-"].draw_polygon(hullpoints,line_width=2,line_color=cluster_color,fill_color=canvas)
							splash["-GRAPH-"].send_figure_to_back(cluster)
							clusters.append(cluster)
						except:
							pass
		return splash
	
	# USE HARD CODED DEFAULTS
	def reset_defaults():
		print ("\nUsing default GUI settings...")
		global acquisition_time, frame_time, x_size, y_size, seed_num, radius, min_traj_num, max_traj_num, orbit, min_traj_length, max_traj_length, steplength, noise, unconst, hotspotprobability, hotspotmax, trxyt_num 
		acquisition_time = 320 
		frame_time = 0.02
		x_size = 10
		y_size = 10
		seed_num = 80
		radius = 0.1
		min_traj_num = 4
		max_traj_num = 12
		orbit = True
		min_traj_length = 8
		max_traj_length = 30
		steplength = 0.1
		noise = 1000
		unconst = 2.0
		hotspotprobability = 0.2
		hotspotmax = 3
		trxyt_num = 1
		return 
	
	# SAVE SETTINGS
	def save_defaults():
		print ("\nSaving GUI settings to synthetic_data_generator_gui.defaults...")
		with open("synthetic_data_generator_gui.defaults","w") as outfile:
			outfile.write("{}\t{}\n".format("Acquisition time (s)",acquisition_time))
			outfile.write("{}\t{}\n".format("Frame time (s)",frame_time))
			outfile.write("{}\t{}\n".format("x-axis size (um)",x_size))
			outfile.write("{}\t{}\n".format("y-axis size (um)",y_size))
			outfile.write("{}\t{}\n".format("Seed number",seed_num))
			outfile.write("{}\t{}\n".format("Radius (um)",radius))
			outfile.write("{}\t{}\n".format("Minimum trajectory number",min_traj_num))
			outfile.write("{}\t{}\n".format("Maximum trajectory number",max_traj_num))
			outfile.write("{}\t{}\n".format("Trajectories orbit seed",orbit))
			outfile.write("{}\t{}\n".format("Minimum trajectory length",min_traj_length))
			outfile.write("{}\t{}\n".format("Maximum trajectory length",max_traj_length))
			outfile.write("{}\t{}\n".format("Clustered step length (um)",steplength))
			outfile.write("{}\t{}\n".format("Unclustered trajectory number",noise))
			outfile.write("{}\t{}\n".format("Step length multiplier",unconst))
			outfile.write("{}\t{}\n".format("Hotspot probability",hotspotprobability))
			outfile.write("{}\t{}\n".format("Maximum hotspot trajectory number",hotspotmax))
			outfile.write("{}\t{}\n".format("Number of trxyt files to generate",trxyt_num))
		return
	
	# LOAD DEFAULTS
	def load_defaults():
		global acquisition_time, frame_time, x_size, y_size, seed_num, radius, min_traj_num, max_traj_num, orbit, min_traj_length, max_traj_length, steplength, noise, unconst, hotspotprobability, hotspotmax, trxyt_num
		try:
			with open ("synthetic_data_generator_gui.defaults","r") as infile:
				print ("\nLoading GUI settings from synthetic_data_generator_gui.defaults...")
				defaultdict = {}
				for line in infile:
					spl = line.split("\t")
					defaultdict[spl[0]] = spl[1].strip()
			acquisition_time = int(defaultdict["Acquisition time (s)"])
			frame_time = float(defaultdict["Frame time (s)"])
			x_size = int(defaultdict["x-axis size (um)"])
			y_size = int(defaultdict["y-axis size (um)"])
			seed_num = int(defaultdict["Seed number"])
			radius = float(defaultdict["Radius (um)"])
			min_traj_num = int(defaultdict["Minimum trajectory number"])
			max_traj_num = int(defaultdict["Maximum trajectory number"])
			orbit = defaultdict["Trajectories orbit seed"]
			min_traj_length = int(defaultdict["Minimum trajectory length"])
			max_traj_length = int(defaultdict["Maximum trajectory length"])
			steplength = float(defaultdict["Clustered step length (um)"])
			noise = int(defaultdict["Unclustered trajectory number"])
			unconst = float(defaultdict["Step length multiplier"])
			hotspotprobability = float(defaultdict["Hotspot probability"])
			hotspotmax = int(defaultdict["Maximum hotspot trajectory number"])
			trxyt_num = int(defaultdict["Number of trxyt files to generate"])
		except:
			print ("\nSettings could not be loaded")
		return
		
	# UPDATE GUI BUTTONS
	def update_buttons():
		window.Element("-ACQ_TIME-").update(acquisition_time)
		window.Element("-FRAME_TIME-").update(frame_time)
		window.Element("-SEED-").update(seed_num)
		window.Element("-MIN_TRAJ_NUM-").update(min_traj_num)
		window.Element("-MAX_TRAJ_NUM-").update(max_traj_num)
		window.Element("-MIN_TRAJ_LEN-").update(min_traj_length)
		window.Element("-MAX_TRAJ_LEN-").update(max_traj_length)
		window.Element("-X_SIZE-").update(x_size)
		window.Element("-Y_SIZE-").update(y_size)
		window.Element("-RADIUS-").update(radius)
		window.Element("-STEP_LEN-").update(steplength)
		window.Element("-NOISE-").update(noise)
		window.Element("-STEP_LEN_UNCLUST-").update(unconst)
		window.Element("-HOTSPOT_PROB-").update(hotspotprobability)
		window.Element("-MAX_HOTSPOT_TRAJ-").update(hotspotmax)
		window.Element("-ORBIT-").update(orbit)
		window.Element("-TRXYT_NUM-").update(trxyt_num)
		return
	
	# CHECK VARIABLES
	def check_variables():
		global acquisition_time, frame_time, seed_num, min_traj_num, max_traj_num, min_traj_length, max_traj_length, x_size, y_size, radius, steplength, noise, unconst, hotspotprobability, hotspotmax, orbit, trxyt_num
		
		try:
			acquisition_time = int(acquisition_time)
			if acquisition_time < 1:
				acquisition_time = 1
		except:
			acquisition_time = 320
		
		try:
			frame_time = float(frame_time)
			if frame_time <= 0:
				frame_time = 0.02
		except:
			frame_time = 0.02
		
		try: 
			seed_num = int(seed_num)
			if seed_num < 1:
				seed_num = 80
		except:
			seed_num = 80
		
		try: 
			min_traj_num = int(min_traj_num)
			if min_traj_num < 4:
				min_traj_num = 4
		except:
			min_traj_num = 4
		
		try:
			max_traj_num = int(max_traj_num)
			if max_traj_num <= min_traj_num:
				max_traj_num = min_traj_num + 1
			if max_traj_num > 10000:
				max_traj_num = 10000
		except:
			max_traj_num = 12
			
		try:
			min_traj_length = int(min_traj_length)
			if min_traj_length < 8:
				min_traj_length = 8
		except:
			min_traj_length = 8
			
		try:
			max_traj_length = int(max_traj_length)
		except:
			max_traj_length = 100
			
		if max_traj_length < min_traj_length:
			min_traj_length = 8
			max_traj_length = 100
			
		try:
			x_size = int(x_size)
			if x_size < 1:
				x_size = 10
		except:
			x_size = 10
			
		try:
			y_size = int(y_size)
			if y_size < 1:
				y_size = 10
		except:
			y_size = 10
			
		try:
			radius = float(radius)
			if radius <= 0:
				radius = 0.1
		except:
			radius = 0.1
			
		try: 
			steplength = float(steplength)
			if steplength <= 0:
				steplength = 0.1
		except:
			steplength = 0.1
			
		try:
			noise = int(noise)
			if noise < 0:
				noise = 0
		except:
			noise = 1000
		
		try:
			unconst = float(unconst)
			if unconst < 1.0:
				unconst = 1.0
		except:
			unconst = 2.0
			
		try:
			hotspotprobability = float(hotspotprobability)
			if hotspotprobability <= 0:
				hotspotprobability = 0.2
		except:
			hotspotprobability = 0.2
			
		try:
			hotspotmax = int(hotspotmax)
			if hotspotmax < 2:
				hotspotmax = 2
		except:
			hotspotmax = 3
		
		try:
			trxyt_num = int(trxyt_num)
			if trxyt_num <= 0:
				trxyt_num = 1
		except:
			trxyt_num = 1
			
		return
		
	# VALS
	clusterdict = {} # dictionary holding clusters
	
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

	# GENERATE TRXYT
	def generate():
		global trajectories, clusttrajcounter, clustercounter, traj_nums, radii
		# Seeds
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

		print ("Total traj:",clusttrajcounter + noise)
		print ("Clustered traj:",clusttrajcounter)
		print ("Total clusters:", clustercounter)
		print ("Avg traj per cluster:", np.average(traj_nums))
		print ("Avg cluster radius:", np.average(radii))
		
		return()
		
	# Output
	def output(traj_num_ct, stamp):
		print ("Writing TRXYT...")
		with open("synthetic_data_output_{}/synthetic_data_{}_{}.trxyt".format(stamp, stamp,traj_num_ct+1),"w") as outfile: 
			for tr,traj in enumerate(trajectories,start=1):
				for seg in traj:
					x,y,t = seg
					outline = "{} {} {} {}\n".format(tr,x,y,t)
					outfile.write(outline)
		print ("Writing metrics...")
		with open("synthetic_data_output_{}/synthetic_data_{}_{}_metrics.tsv".format(stamp,stamp,traj_num_ct+1),"w") as outfile: 
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

			outfile.write("\nGENERATED METRICS:\n=================\n")
			outfile.write("TOTAL TRAJECTORIES: {}\n".format(clusttrajcounter + noise))
			outfile.write("CLUSTERED TRAJECTORIES: {}\n".format(clusttrajcounter))	
			outfile.write("UNCLUSTERED TRAJECTORIES: {}\n".format(noise))	
			outfile.write("TOTAL CLUSTERS: {}\n".format(clustercounter))
			outfile.write("SINGLETON CLUSTERS: {}\n".format(seed_num))
			outfile.write("AVERAGE TRAJECTORIES PER CLUSTER: {} +/- {}\n".format(np.average(traj_nums),np.std(traj_nums)/math.sqrt(len(traj_nums))))	
			outfile.write("AVERAGE CLUSTER RADIUS: {} +/- {}\n".format(np.average(radii),np.std(radii)/math.sqrt(len(radii))))
		return ()
			
	# GET INITIAL VALUES FOR GUI
	cwd = os.path.dirname(os.path.abspath(__file__))
	os.chdir(cwd)
	initialdir = cwd
	if os.path.isfile("synthetic_data_generator_gui.defaults"):
		load_defaults()
	else:
		reset_defaults()
		save_defaults()
	
	# GUI LAYOUT
	appFont = ("Any 12")
	sg.set_options(font=appFont)
	sg.theme('DARKGREY11')
	
	# Menu
	menu_def = [
		['&File', ['&Load settings', '&Save settings','&Default settings','&Exit']],
		['&Info', ['&About', '&Help','&Licence','&Updates']],
		]
		
	syn_acq_layout = [
		[sg.T("    Acquisition time (s):", tooltip = "Pretend length of time taken to 'acquire' all frames\nUnits = seconds"), sg.In(acquisition_time, size = (7,1), key = '-ACQ_TIME-'),sg.Push(),sg.T("Frame time (s):", tooltip = "Pretend length of time taken to 'acquire' a single frame\nUnits = seconds"), sg.In(frame_time, size = (7,1), key = '-FRAME_TIME-')],
		[sg.T("         x-axis size (um):", tooltip = "Pretend size of the frame along the x-axis\nUnits = microns"), sg.In(x_size, size = (7,1), key = '-X_SIZE-'), sg.Push(), sg.T("y-axis size (um):", tooltip = "Pretend size of the frame along the y-axis\nUnits = microns"), sg.In(y_size, size = (7,1), key = '-Y_SIZE-')],
		]
		
	syn_seed_layout = [
		[sg.T("                      Seed #:", tooltip = "Number of spawn points (seeds) from which a single cluster will occur"), sg.In(seed_num, size = (7,1), key = '-SEED-'), sg.Push(), sg.T("Seed radius (um):", tooltip = "Radius around each seed to make trajectories\nUnits = microns"), sg.In(radius, size = (7,1), key = '-RADIUS-')],
		[sg.T("                   Min traj #:", tooltip = "Minimum number of trajectories around each seed"), sg.In(min_traj_num, size = (7,1), key = '-MIN_TRAJ_NUM-'),sg.Push(),sg.T("          Max traj #:",tooltip = "Maximum number of trajectories around each seed"), sg.In(max_traj_num, size = (7,1), key = '-MAX_TRAJ_NUM-')],
		[sg.T("           Min traj length:", tooltip = "Minimum number of trajectory steps"), sg.In(min_traj_length, size = (7,1), key = '-MIN_TRAJ_LEN-'), sg.Push(), sg.T("Max traj length:", tooltip = "Maximum number of trajectory steps"), sg.In(max_traj_length, size = (7,1), key = '-MAX_TRAJ_LEN-')],
		[sg.T("       Step length (um):", tooltip = "Maximum step length within a trajectory\nUnits = microns"), sg.In(steplength, size = (7,1), key = '-STEP_LEN-')],
		[sg.T("       Clustered traj orbit seed:", tooltip = "Clustered trajectories orbit their spawn point rather than random walking"), sg.Checkbox("", default = True, key = '-ORBIT-')],
		]
	
	syn_unclustered_traj_layout = [
		[sg.T("      Unclustered traj #:", tooltip = "Number of unclustered trajectories"), sg.In(noise, size = (7,1), key = '-NOISE-')],
		[sg.T("Step length multiplier:", tooltip = "Step length multiplier used to calculate unclustered trajectory step length"), sg.In(unconst, size = (7,1), key = '-STEP_LEN_UNCLUST-')],
		]
		
	syn_hotspot_layout = [
		[sg.T("             Hotspot prob:", tooltip = "Chance of a given seed point generating multiple spatially overlapping but temporally distinct clusters"), sg.In(hotspotprobability, size = (7,1), key = '-HOTSPOT_PROB-')],
		[sg.T("     Max hotspot traj #:", tooltip = "Maximum number of temporal clusters at a given hotspot"), sg.In(hotspotmax, size = (7,1), key = '-MAX_HOTSPOT_TRAJ-')],
		]
		
	layout = [
		[sg.Menu(menu_def)],
		[sg.T("SYNTHETIC",font=("Any",20))],
		[sg.T("Generate synthetic TRXYT data using the below parameters:", font = ("Any 12 italic"))],
		[sg.Frame("ACQUISITION", syn_acq_layout, expand_x = True, pad = ((0,0), (15,15)))],
		[sg.Frame("CLUSTERED TRAJECTORIES", syn_seed_layout, expand_x = True, pad = ((0,0), (0,15)))],
		[sg.Frame("UNCLUSTERED TRAJECTORIES",syn_unclustered_traj_layout, expand_x = True, pad = ((0,0), (0,15)))],
		[sg.Frame("HOTSPOTS", syn_hotspot_layout, expand_x = True, pad = ((0,0), (0,15)))],
		[sg.T("# of TRXYT files to generate:"), sg.In(trxyt_num, key = '-TRXYT_NUM-', size = (7,1)),sg.Push(),sg.B("GENERATE TRXYT", key = '-GENERATE-'),sg.Push()],
	]

	window = sg.Window('SYNTHETIC Data Generator v{}'.format(last_changed),layout)
	popup.close()

	# MAIN LOOP
	while True:
		# Read events and values
		event, values = window.read(timeout = 5000)
		
		# Exit	
		if event == sg.WIN_CLOSED or event == 'Exit':  
			break
		
		# Values
		acquisition_time = values['-ACQ_TIME-'] # pretend length of acquisition
		frame_time = values['-FRAME_TIME-'] # sec 
		seed_num = values['-SEED-'] # number of seed points, where each point will be a single cluster
		min_traj_num = values['-MIN_TRAJ_NUM-'] # min number of trajectories around each seed (default 4)
		max_traj_num = values['-MAX_TRAJ_NUM-'] # max number of trajectories around each seed (default 16)
		min_traj_length = values['-MIN_TRAJ_LEN-'] # min number of trajectory steps (default 8)
		max_traj_length = values['-MAX_TRAJ_LEN-'] # max number of trajectory steps (default 30)
		x_size = values['-X_SIZE-'] # pretend microns
		y_size = values['-Y_SIZE-'] # pretend microns
		radius = values['-RADIUS-'] # radius around each seed to make trajectories (um)
		steplength = values['-STEP_LEN-'] # maximum step length within trajectory (um)
		noise = values['-NOISE-'] # number of unclustered trajectories 
		unconst = values['-STEP_LEN_UNCLUST-'] # steplength multiplier of unclustered trajectories
		hotspotprobability = values['-HOTSPOT_PROB-'] # chance of a given seed point generating multiple spatially overlapping but temporally distinct clusters
		hotspotmax = values['-MAX_HOTSPOT_TRAJ-'] # maximum number of temporal clusters at a given hotspot
		orbit = values['-ORBIT-'] # clustered trajectories orbit their spawn point rather than random walking
		trxyt_num = values['-TRXYT_NUM-']

		# Check variables
		check_variables()
		
		# Reset to hard coded default values
		if event == 'Default settings':
			reset_defaults()
			update_buttons()
				
		# Save settings
		if event == 'Save settings':
			save_defaults()
				
		# Load settings
		if event == 'Load settings':
			load_defaults()
			update_buttons()
						
		# About
		if event == 'About':
			splash = create_splash()	
			splash.close()
			
		# Help	
		if event == 'Help':
			sg.Popup(
				"Help",
				"This program generates synthetic .trxyt files based on parameters defined by the user, for use in nanoscale spatiotemporal indexing clustering (NASTIC and segNASTIC) or 3D DBSCAN (BOOSH).",
				"ACQUSITION - parameters defining the dimensions of the \n     pretend acquisition data.",
				"CLUSTERED TRAJECTORIES - parameters defining the behaviour \n     of clustered trajectories. \n     Seed = point at which a single cluster will form. \n     Step length = step size within a clustered trajectory.",
				"UNCLUSTERED TRAJECTORIES - parameters defining the behaviour \n     of unclustered trajectories. \n     Step length multiplier = value multiplied by the \n     clustered trajectory step length to obtain the \n     step size within unclustered trajectories.",
				"HOTSPOTS - parameters defining the behaviour of hotspots. \n     Hotspot = point at which multiple spatially overlapping \n     but temporally distinct clusters form.",
				"GENERATE TRXYT - generates synthetic .trxyt files based on \n     the selected parameters, and a corresponding \n     metrics.tsv file containing the parameters that were \n     selected, and the metrics which they generated. Both \n     file types are saved in the same directory as the \n     synthetic_data_generator_gui.py python script, in a \n     folder that is generated with the naming format \n     'synthetic_data_output_YYYYMMDD-HHMMSS'.",
				"Tristan Wallis, Alex McCann {}".format(last_changed),
				no_titlebar = True,
				grab_anywhere = True,
				keep_on_top = True,
				)	
		# Check for updates
		if event == 'Updates':
			webbrowser.open("https://github.com/tristanwallis/smlm_clustering/releases",new=2)
			
		# Licence	
		if event == 'Licence':
			sg.Popup(
				"Licence",
				"Creative Commons CC BY 4.0",
				"https://creativecommons.org/licenses/by/4.0/legalcode", 
				no_titlebar = True,
				grab_anywhere = True,
				keep_on_top = True,
				)				
				
		if event == '-GENERATE-':
			time.sleep(0.5)
			traj_num_ct = 0
			stamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now()) # datestamp
			try:
				os.mkdir("synthetic_data_output_{}".format(stamp))
			except:
				sg.Popup("Alert", "Error generating output folder (folder name already exists). Please try again")
				
			while traj_num_ct < trxyt_num:
				print("\nGenerating trxyt file # {}\n------------------------------------------------------------".format(traj_num_ct+1))
				generate()
				output(traj_num_ct, stamp)
				traj_num_ct+=1
			print("\nDONE!")
			print("\nTRXYT and metrics files saved to: {}".format(cwd) + "\\synthetic_data_output_{}".format(stamp) + "\n")
		
		# Update buttons	
		if event: 
			update_buttons()
			
	print ("\nExiting...")
	window.close()	
	quit()