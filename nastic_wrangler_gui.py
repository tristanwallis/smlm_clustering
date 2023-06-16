'''
NASTIC WRANGLER
PYSIMPLEGUI BASED GUI TO PERFORM META ANALYSIS OF METRICS PRODUCED BY NANOSCALE SPATIOTEMPORAL INDEXING CLUSTERING (NASTIC AND SEGNASTIC) OR 3D DBSCAN (BOOSH)

Design and coding: Tristan Wallis
Additional coding: Alex McCann
Queensland Brain Institute
The University of Queensland
Fred Meunier: f.meunier@uq.edu.au

REQUIRED:
Python 3.8 or greater
python -m pip install colorama matplotlib numpy pandas scipy Pillow pysimplegui seaborn scikit-learn  

INPUT:
metrics.tsv files inside the directories produced by NASTIC/segNASTIC/BOOSH 
The directories need to be sorted based on the experimental conditions being compared.

eg 
/control/sample1_NASTIC_20230607-123456/metrics.tsv
/control/sample2_NASTIC_20230607-134562/metrics.tsv
/control/sample3_NASTIC_20230607-143456/metrics.tsv
etc

/stimulation/sample1_NASTIC_20230608-091234/metrics.tsv
/stimulation/sample2_NASTIC_20230608-114523/metrics.tsv
/stimulation/sample3_NASTIC_20230608-123412/metrics.tsv
etc

The program will recursively search through all directories and subdirectories in "control" and "stimulation".

NOTES:
This script has been tested and will run as intended on Windows 7/10/11, with minor interface anomalies on Linux, and possible tk GUI performance issues on MacOS. If GUI display issues are encountered, changing screen resolution may resolve these issues. If there are issues with clicking on the tick boxes in the 'Files to include' list, adding a short delay (~1s) between clicks should resolve this issue.  
Feedback, suggestions and improvements are welcome. Sanctimonious critiques on the pythonic inelegance of the coding are not.
'''



last_changed = 20230615

# MAIN PROG AND FUNCTIONS
if __name__ == "__main__":

	# LOAD MODULES
	import PySimpleGUI as sg
	import os
	from colorama import init as colorama_init
	from colorama import Fore
	from colorama import Style
	
	sg.set_options(dpi_awareness=True) # turns on DPI awareness (Windows only)
	sg.theme('DARKGREY11')
	colorama_init()
	os.system('cls' if os.name == 'nt' else 'clear')
	
	print(f'{Fore.GREEN}=================================================={Style.RESET_ALL}')
	print(f'{Fore.GREEN}NASTIC WRANGLER {last_changed} initialising...{Style.RESET_ALL}')
	print(f'{Fore.GREEN}=================================================={Style.RESET_ALL}')
	
	# POPUP WINDOW
	popup = sg.Window("Initialising...",[[sg.T("NASTIC WRANGLER initialising\nLots of modules...",font=("Arial bold",18))]],finalize=True,no_titlebar = True,alpha_channel=0.9)

	import random
	from scipy.stats import ttest_ind, ttest_ind_from_stats
	from scipy.spatial import ConvexHull
	from sklearn.cluster import DBSCAN
	from sklearn import manifold, datasets, decomposition, ensemble, random_projection
	import numpy as np
	import matplotlib
	matplotlib.use('TkAgg') # prevents Matplotlib related crashes --> self.tk.call('image', 'delete', self.name)
	import matplotlib.pyplot as plt 
	from functools import reduce
	import math
	import datetime
	import glob
	from functools import reduce
	import seaborn as sns
	import pandas as pd
	from io import BytesIO
	from PIL import Image, ImageDraw
	import warnings
	
	warnings.filterwarnings("ignore")
	
	# FUNCTIONS 
	
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
		graph.DrawText("N A S T I C  W R A N G L E R",(0,130),color="white",font=("Any",16),text_location="center")
		graph.DrawText("v{}".format(last_changed),(0,100),color="white",font=("Any",16),text_location="center")
		graph.DrawText("Design and coding: Tristan Wallis",(0,50),color="white",font=("Any",10),text_location="center")
		graph.DrawText("GUI: Alex McCann",(0,20),color="white",font=("Any",10),text_location="center")
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
		global cond1, cond2, col1, col2, avg_check, agg_check, out_check, pca_check
		cond1 = "COND1" # shorthand name for condition 1 (plot)
		cond2 = "COND2" # shorthand name for condition 2 (plot)
		col1 = "royalblue" # colour for condition 1 (plot)
		col2 = "orange" # colour for condition 2 plot (plot)
		avg_check = True # generate average metrics plots
		agg_check = True # generate aggregate metrics plots
		out_check = True # use median filtering to remove outliers from aggregate data
		pca_check = True # generate 2D PCA plots
		return 
	
	# SAVE SETTINGS
	def save_defaults():
		print ("\nSaving GUI settings to nastic_wrangler_gui.defaults...")
		with open("nastic_wrangler_gui.defaults","w") as outfile:
			outfile.write("{}\t{}\n".format("Short name condition 1",cond1))
			outfile.write("{}\t{}\n".format("Short name condition 2",cond2))
			outfile.write("{}\t{}\n".format("Color condition 1",col1))
			outfile.write("{}\t{}\n".format("Color condition 2",col2))
			outfile.write("{}\t{}\n".format("Plot average data",avg_check))
			outfile.write("{}\t{}\n".format("Plot aggregate data",agg_check))
			outfile.write("{}\t{}\n".format("Outlier removal",out_check))
			outfile.write("{}\t{}\n".format("Plot 2D PCA data",pca_check))
		return
	
	# LOAD DEFAULTS
	def load_defaults():
		global cond1, cond2, col1, col2, avg_check, agg_check, out_check, pca_check
		try:
			with open ("nastic_wrangler_gui.defaults","r") as infile:
				print ("\nLoading GUI settings from nastic_wrangler_gui.defaults...")
				defaultdict = {}
				for line in infile:
					spl = line.split("\t")
					defaultdict[spl[0]] = spl[1].strip()
			cond1 = defaultdict["Short name condition 1"]
			cond2 = defaultdict["Short name condition 2"]
			col1 = defaultdict["Color condition 1"]
			col2 = defaultdict["Color condition 2"]
			avg_check = defaultdict["Plot average data"]
			agg_check = defaultdict["Plot aggregate data"]
			out_check = defaultdict["Outlier removal"]
			pca_check = defaultdict["Plot 2D PCA data"]
			if avg_check == "True":
				avg_check = True
			else:
				avg_check = False
			if agg_check == "True":
				agg_check = True
			else:
				agg_check = False
			if out_check == "True":
				out_check = True
			else:
				out_check = False
			if pca_check == "True":
				pca_check = True
			else:
				pca_check = False
		except:
			print ("\nSettings could not be loaded")
		return
		
	# UPDATE GUI BUTTONS
	def update_buttons():
		
		# Toggle Find files button for condition 1
		if find_files1 == True:
			window.Element('-FIND1-').update(disabled = False)
		else:
			window.Element('-FIND1-').update(disabled = True)
		
		# Toggle Find files button for condition 2	
		if find_files2 == True:
			window.Element('-FIND2-').update(disabled = False)
		else:
			window.Element('-FIND2-').update(disabled = True)
		
		# Toggle LOAD DATA button
		if load_data_button == True and files_loaded1 == True and files_loaded2 == True:
			window.Element("-LOAD-").update(disabled = False)
		else:
			window.Element("-LOAD-").update(disabled = True)
			window.Element("-PLOT_DATA-").update(disabled = True)
		
		# Toggle PLOT DATA button
		if cond1 != "" and cond2 != "" and find_files1 == True and find_files2 == True and load_data_button == True and len(datadict1) >0 and len(datadict2) > 0 and files_loaded == True and files_loaded1 == True and files_loaded2 == True:
			window.Element("-PLOT_DATA-").update(disabled = False)
		else:
			window.Element("-PLOT_DATA-").update(disabled = True)
		
		if files_loaded == False:
			window.Element("-PLOT_DATA-").update(disabled = True)
		
		# Update shorthand name for condition 1 and 2
		window.Element("-SHORTNAME_COND1-").update(cond1)
		window.Element("-SHORTNAME_COND2-").update(cond2)
		
		# Update color for condition 1
		if col1 != "None":
			window.Element("-COLOR1-").update(col1)
			window.Element("-CHOOSE1-").update("Condition 1", col1)
		
		# Update color for condition 2
		if col2 != "None":
			window.Element("-COLOR2-").update(col2)
			window.Element("-CHOOSE2-").update("Condition 2", col2)
		
		# Toggle analysis checkboxes
		window.Element("-AVG_CHECKBOX-").update(avg_check)
		window.Element("-AGG_CHECKBOX-").update(agg_check)
		window.Element("-OUT_CHECKBOX-").update(out_check)
		window.Element("-PCA_CHECKBOX-").update(pca_check)
		return
	
	# CHECK VARIABLES
	def check_variables():
		global col1, col2, cond1, cond2
		
		# Try to reset color for condition 1
		if col1 == "None":
			try: 
				col1 = defaultdict["Color condition 1"]
			except:
				col1 = "royalblue"
		
		# Try to reset color for condition 2
		if col2 == "None":
			try:
				col2 = defaultdict["Color condition 2"]
			except:
				col2 = "orange"

		return
		
	# VALS
	cond1 = "Cond1" # short name for condition 1
	cond2 = "Cond2" # short name for condition 2
	dir1 = "" # directory for condition 1
	dir2 = "" # directory for condition 2
	find_files1 = False # used to enable/disable Find files button for condition 1
	find_files2 = False # used to enable/disable Find files button for condition 2
	load_data_button = False # used to enable/disable LOAD DATA button
	tree_icon_dict = {} # dictionary keeping track of which condition 1 files have been ticked/unticked
	tree_icon_dict2 = {} # dictionary keeping track of which condition 2 files have been ticked/unticked
	datadict1 = {} # dictionary holding all data for condition 1
	datadict2 = {} # dictionary holding all data for condition 2
	col1 = "royalblue" # colour for condition 1 plot
	col2 = "orange" # colour for condition 2 plot
	stamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now()) # datestamp
	metrics_dict = {} 
	files_loaded = False # used to enable/disable PLOT DATA button
	files_loaded1 = False # used to enable/disable PLOT DATA button
	files_loaded2 = False # used to enable/disable PLOT DATA button


	# FUNCTIONS		
	
	# READ DATA IN FROM EACH METRICS.TSV FILE
	def load_data(files1,files2):
		global datadict1, datadict2
		print("\n\nReading Condition 1 files...")
		datadict1  = {} # dictionary holding all data for condition 1
		print("\nCondition 1 files selected for analysis:\n----------------------------------------")
		for ct,infilename in enumerate(files1,start=1):
			if tree_icon_dict[ct] == "True":	
				datadict1[ct] = {"CLUSTERS":[]}
				print ("\nFile 1.",ct,infilename)
				with open (infilename,"r") as infile:
					for line in infile:
						line = line.strip()
						spl = line.split("\t")
						try:
							# GRAB CLUSTER INFO
							if int(spl[0]) > -1:
								datadict1[ct]["CLUSTERS"].append(spl) 
						except:
							# GRAB GENERAL INFO
							if len(spl)>1:
								datadict1[ct][spl[0].replace(":","")] = spl[1:]
								# GRAB AVERAGE METRICS FOR EACH SAMPLE
								if spl[0] == "AVG":
									avdata= spl[1:]
									for n,metric in enumerate(["AVERAGE MEMBERSHIP (traj/cluster)","AVERAGE LIFETIME (s)","AVERAGE MSD (um^2/s)","AVERAGE AREA (um^2)","AVERAGE RADIUS (um)","AVERAGE DENSITY (traj/um^2)","AVERAGE RATE (traj/sec)","AVERAGE TIME (s)"]):
										datadict1[ct][metric] = [float(avdata[n])]
							
				# ALL CLUSTER METRICS				
				clustzip = list(zip(*datadict1[ct]["CLUSTERS"])) # convert from 1 cluster per line to 1 cluster per column
				datadict1[ct]["cluster"] = list(clustzip[0])	
				datadict1[ct]["membership"] = list(clustzip[1])	
				datadict1[ct]["lifetime"] = list(clustzip[2])	
				datadict1[ct]["avmsd"] = list(clustzip[3])	
				datadict1[ct]["area"] = list(clustzip[4])	
				datadict1[ct]["radius"] = list(clustzip[5])	
				datadict1[ct]["density"] = list(clustzip[6])	
				datadict1[ct]["rate"] = list(clustzip[7])	
				datadict1[ct]["avtime"] = list(clustzip[8])
		if len(datadict1) == 1:
			print("\nCondition 1 files read (1 file)")
		else:
			print("\nCondition 1 files read ({} files)".format(len(datadict1)))
		
		print("\n\nReading Condition 2 files...")
		datadict2  = {} # dictionary holding all data for condition 2
		print("\nCondition 2 files selected for analysis:\n----------------------------------------")
		for ct,infilename in enumerate(files2,start=1):
			if tree_icon_dict2[ct] == "True":
				datadict2[ct] = {"CLUSTERS":[]}
				print ("\nFile 2.",ct,infilename)
				with open (infilename,"r") as infile:
					for line in infile:
						line = line.strip()
						spl = line.split("\t")
						try:
							# GRAB CLUSTER INFO
							if int(spl[0]) > -1:
								datadict2[ct]["CLUSTERS"].append(spl) 
						except:
							# GRAB GENERAL INFO
							if len(spl)>1:
								datadict2[ct][spl[0].replace(":","")] = spl[1:]
								# GRAB AVERAGE METRICS FOR EACH SAMPLE
								if spl[0] == "AVG":
									avdata= spl[1:]
									for n,metric in enumerate(["AVERAGE MEMBERSHIP (traj/cluster)","AVERAGE LIFETIME (s)","AVERAGE MSD (um^2/s)","AVERAGE AREA (um^2)","AVERAGE RADIUS (um)","AVERAGE DENSITY (traj/um^2)","AVERAGE RATE (traj/sec)","AVERAGE TIME (s)"]):
										datadict2[ct][metric] = [float(avdata[n])]
								
				# ALL CLUSTER METRICS				
				clustzip = list(zip(*datadict2[ct]["CLUSTERS"])) # convert from 1 cluster per line to 1 cluster per column
				datadict2[ct]["cluster"] = list(clustzip[0])	
				datadict2[ct]["membership"] = list(clustzip[1])	
				datadict2[ct]["lifetime"] = list(clustzip[2])	
				datadict2[ct]["avmsd"] = list(clustzip[3])	
				datadict2[ct]["area"] = list(clustzip[4])	
				datadict2[ct]["radius"] = list(clustzip[5])	
				datadict2[ct]["density"] = list(clustzip[6])	
				datadict2[ct]["rate"] = list(clustzip[7])	
				datadict2[ct]["avtime"] = list(clustzip[8])	
		if len(datadict2) == 1:
			print("\nCondition 2 files read (1 file)")
		else:
			print("\nCondition 2 files read ({} files)".format(len(datadict2)))
		window["-TABGROUP-"].Widget.select(1)
		return(datadict1,datadict2)
		
	# COMPARE AVERAGE METRICS
	def compare_average_metrics(datadict1,datadict2):
		global metrics_dict
		print("\nAVERAGE METRICS\n===============")
		for metric in [
		"SELECTION AREA (um^2)",
		"SELECTED TRAJECTORIES",
		"CLUSTERED TRAJECTORIES",
		"UNCLUSTERED TRAJECTORIES",
		"VAR CONFINED TRAJECTORIES",
		"VAR UNCONFINED TRAJECTORIES",
		"TOTAL CLUSTERS",
		"CLUSTERED TRAJECTORIES AVERAGE INSTANTANEOUS DIFFUSION COEFFICIENT (um^2/s)",
		"UNCLUSTERED TRAJECTORIES AVERAGE INSTANTANEOUS DIFFUSION COEFFICIENT (um^2/s)",
		"HOTSPOTS (CLUSTER SPATIAL OVERLAP AT 1/2 AVERAGE RADIUS)",
		"TOTAL CLUSTERS IN HOTSPOTS",
		"AVERAGE CLUSTERS PER HOTSPOT",
		"PERCENTAGE OF CLUSTERS IN HOTSPOTS",
		"AVERAGE MEMBERSHIP (traj/cluster)",
		"AVERAGE LIFETIME (s)",
		"AVERAGE MSD (um^2/s)",
		"AVERAGE AREA (um^2)",
		"AVERAGE RADIUS (um)",
		"AVERAGE DENSITY (traj/um^2)",
		"AVERAGE RATE (traj/sec)",
		"AVERAGE TIME (s)"
		]:
			c1,c2,p = average_data(metric)
			if c1[1] != 0:
				metrics_dict[metric] = {cond1:c1,cond2:c2,"p":p}
				print ("{} p={}".format(metric,p))
		return(metrics_dict)
				
	def average_data(header):
		c1 = []
		for sample in datadict1:
			try:
				c1.append(datadict1[sample][header][0])
			except:
				c1.append(0)
		c2 = []
		for sample in datadict2:
			try:
				c2.append(datadict2[sample][header][0])
			except:
				c2.append(0)
		
		c1=[float(x) for x in c1]	
		c2=[float(x) for x in c2]		
		t, p = ttest_ind(c2,c1, equal_var=False)
		
		n = len(c1)
		avg = np.average(c1)
		sem = np.std(c1)/math.sqrt(len(c1))
		c1 = [n,avg,sem] + c1
		
		n = len(c2)
		avg = np.average(c2)
		sem = np.std(c2)/math.sqrt(len(c2))
		c2 = [n,avg,sem] + c2
		return c1,c2,p

	# HIGHER ORDER METRICS DRIVED FROM AVERAGE METRICS
	def higher_order_average(metrics_dict):		
		print("\nHYBRID METRICS\n==============")
		cl1 = np.array(metrics_dict["CLUSTERED TRAJECTORIES"][cond1][3:])
		ucl1 = np.array(metrics_dict["SELECTED TRAJECTORIES"][cond1][3:])	
		perc_clust1 = list(100*cl1/ucl1)		
		cl2 = np.array(metrics_dict["CLUSTERED TRAJECTORIES"][cond2][3:])
		ucl2 = np.array(metrics_dict["SELECTED TRAJECTORIES"][cond2][3:])	
		perc_clust2 = list(100*cl2/ucl2)		
		t, p = ttest_ind(perc_clust2,perc_clust1, equal_var=False)
		n = len(perc_clust1)
		avg = np.average(perc_clust1)
		sem = np.std(perc_clust1)/math.sqrt(len(perc_clust1))
		perc_clust1 = [n,avg,sem] + perc_clust1
		n = len(perc_clust2)
		avg = np.average(perc_clust2)
		sem = np.std(perc_clust2)/math.sqrt(len(perc_clust2))
		perc_clust2 = [n,avg,sem] + perc_clust2
		metrics_dict["PERCENTAGE CLUSTERED TRAJECTORIES"] = {cond1:perc_clust1,cond2:perc_clust2,"p":p}
		print("PERCENTAGE CLUSTERED TRAJECTORIES p={}".format(p))

		tc1 = np.array(metrics_dict["TOTAL CLUSTERS"][cond1][3:])
		sa1 = np.array(metrics_dict["SELECTION AREA (um^2)"][cond1][3:])	
		clust_dens1 = list(tc1/sa1)		
		tc2 = np.array(metrics_dict["TOTAL CLUSTERS"][cond2][3:])
		sa2 = np.array(metrics_dict["SELECTION AREA (um^2)"][cond2][3:])	
		clust_dens2 = list(tc2/sa2)		
		t, p = ttest_ind(clust_dens2,clust_dens1, equal_var=False)

		n = len(clust_dens1)
		avg = np.average(clust_dens1)
		sem = np.std(clust_dens1)/math.sqrt(len(clust_dens1))
		clust_dens1 = [n,avg,sem] + clust_dens1

		n = len(clust_dens2)
		avg = np.average(clust_dens2)
		sem = np.std(clust_dens2)/math.sqrt(len(clust_dens2))
		clust_dens2 = [n,avg,sem] + clust_dens2

		metrics_dict["CLUSTER DENSITY (clusters/um^2)"] = {cond1:clust_dens1,cond2:clust_dens2,"p":p}
		print("CLUSTER DENSITY p={}".format(p))

	# COMPARE AGGREGATE METRICS
	def compare_aggregate_metrics(datadict1,datadict2):
		global metrics_dict
		print("\nAGGREGATE METRICS\n=================")
		for metric in [
		"membership",
		"lifetime",
		"avmsd",
		"area",
		"radius",
		"density",
		"rate",
		"avtime"]:
			c1,c2,p = aggregate_data(metric)
			if c1[1] != 0:	
				metrics_dict[metric] = {cond1:c1,cond2:c2,"p":p}
				print ("{} p={}".format(metric.upper(),p))
				
	def aggregate_data(header):
		c1 = []
		for sample in datadict1:
			try:
				[c1.append(float(x)) for x in datadict1[sample][header]]
			except:
				c1.append(0)
		c2 = []
		for sample in datadict2:
			try:	
				[c2.append(float(x)) for x in datadict2[sample][header]]
			except:
				c2.append(0)
		
		c1=[float(x) for x in c1]	
		c2=[float(x) for x in c2]
		
		if out_check == True:
			c1 = list(reject_outliers(np.array(c1)))
			c2 = list(reject_outliers(np.array(c2)))
		
		t, p = ttest_ind(c2,c1, equal_var=False)
		
		n = len(c1)
		avg = np.average(c1)
		sem = np.std(c1)/math.sqrt(len(c1))
		c1 = [n,avg,sem] + c1
		
		n = len(c2)
		avg = np.average(c2)
		sem = np.std(c2)/math.sqrt(len(c2))
		c2 = [n,avg,sem] + c2
		return c1,c2,p

	def normalize(lst):
		lst = [x + 0.00000001 for x in lst] # won't spit the dummy if all zeros
		s = sum(lst)
		return list(map(lambda x: float(x)/s, lst))		
	
	def reject_outliers(data, m = 2.5):
		d = np.abs(data - np.median(data))
		mdev = np.median(d)
		s = d/(mdev if mdev else 1.)
		return data[s<m]	

	# WRITE OUTPUT FILE
	def write_output_file():
		os.mkdir("nastic_wrangler_output_{}".format(stamp))
		with open("nastic_wrangler_output_{}/processed_metrics.tsv".format(stamp),"w",encoding='utf8') as outfile:
			outfile.write("#NASTIC / SEGNASTIC / BOOSH WRANGLER - Tristan Wallis t.wallis@uq.edu.au\n")
			outfile.write("#ANALYSED:\t{}\n".format(stamp))
			outfile.write("#{}:\t{}\n".format(cond1,dir1))
			outfile.write("#{}:\t{}\n".format(cond2,dir2))
			outfile.write("#{}:\t{}\n".format(cond1,col1))
			outfile.write("#{}:\t{}\n".format(cond2,col2))
			outfile.write("#OUTLIER REDUCTION:\t{}\n".format(out_check))
			outfile.write("\n#AVERAGE METRICS\n")
			outfile.write("#METRIC\tCONDITION\tN\tAVG\tSEM\tDATA\n")
			for metric in ["SELECTION AREA (um^2)",
			"SELECTED TRAJECTORIES",
			"CLUSTERED TRAJECTORIES",
			"UNCLUSTERED TRAJECTORIES",
			"PERCENTAGE CLUSTERED TRAJECTORIES",
			"VAR CONFINED TRAJECTORIES",
			"VAR UNCONFINED TRAJECTORIES",
			"TOTAL CLUSTERS",
			"CLUSTERED TRAJECTORIES AVERAGE INSTANTANEOUS DIFFUSION COEFFICIENT (um^2/s)",
			"UNCLUSTERED TRAJECTORIES AVERAGE INSTANTANEOUS DIFFUSION COEFFICIENT (um^2/s)",
			"HOTSPOTS (CLUSTER SPATIAL OVERLAP AT 1/2 AVERAGE RADIUS)",
			"TOTAL CLUSTERS IN HOTSPOTS",
			"AVERAGE CLUSTERS PER HOTSPOT",
			"PERCENTAGE OF CLUSTERS IN HOTSPOTS",
			"CLUSTER DENSITY (clusters/um^2)",
			"AVERAGE MEMBERSHIP (traj/cluster)",
			"AVERAGE LIFETIME (s)",
			"AVERAGE MSD (um^2/s)",
			"AVERAGE AREA (um^2)",
			"AVERAGE RADIUS (um)",
			"AVERAGE DENSITY (traj/um^2)",
			"AVERAGE RATE (traj/sec)",
			"AVERAGE TIME (s)"
			]:
				try:
					c1,c2,p = metrics_dict[metric][cond1],metrics_dict[metric][cond2],metrics_dict[metric]["p"]
					c1 = reduce(lambda x, y: str(x) + "\t" + str(y), c1)
					c2 = reduce(lambda x, y: str(x) + "\t" + str(y), c2)
					outfile.write("{}:\t{}\t{}\n".format(metric,cond1,c1))	
					outfile.write("p={}:\t{}\t{}\n".format(p,cond2,c2))
				except:	
					pass	
			outfile.write("\n#AGGREGATE METRICS\n")
			outfile.write("#METRIC\tCONDITION\tN\tAVG\tSEM\tDATA\n")
			for metric in [["membership","MEMBERSHIP (traj/cluster)"],
			["lifetime","APPARENT CLUSTER LIFETIME (s)"],
			["avmsd","CLUSTER AVG. MSD (um^2)"],
			["area","CLUSTER AREA (um^2)"],
			["radius","CLUSTER RADIUS (um)"],
			["density","CLUSTER DENSITY (traj/um^2)"],
			["rate","RATE (traj/s)"],
			["avtime","CLUSTER AVG. TIME (s)"]]:
				try:
					c1,c2,p = metrics_dict[metric[0]][cond1],metrics_dict[metric[0]][cond2],metrics_dict[metric[0]]["p"]
					c1 = reduce(lambda x, y: str(x) + "\t" + str(y), c1)
					c2 = reduce(lambda x, y: str(x) + "\t" + str(y), c2)
					outfile.write("{}:\t{}\t{}\n".format(metric[1],cond1,c1))	
					outfile.write("p={}:\t{}\t{}\n".format(p,cond2,c2))
				except:	
					pass
			outfile.write("\n#PCA PLOT\n")
			outfile.write("#CONDITION\tPCA PLOT #\tFILE DIRECTORY\n")
			ct1 = 1
			for ct,infilename in enumerate(files1,start=1):
				if tree_icon_dict[ct] == "True":	
					outfile.write(cond1 + "\t" + str(1) + "." + str(ct1) + "\t" + infilename + "\n")
				ct1 += 1
			ct2 = 1
			for ct, infilename in enumerate(files2,start=1):
				if tree_icon_dict2[ct] == "True":	
					outfile.write(cond2 + "\t" + str(2) + "." + str(ct2) + "\t" + infilename + "\n")
				ct2 += 1
				

	# PLOTTING	
	def plotting():
		print ("\n\nPlotting graphs...")
		font = {"family" : "Arial","size": 16} 
		matplotlib.rc('font', **font)
			
	# AVERAGE METRICS PLOT
	def average_metrics():
		os.mkdir("nastic_wrangler_output_{}/average_plots".format(stamp))			
		print ("\nAverage metrics plots:\n======================")
		for metric in [
		["SELECTION AREA (um^2)","Selection area",u"Area (μm²)"],
		["SELECTED TRAJECTORIES","Selected trajectories","Selected trajectories"],
		["CLUSTERED TRAJECTORIES","Clustered trajectories","Clustered trajectories"],
		["UNCLUSTERED TRAJECTORIES","Unclustered trajectories","Unclustered trajectories"],
		["PERCENTAGE CLUSTERED TRAJECTORIES","Percentage clustered trajectories","% clustered trajectories"],
		["VAR CONFINED TRAJECTORIES","VAR confined trajectories","VAR confined trajectories"],
		["VAR UNCONFINED TRAJECTORIES","VAR unconfined trajectories","VAR unconfined trajectories"],
		["TOTAL CLUSTERS","Total clusters","Total clusters"],
		["CLUSTERED TRAJECTORIES AVERAGE INSTANTANEOUS DIFFUSION COEFFICIENT (um^2/s)","Clustered trajectories Inst diff coeff",u"Clustered Inst diff coeff (μm²)"],
		["UNCLUSTERED TRAJECTORIES AVERAGE INSTANTANEOUS DIFFUSION COEFFICIENT (um^2/s)","Unclustered trajectories Inst diff coeff",u"Unclustered Inst diff coeff (μm²)"],
		["HOTSPOTS (CLUSTER SPATIAL OVERLAP AT 1/2 AVERAGE RADIUS)","Hotspots","Hotspots"],
		["TOTAL CLUSTERS IN HOTSPOTS","Total clusters in hotspots","Total clusters in hotspots"],
		["AVERAGE CLUSTERS PER HOTSPOT","Average clusters per hotspot","Average clusters/hotspot"],
		["PERCENTAGE OF CLUSTERS IN HOTSPOTS","Percentage clusters in hotspots","% clusters in hotspots"],
		["CLUSTER DENSITY (clusters/um^2)","Cluster density",u"Cluster density (clusters/μm²)"],
		["AVERAGE MEMBERSHIP (traj/cluster)","Average cluster membership","Cluster membership (traj/cluster)"],
		["AVERAGE LIFETIME (s)","Average cluster lifetime","Average cluster lifetime","Cluster lifetime (s)"],
		["AVERAGE MSD (um^2/s)","Average MSD of clustered trajectories","Clustered traj MSD (μm²)"],
		["AVERAGE AREA (um^2)","Average cluster area","Cluster area (μm²)"],
		["AVERAGE RADIUS (um)","Average cluster radius","Cluster radius (μm)"],
		["AVERAGE DENSITY (traj/um^2)","Average density of clustered trajectories","Clustered traj density (traj/μm²)"],
		["AVERAGE RATE (traj/sec)","Average cluster rate","Rate (traj/sec)"],
		["AVERAGE TIME (s)","Average cluster time centroid","Acquisition time (s)"]
		]:
			try:
				met,title,label = metric[0],metric[1],metric[2]
				fig = plt.figure(figsize=(3,4))
				ax0 = plt.subplot(111) 
				c1,c2,p = metrics_dict[met][cond1],metrics_dict[met][cond2],metrics_dict[met]["p"]
				print (met)
				
				# Bars
				avg_cond1 = c1[1]
				avg_cond1_sem = c1[2]
				avg_cond2 = c2[1]
				avg_cond2_sem = c2[2]
				bars = [cond1,cond2]
				avgs = [avg_cond1,avg_cond2]
				sems = [avg_cond1_sem,avg_cond2_sem]
				color=[col1,col2]
				ax0.bar(bars, avgs, yerr=sems, align='center',color=color,edgecolor="#444",linewidth=1.5, alpha=1,error_kw=dict(ecolor="#444",elinewidth=1.5,antialiased=True,capsize=5,capthick=1.5,zorder=1000))
				
				# Swarm	
				rows = []	
				for val in c1[3:]:
					rows.append({"condition":cond1,"val":val})
				for val in c2[3:]:
					rows.append({"condition":cond2,"val":val})	
				df = pd.DataFrame(rows)
				ax0 = sns.swarmplot(x="condition", y="val", data=df,alpha=0.9,size=5,order=bars,palette=["k","k"])
				# Significance
				star = "ns"
				if p <= 0.05:
					star = "*"
				if p <= 0.01:
					star = "**"		
				if p <= 0.001:
					star = "***"
				if p <= 0.0001:
					star = "****"
				if star != "ns":	
					ax0.text(0.5, 0.90, star, ha='center', va='bottom', color="k",transform=ax0.transAxes)
				else:
					ax0.text(0.5, 0.92, star, ha='center', va='bottom', color="k",transform=ax0.transAxes)
				ax0.plot([0.25,0.25, 0.75, 0.75], [0.9, 0.92, 0.92, 0.9], lw=1.5,c="k",transform=ax0.transAxes)
				ax0.set_xticklabels([cond1 + "\nN={}".format(c1[0]),cond2+ "\nN={}".format(c2[0])])
				plt.ylabel(label)
				plt.xlabel("")
				ylim = ax0.get_ylim()[1]
				plt.ylim(0,ylim*1.1)
				plt.tight_layout()
				plt.savefig("nastic_wrangler_output_{}/average_plots/{}.png".format(stamp,title.replace(" ","_")))
				plt.close()
			except:
				pass


	# AGGREGATE METRICS PLOT
	def aggregate_metrics():
		print ("\nAggregate metrics plots:\n========================")
		os.mkdir("nastic_wrangler_output_{}/aggregate_plots".format(stamp))			

		for metric in [["membership","Membership (traj/cluster)"],
			["lifetime","Apparent cluster lifetime (s)"],
			["avmsd",u"Cluster average MSD (μm²)"],
			["area",u"Cluster area (μm²)"],
			["radius","Cluster radius (μm)"],
			["density",u"Cluster density (traj/μm²)"],
			["rate","Rate (traj/s)"],
			["avtime","Cluster average time (s)"]]:

			try:
				met,label = metric[0],metric[1]
				fig = plt.figure(figsize=(3,4))
				ax0 = plt.subplot(111) 
				c1,c2,p = metrics_dict[met][cond1],metrics_dict[met][cond2],metrics_dict[met]["p"]
				print (met.upper())
				color=[col1,col2]

				# Violin	
				sns.violinplot(data=[c1[3:],c2[3:]],palette=color)
				ax0.set_xticklabels([cond1 + "\nN={}".format(c1[0]),cond2+ "\nN={}".format(c2[0])])
				plt.ylabel(label)
				plt.xlabel("")
				star = "ns"
				if p <= 0.05:
					star = "*"
				if p <= 0.01:
					star = "**"		
				if p <= 0.001:
					star = "***"
				if p <= 0.0001:
					star = "****"
				if star != "ns":	
					ax0.text(0.5, 0.90, star, ha='center', va='bottom', color="k",transform=ax0.transAxes)
				else:
					ax0.text(0.5, 0.92, star, ha='center', va='bottom', color="k",transform=ax0.transAxes)
				ax0.plot([0.25,0.25, 0.75, 0.75], [0.9, 0.92, 0.92, 0.9], lw=1.5,c="k",transform=ax0.transAxes)
				ylim = ax0.get_ylim()[1]
				plt.ylim(0,ylim*1.1)
				plt.tight_layout()
				plt.savefig("nastic_wrangler_output_{}/aggregate_plots/{}.png".format(stamp,met))
			except:
				pass

	# 2D PCA OF AVERAGED METRICS PLOT
	def PCA_metrics():
		global metrics_dict
		print ("\nPCA plot:\n=========")
		c1data = []
		c2data = []
		for metric in [
		"PERCENTAGE CLUSTERED TRAJECTORIES",
		"VAR CONFINED TRAJECTORIES",
		"VAR UNCONFINED TRAJECTORIES",
		"CLUSTERED TRAJECTORIES AVERAGE INSTANTANEOUS DIFFUSION COEFFICIENT (um^2/s)",
		"UNCLUSTERED TRAJECTORIES AVERAGE INSTANTANEOUS DIFFUSION COEFFICIENT (um^2/s)",
		"AVERAGE CLUSTERS PER HOTSPOT",
		"PERCENTAGE OF CLUSTERS IN HOTSPOTS",
		"CLUSTER DENSITY (clusters/um^2)",
		"AVERAGE MEMBERSHIP (traj/cluster)",
		"AVERAGE LIFETIME (s)",
		"AVERAGE MSD (um^2/s)",
		"AVERAGE AREA (um^2)",
		"AVERAGE RADIUS (um)",
		"AVERAGE DENSITY (traj/um^2)",
		"AVERAGE RATE (traj/sec)",
		"AVERAGE TIME (s)"
		]:
			try:
				print (metric)
				c1,c2 = metrics_dict[metric][cond1],metrics_dict[metric][cond2]
				c1data.append(c1[3:])
				c2data.append(c2[3:])
			except:
				pass
		c1data = [normalize(x) for x in c1data]
		c2data = [normalize(x) for x in c2data]
		c1data = list(zip(*c1data))
		c2data = list(zip(*c2data))
		alldata = c1data+c2data
		
		colors = []
		[colors.append(col1) for x in datadict1]
		[colors.append(col2) for x in datadict2]	

		names  = []	
		ct1 = 1
		for ct,infilename in enumerate(files1,start=1):
			if tree_icon_dict[ct] == "True":	
				names.append("1.{}".format(ct1))
			ct1 += 1
		ct2 = 1
		for ct, infilename in enumerate(files2,start=1):
			if tree_icon_dict2[ct] == "True":	
				names.append("2.{}".format(ct2))
			ct2 += 1
		
		mapdata = decomposition.TruncatedSVD(n_components=2).fit_transform(alldata) 
		fig = plt.figure(figsize=(4,4))
		ax0 = plt.subplot(111)
		ax0.scatter(mapdata[:, 0], mapdata[:, 1],c=colors)
		for i in range(mapdata.shape[0]):
			ax0.text(mapdata[i, 0], mapdata[i, 1],names[i],alpha=0.75)
		ax0.set_xlabel('Dimension 1')
		ax0.set_ylabel('Dimension 2')
		plt.tight_layout()
		plt.savefig("nastic_wrangler_output_{}/pca_labels.png".format(stamp))
		
		fig = plt.figure(figsize=(4,4))
		ax0 = plt.subplot(111)
		ax0.scatter(mapdata[:, 0], mapdata[:, 1],c=colors)
		ax0.set_xlabel('Dimension 1')
		ax0.set_ylabel('Dimension 2')
		plt.tight_layout()
		plt.savefig("nastic_wrangler_output_{}/pca.png".format(stamp))

	# DRAW CHECKBOX ICON FOR TREE
	def icon(check):
		box = (20, 20)
		background = (255, 255, 255, 0)
		rectangle = (1, 1, 19, 19)
		line = ((3,10), (10,20), (20,1))
		im = Image.new('RGBA', box, "grey")
		draw = ImageDraw.Draw(im, 'RGBA')
		draw.rectangle(rectangle, outline='white', width=1)
		
		if check == 1:
			draw.line(line, fill='white', width=2, joint='curve')
		elif check == 2:
			draw.line(line, fill='grey', width=2, joint='curve')
		with BytesIO() as output:
			im.save(output, format="PNG")
			png = output.getvalue()
		return png
	
	# GET INITIAL VALUES FOR GUI
	cwd = os.path.dirname(os.path.abspath(__file__))
	os.chdir(cwd)
	initialdir = cwd
	if os.path.isfile("nastic_wrangler_gui.defaults"):
		load_defaults()
	else:
		reset_defaults()
		save_defaults()
		
	plt.rcdefaults() 
	font = {"family" : "Arial","size": 14} 
	matplotlib.rc('font', **font)
	
	# GUI LAYOUT
	appFont = ("Any 12")
	sg.set_options(font=appFont)
	sg.theme('DARKGREY11')
	
	# Menu
	menu_def = [
		['&File', ['&Load settings', '&Save settings','&Default settings','&Exit']],
		['&Info', ['&About', '&Help','&Licence' ]],
		]

	# LOAD FILES TAB
	
	# Tree 
	check = [icon(0), icon(1), icon(2)]
	headings1 = ['Condition 1 File names']
	headings2 = ['Condition 2 File names',]
	treedata1 = sg.TreeData()
	treedata2 = sg.TreeData()
	
	# Condition 1 frame
	directory1_frame_layout = [
		[sg.FolderBrowse("Browse",key="-BROWSE_DIR1-",target="-DIRECTORYNAME1-", tooltip = "Select directory containing metrics.tsv files", initial_folder=initialdir),sg.In("Select condition 1 directory", key = '-DIRECTORYNAME1-', size = (33,1), enable_events = True)],
		[sg.B("Find files", key = '-FIND1-', tooltip = "Recursively search selected directory for metrics.tsv files", disabled = True),sg.T("Files to include:")],
		[sg.Tree(data=treedata1, headings = headings1, row_height=25, num_rows = 7,select_mode = sg.TABLE_SELECT_MODE_BROWSE,key = '-TREE-', tooltip = "Untick files to exclude them from analysis\nUse ~1s delay between clicks", metadata = [],vertical_scroll_only=True, justification='left', enable_events=True, auto_size_columns = True, expand_x = True, col0_width = 1)],
	]

	directory1_frame_layout_col = [
		[sg.Frame("Condition 1", directory1_frame_layout)],
	]
	
	# Condition 2 frame
	directory2_frame_layout = [
		[sg.FolderBrowse("Browse",key="-BROWSE_DIR2-", target = "-DIRECTORYNAME2-", tooltip = "Select directory containing metrics.tsv files", initial_folder=initialdir), sg.In("Select condition 2 directory", size = (33,1), key = '-DIRECTORYNAME2-', enable_events = True)],
		[sg.B("Find files", key = '-FIND2-', tooltip = "Recursively search selected directory for metrics.tsv files", disabled = True), sg.T("Files to include:")],
		[sg.Tree(data=treedata2, headings = headings2, num_rows=7, row_height=25, select_mode = sg.TABLE_SELECT_MODE_BROWSE, key = '-TREE2-', tooltip = "Untick files to exclude them from analysis\nUse ~1s delay between clicks", metadata = [],vertical_scroll_only=True,justification='left', enable_events=True, auto_size_columns = True, expand_x = True, col0_width = 1)],
	]

	directory2_frame_layout_col = [
		[sg.Frame("Condition 2", directory2_frame_layout)],
	]

	# Load files tab layout
	load_file_layout = [
		[sg.Column(directory1_frame_layout_col),sg.Column(directory2_frame_layout_col)],
		[sg.Push(),sg.B("LOAD DATA",key = "-LOAD-", tooltip = "Reads data from each metrics.tsv file selected", size=(25,2),button_color=("white","gray"),disabled=True),sg.Push()],
	]

	# PLOT DATA TAB
	
	# Condition 1 frame
	directory1_plot_frame_layout = [
		[sg.T("Shorthand name to show on plot:"),sg.Input("COND1",key="-SHORTNAME_COND1-",size=(13,1), enable_events = True)],
		[sg.T("Select color"),sg.ColorChooserButton("Condition 1",key="-CHOOSE1-",target="-COLOR1-",button_color=("white",col1)),sg.Input(col1,key ="-COLOR1-",enable_events=True,visible=False)],
	]

	directory1_plot_frame_layout_col = [
		[sg.Frame("Condition 1", directory1_plot_frame_layout)],
	]
	
	# Condition 2 frame
	directory2_plot_frame_layout = [
		[sg.T("Shorthand name to show on plot:"),sg.Input("COND2",key="-SHORTNAME_COND2-",size=(13,1), enable_events = True)],
		[sg.T("Select color"),sg.ColorChooserButton("Condition 2",key="-CHOOSE2-",target="-COLOR2-",button_color=("white",col2)),sg.Input(col2,key ="-COLOR2-",enable_events=True,visible=False)],
	]

	directory2_plot_frame_layout_col = [
		[sg.Frame("Condition 2", directory2_plot_frame_layout)],
	]
	
	# Plot data tab layout
	plot_data_layout = [
		[sg.Col(directory1_plot_frame_layout_col),sg.Col(directory2_plot_frame_layout_col)],
		[sg.T("     "),sg.Checkbox("Outlier Removal", default = True,key = '-OUT_CHECKBOX-'),sg.T("   - median filtering outlier reduction for aggregate data", font = "Any 12 italic")],
		[sg.T("Data to Plot:")],
		[sg.T("     "),sg.Checkbox("Aggregate Data", default = True, key = '-AGG_CHECKBOX-'), sg.T("   - aggregates individual cluster metrics across all samples (N = total # of clusters)", font = "Any 12 italic")],
		[sg.T("     "),sg.Checkbox("Average Data", default = True, key = '-AVG_CHECKBOX-'), sg.T("      - averages cluster metrics for each sample (N = # of samples)", font = "Any 12 italic")], 
		[sg.T("     "),sg.Checkbox("2D PCA analysis", default = True, key = '-PCA_CHECKBOX-'), sg.T(" - determine the overall relationships between samples",font = "Any 12 italic")],
		[sg.Push(),sg.B("PLOT DATA",key = "-PLOT_DATA-", tooltip = "Generate and save selected plots and \ndatestamped TSV of raw data used for plots",size=(10,2),button_color=("white","gray"),disabled=True, enable_events = True),sg.Push()],
	]

	# LAYOUT
	layout = [
		[sg.Menu(menu_def)],
		[sg.T("NASTIC Wrangler",font=("Any",20))],
		[sg.TabGroup([
			[sg.Tab("Load Files", load_file_layout)],
			[sg.Tab("Plot Data", plot_data_layout)],
			],key = '-TABGROUP-'),
		],
	]

	window = sg.Window('NASTIC WRANGLER v{}'.format(last_changed),layout)
	tree = window["-TREE-"]
	tree2 = window['-TREE2-']
	popup.close()

	# MAIN LOOP
	while True:
		# Read events and values
		event, values = window.read(timeout = 250)
		
		# Exit	
		if event == sg.WIN_CLOSED or event == 'Exit':  
			break
		
		# Values
		dir1 = values["-DIRECTORYNAME1-"]
		dir2 = values["-DIRECTORYNAME2-"]
		tree_vals = values["-TREE-"]
		tree2_vals = values["-TREE2-"]
		cond1 = values["-SHORTNAME_COND1-"]
		cond2 = values["-SHORTNAME_COND2-"]
		col1 = values["-COLOR1-"]
		col2 = values["-COLOR2-"]
		avg_check = values['-AVG_CHECKBOX-']
		agg_check = values['-AGG_CHECKBOX-']
		out_check = values['-OUT_CHECKBOX-']
		pca_check = values['-PCA_CHECKBOX-']
		
		# Check variables
		check_variables()

		# Toggle Find files button for condition 1
		if dir1 != "" and dir1 != "Select condition 1 directory":
			find_files1 = True
			update_buttons()
		else:
			find_files1 = False
			update_buttons()
		
		# Toggle Find files button for condition 2
		if dir2 != "" and dir2 != "Select condition 2 directory":
			find_files2 = True
			update_buttons()
		else:
			find_files2 = False
			update_buttons()
		
		# Toggle LOAD DATA button
		if event == '-DIRECTORYNAME1-':
			load_data_button = False
			files_loaded1 = False
			files_loaded = False
			update_buttons()
		
		if event == '-DIRECTORYNAME2-':
			load_data_button = False
			files_loaded2 = False
			files_loaded = False
			update_buttons()
			
		# RECURSIVELY SEARCH FOR METRICS.TSV FILES FOR EACH CONDITION
		
		# Condition 1
		if event == '-FIND1-':
			if dir1 != "" and dir1 != "Select condition 1 directory":
				split_files1_list = []
				files1_list = []
				files1 = glob.glob(dir1 + '/**/metrics.tsv', recursive = True)
				files1 = [file.replace("\\","/") for file in files1] # get all paths into proper forward slash style!
				for file in files1:
					files1_list.append(file)
					split_files1 = file.split("/")
					filename_split_files1 = split_files1[-2]
					split_files1_list.append(filename_split_files1)
				combolist1 = [""]+[x for x in range(1,len(files1)+1)]
				treedata1 = sg.TreeData()
				ct = 1
				if len(files1) == 0:
					print("\n\nNo Condition 1 files found. Make sure selected directory 1 folder contains metrics.tsv files")
				elif len(files1) == 1:
					print("\n\nCondition 1 files found: (1 file)\n----------------------------------")
				elif len(files1) > 1:
					print("\n\nCondition 1 files found: ({} files)\n----------------------------------".format(len(files1)))
				for selectedfile in split_files1_list:
					treedata1.insert("",combolist1[ct], combolist1[ct],values = [selectedfile], icon = check[1])
					print("\nFile 1.", combolist1[ct],files1_list[ct-1])
					tree.update(treedata1)
					tree_icon_dict.update({ct:"True"})
					ct +=1
				if "True" in tree_icon_dict.values() and "True" in tree_icon_dict2.values():
					load_data_button = True
					files_loaded1 = True
					update_buttons()
				else:
					load_data_button = False
					files_loaded1 = True
					update_buttons()					
						
		# Condition 2
		if event == '-FIND2-':
			if dir2 != "" and dir2 != "Select condition 2 directory":
				split_files2_list = []
				files2_list = []
				files2 = glob.glob(dir2 + '/**/metrics.tsv', recursive = True)
				files2 = [file.replace("\\","/") for file in files2]
				for file in files2:
					files2_list.append(file)
					split_files2 = file.split("/")
					filename_split_files2 = split_files2[-2]
					split_files2_list.append(filename_split_files2)
				combolist2 = [""]+[x for x in range(1,len(files2)+1)]
				treedata2 = sg.TreeData()
				ct = 1
				if len(files1) == 0:
					print("\n\nNo Condition 2 files found. Make sure selected directory 2 folder contains metrics.tsv files")
				elif len(files2) == 1:
					print("\n\nCondition 2 files found: (1 file)\n----------------------------------")
				elif len(files2) > 1:
					print("\n\nCondition 2 files found: ({} files)\n----------------------------------".format(len(files2)))
				for selectedfile in split_files2_list:
					treedata2.insert("",combolist2[ct], combolist2[ct],values = [selectedfile], icon = check[1])
					print("\nFile 2.", combolist2[ct],files2_list[ct-1])
					tree2.update(treedata2)
					tree_icon_dict2.update({ct:"True"})
					ct +=1
				if "True" in tree_icon_dict.values() and "True" in tree_icon_dict2.values():
					load_data_button = True
					files_loaded2 = True
					update_buttons()
				else:
					load_data_button = False
					files_loaded2 = True
					update_buttons()					
		
		# Tree that shows files found for condition 1
		if event == '-TREE-':
			try:
				filenumber = values['-TREE-'][0]
				ct = filenumber
				if filenumber in tree.metadata:
					tree.metadata.remove(filenumber)
					tree_icon_dict.update({ct:"False"})
					tree.update(key=filenumber, icon=check[0])		
				else:
					tree.metadata.append(filenumber)
					tree_icon_dict.update({ct:"True"})
					tree.update(key=filenumber, icon=check[1])
				if "True" in tree_icon_dict.values() and "True" in tree_icon_dict2.values():
					load_data_button = True
					files_loaded = False
					update_buttons()
				else:
					load_data_button = False
					files_loaded = False
					update_buttons()
			except:
				pass
		
		# Tree that shows files found for condition 2
		if event == '-TREE2-':
			try:
				filenumber = values['-TREE2-'][0]
				ct = filenumber
				if filenumber in tree2.metadata:
					tree2.metadata.remove(filenumber)
					tree2.update(key=filenumber, icon=check[0])
					tree_icon_dict2.update({ct:"False"})
				else:
					tree2.metadata.append(filenumber)
					tree2.update(key=filenumber, icon=check[1])
					tree_icon_dict2.update({ct:"True"})
				if "True" in tree_icon_dict.values() and "True" in tree_icon_dict2.values():
					load_data_button = True
					files_loaded = False
					update_buttons()
				else:
					load_data_button = False
					files_loaded = False
					update_buttons()					
			except:
				pass
				
		# LOAD DATA
		if event == '-LOAD-':
			load_data(files1,files2)
			files_loaded = True
			update_buttons()
		
		# Select color Condition 1
		if event == '-CHOOSE1-':
			update_buttons()
		
		# Select color Condition 2
		if event == '-CHOOSE2-':
			update_buttons()
			
		# Toggle checkboxes for analysis plots to show
		if event == '-AGG_CHECKBOX-' or '-AVG_CHECKBOX-' or '-PCA_CHECKBOX-':
			update_buttons()
		
		# PLOT DATA
		if event == '-PLOT_DATA-':
			if cond1 == cond2:
				print("\nPlease use different shorthand names for condition 1 and 2.")
			else:
				print("\n\nAveraging metrics...")
				compare_average_metrics(datadict1,datadict2)
				higher_order_average(metrics_dict)		
				print("\n\nAggregating metrics...")
				compare_aggregate_metrics(datadict1,datadict2)		
				plotting()
				write_output_file()	
				if avg_check == True:
					average_metrics()
				if agg_check == True:
					aggregate_metrics()	
				if pca_check == True:
					PCA_metrics()
				print("\n\nAll plots and metrics have been saved to: {}".format(cwd) + "\\nastic_wrangler_output_{}".format(stamp) + "\n")
				stamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now()) # datestamp

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
				"This program allows the visualisation of the metrics.tsv files produced by NASTIC and SEGNASTIC. \nComparison bar plots for each metric and statistical significance (t-test) are shown.",
				"Load Files tab:",
				"     Browse: Select a directory for each condition.",
				"     Find files: Recursively search directories for \n     metrics.tsv files.", 
				"     Untick files to exclude from analysis (use ~1s delay \n     between clicks).",
				"     LOAD DATA: Load selected files and extract information.",
				"Plot Data tab:",
				"     Shorthand name: for each condition to appear on plots.",
				"     Select color: for each condition to appear on plots.",
				"     Outlier removal: uses median filtering to remove \n     outliers from aggregate data.",
				"     Aggregate data: aggregates individual cluster metrics \n     across all samples (N = total number of clusters).",
				"     Average data: averages cluster metrics for each sample \n     (N = number of samples).",
				"     2D PCA analysis allows you to determine the overall \n     relationships between samples.",
				"     PLOT DATA: plot (selected) aggregate, average and PCA. \n     Also saves datestamped TSV of raw data used for plots.",
				"Tristan Wallis, Alex McCann {}".format(last_changed),
				no_titlebar = True,
				grab_anywhere = True,
				keep_on_top = True,
				)	

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
				
	print ("\nExiting...")
	plt.close('all')
	window.close()	
	quit()