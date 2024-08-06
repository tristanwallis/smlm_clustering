# -*- coding: utf-8 -*-
'''
NASTIC_GUI
PYSIMPLEGUI BASED GUI FOR SPATIOTEMPORAL INDEXING CLUSTERING OF MOLECULAR TRAJECTORY DATA

Design and coding: Tristan Wallis
Additional coding: Kyle Young, Alex McCann
Debugging: Sophie Huiyi Hou, Kye Kudo, Alex McCann
Queensland Brain Institute
The University of Queensland
Fred Meunier: f.meunier@uq.edu.au

REQUIRED:
Python 3.8 or greater
python -m pip install scipy numpy matplotlib matplotlib-venn scikit-learn statsmodels rtree pysimplegui colorama

INPUT:
TRXYT trajectory files
Space separated: TRajectory# X-position(um) Y-position(um) Time(sec)  
No headers

1 9.0117 39.86 0.02
1 8.9603 39.837 0.04
1 9.093 39.958 0.06
1 9.0645 39.975 0.08
2 9.1191 39.932 0.1
2 8.9266 39.915 0.12
etc

NOTES:
This script has been tested and will run as intended on Windows 7/10/11, with minor interface anomalies on Linux, and possible tk GUI performance issues on MacOS.
The script will fork to multiple CPU cores for the heavy number crunching routines (this also prevents it from being packaged as an exe using pyinstaller).
Feedback, suggestions and improvements are welcome. Sanctimonious critiques on the pythonic inelegance of the coding are not.

CHECK FOR UPDATES:
https://github.com/tristanwallis/smlm_clustering/releases
'''

last_changed = "20240806"

# MULTIPROCESSING FUNCTIONS
from scipy.spatial import ConvexHull
import multiprocessing
import numpy as np
import math
from math import dist
import functools
import webbrowser
import warnings
warnings.filterwarnings("ignore")

def metrics(data):
	points,minlength,radius_factor,centroid=data
	# MSD over time
	msds = []
	for i in range(1,minlength,1):
		all_diff_sq = []
		for j in range(0,i):
			msdpoints = points[j::i]
			diff = [dist(msdpoints[k][:2],msdpoints[k-1][:2]) for k in range(1,len(msdpoints))] # displacement 
			diff_sq = np.array(diff)**2 # square displacement
			[all_diff_sq.append(x) for x in diff_sq]
		msd = np.average(all_diff_sq)
		msds.append(msd)
		
	# Instantaneous diffusion coefficient
	diffcoeff = (msds[3]-msds[0])
		
	# Area
	points2d =[sublist[:2] for sublist in points] # only get 2D hull
	try: 
		area =ConvexHull(points2d).volume
	except:
		area = 0
	# Bounding Box	
	radius = math.sqrt(area/math.pi)*radius_factor 
	dx,dy,dt = centroid 
	px,py,pt=zip(*points)
	left,bottom,early,right,top,late = dx-radius,dy-radius,min(pt),dx+radius,dy+radius,max(pt) 
	bbox = [left,bottom,early,right,top,late]
	return [points,msds,area,radius,bbox,centroid,diffcoeff]
	
def multi(allpoints):
	with multiprocessing.Pool() as pool:
		allmetrics = pool.map(metrics,allpoints)
	return allmetrics

# MAIN PROG AND FUNCTIONS
if __name__ == "__main__": # has to be called this way for multiprocessing to work
	
	# LOAD MODULES
	import PySimpleGUI as sg
	from colorama import init as colorama_init
	from colorama import Fore
	from colorama import Style
	import os
	
	sg.set_options(dpi_awareness=True) # turns on DPI awareness (Windows only)
	sg.theme('DARKGREY11')
	colorama_init()
	os.system('cls' if os.name == 'nt' else 'clear')
	print(f'{Fore.GREEN}============================================================={Style.RESET_ALL}')
	print(f'{Fore.GREEN}NASTIC {last_changed} initialising...{Style.RESET_ALL}')
	print(f'{Fore.GREEN}============================================================={Style.RESET_ALL}')
	popup = sg.Window("Initialising...",[[sg.T("NASTIC initialising...",font=("Arial bold",18))]],finalize=True,no_titlebar = True,alpha_channel=0.9)
	
	import random
	from scipy.spatial import ConvexHull
	from scipy.stats import gaussian_kde
	from scipy.stats import variation
	from sklearn.cluster import DBSCAN
	from sklearn import datasets, decomposition, ensemble, random_projection
	import numpy as np
	from rtree import index
	import matplotlib
	matplotlib.use('TkAgg') # prevents Matplotlib related crashes --> self.tk.call('image', 'delete', self.name)
	import matplotlib.pyplot as plt
	from matplotlib.widgets import LassoSelector
	from matplotlib import path
	from mpl_toolkits.mplot3d import art3d
	import math
	from math import dist
	import time
	import datetime
	import pickle
	import io
	from functools import reduce
	import warnings
	import multiprocessing
	
	# VAR stuff
	from scipy.optimize import curve_fit
	from statsmodels.tsa.api import VAR
	from sklearn.cluster import KMeans
	from sklearn import preprocessing
	from matplotlib_venn import venn2
	
	warnings.filterwarnings("ignore")
	
	# ALPHA COEFFICIENT FOR SINGLE TRAJECTORY	
	def anom_diff_f(x, alpha):
		return np.power(x, alpha)	

	def fit_alphas(msd_times,traj):
		msds = seldict[traj]["msds"]
		popt, pcov = curve_fit(anom_diff_f, msd_times, msds)
		alpha_coeff =  popt[0]
		seldict[traj]["alpha"] = alpha_coeff
		return

	# VAR
	def vector_autoregression(traj):
		curr_points = seldict[traj]["points"]
		curr_points = [[x[0],x[1]] for x in curr_points] # just X and Y
		model = VAR(curr_points)
		results = model.fit(1, trend='n')
		cov_matrix = results.sigma_u
		coeff_matrix = results.coefs
		cov_norm = np.linalg.norm(cov_matrix, ord='fro')
		coeff_norm = np.linalg.norm(coeff_matrix.reshape(2, 2), ord='fro')
		seldict[traj]["cov_norm"] = cov_norm
		seldict[traj]["coeff_norm"] = coeff_norm
		return

	# ASSIGN TRAJECTORIES AS VAR CONFINED/UNCONFINED		
	def var_confine():
		confinedindices = []
		unconfinedindices = []		
		print ("Trajectory vector autoregression analysis...")
		t1 = time.time()

		# Derive vector autoregression mobility metrics
		allindices = range(len(seldict))
		msd_times = [frame_time*x for x in range(1,minlength,1)]
		var_metrics = []	
		for num,traj in enumerate(allindices): 
			if num%10 == 0:
				try: 
					bar = 100*num/(len(allindices)-1)
					window['-PROGBAR-'].update_bar(bar)
				except:
					pass
			fit_alphas(msd_times,traj)
			vector_autoregression(traj)
			var_metrics.append([seldict[traj]["alpha"],seldict[traj]["cov_norm"],seldict[traj]["coeff_norm"],seldict[traj]["area"]])
		X = np.array(var_metrics)
		scaler = preprocessing.StandardScaler().fit(X)
		X_scaled = scaler.transform(X)
		kmeans = KMeans(n_clusters=2, n_init=100, max_iter=500).fit(X_scaled)
		kmeanslabels = kmeans.labels_
		for num,traj in enumerate(allindices): 
			seldict[traj]["kmeans_group"] = kmeanslabels[num]	
		window['-PROGBAR-'].update_bar(0)
		t2 = time.time()
		print ("VAR completed in {} sec".format(round(t2-t1,3)))
	
		# Determine which VAR kmeans group contains clustered trajectories
		clustgroups = [seldict[x]["kmeans_group"] for x in clustindices]
		clustgroups = round(np.average(clustgroups),0)
		if clustgroups == 0:
			var_cols = [var_color1,var_color2]
		else:
			var_cols = [var_color2,var_color1]
		for i in allindices:
			if seldict[i]["kmeans_group"] ==clustgroups:
				confinedindices.append(i)
			else:
				unconfinedindices.append(i)	
		return 	confinedindices,unconfinedindices,var_cols
		
	# NORMALIZE
	def normalize(lst):
		s = sum(lst)
		return map(lambda x: float(x)/s, lst)

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
		xmin =-300
		xmax=300
		ymin=-100
		ymax=100
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
		splash = sg.Window("Cluster Sim",layout, no_titlebar = True,finalize=True,alpha_channel=0.9,grab_anywhere=True,element_justification="c")
		obj_list=initialise_particles(graph)
		graph.DrawText("N A S T I C v{}".format(last_changed),(0,70),color="white",font=("Any",16),text_location="center")
		graph.DrawText("Design and coding: Tristan Wallis",(0,45),color="white",font=("Any",10),text_location="center")
		graph.DrawText("Additional coding: Kyle Young, Alex McCann",(0,30),color="white",font=("Any",10),text_location="center")
		graph.DrawText("Debugging: Sophie Huiyi Hou, Kye Kudo, Alex McCann",(0,15),color="white",font=("Any",10),text_location="center")
		graph.DrawText("Queensland Brain Institute",(0,0),color="white",font=("Any",10),text_location="center")	
		graph.DrawText("The University of Queensland",(0,-15),color="white",font=("Any",10),text_location="center")	
		graph.DrawText("Fred Meunier f.meunier@uq.edu.au",(0,-30),color="white",font=("Any",10),text_location="center")	
		graph.DrawText("PySimpleGUI: https://pypi.org/project/PySimpleGUI/",(0,-55),color="white",font=("Any",10),text_location="center")	
		graph.DrawText("Rtree: https://pypi.org/project/Rtree/",(0,-75),color="white",font=("Any",10),text_location="center")
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
		print ("Using default GUI settings...")
		global traj_prob,detection_alpha,minlength,maxlength,acq_time,time_threshold,radius_factor,cluster_threshold,canvas_color,plot_trajectories,plot_centroids,plot_clusters,plot_colorbar,line_width,line_alpha,line_color,centroid_size,centroid_alpha,centroid_color,cluster_alpha,cluster_linetype,cluster_width,saveformat,savedpi,savetransparency,savefolder,selection_density,autoplot,autocluster,radius_thresh,cluster_fill,auto_metric,plotxmin,plotxmax,plotymin,plotymax,msd_filter,frame_time,tmin,tmax,plot_hotspots,hotspot_alpha,hotspot_linetype,hotspot_width,hotspot_color,hotspot_radius,var_color,axes_3d,msd_color,clust_color,var_color1,var_color2,msd_color1,msd_color2,pixel
		traj_prob = 1
		detection_alpha = 0.05
		selection_density = 0
		minlength = 5
		maxlength = 1000
		acq_time = 320
		frame_time = 0.02
		time_threshold = 20
		radius_factor = 1.2
		cluster_threshold = 3
		canvas_color = "black"
		plot_trajectories = True
		plot_centroids = False
		plot_clusters = True
		plot_hotspots = True
		plot_colorbar = True
		line_width = 1.5	
		line_alpha = 0.25	
		line_color = "white"	
		centroid_size = 5	
		centroid_alpha = 0.75
		centroid_color = "white"
		cluster_width = 2	
		cluster_alpha = 1	
		cluster_linetype = "solid"
		cluster_fill = False	
		saveformat = "png"
		savedpi = 300	
		savetransparency = False
		autoplot=True
		autocluster=True
		radius_thresh=0.2
		auto_metric=False
		plotxmin=""
		plotxmax=""
		plotymin=""
		plotymax=""
		msd_filter = False
		hotspot_width = 2.5	
		hotspot_alpha = 1	
		hotspot_linetype = "dotted"
		hotspot_color = "white"				
		hotspot_radius = 1.0
		var_color =	False
		msd_color = False
		clust_color = False
		axes_3d	= True	
		var_color1 = "orange"
		var_color2 = "purple"
		msd_color1 = "cyan"
		msd_color2 = "magenta"
		pixel = 0.106
		return 

	# SAVE SETTINGS
	def save_defaults():
		print ("Saving GUI settings to nastic_gui.defaults...")
		with open("nastic_gui.defaults","w") as outfile:
			outfile.write("{}\t{}\n".format("Trajectory probability",traj_prob))
			outfile.write("{}\t{}\n".format("Raw trajectory detection plot opacity",detection_alpha))
			outfile.write("{}\t{}\n".format("Selection density",selection_density))
			outfile.write("{}\t{}\n".format("Trajectory minimum length",minlength))
			outfile.write("{}\t{}\n".format("Trajectory maximum length",maxlength))
			outfile.write("{}\t{}\n".format("Acquisition time (s)",acq_time))	
			outfile.write("{}\t{}\n".format("Frame time (s)",frame_time))				
			outfile.write("{}\t{}\n".format("Time threshold (s)",time_threshold))
			outfile.write("{}\t{}\n".format("Radius factor",radius_factor))
			outfile.write("{}\t{}\n".format("Cluster threshold",cluster_threshold))			
			outfile.write("{}\t{}\n".format("Canvas color",canvas_color))	
			outfile.write("{}\t{}\n".format("Plot trajectories",plot_trajectories))
			outfile.write("{}\t{}\n".format("Plot centroids",plot_centroids))
			outfile.write("{}\t{}\n".format("Plot clusters",plot_clusters))
			outfile.write("{}\t{}\n".format("Plot hotspots",plot_hotspots))
			outfile.write("{}\t{}\n".format("Plot colorbar",plot_colorbar))				
			outfile.write("{}\t{}\n".format("Trajectory line width",line_width))
			outfile.write("{}\t{}\n".format("Trajectory line color",line_color))
			outfile.write("{}\t{}\n".format("Trajectory line opacity",line_alpha))
			outfile.write("{}\t{}\n".format("Centroid size",centroid_size))
			outfile.write("{}\t{}\n".format("Centroid color",centroid_color))
			outfile.write("{}\t{}\n".format("Centroid opacity",centroid_alpha))
			outfile.write("{}\t{}\n".format("Cluster fill",cluster_fill))		
			outfile.write("{}\t{}\n".format("Cluster line width",cluster_width))			
			outfile.write("{}\t{}\n".format("Cluster line opacity",cluster_alpha))
			outfile.write("{}\t{}\n".format("Cluster line type",cluster_linetype))
			outfile.write("{}\t{}\n".format("Hotspot line width",hotspot_width))			
			outfile.write("{}\t{}\n".format("Hotspot line opacity",hotspot_alpha))
			outfile.write("{}\t{}\n".format("Hotspot line type",hotspot_linetype))	
			outfile.write("{}\t{}\n".format("Hotspot radius",hotspot_radius))	
			outfile.write("{}\t{}\n".format("Hotspot color",hotspot_color))				
			outfile.write("{}\t{}\n".format("Plot save format",saveformat))
			outfile.write("{}\t{}\n".format("Plot save dpi",savedpi))
			outfile.write("{}\t{}\n".format("Plot background transparent",savetransparency))
			outfile.write("{}\t{}\n".format("Auto cluster",autocluster))
			outfile.write("{}\t{}\n".format("Auto plot",autoplot))
			outfile.write("{}\t{}\n".format("Cluster size screen",radius_thresh))	
			outfile.write("{}\t{}\n".format("Auto metric",auto_metric))	
			outfile.write("{}\t{}\n".format("MSD filter",msd_filter))
			outfile.write("{}\t{}\n".format("VAR color",var_color))
			outfile.write("{}\t{}\n".format("MSD color",msd_color))
			outfile.write("{}\t{}\n".format("Cluster color",clust_color))
			outfile.write("{}\t{}\n".format("3D axes",axes_3d))
			outfile.write("{}\t{}\n".format("VAR color confined",var_color1))
			outfile.write("{}\t{}\n".format("VAR color unconfined",var_color2))
			outfile.write("{}\t{}\n".format("MSD color < AVG",msd_color1))
			outfile.write("{}\t{}\n".format("MSD color > AVG",msd_color2))
			outfile.write("{}\t{}\n".format("Pixel size (um)",pixel))
		return
		
	# LOAD DEFAULTS
	def load_defaults():
		global defaultdict,traj_prob,detection_alpha,minlength,maxlength,acq_time,time_threshold,radius_factor,cluster_threshold,canvas_color,plot_trajectories,plot_centroids,plot_clusters,plot_colorbar,line_width,line_alpha,line_color,centroid_size,centroid_alpha,centroid_color,cluster_alpha,cluster_linetype,cluster_width,saveformat,savedpi,savetransparency,savefolder,selection_density,autoplot,autocluster,radius_thresh,cluster_fill,auto_metric,plotxmin,plotxmax,plotymin,plotymax,msd_filter,frame_time,tmin,tmax,plot_hotspots,hotspot_alpha,hotspot_linetype,hotspot_width,hotspot_color,hotspot_radius,var_color,axes_3d,msd_color,clust_color,var_color1,var_color2,msd_color1,msd_color2,pixel
		try:
			with open ("nastic_gui.defaults","r") as infile:
				print ("Loading GUI settings from nastic_gui.defaults...")
				defaultdict = {}
				for line in infile:
					spl = line.split("\t")
					defaultdict[spl[0]] = spl[1].strip()
			traj_prob = float(defaultdict["Trajectory probability"])
			detection_alpha = float(defaultdict["Raw trajectory detection plot opacity"])
			selection_density = float(defaultdict["Selection density"])
			minlength = int(defaultdict["Trajectory minimum length"])
			maxlength = int(defaultdict["Trajectory maximum length"])
			acq_time = int(defaultdict["Acquisition time (s)"])
			frame_time = float(defaultdict["Frame time (s)"])
			time_threshold = int(defaultdict["Time threshold (s)"])
			radius_factor = float(defaultdict["Radius factor"])
			cluster_threshold = int(defaultdict["Cluster threshold"])
			canvas_color = defaultdict["Canvas color"]
			plot_trajectories = defaultdict["Plot trajectories"]
			if plot_trajectories == "True":
				plot_trajectories = True
			if plot_trajectories == "False":
				plot_trajectories = False			
			plot_centroids = defaultdict["Plot centroids"]
			if plot_centroids == "True":
				plot_centroids = True
			if plot_centroids == "False":
				plot_centroids = False
			plot_clusters = defaultdict["Plot clusters"]
			if plot_clusters == "True":
				plot_clusters = True
			if plot_clusters == "False":
				plot_clusters = False
			plot_colorbar = defaultdict["Plot colorbar"]
			if plot_colorbar == "True":
				plot_colorbar = True
			if plot_colorbar == "False":
				plot_colorbar = False
			plot_hotspots = defaultdict["Plot hotspots"]
			if plot_hotspots == "True":
				plot_hotspots = True
			if plot_hotspots == "False":
				plot_hotspots = False					
			line_width = float(defaultdict["Trajectory line width"])	
			line_alpha = float(defaultdict["Trajectory line opacity"])	
			line_color = defaultdict["Trajectory line color"]	
			centroid_size = int(defaultdict["Centroid size"])	
			centroid_alpha = float(defaultdict["Centroid opacity"])	
			centroid_color = defaultdict["Centroid color"]
			cluster_width = float(defaultdict["Cluster line width"])	
			cluster_alpha = float(defaultdict["Cluster line opacity"])	
			cluster_linetype = defaultdict["Cluster line type"]		
			cluster_fill = defaultdict["Cluster fill"]
			if cluster_fill == "True":
				cluster_fill = True
			if cluster_fill == "False":
				cluster_fill = False	
			hotspot_color = defaultdict["Hotspot color"]
			hotspot_radius = defaultdict["Hotspot radius"]
			hotspot_width = float(defaultdict["Hotspot line width"])	
			hotspot_alpha = float(defaultdict["Hotspot line opacity"])	
			hotspot_linetype = defaultdict["Hotspot line type"]					
			saveformat = defaultdict["Plot save format"]
			savedpi = defaultdict["Plot save dpi"]
			savetransparency = defaultdict["Plot background transparent"]
			if savetransparency == "True":
				savetransparency = True
			if savetransparency == "False":
				savetransparency = False
			autoplot = defaultdict["Auto plot"]
			if autoplot == "True":
				autoplot = True
			if autoplot == "False":
				autoplot = False	
			autocluster = defaultdict["Auto cluster"]
			if autocluster == "True":
				autocluster = True
			if autocluster == "False":
				autocluster = False
			radius_thresh = defaultdict["Cluster size screen"]
			auto_metric = defaultdict["Auto metric"]	
			if auto_metric == "True":
				auto_metric = True
			if auto_metric == "False":
				auto_metric = False	
			plotxmin=""
			plotxmax=""
			plotymin=""
			plotymax=""
			msd_filter = defaultdict["MSD filter"]
			if msd_filter == "True":
				msd_filter = True
			if msd_filter == "False":
				msd_filter = False
			var_color = defaultdict["VAR color"]
			if var_color == "True":
				var_color = True
			if var_color == "False":
				var_color = False	
			axes_3d = defaultdict["3D axes"]
			if axes_3d == "True":
				axes_3d = True
			if axes_3d == "False":
				axes_3d = False		
			msd_color = defaultdict["MSD color"]
			if msd_color == "True":
				msd_color = True
			if msd_color == "False":
				msd_color = False		
			clust_color = defaultdict["Cluster color"]
			if clust_color == "True":
				clust_color = True
			if clust_color == "False":
				clust_color = False		
			var_color1 = defaultdict["VAR color confined"]		
			var_color2 = defaultdict["VAR color unconfined"]	
			msd_color1 = defaultdict["MSD color < AVG"]		
			msd_color2 = defaultdict["MSD color > AVG"]
			pixel = defaultdict["Pixel size (um)"]
		except:
			print ("Settings could not be loaded")
		return
		
	# UPDATE GUI BUTTONS
	def update_buttons():
		if len(infilename) > 0:  
			window.Element("-PLOTBUTTON-").update(button_color=("white","#111111"),disabled=False)
			window.Element("-INFILE-").InitialFolder = os.path.dirname(infilename)			
		else:
			window.Element("-PLOTBUTTON-").update(button_color=("white","gray"),disabled=True)	
		if len(trajdict) > 0:
			for buttonkey in ["-R1-","-R2-","-R3-","-R4-","-R5-","-R6-","-R7-","-R8-"]:
				window.Element(buttonkey).update(disabled=False)
		else:
			for buttonkey in ["-R1-","-R2-","-R3-","-R4-","-R5-","-R6-","-R7-","-R8-"]:
				window.Element(buttonkey).update(disabled=True)	
		if len(roi_list) > 0:  
			window.Element("-SELECTBUTTON-").update(button_color=("white","#111111"),disabled=False)
		else:  
			window.Element("-SELECTBUTTON-").update(button_color=("white","gray"),disabled=True)			
		if len(sel_traj) > 0:  
			window.Element("-CLUSTERBUTTON-").update(button_color=("white","#111111"),disabled=False)
		else:  
			window.Element("-CLUSTERBUTTON-").update(button_color=("white","gray"),disabled=True)	
		if len(clusterdict) > 0:  
			window.Element("-DISPLAYBUTTON-").update(button_color=("white","#111111"),disabled=False)
			if plotflag:
				window.Element("-SAVEBUTTON-").update(button_color=("white","#111111"),disabled=False)
			window.Element("-CANVASCOLORCHOOSE-").update(disabled=False)
			window.Element("-LINECOLORCHOOSE-").update(disabled=False)
			window.Element("-CENTROIDCOLORCHOOSE-").update(disabled=False)
			window.Element("-HOTSPOTCOLORCHOOSE-").update(disabled=False)
			window.Element("-SAVEANALYSES-").update(button_color=("white","#111111"),disabled=False)
			window.Element("-VARCOLOR1CHOOSE-").update(disabled=False)
			window.Element("-VARCOLOR2CHOOSE-").update(disabled=False)
			window.Element("-MSDCOLOR1CHOOSE-").update(disabled=False)
			window.Element("-MSDCOLOR2CHOOSE-").update(disabled=False)
			for buttonkey in ["-M1-","-M2-","-M3-","-M4-","-M5-","-M6-","-M7-","-M8-"]:
				window.Element(buttonkey).update(disabled=False)
		else:  
			window.Element("-DISPLAYBUTTON-").update(button_color=("white","gray"),disabled=True)
			window.Element("-SAVEBUTTON-").update(button_color=("white","gray"),disabled=True)
			window.Element("-CANVASCOLORCHOOSE-").update(disabled=True)
			window.Element("-LINECOLORCHOOSE-").update(disabled=True)
			window.Element("-CENTROIDCOLORCHOOSE-").update(disabled=True)
			window.Element("-HOTSPOTCOLORCHOOSE-").update(disabled=True)
			window.Element("-SAVEANALYSES-").update(button_color=("white","gray"),disabled=True)	
			window.Element("-VARCOLOR1CHOOSE-").update(disabled=True)
			window.Element("-VARCOLOR2CHOOSE-").update(disabled=True)
			window.Element("-MSDCOLOR1CHOOSE-").update(disabled=True)
			window.Element("-MSDCOLOR2CHOOSE-").update(disabled=True)
			for buttonkey in ["-M1-","-M2-","-M3-","-M4-","-M5-","-M6-","-M7-","-M8-"]:
				window.Element(buttonkey).update(disabled=True)	
		window.Element("-TRAJPROB-").update(traj_prob)
		window.Element("-DETECTIONALPHA-").update(detection_alpha)	
		window.Element("-SELECTIONDENSITY-").update(selection_density)	
		window.Element("-MINLENGTH-").update(minlength)
		window.Element("-MAXLENGTH-").update(maxlength)	
		window.Element("-ACQTIME-").update(acq_time)
		window.Element("-FRAMETIME-").update(frame_time)	
		window.Element("-TIMETHRESHOLD-").update(time_threshold)	
		window.Element("-RADIUSFACTOR-").update(radius_factor)	
		window.Element("-CLUSTERTHRESHOLD-").update(cluster_threshold)
		window.Element("-CANVASCOLORCHOOSE-").update("Choose",button_color=("gray",canvas_color))	
		window.Element("-CANVASCOLOR-").update(canvas_color)	
		window.Element("-TRAJECTORIES-").update(plot_trajectories)
		window.Element("-CENTROIDS-").update(plot_centroids)
		window.Element("-CLUSTERS-").update(plot_clusters)
		window.Element("-HOTSPOTS-").update(plot_hotspots)			
		window.Element("-COLORBAR-").update(plot_colorbar)	
		window.Element("-LINEWIDTH-").update(line_width)
		window.Element("-LINEALPHA-").update(line_alpha)
		window.Element("-LINECOLORCHOOSE-").update("Choose",button_color=("gray",line_color))
		window.Element("-LINECOLOR-").update(line_color)		
		window.Element("-CENTROIDSIZE-").update(centroid_size)
		window.Element("-CENTROIDALPHA-").update(centroid_alpha)
		window.Element("-CENTROIDCOLORCHOOSE-").update("Choose",button_color=("gray",centroid_color))
		window.Element("-CENTROIDCOLOR-").update(centroid_color)
		window.Element("-CLUSTERWIDTH-").update(cluster_width)
		window.Element("-CLUSTERALPHA-").update(cluster_alpha)
		window.Element("-CLUSTERLINETYPE-").update(cluster_linetype)
		window.Element("-CLUSTERFILL-").update(cluster_fill)
		window.Element("-HOTSPOTCOLORCHOOSE-").update("Choose",button_color=("gray",hotspot_color))
		window.Element("-HOTSPOTCOLOR-").update(hotspot_color)		
		window.Element("-HOTSPOTWIDTH-").update(hotspot_width)
		window.Element("-HOTSPOTALPHA-").update(hotspot_alpha)
		window.Element("-HOTSPOTLINETYPE-").update(hotspot_linetype)		
		window.Element("-HOTSPOTRADIUS-").update(hotspot_radius)				
		window.Element("-SAVEFORMAT-").update(saveformat)	
		window.Element("-SAVETRANSPARENCY-").update(savetransparency)
		window.Element("-SAVEDPI-").update(savedpi)
		window.Element("-SAVEFOLDER-").update(savefolder)
		window.Element("-RADIUSTHRESH-").update(radius_thresh)
		window.Element("-AUTOCLUSTER-").update(autocluster) 
		window.Element("-AUTOPLOT-").update(autoplot) 
		window.Element("-AUTOMETRIC-").update(auto_metric)
		window.Element("-PLOTXMIN-").update(plotxmin)
		window.Element("-PLOTXMAX-").update(plotxmax)
		window.Element("-PLOTYMIN-").update(plotymin)
		window.Element("-PLOTYMAX-").update(plotymax)
		window.Element("-MSDFILTER-").update(msd_filter)
		window.Element("-VARCOLOR-").update(var_color)
		window.Element("-MSDCOLOR-").update(msd_color)
		window.Element("-CLUSTCOLOR-").update(clust_color)
		window.Element("-TMIN-").update(tmin)
		window.Element("-TMAX-").update(tmax)	
		window.Element("-AXES3D-").update(axes_3d)
		window.Element("-VARCOLOR1CHOOSE-").update(" < ",button_color=("gray",var_color1))		
		window.Element("-VARCOLOR2CHOOSE-").update(" > ",button_color=("gray",var_color2))
		window.Element("-MSDCOLOR1CHOOSE-").update(" < ",button_color=("gray",msd_color1))		
		window.Element("-MSDCOLOR2CHOOSE-").update(" > ",button_color=("gray",msd_color2))		
		window.Element("-PIXEL-").update(pixel)	
		return	
		
	# CHECK VARIABLES
	def check_variables():
		global traj_prob,detection_alpha,minlength,maxlength,acq_time,time_threshold,radius_factor,cluster_threshold,canvas_color,plot_trajectories,plot_centroids,plot_clusters,line_width,line_alpha,line_color,centroid_size,centroid_alpha,centroid_color,cluster_alpha,cluster_linetype,cluster_width,saveformat,savedpi,savetransparency,savefolder,selection_density,radius_thresh,plotxmin,plotxmax,plotymin,plotymax,frame_time,tmin,tmax,plot_hotspots,hotspot_alpha,hotspot_linetype,hotspot_width,hotspot_color,hotspot_radius,msd_color,var_color,clust_color,var_color1,var_color2,msd_color1,msd_color2,pixel

		if traj_prob not in [0.01,0.05,0.1,0.25,0.5,0.75,1.0]:
			traj_prob = 1.0
		if detection_alpha not in [0.01,0.05,0.1,0.25,0.5,0.75,1.0]:
			detection_alpha = 0.25 
		try:
			selection_density = float(selection_density)
			if selection_density < 0:
				selection_density = 0
		except:
			selection_density = 0
		try:
			minlength = int(minlength)
			if minlength < 5:
				minlength = 5
		except:
			minlength = 5
		try:
			maxlength = int(maxlength)
		except:
			maxlength = 1000
		if minlength > maxlength:
			minlength = 5
			maxlength = 1000
		try:
			pixel = float(pixel)
			if pixel <= 0:
				pixel = 0.106
		except:
			pixel = 0.106	
		try:
			acq_time = int(acq_time)
			if acq_time < 1:
				acq_time = 1
		except:
			acq_time = 320
		try:
			frame_time = float(frame_time)
			if frame_time <= 0:
				frame_time = 0.02
		except:
			frame_time = 0.02			
		try:
			time_threshold = int(time_threshold)
			if time_threshold < 1:
				time_threshold = 1
		except:
			time_threshold = 20
		try:
			radius_factor = float(radius_factor)
			if radius_factor < 0.01:
				radius_factor = 1.2
		except:
			radius_factor = 1.2		
		try:
			cluster_threshold = int(cluster_threshold)
			if cluster_threshold < 2:
				cluster_threshold = 2
		except:
			cluster_threshold = 3	
		try:
			radius_thresh = float(radius_thresh)
			if radius_thresh < 0.001:
				radius_thresh = 0.2
		except:
			radius_thresh = 0.2
		if line_width not in [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]:
			line_width = 0.25 
		if line_alpha not in [0.01,0.05,0.1,0.25,0.5,0.75,1.0]:
			line_alpha = 0.25 		
		if centroid_size not in [1,2,5,10,20,50]:
			centroid_size = 5 
		if centroid_alpha not in [0.01,0.05,0.1,0.25,0.5,0.75,1.0]:
			centroid_alpha = 0.75 
		if cluster_width not in [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]:
			cluster_width = 1.5 
		if cluster_alpha not in [0.01,0.05,0.1,0.25,0.5,0.75,1.0]:
			cluster_alpha = 1.0 
		if cluster_linetype not in ["solid","dotted","dashed"]:
			cluster_linetype = "solid" 	
		if hotspot_width not in [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]:
			hotspot_width = 1.5 
		if hotspot_alpha not in [0.01,0.05,0.1,0.25,0.5,0.75,1.0]:
			hotspot_alpha = 1.0 
		if hotspot_linetype not in ["solid","dotted","dashed"]:
			hotspot_linetype = "dotted"
		if hotspot_radius not in [0.1,0.25,0.5,1.0,1.25,1.5,1.75,2.0]:
			hotspot_radius = 1.0
		if saveformat not in ["eps","pdf","png","ps","svg"]:
			saveformat = "png"
		if savedpi not in [50,100,300,600,1200]:
			savedpi = 300	
		if savefolder == "":
			savefolder = os.path.dirname(infilename)
		
		# If user presses cancel when choosing a color 	
		if canvas_color == "None":
			try:
				canvas_color = defaultdict["Canvas color"]
			except:	
				canvas_color = "black"
		if line_color == "None":
			try:
				line_color = defaultdict["Trajectory line color"]
			except:	
				line_color = "white"
		if centroid_color == "None":
			try:
				centroid_color = defaultdict["Centroid color"]
			except:	
				centroid_color = "white"
		if hotspot_color == "None":
			try:
				hotspot_color = defaultdict["Hotspot color"]
			except:	
				hotspot_color = "white"
		try:
			plotxmin = float(plotxmin)
		except:
			plotxmin = ""	
		try:
			plotxmax = float(plotxmax)
		except:
			plotxmax = ""	
		try:
			plotymin = float(plotymin)
		except:
			plotymin = ""	
		try:
			plotymax = float(plotymax)
		except:
			plotymax = ""	
		try:
			tmin = float(tmin)
			if tmin < 0 or tmin > acq_time:
				tmin = 0
		except:
			tmin = 0	
		try:
			tmax = float(tmax)
			if tmax < 0 or tmax > acq_time:
				tmax = acq_time
		except:
			tmin = acq_time			

		if msd_color:
			var_color=False
			clust_color = False
		if var_color:
			msd_color=False
			clust_color = False
		if clust_color:
			msd_color=False
			var_color=False

		if var_color1 == "None":
			try:
				var_color1 = defaultdict["VAR color confined"]
			except:	
				var_color1 = "orange"	
		if var_color2 == "None":
			try:
				var_color2 = defaultdict["VAR color unconfined"]
			except:	
				var_color2 = "purple"	
			
		if msd_color1 == "None":
			try:
				msd_color1 = defaultdict["MSD color < AVG"]
			except:	
				msd_color1 = "cyan"	
		if msd_color2 == "None":
			try:
				msd_color2 = defaultdict["MSD color > AVG"]
			except:	
				msd_color2 = "magenta"
		return

	# GET DIMENSIONS OF ZOOM
	def ondraw(event):
		global selverts
		zx = ax0.get_xlim()
		zy = ax0.get_ylim()
		selverts = [[zx[0],zy[0]],[zx[0],zy[1]],[zx[1],zy[1]],[zx[1],zy[0]],[zx[0],zy[0]]]
		selarea =PolyArea(list(zip(*selverts))[0],list(zip(*selverts))[1])
		return 

	# GET HAND DRAWN REGION
	def onselect(verts):
		global selverts
		p = path.Path(verts)
		selverts = verts[:] 
		selverts.append(selverts[0])
		selarea =PolyArea(list(zip(*selverts))[0],list(zip(*selverts))[1])
		return 	

	# AREA IN POLYGON
	def PolyArea(x,y):
		return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

	# USE SELECTION AREA
	def use_roi(selverts,color):
		all_selverts.append(selverts)
		vx,vy = list(zip(*selverts))
		roi, = ax0.plot(vx,vy,linewidth=2,c=color,alpha=1)
		roi_list.append(roi)
		plt.xlim(xlims)
		plt.ylim(ylims)
		plt.show(block=False)
		if len(roi_list) <= 1:
			window.Element("-SEPARATE-").update(disabled = True)
		elif len(roi_list) >1:
			window.Element("-SEPARATE-").update(disabled = False)
		return
		
	# READ ROI DATA	
	def read_roi():	
		#Check for ROI file type
		roi_file_split = roi_file.split(".")
		if roi_file_split[-1] == "rgn":
			
			#PalmTracer .rgn file
			window.Element("-PIXEL_TEXT-").update(visible = True)
			window.Element("-PIXEL-").update(visible = True)		
			window.Element("-REPLOT_ROI-").update(visible = True)	
			with open (roi_file, "r") as infilename:
				RGN_data = infilename.read()
				RGN_list = [x.split(", ") for x in RGN_data.split("\n")] #Info for each ROI is listed on a new line -> separate ROI info in list
				ROI_n = 0
				ROI_X_Y = np.array([[0,0,0],[0,0,0]])
				for ROI in RGN_list:
					if ROI[0] != "":
						ROI_n = ROI_n +1
						col_6_data = ROI[6]
						space2comma = col_6_data.split(" ") #Data in this column is separated by spaces
						n_coord_pairs = space2comma[1] #This position contains the number of coordinate pairs
						x_coord_pix = []
						y_coord_pix = []
						x_coord_pix.append(space2comma[2::2]) #File contains alternating x,y coordinates
						x_coord_pix = x_coord_pix.pop()
						x_coord_pix = ([int(x) for x in x_coord_pix])
						y_coord_pix.append(space2comma[3::2])
						y_coord_pix = y_coord_pix.pop()
						y_coord_pix = ([int(y) for y in y_coord_pix])
						
						#Convert pixels to microns
						x_coord_micron_list = []
						y_coord_micron_list = []
						ROI_list = []
						for i in range(0, len(x_coord_pix)):
							x_coord_micron_list.append(float(pixel)*x_coord_pix[i])
							y_coord_micron_list.append(float(pixel)*y_coord_pix[i])		
							ROI_list.append(0)
						int_n_coord_pairs = int(n_coord_pairs)
						output_file = np.zeros((int_n_coord_pairs+1, 3))
						cnt = 0
						while cnt < int_n_coord_pairs:
							for r in output_file:
								output_file[cnt,0] = int(ROI_n-1)
								cnt+=1
						if cnt == int_n_coord_pairs:
							output_file[cnt,0] = int(ROI_n-1)
						cnt = 0
						while cnt < len(x_coord_micron_list):
							for x in x_coord_micron_list:
								output_file[cnt,1] = x
								cnt+=1
						if cnt == int_n_coord_pairs:
							output_file[cnt,1] = x_coord_micron_list[0]
						cnt = 0 
						while cnt < len(y_coord_micron_list):
							for y in y_coord_micron_list:
								output_file[cnt,2] = y
								cnt+=1
						if cnt == int_n_coord_pairs:
							output_file[cnt,2] = y_coord_micron_list[0]						
						ROI_X_Y = np.append(ROI_X_Y, output_file, axis = 0)
					else:
						break
				ROI_X_Y = np.delete(ROI_X_Y, 0, 0)
				ROI_X_Y = np.delete(ROI_X_Y, 0, 0)
				roidict = {}
				spl = []
				for line in ROI_X_Y:
					try:
						roi = int(float(line[0]))
						x = float(line[1])
						y = float(line[2])
						try:
							roidict[roi].append([x,y])
						except:
							roidict[roi] = []
							roidict[roi].append([x,y])
					except:
						pass
			if len(roidict) == 0:
				sg.Popup("Alert", "No ROIs found")
			else:
				for roi in roidict:
					selverts = roidict[roi]
					use_roi(selverts,"orange")
				return
			return
		
		else:
			#NASTIC roi_coordinates.tsv file / SEGNASTIC roi_coordinates.tsv file 
			window.Element("-PIXEL_TEXT-").update(visible=False)
			window.Element("-PIXEL-").update(visible=False)
			window.Element("-REPLOT_ROI-").update(visible = False)
			roidict = {}
			try: 
				with open (roi_file,"r") as infile:
					for line in infile:
						spl = line.split("\t")
						try:
							roi = int(spl[0])
							x = float(spl[1])
							y = float(spl[2])
							try:
								roidict[roi].append([x,y])
							except:	
								roidict[roi] = []
								roidict[roi].append([x,y])
						except:
							pass
				if len(roidict) == 0:
					sg.Popup("Alert","No ROIs found")
				else:	
					for roi in roidict:			
						selverts =roidict[roi]	
						use_roi(selverts,"orange")
			except:
				pass
		return 

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

	# DISTILL OVERLAPPING LISTS	
	def distill_list(overlappers):
		sets = [set(x) for x in overlappers]
		allelts = set.union(*sets)
		components = {x: {x} for x in allelts}
		component = {x: x for x in allelts}
		for s in sets:
			comp = sorted({component[x] for x in s})
			mergeto = comp[0]
			for mergefrom in comp[1:]:
				components[mergeto] |= components[mergefrom]
				for x in components[mergefrom]:
					component[x] = mergeto
				del components[mergefrom]
		distilled =  components.values()
		distilled = [list(x) for x in distilled]	
		return distilled

	# FIND TRAJECTORIES WHOSE BOUNDING BOXES OVERLAP IN SPACE AND TIME	
	def trajectory_overlap(indices,time_threshold,av_msd):
		# Create and populate 3D r-tree
		p = index.Property()
		p.dimension=3
		idx_3d = index.Index(properties=p)
		intree = []		
		
		# Analyse all trajectories regardless of mobility 
		for idx in indices:
			if seldict[idx]["msds"][0] < av_msd:
				idx_3d.insert(idx,seldict[idx]["bounding_box"])
				intree.append(idx)	
				
		# Query the r-tree
		overlappers = []
		for idx in intree:
			if idx%10 == 0: 
				try: 
					bar = 100*idx/(len(intree)-10)
					window['-PROGBAR-'].update_bar(bar)
				except:
					pass
			bbox = seldict[idx]["bounding_box"]
			left,bottom,early,right,top,late = bbox[0],bbox[1],bbox[2]-time_threshold/2,bbox[3],bbox[4],bbox[5]+time_threshold/2
			intersect = list(idx_3d.intersection([left,bottom,early,right,top,late]))
			overlap = [int(x) for x in intersect]
			overlappers.append(overlap)	
			
		# Distill the list	
		overlappers =  distill_list(overlappers)
		overlappers = [x for x in overlappers if len(x) >= cluster_threshold]
		return overlappers,intree

	# PROBABILITY OF CLUSTER OVERLAP AT A GIVEN DISTANCE	
	def overlap_prob(clustpoints,epsilon):
		labels,clusterlist = dbscan(clustpoints,epsilon,2) # does each cluster centroid form a clustered cluster within dist (epsilon)?
		clusterlist = [x for x in clusterlist]
		unclustered = [x for x in labels if x == -1] # select DBSCAN unclustered centroids	
		p = 1 - float(len(unclustered))/len(labels) # probability of adjacent cluster centroids within dist
		return p

	# LOAD AND PLOT TRXYT TAB
	def trxyt_tab(filter_status):
		# Reset variables
		global all_selverts,all_selareas,roi_list,trajdict,sel_traj,lastfile,seldict,clusterdict,x_plot,y_plot,xlims,ylims,savefolder,buf 
		all_selverts = [] # all ROI vertices
		all_selareas = [] # all ROI areas
		roi_list = [] # ROI artists
		trajdict = {} # Dictionary holding raw trajectory info
		sel_traj = [] # Selected trajectory indices
		lastfile = "" # Force the program to load a fresh TRXYT
		seldict = {} # Selected trajectories and metrics
		clusterdict = {} # Cluster information
		
		# Close all opened windows
		for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
			try:
				plt.close(i)
			except:
				pass
		# Close all buffers		
		try:
			buf0.close()
		except:
			pass	
		try:
			buf1.close()
		except:
			pass	
		try:
			buf2.close()
		except:
			pass	
		try:
			buf3.close()
		except:
			pass	
		try:
			buf4.close()
		except:
			pass	
		try:
			buf5.close()
		except:
			pass	
		try:
			buf6.close()
		except:
			pass	
		try:
			buf7.close()
		except:
			pass	
		try:
			buf8.close()
		except:
			pass
		try:
			buf9.close()
		except:
			pass

		'''
		IMPORTANT: It appears that some matlab processing of trajectory data converts trajectory numbers > 99999 into scientific notation with insufficient decimal points. eg 102103 to 1.0210e+05, 102104 to 1.0210e+05. This can cause multiple trajectories to be incorrectly merged into a  single trajectory.
		For trajectories > 99999 we empirically determine whether detections are within 0.32u of each other, and assign them into a single trajectory accordingly. For trajectories <99999 we honour the existing trajectory number.
		'''		
		trajectory_error = False
		if filter_status == False:
			if infilename != lastfile:
				# Read file into dictionary
				lastfile=infilename
				print("Loading raw trajectory data from {}...".format(infilename))
				t1=time.time()
				rawtrajdict = {}
				ct = 99999
				x0 = -10000
				y0 = -10000
				with open (infilename,"r") as infile:
					for line in infile:
						try:
							line = line.replace("\n","").replace("\r","")
							spl = line.split(" ")
							n = int(float(spl[0]))
							x = float(spl[1])
							y = float(spl[2])
							t = float(spl[3])
							if n > 99999:
								if abs(x-x0) < 0.32 and abs(y-y0) < 0.32:
									rawtrajdict[ct]["points"].append([x,y,t])
									x0 = x
									y0= y
								else:
									ct += 1
									rawtrajdict[ct]= {"points":[[x,y,t]]}	
									x0 = x
									y0=y
							else:
								try:
									rawtrajdict[n]["points"].append([x,y,t])
								except:
									rawtrajdict[n]= {"points":[[x,y,t]]}
						except:
							pass
				print("{} trajectories read".format(len(rawtrajdict))) 			
			
			# Don't bother with anything else if there's no trajectories				
			if len(rawtrajdict) == 0:
				sg.Popup("Alert","No trajectory information found")
				trajectory_error = True 
			else:
				# Screen trajectories by length
				filttrajdict = {} 
				print("Filtering trajectories by length...") 
				for num,traj in enumerate(rawtrajdict):
					if num%1000 == 0: 
						try: 
							bar = 100*num/(len(rawtrajdict)-1) 
							window['-PROGBAR-'].update_bar(bar)
						except:
							pass
					points = rawtrajdict[traj]["points"]
					x,y,t = zip(*points)
					if len(points) >=minlength and len(points) <=maxlength and variation(x) > 0.0001 and variation(y) > 0.0001:
						filttrajdict[traj] = rawtrajdict[traj]
				window['-PROGBAR-'].update_bar(0)	
				if len(filttrajdict) == 1:
					print("1 remaining trajectory") 
					sg.Popup("Alert","Not enough trajectories remaining after length filtering")
					trajectory_error = True 
				elif len(filttrajdict) == 0:
					print("0 remaining trajectories") 
					sg.Popup("Alert","No trajectories remaining after length filtering")
					trajectory_error = True 
				elif len(filttrajdict) >1:
					print(len(filttrajdict), "remaining trajectories")								
		else:
			t1=time.time()
		if trajectory_error == False:
			trajdict = filttrajdict
			# Display detections
			print("Plotting detections...")
			ct = 0
			ax0.cla() # clear last plot if present
			detpoints = []
			for num,traj in enumerate(trajdict):
				if num%10 == 0:
					try:
						bar = 100*num/(len(trajdict))
						window['-PROGBAR-'].update_bar(bar)
					except:
						pass
				if random.random() <= traj_prob:
					ct+=1
					[detpoints.append(i) for i in trajdict[traj]["points"]]
			x_plot,y_plot,t_plot=zip(*detpoints)
			ax0.scatter(x_plot,y_plot,c="w",s=3,linewidth=0,alpha=detection_alpha)	
			ax0.set_facecolor("k")	
			ax0.set_xlabel("X")
			ax0.set_ylabel("Y")	
			xlims = plt.xlim()
			ylims = plt.ylim()	

			# Force correct aspect using imshow - very proud of discovering this by accident
			ax0.imshow([[0,1], [0,1]], 
			extent = (xlims[0],xlims[1],ylims[0],ylims[1]),
			cmap = cmap, 
			interpolation = 'bicubic',
			alpha=0)
			plt.tight_layout()
			plt.show(block=False)		
			window['-PROGBAR-'].update_bar(0)
			t2 = time.time()
			print("{} detections from {} trajectories plotted in {} sec".format(len(x_plot),ct,round(t2-t1,3)))
			
			# Pickle this raw image
			buf = io.BytesIO()
			pickle.dump(ax0, buf)
			buf.seek(0)
			
			# Clear the variables for ROI selection
			all_selverts = [] # all ROI vertices
			all_selareas = [] # all ROI areas
			roi_list = [] # ROI artists
			window["-TABGROUP-"].Widget.select(1)
		return 
		
	# ROI SELECTION TAB
	def roi_tab():
		global selverts,all_selverts,all_selareas,roi_list,trajdict,sel_traj,sel_centroids,all_selverts_copy,all_selverts_bak,prev_roi_file
		
		# Load and apply ROIs	
		if event ==	"-R2-" and roi_file != "Load previously defined ROIs" and roi_file != prev_roi_file and os.path.isfile(roi_file) == True:
			prev_roi_file = roi_file
			all_selverts_bak = [x for x in all_selverts]
			try:
				selverts_reset = [x for x in all_selverts_copy]
			except:
				selverts_reset = []
			if len(selverts) >3:
				if len(selverts_reset) == 0:
					selverts_reset = [x for x in all_selverts]
				filter_status = True 
				trxyt_tab(filter_status)
				window.Element('-RESET-').update(disabled = True)
			if len(roi_list) <= 1:
				window.Element('-SEPARATE-').update(disabled = True)
			elif len(roi_list) > 1:
				window.Element('-SEPARATE-').update(disabled = False)
				for roi in roi_list:
					roi.remove()
					roi_list = []			
					all_selverts = []
					selverts = []
					sel_traj = []
			# Close all opened windows
			for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
				try:
					plt.close(i)
				except:
					pass
			# Close all buffers		
			try:
				buf0.close()
			except:
				pass	
			try:
				buf1.close()
			except:
				pass	
			try:
				buf2.close()
			except:
				pass	
			try:
				buf3.close()
			except:
				pass	
			try:
				buf4.close()
			except:
				pass	
			try:
				buf5.close()
			except:
				pass	
			try:
				buf6.close()
			except:
				pass	
			try:
				buf7.close()
			except:
				pass	
			try:
				buf8.close()
			except:
				pass
			try:
				buf9.close()
			except:
				pass
			roidict = read_roi()
				
		# Clear all ROIs
		if event ==	"-R3-" and len(roi_list) > 0:
			all_selverts_bak = [x for x in all_selverts]
			for roi in roi_list:
				roi.remove()
			roi_list = []			
			all_selverts = []
			selverts = []
			sel_traj = []
			plt.show(block=False)
			window.Element("-SEPARATE-").update(disabled=True)
			window.Element("-PIXEL_TEXT-").update(visible=False)
			window.Element("-PIXEL-").update(visible=False)
			window.Element("-REPLOT_ROI-").update(visible=False)

		# Remove last added ROI
		if event ==	"-R6-" and len(roi_list) > 0:	
			all_selverts_bak = [x for x in all_selverts]
			roi_list[-1].remove()
			roi_list.pop(-1)	
			all_selverts.pop(-1)
			selverts = []
			plt.show(block=False)	
			if len(roi_list) <=1:
				window.Element("-SEPARATE-").update(disabled=True)

		# Add ROI encompassing all detections	
		if event ==	"-R4-":
			all_selverts_bak = [x for x in all_selverts]
			for roi in roi_list:
				roi.remove()
			roi_list = list()
			xmin = min(x_plot)
			xmax = max(x_plot)
			ymin = min(y_plot)
			ymax = max(y_plot)
			all_selverts = []
			selverts = [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax],[xmin,ymin]]
			use_roi(selverts,"orange")
			
		# Add current ROI	
		if event ==	"-R5-" and len(selverts) > 3:
			all_selverts_bak = [x for x in all_selverts]
			if selverts[0][0] != xlims[0] and selverts[0][1] != ylims[0]: # don't add entire plot
				use_roi(selverts,"orange")
			window.Element("-PIXEL_TEXT-").update(visible=False)
			window.Element("-PIXEL-").update(visible=False)
			window.Element("-REPLOT_ROI-").update(visible=False)
				
		# Undo last ROI change	
		if event ==	"-R7-":
			try:
				len(all_selverts_bak)
				if len(all_selverts_bak) >0:
					if len(roi_list) > 0:
						for roi in roi_list:
							roi.remove()
					roi_list = list()
					plt.show(block=False)	
					all_selverts = []	
					for selverts in all_selverts_bak:
						use_roi(selverts,"orange")
					if len(roi_list) <1:
						window.Element("-RESET-").update(disabled=True)
			except:
				pass
				
		# Reset to original view with ROI
		if event == "-RESET-":
			selverts_reset = [x for x in all_selverts_copy]
			if len(selverts) >3:
				if len(selverts_reset) == 0:
					selverts_reset = [x for x in all_selverts]
				filter_status = True
				trxyt_tab(filter_status)
				for selverts in selverts_reset:
					use_roi(selverts,"orange")
				window.Element('-RESET-').update(disabled = True)
		
		# Save current ROIs	
		if event ==	"-R8-" and len(all_selverts) > 0:
			stamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now())
			outpath = os.path.dirname(infilename)
			outdir = outpath + "/" + infilename.split("/")[-1].replace(".trxyt","_NASTIC_ROIs_{}".format(stamp))
			try:
				os.mkdir(outdir)
				roi_directory = outdir
				os.makedirs(roi_directory,exist_ok = True)
				os.chdir(roi_directory)
				roi_save = "{}_roi_coordinates.tsv".format(stamp)
				with open(roi_save,"w") as outfile:
					outfile.write("ROI\tx(um)\ty(um)\n")
					for roi,selverts in enumerate(all_selverts):
						for coord in selverts:
							outfile.write("{}\t{}\t{}\n".format(roi,coord[0],coord[1]))	
				print ("Current ROIs saved as {}_roi_coordinates.tsv".format(stamp))
			except:
				sg.Popup("Alert", "Error with saving ROIs", "Check whether ROIs are already saved")
		
		# Save current ROIs as separate files
		if event == "-SEPARATE-":
			stamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now())
			outpath = os.path.dirname(infilename)
			outdir = outpath + "/" + infilename.split("/")[-1].replace(".trxyt","_NASTIC_ROIs_{}".format(stamp))
			try:
				os.mkdir(outdir)
				roi_directory = outdir
				os.makedirs(roi_directory,exist_ok = True)
				os.chdir(roi_directory)
				for roi,selverts in enumerate(all_selverts):
					roi_save = "{}_roi_coordinates{}.tsv".format(stamp, roi)
					with open(roi_save,"w") as outfile:
						outfile.write("ROI\tx(um)\ty(um)\n")
						for coord in selverts:
							outfile.write("{}\t{}\t{}\n".format(roi,coord[0],coord[1]))	
					print ("ROI{} saved as {}_roi_coordinates{}.tsv".format(roi,stamp,roi))
			except:
				sg.Popup("Alert", "Error with saving ROIs", "Check whether ROIs are already saved")

		# Select trajectories within ROIs			
		if event ==	"-SELECTBUTTON-" and len(roi_list) > 0:
			if len(roi_list) >1: 
				print ("Selecting trajectories within {} ROIs...".format(len(roi_list)))
			else: 
				print("Selecting trajectories within 1 ROI...") 
			t1=time.time()
		
			# Centroids for each trajectory
			all_centroids = []
			for num,traj in enumerate(trajdict):
				if num%10 == 0:
					try:
						bar = 100*num/(len(trajdict))
						window['-PROGBAR-'].update_bar(bar)
					except:
						pass
				points = trajdict[traj]["points"]
				x,y,t=list(zip(*points))
				xmean = np.average(x)
				ymean = np.average(y)	
				tmean = np.average(t)
				centroid = [xmean,ymean,tmean]
				trajdict[traj]["centroid"] = centroid
				all_centroids.append([centroid[0],centroid[1],traj])
			sel_traj = []
			sel_centroids = []
			all_selareas = []
			for selverts in all_selverts:
				selx,sely=list(zip(*selverts))
				minx=min(selx)
				maxx=max(selx)
				miny=min(sely)
				maxy=max(sely)	
				pointarray = [i for i in all_centroids if i[0] > minx and i[0] < maxx and i[1] > miny and i[1] < maxy] # pre screen for centroids in selection bounding box
				
				p = path.Path(selverts)
				pointarray = [i for i in pointarray if p.contains_point(i)] # screen for presecreened centroids actually within selection
				selarea =PolyArea(list(zip(*selverts))[0],list(zip(*selverts))[1])
				all_selareas.append(selarea)
				[sel_traj.append(i[2]) for i in pointarray]	
			sel_traj = list(set(sel_traj)) # remove duplicates from any overlapping
			density = float(len(sel_traj)/sum(all_selareas))
			window.Element("-DENSITY-").update(round(density,2))
			if selection_density > 0:
				thresh = selection_density/density
				sel_traj =[i for i in sel_traj if random.random()< thresh]
			all_selverts_copy = [x for x in all_selverts]
			all_selverts = []
			for roi in roi_list:
				roi.remove()
			roi_list = []
			if len(sel_traj) == 0:
				sg.Popup("Alert", "No trajectories found in selected ROI", "Save ROIs that you want to keep before Removing")
				for selverts in all_selverts_copy:
					use_roi(selverts,"orange")
				window['-PROGBAR-'].update_bar(0)
			else:
				window['-PROGBAR-'].update_bar(0)
				t2=time.time()
				print ("{} trajectories selected in {}um^2, {} sec".format(len(sel_traj),round(sum(all_selareas),2),t2-t1))
				density = float(len(sel_traj)/sum(all_selareas))		
				print ("{} trajectories/um^2".format(round(density,2)))
				window.Element("-DENSITY-").update(round(density,2))
				window["-TABGROUP-"].Widget.select(2)
				if autocluster:
					cluster_tab()
				else:  
					for selverts in all_selverts_copy:
						use_roi(selverts,"orange")
		return
		
	# CLUSTERING TAB	
	def cluster_tab():
		global seldict,clusterdict,allindices,clustindices,unclustindices,spatial_clusters,all_diffcoeffs,av_msd,all_msds,msd_filter_threshold

		# Dictionary of selected trajectories
		print ("Generating bounding boxes of selected trajectories...")
		sel_centroids = []
		seldict = {}
		t1=time.time()
		all_msds = []
		all_diffcoeffs = []
		
		allpoints = [[trajdict[traj]["points"],minlength,radius_factor,trajdict[traj]["centroid"]] for traj in sel_traj]
		
		allmetrics = multi(allpoints) # fork these calculations onto all cores
		for num,metrics in enumerate(allmetrics):
			if num%10 == 0:
				try: 
					bar = 100*num/(len(allmetrics))
					window['-PROGBAR-'].update_bar(bar)
				except:
					pass
			seldict[num]={}
			points,msds,area,radius,bbox,centroid,diffcoeff = metrics
			seldict[num]["points"]=points
			seldict[num]["msds"]=msds
			all_msds.append(msds[0])
			seldict[num]["area"]=area
			seldict[num]["diffcoeff"]=diffcoeff/(frame_time*3)
			all_diffcoeffs.append(abs(diffcoeff))
			seldict[num]["radius"]=radius
			seldict[num]["bounding_box"]=bbox
			seldict[num]["centroid"]=centroid
			sel_centroids.append(centroid)
		window['-PROGBAR-'].update_bar(0)		
		t2=time.time()	
		print ("{} bounding boxes generated in {} sec".format(len(allmetrics),round(t2-t1,3)))	

		# Screen on MSD
		if msd_filter:
			print ("Calculating average MSD...")
			av_msd = np.average(all_msds)
			msd_filter_threshold = np.average(all_msds)
		else:
			av_msd = 10000 # no molecule except in an intergalactic gas cloud has an MSD this big
			msd_filter_threshold = 10000
		
		# Determine overlapping trajectories
		print ("Clustering selected trajectories...")
		indices = range(len(seldict))
		t1 = time.time()
		spatial_clusters,intree =  trajectory_overlap(indices,time_threshold,av_msd)
		t2 = time.time()
		print ("{} trajectories clustered in {} sec".format(len(intree),round(t2-t1,3)))
		
		# Cluster metrics
		print ("Generating metrics of clustered trajectories...")
		t1 = time.time()
		clusterdict = {} # dictionary holding info for each spatial cluster
		for num,cluster in enumerate(spatial_clusters):
			if num%10 == 0: 
				try: 
					bar = 100*num/(len(spatial_clusters)-1) 
					window['-PROGBAR-'].update_bar(bar) 
				except:
					pass
			clusterdict[num] = {"indices":cluster} # indices of trajectories in this cluster
			clusterdict[num]["traj_num"] = len(cluster) # number of trajectories in this cluster
			clustertimes = [seldict[i]["centroid"][2] for i in cluster] # all centroid times in this cluster
			clusterdict[num]["centroid_times"] = clustertimes
			clusterdict[num]["lifetime"] = max(clustertimes) - min(clustertimes) # lifetime of this cluster (sec)
			msds = [seldict[i]["msds"][0] for i in cluster] # MSDs for each trajectory in this cluster
			clusterdict[num]["av_msd"]= np.average(msds) # average trajectory MSD in this cluster
			diffcoeffs = [seldict[i]["diffcoeff"] for i in cluster] # Instantaneous diffusion coefficients for each trajectory in this cluster
			clusterdict[num]["av_diffcoeff"]= np.average(diffcoeffs) # average trajectory inst diff coeff in this cluster
			clusterpoints = [point[:2]  for i in cluster for point in seldict[i]["points"]] # All detection points [x,y] in this cluster
			clusterdict[num]["det_num"] = len(clusterpoints) # number of detections in this cluster	
			try: 
				ext_x,ext_y,ext_area,int_x,int_y,int_area = double_hull(clusterpoints) # Get external/internal hull area 
			except:
				sg.Popup("Alert","Clustering error","Please try different clustering metrics")
				return 
			clusterdict[num]["area"] = int_area # internal hull area as cluster area (um2)
			clusterdict[num]["radius"] = math.sqrt(int_area/math.pi) # radius of cluster (um)
			clusterdict[num]["area_xy"] = [int_x,int_y] # area border coordinates	
			clusterdict[num]["density"] = len(cluster)/int_area # trajectories/um2
			clusterdict[num]["rate"] = len(cluster)/(max(clustertimes) - min(clustertimes)) # accumulation rate (trajectories/sec)
			clustercentroids = [seldict[i]["centroid"] for i in cluster]
			x,y,t = zip(*clustercentroids)
			xmean = np.average(x)
			ymean = np.average(y)
			tmean = np.average(t)
			clusterdict[num]["centroid"] = [xmean,ymean,tmean] # centroid for this cluster
			
		# Screen out large clusters
		allindices = range(len(seldict))
		clustindices = []
		tempclusterdict = {}
		counter = 1
		for num in clusterdict:
			if clusterdict[num]["radius"] < radius_thresh:
				tempclusterdict[counter] = clusterdict[num]
				[clustindices.append(i) for i in clusterdict[num]["indices"]]
				counter +=1
		clusterdict = tempclusterdict.copy()
		
		if len(clusterdict) == 0:
			sg.Popup("Alert","No unique spatiotemporal clusters containing trajectories found in the selected ROI","Please try adjusting the ROI or clustering parameters")
			window.Element("-RESET-").update(disabled=False)
			window['-PROGBAR-'].update_bar(0) 
			for selverts in all_selverts_copy:
				use_roi(selverts,"orange")	
		else:
			window.Element("-RESET-").update(disabled=False)
			unclustindices = [idx for idx in allindices if idx not in clustindices]
			for selverts in all_selverts_copy:
				use_roi(selverts,"green")
			window['-PROGBAR-'].update_bar(0)
			t2 = time.time()
			print ("{} unique spatiotemporal clusters containing {} trajectories identified in {} sec".format(len(clusterdict),len(clustindices),round(t2-t1,3)))
			window["-TABGROUP-"].Widget.select(3)
			if autoplot and len(clusterdict)>0:
				display_tab(xlims,ylims)
		return

	# DISPLAY CLUSTERED DATA TAB
	def	display_tab(xlims,ylims):
		global buf0,plotflag,plotxmin,plotymin,plotxmax,plotymax,var_cols,clustgroups,confinedindices,unconfinedindices
		print ("Plotting...")
		xlims = ax0.get_xlim()
		ylims = ax0.get_ylim()
	
		#Assign trajectories into high and low mobility groups based on Kmeans clustering of vector autoregression metrics - Kyle Young
		confinedindices = []
		unconfinedindices = []		
		if var_color:
			confinedindices,unconfinedindices,var_cols = var_confine()

		# User zoom
		if plotxmin !="" and plotxmax !="" and plotymin !="" and plotymax !="":
			xlims = [plotxmin,plotxmax]
			ylims = [plotymin,plotymax]
		
		# Reset zoom	
		if plotxmin ==0.0 and plotxmax ==0.0 and plotymin ==0.0 and plotymax ==0.0:	
			xlims =	[min(x_plot),max(x_plot)]
			ylims =	[min(y_plot),max(y_plot)]
		plotxmin,plotxmax,plotymin,plotymax="","","",""

		ax0.cla()
		ax0.set_facecolor(canvas_color)
		xcent = []
		ycent = []
		
		# Plot trajectories
		t1=time.time()
		if msd_color:
			av_msd = np.average(all_msds)

		# Unclustered trajectories
		if plot_trajectories:
			print ("Plotting unclustered trajectories...")
		for num,traj in enumerate(unclustindices):
			if num%10 == 0:
				try:
					bar = 100*num/(len(seldict)-1)
					window['-PROGBAR-'].update_bar(bar)
				except:
					pass
			centx=seldict[traj]["centroid"][0]
			centy=seldict[traj]["centroid"][1]
			if centx > xlims[0] and centx < xlims[1] and centy > ylims[0] and centy < ylims[1]:
				# Plot unclustered trajectories
				if plot_trajectories:	
					x,y,t=zip(*seldict[traj]["points"])
					tr = []
					if var_color:
						tr = matplotlib.lines.Line2D(x,y,c=var_cols[seldict[traj]["kmeans_group"]],alpha=line_alpha,linewidth=line_width)
					if msd_color:	
						if seldict[traj]["msds"][0] < av_msd:
							tr = matplotlib.lines.Line2D(x,y,c=msd_color1,alpha=line_alpha,linewidth=line_width)
						else:
							tr = matplotlib.lines.Line2D(x,y,c=msd_color2,alpha=line_alpha,linewidth=line_width)
					if not var_color and not msd_color:		
						tr = matplotlib.lines.Line2D(x,y,c=line_color,alpha=line_alpha,linewidth=line_width)
					ax0.add_artist(tr) 
				# Plot centroids
				if plot_centroids:
					xcent.append(seldict[traj]["centroid"][0])
					ycent.append(seldict[traj]["centroid"][1])	
		window['-PROGBAR-'].update_bar(0)				
		ax0.scatter(xcent,ycent,c=centroid_color,alpha=centroid_alpha,s=centroid_size,linewidth=0,zorder=100)
		
		# Clustered trajectories
		if plot_trajectories:
			print ("Plotting clustered trajectories...")
		for num,traj in enumerate(clustindices):		
			if num%10 == 0:
				try:
					bar = 100*num/(len(seldict)-1)
					window['-PROGBAR-'].update_bar(bar)
				except:
					pass
			centx=seldict[traj]["centroid"][0]
			centy=seldict[traj]["centroid"][1]
			centt=seldict[traj]["centroid"][2]
			if centx > xlims[0] and centx < xlims[1] and centy > ylims[0] and centy < ylims[1]:
				# Plot clustered trajectories
				if plot_trajectories:
					x,y,t=zip(*seldict[traj]["points"])
					tr = []
					if clust_color:
						col = cmap(centt/float(acq_time))
						tr = matplotlib.lines.Line2D(x,y,c=col,alpha=line_alpha,linewidth=line_width)
					if var_color:
						tr = matplotlib.lines.Line2D(x,y,c=var_cols[seldict[traj]["kmeans_group"]],alpha=line_alpha,linewidth=line_width)
					if msd_color:	
						if seldict[traj]["msds"][0] < av_msd:
							tr = matplotlib.lines.Line2D(x,y,c=msd_color1,alpha=line_alpha,linewidth=line_width)
						else:
							tr = matplotlib.lines.Line2D(x,y,c=msd_color2,alpha=line_alpha,linewidth=line_width)
					if not var_color and not msd_color and not clust_color:		
						tr = matplotlib.lines.Line2D(x,y,c=line_color,alpha=line_alpha,linewidth=line_width)
					ax0.add_artist(tr) 
				# Plot centroids
				if plot_centroids:
					xcent.append(seldict[traj]["centroid"][0])
					ycent.append(seldict[traj]["centroid"][1])	
		window['-PROGBAR-'].update_bar(0)				
		ax0.scatter(xcent,ycent,c=centroid_color,alpha=centroid_alpha,s=centroid_size,linewidth=0,zorder=100)		
		
		# Clusters
		if plot_clusters:
			print ("Highlighting clusters...")
			for cluster in clusterdict:
				try:
					bar = 100*cluster/(len(clusterdict))
					window['-PROGBAR-'].update_bar(bar)
				except:
					pass
				centx=clusterdict[cluster]["centroid"][0]
				centy=clusterdict[cluster]["centroid"][1]
				if centx > xlims[0] and centx < xlims[1] and centy > ylims[0] and centy < ylims[1]:
					indices = clusterdict[cluster]["indices"]
					cx,cy,ct = clusterdict[cluster]["centroid"]
					col = cmap(ct/float(acq_time))
					# Unfilled polygon
					bx,by = clusterdict[cluster]["area_xy"]
					cl = matplotlib.lines.Line2D(bx,by,c=col,alpha=cluster_alpha,linewidth=cluster_width,linestyle=cluster_linetype,zorder=100)
					ax0.add_artist(cl)
					# Filled polygon
					if cluster_fill:
						vertices = list(zip(*clusterdict[cluster]["area_xy"]))
						cl = plt.Polygon(vertices,facecolor=col,edgecolor=col,alpha=cluster_alpha,zorder=-ct)
						ax0.add_patch(cl) 
			window['-PROGBAR-'].update_bar(0)	
			
		# Hotspots info
		if plot_hotspots:	
			print ("Plotting hotspots of overlapping clusters...")	
			radii = []
			for cluster in clusterdict:
				radius  = clusterdict[cluster]["radius"]
				radii.append(radius)
			av_radius = np.average(radii)*hotspot_radius
			clustpoints = [clusterdict[i]["centroid"][:2] for i in clusterdict]
			overlapdict = {} # dictionary of overlapping clusters at av_radius
			labels,clusterlist = dbscan(clustpoints,av_radius,2) # does each cluster centroid any other cluster centroids within av_radius (epsilon)?
			clusterlist = [x for x in clusterlist]
			try:
				clusterlist.remove(-1)
			except:
				pass
			for cluster in clusterlist:
				overlapdict[cluster] = {}
				overlapdict[cluster]["clusters"]=[]
			for num,label in enumerate(labels):
				if label > -1:
					overlapdict[label]["clusters"].append(num+1)
			overlappers = [overlapdict[x]["clusters"] for x in clusterlist]
			if len(overlappers) > 0:
				for num,overlap in enumerate(overlappers):
					try:
						bar = 100*num/len(overlappers)
						window['-PROGBAR-'].update_bar(bar)
					except:
						pass
					clusterpoints = []
					for cluster in overlap:
						centx=clusterdict[cluster]["centroid"][0]
						centy=clusterdict[cluster]["centroid"][1]
						if centx > xlims[0] and centx < xlims[1] and centy > ylims[0] and centy < ylims[1]:	
							points = zip(*clusterdict[cluster]["area_xy"])
							[clusterpoints.append(point) for point in points]
					if len(clusterpoints) > 0:	
						ext_x,ext_y,ext_area,int_x,int_y,int_area = double_hull(clusterpoints)
						cl = matplotlib.lines.Line2D(ext_x,ext_y,c=hotspot_color,alpha=hotspot_alpha,linewidth=hotspot_width,linestyle=hotspot_linetype,zorder=15000)
						ax0.add_artist(cl) 	
				window['-PROGBAR-'].update_bar(0)			
		ax0.set_xlabel("X")
		ax0.set_ylabel("Y")
		window['-PROGBAR-'].update_bar(0)
		selverts = [y for x in all_selverts_copy for y in x]
		selx,sely=list(zip(*selverts))
		minx=min(selx)
		maxx=max(selx)
		miny=min(sely)
		maxy=max(sely)	
		if minx > xlims[0] and maxx < xlims[1] and miny > ylims[0] and maxy < ylims[1]:
			ax0.set_xlim(minx,maxx)
			ax0.set_ylim(miny,maxy)
		else:	
			ax0.set_xlim(xlims)
			ax0.set_ylim(ylims)
		plt.tight_layout()
		plt.show(block=False)

		# Colorbar
		if plot_colorbar:
			print ("Plotting colorbar...")
			xlims = ax0.get_xlim()
			ylims = ax0.get_ylim()
			x_perc = (xlims[1] - xlims[0])/100
			y_perc = (ylims[1] - ylims[0])/100
			ax0.imshow([[0,1], [0,1]], 
			extent = (xlims[0] + x_perc*2,xlims[0] + x_perc*27,ylims[0] + x_perc*2,ylims[0] + x_perc*4),
			cmap = cmap, 
			interpolation = 'bicubic',
			zorder=1000
			)		
		plt.tight_layout()
		plt.show(block=False)
		# Pickle
		buf0 = io.BytesIO()
		pickle.dump(ax0, buf0)
		buf0.seek(0)	
		if auto_metric:
			window["-TABGROUP-"].Widget.select(4)
		plotflag=True
		t2 = time.time()
		print ("Plot complete in {} sec, please wait for display...".format(round(t2-t1,3)))
		return

	# METRICS TAB
	def metrics_tab():
		global buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9,av_msd,all_msds,confinedindices,unconfinedindices,allindices
		
		# MSD for clustered and unclustered detections
		if event == "-M1-":
			print ("Plotting MSD curves...")
			t1=time.time()
			fig1 = plt.figure(1,figsize=(4,4))
			ax1 = plt.subplot(111)
			ax1.cla()
			clust_msds = [seldict[x]["msds"] for x in clustindices]
			unclust_msds = [seldict[x]["msds"] for x in unclustindices]
			clust_vals = []
			unclust_vals = []
			for i in range(minlength-1):
				clust_vals.append([])
				unclust_vals.append([])
				[clust_vals[i].append(x[i]) for x in clust_msds if x[i] == x[i]]# don't append NaNs
				[unclust_vals[i].append(x[i]) for x in unclust_msds if x[i] == x[i]]
			clust_av = [np.average(x) for x in clust_vals]	
			clust_sem = [np.std(x)/math.sqrt(len(x)) for x in clust_vals]
			unclust_av = [np.average(x) for x in unclust_vals]	
			unclust_sem = [np.std(x)/math.sqrt(len(x)) for x in unclust_vals]
			msd_times = [frame_time*x for x in range(1,minlength,1)]
			ax1.scatter(msd_times,clust_av,s=10,c="orange")
			ax1.errorbar(msd_times,clust_av,clust_sem,c="orange",label="Clustered: {}".format(len(clust_msds)),capsize=5)
			ax1.scatter(msd_times,unclust_av,s=10,c="blue")
			ax1.errorbar(msd_times,unclust_av,unclust_sem,c="blue",label="Unclustered: {}".format(len(unclust_msds)),capsize=5)
			ax1.legend()
			plt.xlabel("Time (s)")
			plt.ylabel(u"MSD (μm²)")
			plt.tight_layout()
			fig1.canvas.manager.set_window_title('MSD Curves')
			plt.show(block=False)
			print(reduce(lambda x, y: str(x) + "\t" + str(y), ["TIME (S):"] + msd_times))
			print(reduce(lambda x, y: str(x) + "\t" + str(y), ["UNCLUST MSD (um^2):"] + unclust_av))
			print(reduce(lambda x, y: str(x) + "\t" + str(y), ["UNCLUST SEM:"] + unclust_sem))
			print(reduce(lambda x, y: str(x) + "\t" + str(y), ["CLUST MSD (um^2):"] + clust_av))
			print(reduce(lambda x, y: str(x) + "\t" + str(y), ["CLUST SEM:"] + clust_sem))
			t2=time.time()
			print ("MSD plot completed in {} sec".format(round(t2-t1,3)))
			# Pickle
			buf1 = io.BytesIO()
			pickle.dump(ax1, buf1)
			buf1.seek(0)
			
		# Spatial and temporal cluster probability stuff
		if event == "-M2-":	
			print ("Determining spatio temporal cluster probabilities...")
			t1=time.time()
			radii = []
			for cluster in clusterdict:
				radius  = clusterdict[cluster]["radius"]
				radii.append(radius)
			av_radius = np.average(radii)
			clustpoints = [clusterdict[i]["centroid"][:2] for i in clusterdict]
			total = len(clustpoints)
			distances = np.arange(0.001,av_radius,0.001)
			logdistances = [x*1000 for x in distances] 
			hotspots = [] # number of hotspots at each distance
			hotspot_probs = [] # probabilities of hotspots at each distance 
			cluster_numbers = [] # average number of clusters in each hotspot at each distance
			intercluster_times = [] # average time between clusters in hotspots at each distance
			for dist in distances:
				overlapdict = {} # dictionary of overlapping clusters at this distance
				c_nums = [] # number of clusters per hotspot at this distance
				timediffs = [] # hotspot intercluster times at this distance
				labels,clusterlist = dbscan(clustpoints,dist,2) # does each cluster centroid form a clustered cluster within dist (epsilon)?
				unclustered = [x for x in labels if x == -1] # select DBSCAN unclustered centroids	
				p = 1 - float(len(unclustered))/total # probability of adjacent cluster centroids within dist 
				#labels = [x for x in labels if x > -1] # only worry about clustered centroids
				clusterlist = [x for x in clusterlist]
				try:
					clusterlist.remove(-1)
				except:
					pass
				for cluster in clusterlist:
					overlapdict[cluster] = {}
					overlapdict[cluster]["clusters"]=[]
				for num,label in enumerate(labels):
					if label > -1:
						overlapdict[label]["clusters"].append(num)
				if len(overlapdict) > 0:
					for overlap in overlapdict:
						clusters = overlapdict[overlap]["clusters"]
						c_nums.append(len(clusters))
						times = [clusterdict[i+1]["centroid"][2] for i in clusters]
						times.sort()
						diffs = np.diff(times)
						[timediffs.append(t) for t in diffs]
				else:
					c_nums.append(0)
					timediffs.append(0)
				hotspots.append(len(clusterlist))
				hotspot_probs.append(p)	
				intercluster_times.append(np.average(timediffs))
				cluster_numbers.append(np.average(c_nums))
			fig2 =plt.figure(2,figsize=(8,8))	
			ax2 = plt.subplot(221)
			ax2.cla()
			ax2.plot(logdistances,hotspot_probs,c="blue")
			ax2.set_xlabel(u"Distance (nm)")
			ax2.set_ylabel("Probability")
			ax2.set_title("Overlap probability")
			ax2.axvline(av_radius*1000,linewidth=1.5,linestyle="dotted",c="k")
			ax2.axvline(1,linewidth=1.5,linestyle="dotted",c="k")
			ax2.set_xlim(0,)
			ax2.set_ylim(0,)
			
			# Monte Carlo simulation
			print ("Monte Carlo simulation...")
			clusternum = len(clusterdict)
			xrange = math.sqrt(sum(all_selareas))
			yrange = math.sqrt(sum(all_selareas))
			allprobs = []
			for i in range(50):
				bar = 100*i/49
				window['-PROGBAR-'].update_bar(bar)
				clustpoints = []
				prob = []
				for j in range(clusternum):
					x = random.random()*xrange
					y = random.random()*yrange
					clustpoints.append([x,y])
				for dist in distances:
					p = overlap_prob(clustpoints,dist)
					prob.append(p)	
				allprobs.append(prob)
			window['-PROGBAR-'].update_bar(0)	
			allprobs = list(zip(*allprobs))
			probs = np.array([np.average(x) for x in allprobs])	
			errs = np.array([np.std(x)/math.sqrt(len(x)) for x in allprobs])
			ax2.plot(logdistances,probs,c="r",linestyle="dotted",alpha=1, label = "Sim 1")
			ax2.fill_between(logdistances, probs-errs, probs+errs,facecolor="r",alpha=0.2,edgecolor="r")
			ax3 = plt.subplot(222,sharex=ax2)
			ax3.cla()
			ax3.plot(logdistances,cluster_numbers,c="orange")
			ax3.set_xlabel(u"Distance (nm)")
			ax3.set_ylabel("Clusters per hotspot")
			ax3.set_title("Hotspot membership")
			ax3.axvline(av_radius*1000,linewidth=1.5,linestyle="dotted",c="k")
			ax3.axvline(1,linewidth=1.5,linestyle="dotted",c="k")
			ax3.set_ylim(0,)
			ax4 = plt.subplot(223,sharex=ax2)
			ax4.cla()
			ax4.plot(logdistances,intercluster_times,c="green")
			ax4.set_xlabel(u"Distance (nm)")
			ax4.set_ylabel("Time (s)")
			ax4.set_title("Hotspot intercluster time")
			ax4.axvline(av_radius*1000,linewidth=1.5,linestyle="dotted",c="k")
			ax4.axvline(1,linewidth=1.5,linestyle="dotted",c="k")
			ax4.set_ylim(0,)
			ax5 = plt.subplot(224)
			ax5.cla()
			cluster_per_time = []
			clustertimes = [[min(clusterdict[i]["centroid_times"]),max(clusterdict[i]["centroid_times"])] for i in clusterdict]
			for timepoint in range(acq_time):
				count=0
				for ctime in clustertimes:
					if ctime[0]< timepoint and ctime[1] > timepoint: 
						count+=1
				cluster_per_time.append(count/sum(all_selareas))
				
			ax5.plot(cluster_per_time,c="red")	
			ax5.set_xlabel("Acq. time (s)")
			ax5.set_ylabel(u"Clusters/μm²")
			ax5.set_title("Cluster number")
			
			plt.tight_layout()
			fig2.canvas.manager.set_window_title('Overlap metrics')
			plt.show(block=False)
			t2=time.time()
			print ("Cluster probability plot completed in {} sec".format(round(t2-t1,3)))
			# Pickle
			buf2 = io.BytesIO()
			pickle.dump(fig2, buf2)
			buf2.seek(0)
			
		# Dimensionality reduction
		if event == "-M3-":	
			print ("Dimensionality reduction of cluster metrics...")
			t1 = time.time()
			metrics_array = []
			col_array = []
			for num in clusterdict:
				traj_num=clusterdict[num]["traj_num"] # number of trajectories in this cluster
				lifetime = clusterdict[num]["lifetime"]  # lifetime of this cluster (sec)
				av_msd = clusterdict[num]["av_msd"] # Average trajectory MSD in this cluster
				area = clusterdict[num]["area"] # Use internal hull area as cluster area (um2)
				radius = clusterdict[num]["radius"] # cluster radius um
				density = clusterdict[num]["density"] # trajectories/um2
				rate = clusterdict[num]["rate"] # accumulation rate (trajectories/sec)
				cltime = float(clusterdict[num]["centroid"][2]/acq_time) 
				clustarray = [traj_num,lifetime,av_msd,area,radius,density,rate]	
				metrics_array.append(clustarray)
				col_array.append(cmap(cltime))				
			
			# Normalise each column	
			metrics_array = list(zip(*metrics_array))
			metrics_array = [normalize(x) for x in metrics_array]
			metrics_array = list(zip(*metrics_array))			
			mapdata = decomposition.TruncatedSVD(n_components=3).fit_transform(metrics_array)
			try:
				fig3 =plt.figure(3,figsize=(4,4))			
				ax6 = plt.subplot(111,projection='3d')
				ax6.cla()
				ax6.scatter(mapdata[:, 0], mapdata[:, 1],mapdata[:, 2],c=col_array)
				ax6.set_xticks([])
				ax6.set_yticks([])
				ax6.set_zticks([])
				ax6.set_xlabel('Dimension 1')
				ax6.set_ylabel('Dimension 2')
				ax6.set_zlabel('Dimension 3')
				plt.tight_layout()
				fig3.canvas.manager.set_window_title('PCA - all metrics')
				plt.show(block=False)	
				t2=time.time()
				# Pickle
				buf3 = io.BytesIO()
				pickle.dump(ax6, buf3)
				buf3.seek(0)
				print ("Plot completed in {} sec".format(round(t2-t1,3)))
			except:
				print("Alert: PCA plot could not be completed - not enough clusters")
				sg.Popup("Alert","Not enough clusters to generate PCA plot")
				plt.close()
				
		# 3D plot
		if event == "-M4-":	
			print ("3D [x,y,t] plot of trajectories...")
			t1 = time.time()
			fig4 =plt.figure(4,figsize=(8,8))
			ax7 = plt.subplot(111,projection='3d')
			ax7.cla()
			xlims = ax0.get_xlim()
			ylims = ax0.get_ylim()
			
			if msd_color:	
				av_msd = np.average(all_msds)
				
			if var_color:
				confinedindices,unconfinedindices,var_cols = var_confine()
				
			xcent = []
			ycent = []
			tcent = []
			
			# Plot unclustered trajectories
			if plot_trajectories:
				print ("Plotting unclustered trajectories...")
			for num,traj in enumerate(unclustindices):
				if num%10 == 0:
					try:
						bar = 100*num/(len(unclustindices)-1)
						window['-PROGBAR-'].update_bar(bar)
					except:
						pass
				centx=seldict[traj]["centroid"][0]
				centy=seldict[traj]["centroid"][1]
				centt=seldict[traj]["centroid"][2]
				if centx > xlims[0] and centx < xlims[1] and centy > ylims[0] and centy < ylims[1] and  centt > tmin and centt < tmax:
					# Plot unclustered trajectories
					if plot_trajectories:
						x,t,y=zip(*seldict[traj]["points"])
						tr = []
						if var_color:
							tr = art3d.Line3D(x,y,t,c=var_cols[seldict[traj]["kmeans_group"]],alpha=line_alpha,linewidth=line_width,zorder=acq_time - np.average(y))
						if msd_color:
							if seldict[traj]["msds"][0] < av_msd:
								tr = art3d.Line3D(x,y,t,c=msd_color1,alpha=line_alpha,linewidth=line_width,zorder=acq_time - np.average(y))
							else:
								tr = art3d.Line3D(x,y,t,c=msd_color2,alpha=line_alpha,linewidth=line_width,zorder=acq_time - np.average(y))
						if not var_color and not msd_color:
							if axes_3d:
								tr = art3d.Line3D(x,y,t,c="k",alpha=line_alpha,linewidth=line_width,zorder=acq_time - np.average(y))
							else:	
								tr = art3d.Line3D(x,y,t,c=line_color,alpha=line_alpha,linewidth=line_width,zorder=acq_time - np.average(y))	
						ax7.add_artist(tr) 
					# Plot centroids
					if plot_centroids:
						xcent.append(centx)
						ycent.append(centy)	
						tcent.append(centt)	
						
			window['-PROGBAR-'].update_bar(0)			
			
			# Plot clustered trajectories
			if plot_trajectories:
				print ("Plotting clustered trajectories...")
			for num,traj in enumerate(clustindices):
				if num%10 == 0:
					try:
						bar = 100*num/(len(clustindices)-1)
						window['-PROGBAR-'].update_bar(bar)
					except:
						pass
				centx=seldict[traj]["centroid"][0]
				centy=seldict[traj]["centroid"][1]
				centt = seldict[traj]["centroid"][2]
				if centx > xlims[0] and centx < xlims[1] and centy > ylims[0] and centy < ylims[1] and  centt > tmin and centt < tmax:
					# Plot clustered trajectories
					if plot_trajectories:
						x,t,y=zip(*seldict[traj]["points"])
						tr = []
						if clust_color:
							col = cmap(centt/float(acq_time))
							tr = art3d.Line3D(x,y,t,c=col,alpha=line_alpha,linewidth=line_width,zorder=acq_time - np.average(y))
						if var_color:
							tr = art3d.Line3D(x,y,t,c=var_cols[seldict[traj]["kmeans_group"]],alpha=line_alpha,linewidth=line_width,zorder=acq_time - np.average(y))
						if msd_color:
							if seldict[traj]["msds"][0] < av_msd:
								tr = art3d.Line3D(x,y,t,c=msd_color1,alpha=line_alpha,linewidth=line_width,zorder=acq_time - np.average(y))
							else:
								tr = art3d.Line3D(x,y,t,c=msd_color2,alpha=line_alpha,linewidth=line_width,zorder=acq_time - np.average(y))						
						if not var_color and not msd_color and not clust_color:
							if axes_3d:
								tr = art3d.Line3D(x,y,t,c="k",alpha=line_alpha,linewidth=line_width,zorder=acq_time - np.average(y))
							else:	
								tr = art3d.Line3D(x,y,t,c=line_color,alpha=line_alpha,linewidth=line_width,zorder=acq_time - np.average(y))	
						ax7.add_artist(tr)
					# Plot centroids
					if plot_centroids:
						xcent.append(centx)
						ycent.append(centy)	
						tcent.append(centt)			
			ax7.scatter(xcent,tcent,ycent,c=centroid_color,alpha=centroid_alpha,s=centroid_size,linewidth=0)						
			window['-PROGBAR-'].update_bar(0)			
			
			# Plot clusters	
			if plot_clusters:	
				print ("Plotting clusters...")			
				for cluster in clusterdict:
					try: 
						bar = 100*cluster/(len(clusterdict))
						window['-PROGBAR-'].update_bar(bar)
					except:
						pass
					centx=clusterdict[cluster]["centroid"][0]
					centy=clusterdict[cluster]["centroid"][1]
					centt=clusterdict[cluster]["centroid"][2] 
					if centx > xlims[0] and centx < xlims[1] and centy > ylims[0] and centy < ylims[1] and centt > tmin and centt < tmax: 
						cx,cy,ct = clusterdict[cluster]["centroid"]
						col = cmap(ct/float(acq_time))
						bx,by = clusterdict[cluster]["area_xy"]
						bt = [ct for x in bx]
						cl = art3d.Line3D(bx,bt,by,c=col,alpha=cluster_alpha,linewidth=cluster_width,linestyle=cluster_linetype,zorder=5+acq_time - ct)
						ax7.add_artist(cl)	
				window['-PROGBAR-'].update_bar(0)
				
			# Labels etc
			if axes_3d:
				ax7.set_xlabel("X")
				ax7.set_ylabel("T")
				ax7.set_zlabel("Y")
			else:		
				ax7.set_facecolor(canvas_color)				
				ax7.set_xticks([])
				ax7.set_yticks([])
				ax7.set_zticks([])
				ax7.set_axis_off()			
			ax7.set_xlim(xlims)
			ax7.set_ylim(tmin,tmax)
			ax7.set_zlim(ylims)
			# The next 2 lines help keep the correct x:y aspect ratio in the 3D plot
			try:
				xy_ratio = (xlims[1] - xlims[0])/(ylims[1] - ylims[0])
				ax7.set_box_aspect(aspect=(xy_ratio,1,1))
			except:
				pass
			fig4.canvas.manager.set_window_title('3D plot')
			plt.tight_layout()	
			plt.show(block=False)
			window['-PROGBAR-'].update_bar(0)
			t2=time.time()
			# Pickle
			buf4 = io.BytesIO()
			pickle.dump(ax7, buf4)
			buf4.seek(0)
			print ("Plot completed in {} sec".format(round(t2-t1,3)))						

		# KDE
		if event == "-M5-":	
			print ("2D Kernel density estimation of all detections...")
			t1 = time.time()
			fig5 =plt.figure(5,figsize=(8,8))
			ax8 = plt.subplot(111)				
			ax8.cla()
			ax8.set_facecolor("k")	
			xlims = ax0.get_xlim()
			ylims = ax0.get_ylim()
			allpoints = [point[:2]  for i in seldict for point in seldict[i]["points"]] # All detection points
			allpoints = [i for i in allpoints if i[0] > xlims[0] and i[0] < xlims[1] and i[1] > ylims[0] and i[1] < ylims[1]] # Detection points within zoom
			kde_method = 0.10 # density estimation method. Larger for smaller amounts of data (0.05 - 0.15 should be ok)
			kde_res = 0.6 # resolution of density map (0.5-0.9). Larger = higher resolution
			x = np.array(list(zip(*allpoints))[0])
			y = np.array(list(zip(*allpoints))[1])
			k = gaussian_kde(np.vstack([x, y]),bw_method=kde_method)
			xi, yi = np.mgrid[x.min():x.max():x.size**kde_res*1j,y.min():y.max():y.size**kde_res*1j]
			zi = k(np.vstack([xi.flatten(), yi.flatten()]))
			ax8.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=1,cmap="inferno",zorder=-100)
			ax8.set_xlabel("X")
			ax8.set_ylabel("Y")
			x_perc = (xlims[1] - xlims[0])/100
			y_perc = (ylims[1] - ylims[0])/100
			ax8.imshow([[0,1], [0,1]], 
			extent = (xlims[0] + x_perc*2,xlims[0] + x_perc*27,ylims[0] + x_perc*2,ylims[0] + x_perc*4),
			cmap = "inferno", 
			interpolation = 'bicubic',
			zorder=1000)
			ax8.set_xlim(xlims)
			ax8.set_ylim(ylims)
			fig5.canvas.manager.set_window_title('2D KDE')
			plt.tight_layout()	
			plt.show(block=False)
			t2=time.time()
			# Pickle
			buf5 = io.BytesIO()
			pickle.dump(ax8, buf5)
			buf5.seek(0)
			print ("Plot completed in {} sec".format(round(t2-t1,3)))	
			
		# Diffusion coefficient	
		if event == "-M6-":	
			print ("Instantaneous diffusion coefficient of trajectories...")
			t1 = time.time()
			fig6 =plt.figure(6,figsize=(8,8))
			ax9 = plt.subplot(111)
			ax9.cla()
			ax9.set_facecolor("k")	
			xlims = ax0.get_xlim()
			ylims = ax0.get_ylim()
			maxdiffcoeff = math.log(max(all_diffcoeffs)/(3*frame_time),10)
			mindiffcoeff = math.log(min(all_diffcoeffs)/(3*frame_time),10)
			print ("Minimum Inst Diff Coeff (log10 um^2/s):",mindiffcoeff)
			print ("Maximum Inst Diff Coeff (log10 um^2/s):",maxdiffcoeff)
			dcrange = abs(maxdiffcoeff-mindiffcoeff)
			cmap3 = matplotlib.cm.get_cmap('viridis_r')
			for num,traj in enumerate(allindices): 
				if num%10 == 0:
					try:  
						bar = 100*num/(len(allindices)-1)
						window['-PROGBAR-'].update_bar(bar)
					except:
						pass
				centx=seldict[traj]["centroid"][0]
				centy=seldict[traj]["centroid"][1]
				if centx > xlims[0] and centx < xlims[1] and centy > ylims[0] and centy < ylims[1]:
					x,y,t=zip(*seldict[traj]["points"])
					diffcoeff  = abs(seldict[traj]["diffcoeff"])
					dcnorm = (math.log(diffcoeff,10)-mindiffcoeff)/dcrange # normalise color 0-1  
					col = cmap3(dcnorm)
					tr = matplotlib.lines.Line2D(x,y,c=col,alpha=0.75,linewidth=line_width,zorder=1-dcnorm)
					ax9.add_artist(tr) 
			window['-PROGBAR-'].update_bar(0)			
			ax9.set_xlabel("X")
			ax9.set_ylabel("Y")
			ax9.set_xlim(xlims)
			ax9.set_ylim(ylims)
			x_perc = (xlims[1] - xlims[0])/100
			y_perc = (ylims[1] - ylims[0])/100
			ax9.imshow([[0,1], [0,1]], 
			extent = (xlims[0] + x_perc*2,xlims[0] + x_perc*27,ylims[0] + x_perc*2,ylims[0] + x_perc*4),
			cmap = "viridis_r", 
			interpolation = 'bicubic',
			zorder=1000)	
			fig6.canvas.manager.set_window_title('Diffusion coefficient')
			plt.tight_layout()	
			plt.show(block=False)	

			# DIFF COEFF TIME PLOT		
			fig7 =plt.figure(7,figsize=(6,3))
			ax10 = plt.subplot(211)
			ax11 = plt.subplot(212,sharex=ax10,sharey=ax10)	
			ax10.cla()
			ax10.set_facecolor("k")	
			ax11.cla()
			ax11.set_facecolor("k")		
			clustcols = []
			diffcols = []
			times = []
			zorders = []
			for num,traj in enumerate(clustindices): 
				if num%10 == 0:
					try:
						bar = 100*num/(len(clustindices)-1)
						window['-PROGBAR-'].update_bar(bar)
					except:
						pass
				centx=seldict[traj]["centroid"][0]
				centy=seldict[traj]["centroid"][1]
				centt=seldict[traj]["centroid"][2]
				if centx > xlims[0] and centx < xlims[1] and centy > ylims[0] and centy < ylims[1]:
					diffcoeff  = abs(seldict[traj]["diffcoeff"])
					clustcol = cmap(centt/float(acq_time))
					dcnorm = (math.log(diffcoeff,10)-mindiffcoeff)/dcrange # normalise color 0-1 
					diffcol = cmap3(dcnorm)
					times.append(centt)
					clustcols.append(clustcol)	
					diffcols.append(diffcol)
					zorders.append(1000)
			window['-PROGBAR-'].update_bar(0)			
			for num,traj in enumerate(unclustindices): 
				if num%10 == 0:
					try:
						bar = 100*num/(len(unclustindices)-1)
						window['-PROGBAR-'].update_bar(bar)
					except:
						pass
				centx=seldict[traj]["centroid"][0]
				centy=seldict[traj]["centroid"][1]
				centt=seldict[traj]["centroid"][2]
				if centx > xlims[0] and centx < xlims[1] and centy > ylims[0] and centy < ylims[1]:
					diffcoeff  = abs(seldict[traj]["diffcoeff"])
					dcnorm = (math.log(diffcoeff,10)-mindiffcoeff)/dcrange # normalise color 0-1 
					clustcol = "dimgray"
					diffcol = cmap3(dcnorm)
					times.append(centt)
					clustcols.append(clustcol)	
					diffcols.append(diffcol)
					zorders.append(100)
			window['-PROGBAR-'].update_bar(0)		
			for i,t in enumerate(times):
				ax10.axvline(t,linewidth=1.5,c=clustcols[i],alpha = 0.75,zorder = zorders[i])
				ax11.axvline(t,linewidth=1.5,c=diffcols[i],alpha = 0.75)

			ax10.set_ylabel("Cluster")
			ax11.set_ylabel("D Coeff")				
			ax11.set_xlabel("time (s)")	
			ax10.tick_params(axis = "both",left = False, labelleft = False,bottom=False,labelbottom=False)
			ax11.tick_params(axis = "both",left = False, labelleft = False)			
			fig7.canvas.manager.set_window_title('Diffusion coefficient time plot')
			plt.tight_layout()	
			plt.show(block=False)	
			t2=time.time()
			# Pickle
			buf6 = io.BytesIO()
			pickle.dump(ax9, buf6)
			buf6.seek(0)
			buf7 = io.BytesIO()
			pickle.dump(fig7, buf7)
			buf7.seek(0)			
			print ("Plots completed in {} sec".format(round(t2-t1,3)))	
			
		# Density
		if event == "-M7-":	
			print ("Detection density over time...")
			t1 = time.time()		
			alltimes = []
			for num,traj in enumerate(allindices): 
				if num%10 == 0:
					try:
						bar = 100*num/(len(allindices)-1)
						window['-PROGBAR-'].update_bar(bar)
					except:
						pass
				points=seldict[traj]["points"]
				[alltimes.append(x[2]) for x in points]
			window['-PROGBAR-'].update_bar(0)
			fig8 =plt.figure(8,figsize=(4,4))
			ax12 = plt.subplot(111)
			bin_edges = np.histogram_bin_edges(alltimes,bins=int(acq_time)) # Sort into 1 second bins
			dist,bins =np.histogram(alltimes,bin_edges)
			dist = [float(x)/sum(all_selareas) for x in dist]
			bin_centers = 0.5*(bins[1:]+bins[:-1])
			ax12.plot(bin_centers,dist,c="royalblue")
			plt.ylabel(u"Trajectories/μm² (1 sec bins)")
			plt.xlabel("Acquisition time (s)")
			fig8.canvas.manager.set_window_title('Density')
			plt.tight_layout()	
			plt.show(block=False)
			t2=time.time()
			# Pickle
			buf8 = io.BytesIO()
			pickle.dump(fig8, buf8)
			buf8.seek(0)			
			print ("Plot completed in {} sec".format(round(t2-t1,3)))	

		# Vector autoregression
		if event == "-M8-":	
			print ("Vector autoregression metrics...")
			t1=time.time()
			confinedindices,unconfinedindices,var_cols = var_confine()
			confinedintersect = set(clustindices).intersection(set(confinedindices))
			unconfinedintersect = set(unclustindices).intersection(set(unconfinedindices))			

			print ("Clustered", len(clustindices))
			print ("Confined", len(confinedindices))
			print ("Confined intersect", len(confinedintersect))
			print ("Unclustered", len(unclustindices))
			print ("Unconfined", len(unconfinedindices))
			print ("Unconfined intersect", len(unconfinedintersect))
			
			fig9 =plt.figure(9,figsize=(6,6))
			ax13 = plt.subplot(211)
			venn2(subsets=(len(clustindices)-len(confinedintersect), len(confinedindices)-len(confinedintersect),len(confinedintersect)),set_labels=('', ''),set_colors=("green","orange"),alpha=0.9)
			
			ax14 = plt.subplot(212)
			venn2(subsets=(len(unclustindices)-len(unconfinedintersect), len(unconfinedindices)-len(unconfinedintersect),len(unconfinedintersect)),set_labels=('', ''),set_colors=("red","blue"),alpha=0.9)
			fig9.canvas.manager.set_window_title('Vector autoregression')
			plt.show(block = False)
			# Pickle
			buf9 = io.BytesIO()
			pickle.dump(fig9, buf9)
			buf9.seek(0)			
			t2=time.time()
			print ("Plots completed in {} sec".format(round(t2-t1,3)))	
			
		# Save metrics	
		if event == "-SAVEANALYSES-":	
			stamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now()) # datestamp
			outpath = os.path.dirname(infilename)
			outdir = outpath + "/" + infilename.split("/")[-1].replace(".trxyt","_NASTIC_{}".format(stamp))
			os.mkdir(outdir)
			os.chdir(outdir)
			outfilename = "{}/metrics.tsv".format(outdir)
			print ("Saving metrics, ROIs and all open plots to {}...".format(outdir))
			# Metrics
			with open(outfilename,"w") as outfile:
				outfile.write("NASTIC: NANOSCALE SPATIO TEMPORAL INDEXING CLUSTERING - Tristan Wallis t.wallis@uq.edu.au\n") 
				outfile.write("TRAJECTORY FILE:\t{}\n".format(infilename))	
				outfile.write("ANALYSED:\t{}\n".format(stamp))
				outfile.write("TRAJECTORY LENGTH CUTOFFS (steps):\t{} - {}\n".format(minlength,maxlength))
				outfile.write("SELECTION DENSITY:\t{}\n".format(selection_density))
				outfile.write("ACQUISITION TIME (s):\t{}\n".format(acq_time))
				outfile.write("FRAME TIME (s):\t{}\n".format(frame_time))
				outfile.write("TIME THRESHOLD (s):\t{}\n".format(time_threshold))
				outfile.write("RADIUS FACTOR:\t{}\n".format(radius_factor))
				outfile.write("CLUSTER THRESHOLD:\t{}\n".format(cluster_threshold)) 
				outfile.write("CLUSTER MAX RADIUS (um):\t{}\n".format(radius_thresh))
				if msd_filter:
					outfile.write("MSD FILTER THRESHOLD (um^2):\t{}\n".format(msd_filter_threshold))
				else:
					outfile.write("MSD FILTER THRESHOLD (um^2):\tNone\n")
				outfile.write("SELECTION AREA (um^2):\t{}\n".format(sum(all_selareas)))
				outfile.write("SELECTED TRAJECTORIES:\t{}\n".format(len(allindices)))
				outfile.write("CLUSTERED TRAJECTORIES:\t{}\n".format(len(clustindices)))
				outfile.write("UNCLUSTERED TRAJECTORIES:\t{}\n".format(len(unclustindices)))
				if len(confinedindices) > 0 or len(unconfinedindices) > 0:
					outfile.write("VAR CONFINED TRAJECTORIES:\t{}\n".format(len(confinedindices)))
					outfile.write("VAR UNCONFINED TRAJECTORIES:\t{}\n".format(len(unconfinedindices)))				
				outfile.write("TOTAL CLUSTERS:\t{}\n".format(len(clusterdict)))
				
				# INSTANTANEOUS DIFFUSION COEFFICIENT (1ST 4 POINTS)
				clustdiffcoeffs = []
				for i in clustindices:
					clustdiffcoeffs.append(seldict[i]["diffcoeff"])
				outfile.write("CLUSTERED TRAJECTORIES AVERAGE INSTANTANEOUS DIFFUSION COEFFICIENT (um^2/s):\t{}\n".format(np.average(clustdiffcoeffs)))
				unclustdiffcoeffs = []
				for i in unclustindices:
					unclustdiffcoeffs.append(seldict[i]["diffcoeff"])
				outfile.write("UNCLUSTERED TRAJECTORIES AVERAGE INSTANTANEOUS DIFFUSION COEFFICIENT (um^2/s):\t{}\n".format(np.average(unclustdiffcoeffs)))	
				
				# HOTSPOT INFO
				radii = []
				for cluster in clusterdict:
					radius  = clusterdict[cluster]["radius"]
					radii.append(radius)
				av_radius = np.average(radii)/2
				clustpoints = [clusterdict[i]["centroid"][:2] for i in clusterdict]
				total = len(clustpoints)
				overlapdict = {} # dictionary of overlapping clusters at av_radius
				c_nums = [] # number of clusters per hotspot at av_radius
				timediffs = [] # hotspot intercluster times at av_radius
				labels,clusterlist = dbscan(clustpoints,av_radius,2) # does each cluster centroid any other cluster centroids within av_radius (epsilon)?
				unclustered = [x for x in labels if x == -1] # select DBSCAN unclustered centroids	
				p = 1 - float(len(unclustered))/total # probability of adjacent cluster centroids within dist 
				clusterlist = [x for x in clusterlist]
				try:
					clusterlist.remove(-1)
				except:
					pass
				for cluster in clusterlist:
					overlapdict[cluster] = {}
					overlapdict[cluster]["clusters"]=[]
				for num,label in enumerate(labels):
					if label > -1:
						overlapdict[label]["clusters"].append(num)
				if len(overlapdict) > 0:
					for overlap in overlapdict:
						clusters = overlapdict[overlap]["clusters"]
						c_nums.append(len(clusters))
						times = [clusterdict[i+1]["centroid"][2] for i in clusters]
						times.sort()
						diffs = np.diff(times)
						[timediffs.append(t) for t in diffs]
				else:
					c_nums.append(0)
					timediffs.append(0)
				hotspots = len(clusterlist)
				hotspot_prob = p				
				intercluster_time = np.average(timediffs)
				hotspot_total = sum(c_nums)
				hotspot_nums = np.average(c_nums)		
				outfile.write("HOTSPOTS (CLUSTER SPATIAL OVERLAP AT 1/2 AVERAGE RADIUS):\t{}\n".format(hotspots))
				outfile.write("TOTAL CLUSTERS IN HOTSPOTS:\t{}\n".format(hotspot_total))
				outfile.write("AVERAGE CLUSTERS PER HOTSPOT:\t{}\n".format(hotspot_nums))
				outfile.write("PERCENTAGE OF CLUSTERS IN HOTSPOTS:\t{}\n".format(round(100*hotspot_prob,3)))

				# MSD CURVES
				outfile.write("\nMSD CURVE DATA:\n")
				clust_msds = [seldict[x]["msds"] for x in clustindices]
				unclust_msds = [seldict[x]["msds"] for x in unclustindices]
				all_msds = [seldict[x]["msds"] for x in allindices]
				clust_vals = []
				unclust_vals = []
				all_vals = []
				for i in range(minlength-1):
					clust_vals.append([])
					unclust_vals.append([])
					all_vals.append([])
					[clust_vals[i].append(x[i]) for x in clust_msds if x[i] == x[i]]# don't append NaNs
					[unclust_vals[i].append(x[i]) for x in unclust_msds if x[i] == x[i]]
					[all_vals[i].append(x[i]) for x in all_msds if x[i] == x[i]]
				clust_av = [np.average(x) for x in clust_vals]	
				clust_sem = [np.std(x)/math.sqrt(len(x)) for x in clust_vals]
				unclust_av = [np.average(x) for x in unclust_vals]	
				unclust_sem = [np.std(x)/math.sqrt(len(x)) for x in unclust_vals]
				all_av = [np.average(x) for x in all_vals]	
				all_sem = [np.std(x)/math.sqrt(len(x)) for x in all_vals]
				msd_times = [frame_time*x for x in range(1,minlength,1)]
				outfile.write(reduce(lambda x, y: str(x) + "\t" + str(y), ["TIME (S):"] + msd_times) + "\n") 
				outfile.write(reduce(lambda x, y: str(x) + "\t" + str(y), ["UNCLUST MSD (um^2):"] + unclust_av) + "\n")
				outfile.write(reduce(lambda x, y: str(x) + "\t" + str(y), ["UNCLUST SEM:"] + unclust_sem) + "\n")
				outfile.write(reduce(lambda x, y: str(x) + "\t" + str(y), ["CLUST MSD (um^2):"] + clust_av) + "\n")
				outfile.write(reduce(lambda x, y: str(x) + "\t" + str(y), ["CLUST SEM:"] + clust_sem) + "\n")	
				outfile.write(reduce(lambda x, y: str(x) + "\t" + str(y), ["ALL MSD (um^2):"] + all_av) + "\n")
				outfile.write(reduce(lambda x, y: str(x) + "\t" + str(y), ["ALL SEM:"] + all_sem) + "\n")	

				# INDIVIDUAL CLUSTER METRICS
				outfile.write("\nINDIVIDUAL CLUSTER METRICS:\n")
				outfile.write("CLUSTER\tMEMBERSHIP\tLIFETIME (s)\tAVG MSD (um^2)\tAREA (um^2)\tRADIUS (um)\tDENSITY (traj/um^2)\tRATE (traj/sec)\tAVG TIME (s)\n")
				trajnums = []
				lifetimes = []
				times = []
				av_msds = []
				areas = []
				radii = []
				densities = []
				rates = []
				for num in clusterdict:
					traj_num=clusterdict[num]["traj_num"] # number of trajectories in this cluster
					lifetime = clusterdict[num]["lifetime"]  # lifetime of this cluster (sec)
					av_msd = clusterdict[num]["av_msd"] # Average trajectory MSD in this cluster
					area = clusterdict[num]["area"] # Use internal hull area as cluster area (um2)
					radius = clusterdict[num]["radius"] # cluster radius um
					density = clusterdict[num]["density"] # trajectories/um2
					rate = clusterdict[num]["rate"] # accumulation rate (trajectories/sec)
					clusttime = clusterdict[num]["centroid"][2] # Time centroid of this cluster 
					outarray = [num,traj_num,lifetime,av_msd,area,radius,density,rate,clusttime]
					outstring = reduce(lambda x, y: str(x) + "\t" + str(y), outarray)
					outfile.write(outstring + "\n")
					trajnums.append(traj_num)
					lifetimes.append(lifetime)
					times.append(clusttime)
					av_msds.append(av_msd)
					areas.append(area)
					radii.append(radius)
					densities.append(density)
					rates.append(rate)
					
				# AVERAGE CLUSTER METRICS	
				outarray = ["AVG",np.average(trajnums),np.average(lifetimes),np.average(av_msds),np.average(areas),np.average(radii),np.average(densities),np.average(rates),np.average(times)]
				outstring = reduce(lambda x, y: str(x) + "\t" + str(y), outarray)
				outfile.write(outstring + "\n")	
				
				# SEMS
				outarray = ["SEM",np.std(trajnums)/math.sqrt(len(trajnums)),np.std(lifetimes)/math.sqrt(len(lifetimes)),np.std(av_msds)/math.sqrt(len(av_msds)),np.std(areas)/math.sqrt(len(areas)),np.std(radii)/math.sqrt(len(radii)),np.std(densities)/math.sqrt(len(densities)),np.std(rates)/math.sqrt(len(rates)),np.std(times)/math.sqrt(len(times))]
				outstring = reduce(lambda x, y: str(x) + "\t" + str(y), outarray)
				outfile.write(outstring + "\n")					
				
			# ALL ROIs
			roi_directory = outdir + "\\saved_ROIs" 
			os.makedirs(roi_directory, exist_ok = True)
			roi_file = "{}/{}_roi_coordinates.tsv".format(roi_directory,stamp)
			with open(roi_file,"w") as outfile:
				outfile.write("ROI\tx(um)\ty(um)\n")
				for roi,selverts in enumerate(all_selverts):
					for coord in selverts:
						outfile.write("{}\t{}\t{}\n".format(roi,coord[0],coord[1]))
			
			# SEPARATE ROIs
			if len(all_selverts) >1:
				for roi,selverts in enumerate(all_selverts):
					roi_save = "{}/{}_roi_coordinates{}.tsv".format(roi_directory, stamp, roi)
					with open(roi_save,"w") as outfile:
						outfile.write("ROI\tx(um)\ty(um)\n")			
						for coord in selverts:
							outfile.write("{}\t{}\t{}\n".format(roi,coord[0],coord[1]))			
			
			# Plots	
			buf.seek(0)		
			fig100=pickle.load(buf)
			for selverts in all_selverts:			
				vx,vy = list(zip(*selverts))
				plt.plot(vx,vy,linewidth=2,c="orange",alpha=1)
			plt.savefig("{}/raw_acquisition.png".format(outdir),dpi=300)
			plt.close()
			try:
				buf0.seek(0)
				fig100=pickle.load(buf0)
				plt.savefig("{}/main_plot.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass		
			try:
				buf1.seek(0)
				fig100=pickle.load(buf1)
				plt.savefig("{}/MSD.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass
			try:
				buf2.seek(0)
				fig100=pickle.load(buf2)
				plt.savefig("{}/overlap.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass	
			try:
				buf3.seek(0)
				fig100=pickle.load(buf3)
				plt.savefig("{}/pca.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass
			try:
				buf4.seek(0)
				fig100=pickle.load(buf4)
				plt.savefig("{}/3d_trajectories.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass	
			try:
				buf5.seek(0)
				fig100=pickle.load(buf5)
				plt.savefig("{}/KDE.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass	
			try:
				buf6.seek(0)
				fig100=pickle.load(buf6)
				plt.savefig("{}/diffusion_coefficient.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass	
			try:
				buf7.seek(0)
				fig100=pickle.load(buf7)
				plt.savefig("{}/diffusion_coefficient_1d.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass
			try:
				buf8.seek(0)
				fig100=pickle.load(buf8)
				plt.savefig("{}/density_vs_time.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass
			try:
				buf9.seek(0)
				fig100=pickle.load(buf9)
				plt.savefig("{}/var_kmeans.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass
			print ("All data saved")	
		return

	# GET INITIAL VALUES FOR GUI
	cwd = os.path.dirname(os.path.abspath(__file__))
	os.chdir(cwd)
	initialdir = cwd
	if os.path.isfile("nastic_gui.defaults"):
		load_defaults()
	else:
		reset_defaults()
		save_defaults()	
	tmin = 0
	tmax = acq_time
		
	# GUI LAYOUT
	appFont = ("Any 12")
	sg.set_options(font=appFont)
	sg.theme('DARKGREY11')
	
	# File tab
	tab1_layout = [
		[sg.FileBrowse(tooltip = "Select a TRXYT file to analyse\nEach line must only contain 4 space separated values\nTRajectory# X-position Y-position Time",file_types=(("Trajectory Files", "*.trxyt"),),key="-INFILE-",initial_folder=initialdir),sg.Input("Select trajectory TRXYT file", key ="-FILENAME-",enable_events=True,size=(55,1),expand_x = True)],
		[sg.T('Minimum trajectory length:',tooltip = "Trajectories must contain at least this many steps"),sg.InputText(minlength,size="50",key="-MINLENGTH-")],
		[sg.T('Maximum trajectory length:',tooltip = "Trajectories must contain fewer steps than this"),sg.InputText(maxlength,size="50",key="-MAXLENGTH-")],
		[sg.T('Probability:',tooltip = "Probability of displaying a trajectory\n1 = all trajectories\nIMPORTANT: only affects display of trajectories,\nundisplayed trajectories can still be selected"),sg.Combo([0.01,0.05,0.1,0.25,0.5,0.75,1.0],default_value=traj_prob,key="-TRAJPROB-")],
		[sg.T('Detection opacity:',tooltip = "Transparency of detection points\n1 = fully opaque"),sg.Combo([0.01,0.05,0.1,0.25,0.5,0.75,1.0],default_value=detection_alpha,key="-DETECTIONALPHA-")],
		[sg.B('PLOT RAW DETECTIONS',size=(25,2),button_color=("white","gray"),key ="-PLOTBUTTON-",disabled=True,tooltip = "Visualise trajectory detections using the above parameters.\nYou may then select regions of interest (ROIs) using the 'ROI' tab.\nThis button will close any other plot windows.")]
	]
	
	# ROI tab
	tab2_layout = [
		[sg.FileBrowse("Load",file_types=(("ROI Files", "roi_coordinates*.tsv *.rgn"),),key="-R1-",target="-R2-",tooltip = "(Optional) Select a region of interest (ROI) file:\n - NASTIC roi_coordinates.tsv file\n - PalmTracer .rgn file", disabled=True),sg.In("Load previously defined ROIs",key ="-R2-",enable_events=True, size = (30,1)),sg.T("Pixel(um):", key = '-PIXEL_TEXT-', tooltip = "Please select a conversion factor\nfor converting pixels to um", visible = False), sg.In(pixel, key = '-PIXEL-', visible = False, size = (6,1)),sg.B("Replot ROIs", key = "-REPLOT_ROI-", visible = False)],
		[sg.B("Save",key="-R8-",tooltip = "Save ROIs together as a single roi_coordinates.tsv file", disabled=True),sg.T("Save currently defined ROIs"), sg.B("Save Separately", key = "-SEPARATE-",tooltip = "Save each ROI separately as individual roi_coordinates.tsv files", disabled = True), sg.T("Save individual ROI files")],
		[sg.B("Clear",key="-R3-",tooltip = "Clear all ROIs from plot",disabled=True),sg.T("Clear all ROIs")],	
		[sg.B("All",key="-R4-",tooltip = "Generate a rectangular ROI that encompases all detections",disabled=True),sg.T("ROI encompassing all detections")],
		[sg.B("Add",key="-R5-",tooltip = "Add ROIs that have been drawn directly on the plot:\n - freehand drawn ROIs (magnifying glass = deselected)\n - zoom-to-rectangle drawn ROIs (magnifying glass = selected)",disabled=True),sg.T("Add selected ROI")],
		[sg.B("Remove",key="-R6-",tooltip = "Remove the last ROI that was added from the plot",disabled=True),sg.T("Remove last added ROI")],
		[sg.B("Undo",key="-R7-",tooltip = "Undo the last ROI change that was made",disabled=True),sg.T("Undo last change"),sg.B("Reset",key="-RESET-", tooltip = "Reset to original detections plot with orange ROI shown",disabled = True),sg.T("Reset to original view with ROI")], 
		[sg.T('Selection density:',tooltip = "Screen out random trajectories to maintain a \nfixed density of selected trajectories (traj/um^2)\n0 = do not adjust density"),sg.InputText(selection_density,size="50",key="-SELECTIONDENSITY-"),sg.T("",key = "-DENSITY-",size=(6,1))],
		[sg.B('SELECT DATA IN ROIS',size=(25,2),button_color=("white","gray"),key ="-SELECTBUTTON-",disabled=True,tooltip = "Select trajectories whose detections lie within the orange ROIs\nYou may then select the clustering parameters using the 'Clustering' tab."),sg.Checkbox("Cluster immediately",key="-AUTOCLUSTER-",default=autocluster,tooltip="Pressing the 'SELECT DATA IN ROIS' button will\nautomatically cluster data within the orange ROIs\nusing predefined parameters in the 'Clustering' tab")]
	]

	# Clustering tab
	tab3_layout = [
		[sg.T('Acquisition time (s):',tooltip = "Time taken to acquire all frames (in seconds)"),sg.InputText(acq_time,size="50",key="-ACQTIME-")],
		[sg.T('Frame time (s):',tooltip = "Time taken to acquire each individual frame (in seconds)"),sg.InputText(frame_time,size="50",key="-FRAMETIME-")],
		[sg.T('Time threshold (s):',tooltip = "Trajectories must be within this many seconds\nof each other to be considered clustered"),sg.InputText(time_threshold,size="50",key="-TIMETHRESHOLD-")],
		[sg.T('Radius factor:',tooltip = "Adjust the radius around each centroid\n to check for overlap"),sg.InputText(radius_factor,size="50",key="-RADIUSFACTOR-")],	
		[sg.T('Cluster threshold:',tooltip = "Clusters must contain at least this\n many overlapping trajectories"),sg.InputText(cluster_threshold,size="50",key="-CLUSTERTHRESHOLD-")],
		[sg.T('Cluster size screen (um):',tooltip = "Clusters with a radius larger than this are ignored\n(in microns)"),sg.InputText(radius_thresh,size="50",key="-RADIUSTHRESH-")],	
		[sg.Checkbox('MSD screen',tooltip = "Exclude trajectories with a mean square displacement\n(MSD) greater than the average MSD of all trajectories",key = "-MSDFILTER-",default=msd_filter)],
		[sg.B('CLUSTER SELECTED DATA',size=(25,2),button_color=("white","gray"),key ="-CLUSTERBUTTON-",disabled=True, tooltip = "Perform spatiotemporal indexing clustering using the above parameters.\nUpon clustering the ROI will turn green.\nIdentified clusters may then be plotted using the parameters in the 'Display' tab.\nThis button will close any other plot windows."),sg.Checkbox("Plot immediately",key="-AUTOPLOT-",default=autoplot,tooltip ="Pressing the 'CLUSTER SELECTED DATA' button will\nautomatically plot the clustered data using the\npredefined parameters in the 'Display' tab.")],
	]

	# Trajectory subtab
	trajectory_layout = [
		[sg.T("Width",tooltip = "Width of plotted trajectory lines"),sg.Combo([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],default_value= line_width,key="-LINEWIDTH-")],
		[sg.T("Opacity",tooltip = "Opacity of plotted trajectory lines"),sg.Combo([0.01,0.05,0.1,0.25,0.5,0.75,1.0],default_value= line_alpha,key="-LINEALPHA-")],
		[sg.T("Color",tooltip = "Trajectory color applies to:\n - all trajectories if 'Cluster' is unticked\n - unclustered trajectories only if 'Cluster' is ticked"),sg.ColorChooserButton("Choose",key="-LINECOLORCHOOSE-",target="-LINECOLOR-",button_color=("gray",line_color),disabled=True),sg.Input(line_color,key ="-LINECOLOR-",enable_events=True,visible=False),sg.Checkbox('Cluster',tooltip = "Color clustered trajectories by the\ncolor of the cluster they belong to.\nUntick MSD and VAR before ticking.",key = "-CLUSTCOLOR-",default=clust_color)],
		[sg.ColorChooserButton(" < ",key="-VARCOLOR1CHOOSE-",target="-VAR1COLOR-",button_color=("gray",var_color1),disabled=True),sg.Input(var_color1,key ="-VAR1COLOR-",enable_events=True,visible=False),sg.ColorChooserButton(" > ",key="-VARCOLOR2CHOOSE-",target="-VAR2COLOR-",button_color=("gray",var_color2),disabled=True),sg.Input(var_color2,key ="-VAR2COLOR-",enable_events=True,visible=False),sg.Checkbox('VAR',tooltip = "Color trajectories by vector autoregression\n(VAR) 'confinement'.\n< = confined, > = unconfined\nUntick Cluster and MSD before ticking.",key = "-VARCOLOR-",default=var_color)],
		[sg.ColorChooserButton(" < ",key="-MSDCOLOR1CHOOSE-",target="-MSD1COLOR-",button_color=("gray",msd_color1),disabled=True),sg.Input(msd_color1,key ="-MSD1COLOR-",enable_events=True,visible=False),sg.ColorChooserButton(" > ",key="-MSDCOLOR2CHOOSE-",target="-MSD2COLOR-",button_color=("gray",msd_color2),disabled=True),sg.Input(msd_color2,key ="-MSD2COLOR-",enable_events=True,visible=False),sg.Checkbox('MSD',tooltip = "Color trajectories according to whether they are less\nthan the average mean square displacement (MSD)\n< = less than average MSD, > = greater than average MSD\nUntick Cluster and VAR before ticking.",key = "-MSDCOLOR-",default=msd_color)]
	]

	# Centroid subtab
	centroid_layout = [
		[sg.T("Size",tooltip = "Size of plotted trajectory centroids"),sg.Combo([1,2,5,10,20,50],default_value= centroid_size,key="-CENTROIDSIZE-")],
		[sg.T("Opacity",tooltip = "Opacity of plotted trajectory centroids"),sg.Combo([0.01,0.05,0.1,0.25,0.5,0.75,1.0],default_value= centroid_alpha,key="-CENTROIDALPHA-")],
		[sg.T("Color",tooltip = "Centroid color"),sg.ColorChooserButton("Choose",key="-CENTROIDCOLORCHOOSE-",target="-CENTROIDCOLOR-",button_color=("gray",centroid_color),disabled=True),sg.Input(centroid_color,key ="-CENTROIDCOLOR-",enable_events=True,visible=False)]
	]

	# Cluster subtab
	cluster_layout = [	
		[sg.T("Opacity",tooltip = "Opacity of plotted clusters"),sg.Combo([0.1,0.25,0.5,0.75,1.0],default_value= cluster_alpha,key="-CLUSTERALPHA-"),sg.Checkbox('Filled',tooltip = "Display clusters as filled polygons",key = "-CLUSTERFILL-",default=cluster_fill)],
		[sg.T("Line width",tooltip = "Width of plotted cluster lines"),sg.Combo([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],default_value= cluster_width,key="-CLUSTERWIDTH-")],
		[sg.T("Line type",tooltip = "Cluster line type"),sg.Combo(["solid","dashed","dotted"],default_value =cluster_linetype,key="-CLUSTERLINETYPE-")]
	]

	# Hotspot subtab
	hotspot_layout = [	
		[sg.T("Radius",tooltip = "Multiply this value by the average cluster radius\nto obtain the hotspot radius"),sg.Combo([0.1,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0],default_value= hotspot_radius,key="-HOTSPOTRADIUS-")],
		[sg.T("Opacity",tooltip = "Opacity of plotted hotspots"),sg.Combo([0.1,0.25,0.5,0.75,1.0],default_value= hotspot_alpha,key="-HOTSPOTALPHA-")],
		[sg.T("Line width",tooltip = "Width of plotted hotspot lines"),sg.Combo([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],default_value= hotspot_width,key="-HOTSPOTWIDTH-")],
		[sg.T("Line type",tooltip = "Hotspot line type"),sg.Combo(["solid","dashed","dotted"],default_value =hotspot_linetype,key="-HOTSPOTLINETYPE-")],
		[sg.T("Color",tooltip = "Hotspot color"),sg.ColorChooserButton("Choose",key="-HOTSPOTCOLORCHOOSE-",target="-HOTSPOTCOLOR-",button_color=("gray",hotspot_color),disabled=True),sg.Input(hotspot_color,key ="-HOTSPOTCOLOR-",enable_events=True,visible=False)]
	]		

	# Export subtab
	export_layout = [
		[sg.T("Format",tooltip = "Format of saved figure"),sg.Combo(["eps","pdf","png","ps","svg"],default_value= saveformat,key="-SAVEFORMAT-"),sg.Checkbox('Transparent background',tooltip = "Useful for making figures",key = "-SAVETRANSPARENCY-",default=False)],
		[sg.T("DPI",tooltip = "Resolution of saved figure"),sg.Combo([50,100,300,600,1200],default_value=savedpi,key="-SAVEDPI-")],
		[sg.T("Directory",tooltip = "Directory for saved figure"),sg.FolderBrowse("Choose",key="-SAVEFOLDERCHOOSE-",target="-SAVEFOLDER-"),sg.Input(key="-SAVEFOLDER-",enable_events=True,size=(43,1))]
	]
	
	# Display tab
	tab4_layout = [
		[sg.T('Canvas',tooltip = "Background color of plotted data"),sg.Input(canvas_color,key ="-CANVASCOLOR-",enable_events=True,visible=False),sg.ColorChooserButton("Choose",button_color=("gray",canvas_color),target="-CANVASCOLOR-",key="-CANVASCOLORCHOOSE-",disabled=True),sg.Checkbox('Traj.',tooltip = "Plot trajectories",key = "-TRAJECTORIES-",default=plot_trajectories),sg.Checkbox('Centr.',tooltip = "Plot trajectory centroids",key = "-CENTROIDS-",default=plot_centroids),sg.Checkbox('Clust.',tooltip = "Plot cluster boundaries",key = "-CLUSTERS-",default=plot_clusters),sg.Checkbox('Hotsp.',tooltip = "Plot cluster hotspots",key = "-HOTSPOTS-",default=plot_hotspots),sg.Checkbox('Colorbar',tooltip = "Plot colorbar for cluster times\nBlue = 0 sec --> green = full acquisition time\nHit 'Plot clustered data' button to refresh colorbar after a zoom",key = "-COLORBAR-",default=plot_colorbar)],
		[sg.TabGroup([
			[sg.Tab("Trajectory",trajectory_layout)],
			[sg.Tab("Centroid",centroid_layout)],
			[sg.Tab("Cluster",cluster_layout)],
			[sg.Tab("Hotspot",hotspot_layout)],
			[sg.Tab("Export",export_layout)]
			])
		],
		[sg.B('PLOT CLUSTERED DATA',size=(25,2),button_color=("white","gray"),key ="-DISPLAYBUTTON-",disabled=True,tooltip="Plot clustered data using the above parameters.\nHit button again after changing parameters to update the plot.\nAdditional metrics can then be plotted using the 'Metrics' tab."),sg.B('SAVE PLOT',size=(25,2),button_color=("white","gray"),key ="-SAVEBUTTON-",disabled=True,tooltip = "Save the current plot using parameters in the 'Export' subtab.\nEach time this button is pressed a new datestamped image will be saved.")],
		[sg.T("Xmin", tooltip = "X-axis minimum"),sg.InputText(plotxmin,size="3",key="-PLOTXMIN-"),sg.T("Xmax", tooltip = "X-axis maximum"),sg.InputText(plotxmax,size="3",key="-PLOTXMAX-"),sg.T("Ymin", tooltip = "Y-axis minimum"),sg.InputText(plotymin,size="3",key="-PLOTYMIN-"),sg.T("Ymax", tooltip = "Y-axis maximum"),sg.InputText(plotymax,size="3",key="-PLOTYMAX-"),sg.Checkbox("Metrics immediately",key="-AUTOMETRIC-",default=auto_metric,tooltip ="Pressing the 'PLOT CLUSTERED DATA' button will\nautomatically swap to the 'Metrics' tab after plotting.")]
	]
	
	# Metrics tab
	tab5_layout = [
		[sg.B("MSD",key="-M1-",tooltip = "Assess whether clustered trajectories have a lower mobility than unclustered trajectories\nusing average mean square displacement (MSD).",disabled=True),sg.T("Plot clustered vs unclustered MSDs")],
		[sg.B("Hotspot",key="-M2-",tooltip = "Assess the likelihood of hotspots occuring.\nVerical dotted line = average cluster radius.\nOverlap probability: red = Monte Carlo simulation.",disabled=True),sg.T("Plot cluster overlap data")],
		[sg.B("PCA",key="-M3-",tooltip = "Use pricinpal component analysis (PCA) to identify whether cluster subpopulations exist.",disabled=True),sg.T("Multidimensional analysis of cluster metrics")],
		[sg.B("3D",key="-M4-",tooltip = "Generate interactive 3D plot based on the 2D plot.",disabled=True),sg.T("X,Y,T plot of trajectories"),sg.T("Tmin:", tooltip = "Minimum time axis value"),sg.InputText(tmin,size="4",key="-TMIN-",tooltip = "Only plot trajectories whose time centroid is greater than this"),sg.T("Tmax", tooltip = "Maximum time axis value"),sg.InputText(tmax,size="4",key="-TMAX-",tooltip = "Only plot trajectories whose time centroid is less than this"),sg.Checkbox('Axes',tooltip = "Ticked = plot axes and grid on white background.\nUnticked = use canvas color as background.",key = "-AXES3D-",default=axes_3d)],
		[sg.B("KDE",key="-M5-",tooltip = "Assess whether clusters correspond with regions of higher detection density.\nBrighter colors = higher densities.\nVery slow - start with 2x2um ROI",disabled=True),sg.T("2D kernel density estimation of all detections (very slow!)")],	
		[sg.B("Diffusion coefficient",key="-M6-",tooltip = "Assess whether clustered trajectories have lower mobilities than unclustered trajectories.\nWarmer colours = lower diffusion coefficient.",disabled=True),sg.T("Instantaneous diffusion coefficient plot of trajectories.")],
		[sg.B("Density over time",key="-M7-",tooltip = "Assess whether the number of detections remains constant over time.\nDensity of trajectories (trajectories/um2) over the course of the acquisition (s)",disabled=True),sg.T("Detection density over the acquisition")],
		[sg.B("Vector autoregression",key="-M8-",tooltip = "Assess whether clustered trajectories are confined and unclustered trajectories are unconfined.\nTop: clustered only (dark green), confined only (orange), both clustered and confined (light green).\nBottom: unclustered only (red), unconfined only (blue), both unclustered and unconfined (purple).",disabled=True),sg.T("VAR confinement vs clustering")],		
		[sg.B("SAVE ANALYSES",key="-SAVEANALYSES-",size=(25,2),button_color=("white","gray"),disabled=True,tooltip = "Save all analysis metrics, ROIs and open plots")]	
	]
	
	# Menu
	menu_def = [
		['&File', ['&Load settings', '&Save settings','&Default settings','&Exit']],
		['&Info', ['&About', '&Help','&Licence','&Updates' ]],
	]

	layout = [
		[sg.Menu(menu_def)],
		[sg.T('NASTIC',font="Any 20")],
		[sg.TabGroup([
			[sg.Tab("File",tab1_layout)],
			[sg.Tab("ROI",tab2_layout)],
			[sg.Tab("Clustering",tab3_layout)],
			[sg.Tab("Display",tab4_layout)],
			[sg.Tab("Metrics",tab5_layout)]
			],key="-TABGROUP-")
		],
		
		[sg.ProgressBar(100, orientation='h',size=(40,20),key='-PROGBAR-',expand_x = True)],
	]
	window = sg.Window('nastic v{}'.format(last_changed), layout)
	popup.close()

	# VARS
	cmap = matplotlib.cm.get_cmap('brg') # colormap for conditional coloring of clusters based on their average acquisition time
	all_selverts = [] # all ROI vertices
	all_selareas = [] # all ROI areas
	roi_list = [] # ROI artists
	trajdict = {} # Dictionary holding raw trajectory info
	sel_traj = [] # Selected trajectory indices
	lastfile = "" # Force the program to load a fresh TRXYT
	prev_roi_file = "" # Force the program to load a fresh ROI file
	seldict = {} # Selected trajectories and metrics
	clusterdict = {} # Cluster information
	plotflag = False # Has clustered data been plotted?
	
	# SET UP PLOTS
	plt.rcdefaults() 
	font = {"family" : "Arial","size": 12} 
	matplotlib.rc('font', **font)
	fig0 = plt.figure(0,figsize=(8,8))
	ax0 = plt.subplot(111)
	# Activate selection functions
	cid = fig0.canvas.mpl_connect('draw_event', ondraw)
	lasso = LassoSelector(ax0,onselect)	
	fig0.canvas.manager.set_window_title('Main display window - DO NOT CLOSE!')

	# MAIN LOOP
	while True:
		#Read events and values
		event, values = window.read(timeout=5000)
		infilename = values["-INFILE-"]	
		minlength = values["-MINLENGTH-"]
		maxlength = values["-MAXLENGTH-"]
		traj_prob = values["-TRAJPROB-"]
		selection_density = values["-SELECTIONDENSITY-"]
		roi_file = values["-R2-"]
		detection_alpha = values["-DETECTIONALPHA-"]
		acq_time = values["-ACQTIME-"]
		frame_time = values["-FRAMETIME-"]
		radius_thresh=values['-RADIUSTHRESH-']
		time_threshold = values["-TIMETHRESHOLD-"]
		radius_factor = values["-RADIUSFACTOR-"]
		cluster_threshold = values["-CLUSTERTHRESHOLD-"]
		canvas_color = values["-CANVASCOLOR-"]
		plot_trajectories = values["-TRAJECTORIES-"]
		plot_centroids = values["-CENTROIDS-"]
		plot_clusters = values["-CLUSTERS-"]
		plot_hotspots = values["-HOTSPOTS-"]
		plot_colorbar = values["-COLORBAR-"]	
		line_width = values["-LINEWIDTH-"]
		line_alpha = values["-LINEALPHA-"]
		line_color = values["-LINECOLOR-"]
		clust_color = values["-CLUSTCOLOR-"]
		var_color = values["-VARCOLOR-"]
		msd_color = values["-MSDCOLOR-"]
		cluster_width = values["-CLUSTERWIDTH-"]
		cluster_alpha = values["-CLUSTERALPHA-"]
		cluster_linetype = values["-CLUSTERLINETYPE-"]
		centroid_size = values["-CENTROIDSIZE-"]
		centroid_alpha = values["-CENTROIDALPHA-"]
		centroid_color = values["-CENTROIDCOLOR-"]
		saveformat = values["-SAVEFORMAT-"]
		savedpi = values["-SAVEDPI-"]
		savetransparency = values["-SAVETRANSPARENCY-"]
		savefolder = values["-SAVEFOLDER-"]
		autocluster = values["-AUTOCLUSTER-"]
		autoplot = values["-AUTOPLOT-"]
		cluster_fill = values['-CLUSTERFILL-']
		auto_metric = values['-AUTOMETRIC-']
		plotxmin = values['-PLOTXMIN-']
		plotxmax = values['-PLOTXMAX-']
		plotymin = values['-PLOTYMIN-']
		plotymax = values['-PLOTYMAX-']	
		msd_filter = values['-MSDFILTER-']	
		tmin = values['-TMIN-']	
		tmax = values['-TMAX-']		
		hotspot_radius = values["-HOTSPOTRADIUS-"]
		hotspot_width = values["-HOTSPOTWIDTH-"]
		hotspot_alpha = values["-HOTSPOTALPHA-"]
		hotspot_linetype = values["-HOTSPOTLINETYPE-"]		
		hotspot_color = values["-HOTSPOTCOLOR-"]
		axes_3d = values["-AXES3D-"]
		var_color1 = values["-VAR1COLOR-"]
		var_color2 = values["-VAR2COLOR-"]
		msd_color1 = values["-MSD1COLOR-"]		
		msd_color2 = values["-MSD2COLOR-"]
		pixel = values['-PIXEL-']

		# Check variables
		check_variables()

		# Exit	
		if event in (sg.WIN_CLOSED, 'Exit'):  
			break
			
		# If main display window is closed
		fignums = [x.num for x in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
		if 0 not in fignums:
			sg.Popup("Main display window closed!","Reinitialising new window","Please restart your analysis")
			fig0 = plt.figure(0,figsize=(8,8))
			ax0 = plt.subplot(111)
			# Activate selection functions
			cid = fig0.canvas.mpl_connect('draw_event', ondraw)
			lasso = LassoSelector(ax0,onselect)	
			fig0.canvas.manager.set_window_title('Main display window - DO NOT CLOSE!')
			
			# Reset variables
			all_selverts = [] # all ROI vertices
			all_selareas = [] # all ROI areas
			roi_list = [] # ROI artists
			trajdict = {} # Dictionary holding raw trajectory info
			sel_traj = [] # Selected trajectory indices
			lastfile = "" # Force the program to load a fresh TRXYT
			prev_roi_file = "" # Force the program to load a fresh ROI file
			seldict = {} # Selected trajectories and metrics
			clusterdict = {} # Cluster information
			
			# Close any other windows
			for i in [1,2,3,4,5,6,7,8,9,10]:
				try:
					plt.close(i)
				except:
					pass

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
				"\n\nFor detailed information regarding usage of the GUI:\n     Please refer to the nastic_user_manual.pdf\n     (downloaded as part of the NASTIC suite).",
				"\nAll buttons have popup tooltips in the meantime!\n", 
				no_titlebar = True,
				grab_anywhere = True	
				)	

		# Licence	
		if event == 'Licence':
			sg.Popup(
				"Licence",
				"Creative Commons CC BY 4.0",
				"https://creativecommons.org/licenses/by/4.0/legalcode", 
				no_titlebar = True,
				grab_anywhere = True	
				)					
		
		# Check for updates
		if event == 'Updates':
			webbrowser.open("https://github.com/tristanwallis/smlm_clustering/releases",new=2)
			
		# Read and plot input file	
		if event == '-PLOTBUTTON-':
			filter_status = False
			trxyt_tab(filter_status)

		# ROI stuff
		if len(trajdict) > 0:
			roi_tab()
		
		if event == '-REPLOT_ROI-':
			if len(roi_list) <= 1:
				window.Element("-SEPARATE-").update(disabled=True)
			elif len(roi_list) > 1:
				window.Element("-SEPARATE-").update(disabled=False)
			if len(roi_list) > 0:
				all_selverts_bak = [x for x in all_selverts]
				roi_list[-1].remove()
				roi_list.pop(-1)	
				all_selverts.pop(-1)
				selverts = []
				plt.show(block=False)
			read_roi()
		
		# Clustering
		if event ==	"-CLUSTERBUTTON-" and len(sel_traj) > 0:
			
			# Close all opened windows
			for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
				try:
					plt.close(i)
				except:
					pass
			# Close all buffers		
			try:
				buf0.close()
			except:
				pass	
			try:
				buf1.close()
			except:
				pass	
			try:
				buf2.close()
			except:
				pass	
			try:
				buf3.close()
			except:
				pass	
			try:
				buf4.close()
			except:
				pass	
			try:
				buf5.close()
			except:
				pass	
			try:
				buf6.close()
			except:
				pass	
			try:
				buf7.close()
			except:
				pass	
			try:
				buf8.close()
			except:
				pass
			try:
				buf9.close()
			except:
				pass
			if len(all_selverts) != 0: 
				all_selverts_copy = [x for x in all_selverts]
			all_selverts = []
			for roi in roi_list:
				roi.remove()
			roi_list = []
			cluster_tab()

		# Display
		if event ==	"-DISPLAYBUTTON-" and len(clusterdict)>0:
			display_tab(xlims,ylims)
			
		# Save
		if event ==	"-SAVEBUTTON-" and len(clusterdict)>0:
			print (savefolder)
			stamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now()) # datestamp
			filename = infilename.split("/")[-1]
			savefile = "{}/{}-{}.{}".format(savefolder,filename,stamp,saveformat)
			print ("Saving {} at {}dpi".format(savefile,savedpi))
			fig0.savefig(savefile,dpi=savedpi,transparent=savetransparency)		

		# Metrics
		if len(clusterdict)>0:
			metrics_tab()	
		
		# Change button colors as appropriate	
		if event: 
			update_buttons()
			
	print("Exiting...")		
	plt.close('all')				
	window.close()
	quit()