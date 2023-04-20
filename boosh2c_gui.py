# -*- coding: utf-8 -*-
'''
PYSIMPLEGUI BASED GUI FOR SPATIOTEMPORAL CLUSTERING OF MOLECULAR TRAJECTORY DATA USING 3D DBSCAN - 2 COLOR VERSION

Design and coding: Tristan Wallis
Additional coding: Kyle Young, Alex McCann
Debugging: Sophie Huiyi Hou, Kye Kudo, Alex McCann
Queensland Brain Institute
The University of Queensland
Fred Meunier: f.meunier@uq.edu.au

REQUIRED:
Python 3.8 or greater
python -m pip install scipy numpy matplotlib scikit-learn pysimplegui colorama

INPUT:
TRXYT trajectory files from Matlab
Space separated: Trajectory X(um) Y(um) T(sec)  
No headers

1 9.0117 39.86 0.02
1 8.9603 39.837 0.04
1 9.093 39.958 0.06
1 9.0645 39.975 0.08
2 9.1191 39.932 0.1
2 8.9266 39.915 0.12
etc

NOTES:
This script has been tested and will run as intended on Windows 7/10, with minor interface anomalies on Linux, and possible tk GUI performance issues on MacOS.
The script will fork to multiple CPU cores for the heavy number crunching routines (this also prevents it from being packaged as an exe using pyinstaller).
Feedback, suggestions and improvements are welcome. Sanctimonious critiques on the pythonic inelegance of the coding are not.
'''
last_changed = "20230414"

# MULTIPROCESSING FUNCTIONS
from scipy.spatial import ConvexHull
import multiprocessing
import numpy as np
import warnings
import math
from math import dist

warnings.filterwarnings("ignore")

def metrics(data):
	points,minlength,centroid=data
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
	area =ConvexHull(points2d).volume

	return [points,msds,centroid,diffcoeff,area]
	
def multi(allpoints):
	with multiprocessing.Pool() as pool:
		allmetrics = pool.map(metrics,allpoints)			
	return allmetrics	

# MAIN PROG AND FUNCTIONS
if __name__ == "__main__": # has to be called this way for multiprocessing to work
	# LOAD MODULES
	import PySimpleGUI as sg
	import os
	from colorama import init as colorama_init
	from colorama import Fore
	from colorama import Style
	sg.theme('DARKGREY11')
	colorama_init()
	os.system('cls' if os.name == 'nt' else 'clear')
	print(f'{Fore.GREEN}=================================================={Style.RESET_ALL}')
	print(f'{Fore.GREEN}BOOSH2C {last_changed} initialising...{Style.RESET_ALL}')
	print(f'{Fore.GREEN}=================================================={Style.RESET_ALL}')
	popup = sg.Window("Initialising...",[[sg.T("BOOSH2C initialising...",font=("Arial bold",18))]],finalize=True,no_titlebar = True,alpha_channel=0.9)
	
	import random
	from scipy.spatial import ConvexHull
	from scipy.stats import gaussian_kde
	from sklearn.cluster import DBSCAN
	from sklearn import manifold, datasets, decomposition, ensemble, random_projection	
	import numpy as np
	import matplotlib
	matplotlib.use('TkAgg') # prevents Matplotlib related crashes --> self.tk.call('image', 'delete', self.name)
	import matplotlib.pyplot as plt
	from matplotlib.widgets import LassoSelector
	from matplotlib import path
	from matplotlib.colors import LinearSegmentedColormap
	import matplotlib.colors as cols
	from mpl_toolkits.mplot3d import Axes3D,art3d
	import math
	import time
	import datetime
	import sys
	import pickle
	import io
	from functools import reduce
	import collections
	import warnings
	import multiprocessing
	warnings.filterwarnings("ignore")

	# NORMALIZE
	def normalize(lst):
		s = sum(lst)
		return map(lambda x: float(x)/s, lst)
		
	# CUSTOM COLORMAP
	def custom_colormap(colorlist,segments):
		cmap = LinearSegmentedColormap.from_list('mycmap', colorlist) # gradient cmap
		N = segments
		colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
		colors_rgba = cmap(colors_i)
		indices = np.linspace(0, 1., N+1)
		cdict = {}
		for ki,key in enumerate(('red','green','blue')):
			cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in range(N+1) ]
		cmap_s = cols.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024) # segmented colormap
		return cmap,cmap_s		

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
		xmin =-180
		xmax=180
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
		graph.DrawText("B O O S H 2 C v{}".format(last_changed),(0,70),color="white",font=("Any",16),text_location="center")
		graph.DrawText("Design and coding: Tristan Wallis",(0,45),color="white",font=("Any",10),text_location="center")
		graph.DrawText("Additional coding: Kyle Young, Alex McCann",(0,30),color="white",font=("Any",10),text_location="center")
		graph.DrawText("Debugging: Sophie Huiyi Hou, Kye Kudo, Alex McCann",(0,15),color="white",font=("Any",10),text_location="center")
		graph.DrawText("Queensland Brain Institute",(0,0),color="white",font=("Any",10),text_location="center")	
		graph.DrawText("The University of Queensland",(0,-15),color="white",font=("Any",10),text_location="center")	
		graph.DrawText("Fred Meunier f.meunier@uq.edu.au",(0,-30),color="white",font=("Any",10),text_location="center")	
		graph.DrawText("PySimpleGUI: https://pypi.org/project/PySimpleGUI/",(0,-55),color="white",font=("Any",10),text_location="center")	
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
		print ("Using default GUI settings...")
		global traj_prob,detection_alpha,minlength,maxlength,acq_time,epsilon,minpts,timewindow,canvas_color,plot_trajectories,plot_centroids,plot_clusters,plot_colorbar,line_width,line_alpha,line_color,line_color2,centroid_size,centroid_alpha,centroid_color,cluster_alpha,cluster_linetype,cluster_width,saveformat,savedpi,savetransparency,savefolder,selection_density,autoplot,autocluster,radius_thresh,cluster_fill,auto_metric,plotxmin,plotxmax,plotymin,plotymax,msd_filter,frame_time,tmin,tmax,cluster_colorby,plot_hotspots,hotspot_alpha,hotspot_linetype,hotspot_width,hotspot_color,hotspot_radius,balance,axes_3d, pixel
		traj_prob = 1
		detection_alpha = 0.05
		selection_density = 0
		minlength = 8
		maxlength = 100
		acq_time = 320
		frame_time = 0.02
		epsilon = 0.05
		minpts = 3
		timewindow = 10
		canvas_color = "black"
		plot_trajectories = True
		plot_centroids = False
		plot_clusters = True
		plot_hotspots = True		
		plot_colorbar = True	
		line_width = 1.5	
		line_alpha = 0.25	
		line_color = "cyan"
		line_color2 = "magenta"		
		centroid_size = 5	
		centroid_alpha = 0.75
		centroid_color = "white"
		cluster_width = 2
		cluster_colorby = "time"
		cluster_alpha = 1	
		cluster_linetype = "solid"
		cluster_fill = False	
		saveformat = "png"
		savedpi = 300	
		savetransparency = False
		autoplot=True
		autocluster=True
		radius_thresh=0.15
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
		hotspot_radius = 1		
		balance = True
		axes_3d = True
		pixel = 0.106
		
		return 

	# SAVE SETTINGS
	def save_defaults():
		print ("Saving GUI settings to boosh2c_gui.defaults...")
		with open("boosh2c_gui.defaults","w") as outfile:
			outfile.write("{}\t{}\n".format("Trajectory probability",traj_prob))
			outfile.write("{}\t{}\n".format("Raw trajectory detection plot opacity",detection_alpha))
			outfile.write("{}\t{}\n".format("Selection density",selection_density))
			outfile.write("{}\t{}\n".format("Trajectory minimum length",minlength))
			outfile.write("{}\t{}\n".format("Trajectory maximum length",maxlength))
			outfile.write("{}\t{}\n".format("Acquisition time (s)",acq_time))	
			outfile.write("{}\t{}\n".format("Frame time (s)",frame_time))				
			outfile.write("{}\t{}\n".format("Epsilon (um)",epsilon))
			outfile.write("{}\t{}\n".format("MinPts",minpts))
			outfile.write("{}\t{}\n".format("Time Window",timewindow))			
			outfile.write("{}\t{}\n".format("Canvas color",canvas_color))	
			outfile.write("{}\t{}\n".format("Plot trajectories",plot_trajectories))
			outfile.write("{}\t{}\n".format("Plot centroids",plot_centroids))
			outfile.write("{}\t{}\n".format("Plot clusters",plot_clusters))
			outfile.write("{}\t{}\n".format("Plot hotspots",plot_hotspots))			
			outfile.write("{}\t{}\n".format("Plot colorbar",plot_colorbar))		
			outfile.write("{}\t{}\n".format("Trajectory line width",line_width))
			outfile.write("{}\t{}\n".format("Trajectory line color 1",line_color))
			outfile.write("{}\t{}\n".format("Trajectory line color 2",line_color2))			
			outfile.write("{}\t{}\n".format("Trajectory line opacity",line_alpha))
			outfile.write("{}\t{}\n".format("Centroid size",centroid_size))
			outfile.write("{}\t{}\n".format("Centroid color",centroid_color))
			outfile.write("{}\t{}\n".format("Centroid opacity",centroid_alpha))
			outfile.write("{}\t{}\n".format("Cluster fill",cluster_fill))		
			outfile.write("{}\t{}\n".format("Cluster color by",cluster_colorby))			
			outfile.write("{}\t{}\n".format("Cluster line width",cluster_width))			
			outfile.write("{}\t{}\n".format("Cluster line opacity",cluster_alpha))
			outfile.write("{}\t{}\n".format("Cluster line type",cluster_linetype))
			outfile.write("{}\t{}\n".format("Hotspot line width",hotspot_width))			
			outfile.write("{}\t{}\n".format("Hotspot line opacity",hotspot_alpha))
			outfile.write("{}\t{}\n".format("Hotspot line type",hotspot_linetype))	
			outfile.write("{}\t{}\n".format("Hotspot color",hotspot_color))			
			outfile.write("{}\t{}\n".format("Hotspot radius",hotspot_radius))			
			outfile.write("{}\t{}\n".format("Plot save format",saveformat))
			outfile.write("{}\t{}\n".format("Plot save dpi",savedpi))
			outfile.write("{}\t{}\n".format("Plot background transparent",savetransparency))
			outfile.write("{}\t{}\n".format("Auto cluster",autocluster))
			outfile.write("{}\t{}\n".format("Auto plot",autoplot))
			outfile.write("{}\t{}\n".format("Cluster size screen",radius_thresh))	
			outfile.write("{}\t{}\n".format("Auto metric",auto_metric))	
			outfile.write("{}\t{}\n".format("MSD filter",msd_filter))	
			outfile.write("{}\t{}\n".format("Color balance",balance))	
			outfile.write("{}\t{}\n".format("3D axes",axes_3d))	
			outfile.write("{}\t{}\n".format("Pixel size (um)",pixel))
			
		return
		
	# LOAD DEFAULTS
	def load_defaults():
		global defaultdict,traj_prob,detection_alpha,minlength,maxlength,acq_time,epsilon,minpts,timewindow,canvas_color,plot_trajectories,plot_centroids,plot_clusters,plot_colorbar,line_width,line_alpha,line_color,line_color2,centroid_size,centroid_alpha,centroid_color,cluster_alpha,cluster_linetype,cluster_width,saveformat,savedpi,savetransparency,savefolder,selection_density,autoplot,autocluster,radius_thresh,cluster_fill,auto_metric,plotxmin,plotxmax,plotymin,plotymax,msd_filter,frame_time,tmin,tmax,cluster_colorby,plot_hotspots,hotspot_alpha,hotspot_linetype,hotspot_width,hotspot_color,hotspot_radius,balance,axes_3d, pixel
		try:
			with open ("boosh2c_gui.defaults","r") as infile:
				print ("Loading GUI settings from boosh2c_gui.defaults...")
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
			epsilon = float(defaultdict["Epsilon (um)"])
			minpts = int(defaultdict["MinPts"])
			timewindow = float(defaultdict["Time Window"])
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
			line_color = defaultdict["Trajectory line color 1"]
			line_color2 = defaultdict["Trajectory line color 2"]			
			centroid_size = int(defaultdict["Centroid size"])	
			centroid_alpha = float(defaultdict["Centroid opacity"])	
			centroid_color = defaultdict["Centroid color"]
			cluster_colorby = defaultdict["Cluster color by"]	
			cluster_width = float(defaultdict["Cluster line width"])	
			cluster_alpha = float(defaultdict["Cluster line opacity"])	
			cluster_linetype = defaultdict["Cluster line type"]		
			cluster_fill = defaultdict["Cluster fill"]
			if cluster_fill == "True":
				cluster_fill = True
			if cluster_fill == "False":
				cluster_fill = False
			hotspot_color = defaultdict["Hotspot color"]
			hotspot_width = float(defaultdict["Hotspot line width"])	
			hotspot_alpha = float(defaultdict["Hotspot line opacity"])	
			hotspot_linetype = defaultdict["Hotspot line type"]		
			hotspot_radius = defaultdict["Hotspot radius"]		
			saveformat = defaultdict["Plot save format"]
			savedpi = defaultdict["Plot save dpi"]	
			savetransparency = defaultdict["Plot background transparent"]
			autoplot = defaultdict["Auto plot"]
			autocluster = defaultdict["Auto cluster"]
			radius_thresh = defaultdict["Cluster size screen"]			
			if savetransparency == "True":
				savetransparency = True
			if savetransparency == "False":
				savetransparency = False
			if autocluster == "True":
				autocluster = True
			if autocluster == "False":
				autocluster = False
			if autoplot == "True":
				autoplot = True
			if autoplot == "False":
				autoplot = False	
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
			balance = defaultdict["Color balance"]
			if balance == "True":
				balance = True
			if balance == "False":
				balance = False	
			axes_3d = defaultdict["3D axes"]
			if axes_3d == "True":
				axes_3d = True
			if axes_3d == "False":
				axes_3d = False	
			pixel = defaultdict["Pixel size (um)"] 	
		except:
			print ("Settings could not be loaded")
		return
		
	# UPDATE GUI BUTTONS
	def update_buttons():
		if len(infilename) > 0 and len(infilename2) > 0:  
			window.Element("-PLOTBUTTON-").update(button_color=("white","#111111"),disabled=False)
			window.Element("-INFILE-").InitialFolder = os.path.dirname(infilename)	
			window.Element("-INFILE2-").InitialFolder = os.path.dirname(infilename2)				
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
			window.Element("-LINECOLORCHOOSE2-").update(disabled=False)
			window.Element("-CENTROIDCOLORCHOOSE-").update(disabled=False)
			window.Element("-HOTSPOTCOLORCHOOSE-").update(disabled=False)			
			window.Element("-SAVEANALYSES-").update(button_color=("white","#111111"),disabled=False)
			for buttonkey in ["-M1-","-M2-","-M3-","-M4-","-M5-","-M6-","-M7-"]:
				window.Element(buttonkey).update(disabled=False)
		else:  
			window.Element("-DISPLAYBUTTON-").update(button_color=("white","gray"),disabled=True)
			window.Element("-SAVEBUTTON-").update(button_color=("white","gray"),disabled=True)
			window.Element("-CANVASCOLORCHOOSE-").update(disabled=True)
			window.Element("-LINECOLORCHOOSE-").update(disabled=True)
			window.Element("-LINECOLORCHOOSE2-").update(disabled=True)
			window.Element("-CENTROIDCOLORCHOOSE-").update(disabled=True)
			window.Element("-HOTSPOTCOLORCHOOSE-").update(disabled=True)
			window.Element("-SAVEANALYSES-").update(button_color=("white","gray"),disabled=True)		
			for buttonkey in ["-M1-","-M2-","-M3-","-M4-","-M5-","-M6-","-M7-"]:
				window.Element(buttonkey).update(disabled=True)	
		window.Element("-TRAJPROB-").update(traj_prob)
		window.Element("-DETECTIONALPHA-").update(detection_alpha)	
		window.Element("-SELECTIONDENSITY-").update(selection_density)	
		window.Element("-MINLENGTH-").update(minlength)
		window.Element("-MAXLENGTH-").update(maxlength)	
		window.Element("-ACQTIME-").update(acq_time)
		window.Element("-FRAMETIME-").update(frame_time)		
		window.Element("-EPSILON-").update(epsilon)	
		window.Element("-MINPTS-").update(minpts)	
		window.Element("-TIMEWINDOW-").update(timewindow)	
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
		window.Element("-LINECOLORCHOOSE2-").update("Choose",button_color=("gray",line_color2))
		window.Element("-LINECOLOR-").update(line_color)
		window.Element("-LINECOLOR2-").update(line_color2)			
		window.Element("-CENTROIDSIZE-").update(centroid_size)
		window.Element("-CENTROIDALPHA-").update(centroid_alpha)
		window.Element("-CENTROIDCOLORCHOOSE-").update("Choose",button_color=("gray",centroid_color))
		window.Element("-CENTROIDCOLOR-").update(centroid_color)
		window.Element("-CLUSTERWIDTH-").update(cluster_width)
		window.Element("-CLUSTERCOLORBY-").update(cluster_colorby)
		window.Element("-CLUSTERALPHA-").update(cluster_alpha)
		window.Element("-CLUSTERLINETYPE-").update(cluster_linetype)
		window.Element("-CLUSTERFILL-").update(cluster_fill)
		window.Element("-HOTSPOTCOLORCHOOSE-").update("Choose",button_color=("gray",hotspot_color))
		window.Element("-HOTSPOTCOLOR-").update(hotspot_color)		
		window.Element("-HOTSPOTWIDTH-").update(hotspot_width)
		window.Element("-HOTSPOTALPHA-").update(hotspot_alpha)
		window.Element("-HOTSPOTRADIUS-").update(hotspot_radius)
		window.Element("-HOTSPOTLINETYPE-").update(hotspot_linetype)
		window.Element("-SAVEFORMAT-").update(saveformat)	
		window.Element("-SAVETRANSPARENCY-").update(savetransparency)
		window.Element("-SAVEDPI-").update(savedpi)
		window.Element("-SAVEFOLDER-").update(savefolder)
		window.Element("-RADIUSTHRESH-").update(radius_thresh)
		window.Element("-AUTOMETRIC-").update(auto_metric)
		window.Element("-PLOTXMIN-").update(plotxmin)
		window.Element("-PLOTXMAX-").update(plotxmax)
		window.Element("-PLOTYMIN-").update(plotymin)
		window.Element("-PLOTYMAX-").update(plotymax)
		window.Element("-MSDFILTER-").update(msd_filter)
		window.Element("-TMIN-").update(tmin)
		window.Element("-TMAX-").update(tmax)	
		window.Element("-BALANCE-").update(balance)	
		window.Element("-AXES3D-").update(axes_3d)	
		window.Element("-PIXEL-").update(pixel)
		
		return	
		
	# CHECK VARIABLES
	def check_variables():
		global traj_prob,detection_alpha,minlength,maxlength,acq_time,epsilon,minpts,timewindow,canvas_color,plot_trajectories,plot_centroids,plot_clusters,line_width,line_alpha,line_color,line_color2,centroid_size,centroid_alpha,centroid_color,cluster_alpha,cluster_linetype,cluster_width,saveformat,savedpi,savetransparency,savefolder,selection_density,radius_thresh,plotxmin,plotxmax,plotymin,plotymax,frame_time,tmin,tmax,cluster_colorby,plot_hotspots,hotspot_alpha,hotspot_linetype,hotspot_width,hotspot_color,hotspot_radius,balance,axes_3d, pixel

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
			if minlength < 8:
				minlength = 8
		except:
			minlength = 8
		try:
			maxlength = int(maxlength)
		except:
			maxlength = 100		
		if minlength > maxlength:
			minlength = 8
			maxlength = 100	
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
				frame_time = 0.02; 
		except:
			frame_time = 0.02			
		try:
			epsilon = float(epsilon)
			if epsilon <= 0: 
				epsilon = 0.00000001
		except:
			epsilon = 0.05
		try:
			minpts = int(minpts)
			if minpts < 3:
				minpts = 3
		except:
			minpts = 3	
		try:
			timewindow = float(timewindow)
			if timewindow < 0.1:
				timewindow = 0.1 
		except:
			timewindow = 10.0		
		try:
			radius_thresh = float(radius_thresh)
			if radius_thresh < 0.001:
				radius_thresh = 0.15
		except:
			radius_thresh = 0.15			
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
		if cluster_colorby not in ["time","composition"]:
			cluster_colorby = "time" 			
		if cluster_alpha not in [0.01,0.05,0.1,0.25,0.5,0.75,1.0]:
			cluster_alpha = 1.0 
		if cluster_linetype not in ["solid","dotted","dashed"]:
			cluster_linetype = "solid" 	
		if hotspot_width not in [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]:
			hotspot_width = 1.5 
		if hotspot_alpha not in [0.01,0.05,0.1,0.25,0.5,0.75,1.0]:
			hotspot_alpha = 1.0 
		if hotspot_linetype not in ["solid","dotted","dashed"]:
			hotspot_linetype = "solid" 	
		if hotspot_radius not in [0.1,0.25,0.5,1.0,1.25,1.5,1.75,2.0]:
			hotspot_radius = 1 					
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
				line_color = defaultdict["Trajectory line color 1"]
			except:	
				line_color = "cyan"
		if line_color2 == "None":
			try:
				line_color2 = defaultdict["Trajectory line color 2"]
			except:	
				line_color2 = "magenta"				
				
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
			#PalmTracer .RGN file
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
				sg.popup("Alert", "No ROIs found")
			else:
				for roi in roidict:
					selverts = roidict[roi]
					use_roi(selverts,"orange")
				return
			return
		
		else:
			window.Element("-PIXEL_TEXT-").update(visible=False)
			window.Element("-PIXEL-").update(visible=False)
			window.Element("-REPLOT_ROI-").update(visible = False)
			
			roidict = {}
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
				sg.popup("Alert","No ROIs found")
			else:	
				for roi in roidict:			
					selverts =roidict[roi]	
					use_roi(selverts,"orange")
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

	# PROBABILITY OF CLUSTER OVERLAP AT A GIVEN DISTANCE	
	def overlap_prob(clustpoints,epsilon):
		labels,clusterlist = dbscan(clustpoints,epsilon,2) # does each cluster centroid form a clustered cluster within dist (epsilon)?
		clusterlist = [x for x in clusterlist]
		unclustered = [x for x in labels if x == -1] # select DBSCAN unclustered centroids	
		p = 1 - float(len(unclustered))/len(labels) # probability of adjacent cluster centroids within dist
		return p

	# LOAD AND PLOT TRXYT TAB
	def trxyt_tab():
		# Reset variables
		global all_selverts,all_selareas,roi_list,trajdict,sel_traj,lastfile,lastfile2,seldict,clusterdict,x_plot1,x_plot2,y_plot1,y_plot2,xlims,ylims,savefolder,buf 
		all_selverts = [] # all ROI vertices
		all_selareas = [] # all ROI areas
		roi_list = [] # ROI artists
		trajdict1 = {} # Dictionary holding raw trajectory info 
		trajdict2 = {} # Dictionary holding raw trajectory info
		sel_traj = [] # Selected trajectory indices
		lastfile = "" # Force the program to load a fresh TRXYT
		lastfile2 = "" # Force the program to load a fresh TRXYT		
		seldict = {} # Selected trajectories and metrics
		clusterdict = {} # Cluster information
		
		# Close all opened windows
		for i in [1,2,3,4,5,6,7,8,9,10]:
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
		try:
			buf10.close()
		except:
			pass				
				
		try:	
			ax0.unshare_x_axes(ax8)	
			ax0.unshare_y_axes(ax8)	
			print ("Unsharing")
		except: 
			pass

		'''
		IMPORTANT: It appears that some matlab processing of trajectory data converts trajectory numbers > 99999 into scientific notation with insufficient decimal points. eg 102103 to 1.0210e+05, 102104 to 1.0210e+05. This can cause multiple trajectories to be incorrectly merged into a single trajectory.
		For trajectories > 99999 we empirically determine whether detections are within 0.32u of each other, and assign them into a single trajectory accordingly. For trajectories <99999 we honour the existing trajectory number.
		'''		
		if infilename != lastfile:
			# Read file into dictionary
			lastfile=infilename
			print("Loading raw trajectory data from {}...".format(infilename))
			t1=time.time()
			rawtrajdict1 = {}
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
								rawtrajdict1[ct]["points"].append([x,y,t]) 
								x0 = x
								y0 = y
							else:
								ct += 1
								rawtrajdict1[ct]= {"points":[[x,y,t]]}	
								x0 = x
								y0 = y
						else:
							try:
								rawtrajdict1[n]["points"].append([x,y,t])
							except:
								rawtrajdict1[n]= {"points":[[x,y,t]]}
					except:
						pass
			
			print("{} trajectories in file 1".format(len(rawtrajdict1)))

		if infilename2 != lastfile2:
			# Read file into dictionary
			lastfile2=infilename2
			print("Loading raw trajectory data from {}...".format(infilename2))
			ct = 10099999
			x0 = -10000
			y0 = -10000
			rawtrajdict2 = {}
			with open (infilename2,"r") as infile:
				for line in infile:
					try:
						line = line.replace("\n","").replace("\r","")
						spl = line.split(" ")
						n = int(float(spl[0])) + 10000000
						x = float(spl[1])
						y = float(spl[2])
						t = float(spl[3])
						if n > 10099999:
							if abs(x-x0) < 0.32 and abs(y-y0) < 0.32:
								rawtrajdict2[ct]["points"].append([x,y,t])
								x0 = x
								y0= y
							else:
								ct += 1
								rawtrajdict2[ct]= {"points":[[x,y,t]]}	
								x0 = x
								y0=y
						else:
							try:
								rawtrajdict2[n]["points"].append([x,y,t])
							except:
								rawtrajdict2[n]= {"points":[[x,y,t]]}
								
					except:
						pass
			print("{} trajectories in file 2".format(len(rawtrajdict2)))					
		
		# Don't bother with anything else if there's no trajectories				
		if len(rawtrajdict1) == 0 or len(rawtrajdict2) == 0:
			sg.popup("Alert","No trajectory information found in file 1 or file 2")
		else:
			
			# Screen and display
			for traj1 in rawtrajdict1: 
				points = rawtrajdict1[traj1]["points"] 
				if len(points) >=minlength and len(points) <=maxlength:
					trajdict1[traj1] = rawtrajdict1[traj1] 
					
			for traj2 in rawtrajdict2:
				points = rawtrajdict2[traj2]["points"]
				if len(points) >=minlength and len(points) <=maxlength:
					trajdict2[traj2] = rawtrajdict2[traj2] 

			trajdict = {**trajdict1,**trajdict2}; 
	
			print("Plotting detections...")
			ct = 0
			ax0.cla() # clear last plot if present
			detpoints1 = [] 
			for num,traj in enumerate(trajdict1): 
				if num%10 == 0:
					bar = 100*num/(len(trajdict1)) 
					window['-PROGBAR-'].update_bar(bar)
				
				if random.random() <= traj_prob:
					ct+=1
					[detpoints1.append(i) for i in trajdict1[traj]["points"]] 
			window['-PROGBAR-'].update_bar(0)		
			detpoints2 = []
			for num,traj in enumerate(trajdict2):
				if num%10 == 0:
					bar = 100*num/(len(trajdict2))
					window['-PROGBAR-'].update_bar(bar)
				
				if random.random() <= traj_prob:
					ct+=1
					[detpoints2.append(i) for i in trajdict2[traj]["points"]]					
			window['-PROGBAR-'].update_bar(0)		
			x_plot1,y_plot1,t_plot1=zip(*detpoints1)
			ax0.scatter(x_plot1,y_plot1,c=line_color,s=3,linewidth=0,alpha=detection_alpha)

			x_plot2,y_plot2,t_plot2=zip(*detpoints2)
			ax0.scatter(x_plot2,y_plot2,c=line_color2,s=3,linewidth=0,alpha=detection_alpha)
			
			ax0.set_facecolor("k")
			#ax0.set_title(infilename.split("/")[-1])		
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
			print("{} detections from {} trajectories plotted in {} sec".format(len(x_plot1+x_plot2),ct,round(t2-t1,3)))
			
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
		global selverts,all_selverts,all_selareas,roi_list,trajdict,sel_traj,sel_centroids,all_selverts_copy,all_selverts_bak

		# Load and apply ROIs	
		if event ==	"-R2-" and roi_file != "Load previously defined ROIs":
			if len(roi_list) <= 1:
				window.Element('-SEPARATE-').update(disabled = True)
			elif len(roi_list) > 1:
				window.Element('-SEPARATE-').update(disabled = False)
			all_selverts_bak = [x for x in all_selverts]
			roidict = read_roi()

		# Clear all ROIs
		if event ==	"-R3-"and len(roi_list) > 0: 
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
			xmin = min(min(x_plot1),min(x_plot2)) 
			xmax = max(max(x_plot1),max(x_plot2)) 
			ymin = min(min(y_plot1),min(y_plot2))
			ymax = max(max(y_plot1),max(y_plot2))
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
				if len(all_selverts_bak) > 0:
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
		
		# Reset view
		if event == "-RESET-":
			selverts_reset = [x for x in all_selverts_copy]
			if len(selverts) >3:
				if len(selverts_reset) == 0:
					selverts_reset = [x for x in all_selverts]
				trxyt_tab()
				for selverts in selverts_reset:
					use_roi(selverts,"orange")
				window.Element('-RESET-').update(disabled = True)
		
		# Save current ROIs	
		if event ==	"-R8-" and len(all_selverts) > 0:
			stamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now())
			outpath = os.path.dirname(infilename)
			outdir = outpath + "/" + infilename.split("/")[-1].replace(".trxyt","_BOOSH2C_ROIs_{}".format(stamp))
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
			outdir = outpath + "/" + infilename.split("/")[-1].replace(".trxyt","_BOOSH2C_ROIs_{}".format(stamp))
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
			print ("Selecting trajectories within {} ROIs...".format(len(roi_list)))
			t1=time.time()
		
			# Centroids for each trajectory
			all_centroids = []
			for num,traj in enumerate(trajdict):
				if num%10 == 0:
					bar = 100*num/(len(trajdict))
					window['-PROGBAR-'].update_bar(bar)
				points = trajdict[traj]["points"]
				x,y,t=list(zip(*points))
				xmean = np.average(x)
				ymean = np.average(y)	
				tmean = np.average(t)
				centroid = [xmean,ymean,tmean]
				trajdict[traj]["centroid"] = centroid
				all_centroids.append([centroid[0],centroid[1],traj])
			sel_traj = []
			all_selareas = []
			sel_centroids = []
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
				sel_traj.sort()
			all_selverts_copy = [x for x in all_selverts]
			all_selverts = []
			for roi in roi_list:
				roi.remove()
			roi_list = []
			
			if len(sel_traj) == 0:
				sg.Popup("Alert","No trajectories found in selected ROI", "Save ROIs that you want to keep before Removing")
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
				if autocluster:
					cluster_tab()
		return
		
	# CLUSTERING TAB	
	def cluster_tab():
		global sel_traj,seldict,clusterdict,allindices,clustindices,unclustindices,spatial_clusters,av_msd,all_diffcoeffs,indices1,balance

		# Dictionary of selected trajectories
		print ("Generating metrics of selected trajectories...")
		t1 = time.time()
		sel_traj.sort()
		indices1 = len([x for x in sel_traj if x < 10000000])

		# Balance trajectory numbers
		if balance:
			print ("Balancing trajectory numbers between colors..")
			col1 = [x for x in sel_traj if x < 10000000]
			
			if len(col1) == 0:
				sg.Popup("Alert", "No trajectories from file 1 found in selected ROI")
				for selverts in all_selverts_copy:
					use_roi(selverts,"orange")
				return
			
			col2 = [x for x in sel_traj if x > 10000000]
			
			if len(col2) == 0: 
				sg.Popup("Alert", "No trajectories from file 2 found in selected ROI")
				for selverts in all_selverts_copy:
					use_roi(selverts,"orange")
				return
			try:
				col_ratio = float(len(col1)/len(col2))
				if col_ratio < 1:
					col2 = [x for x in col2 if random.random() < col_ratio]
				else:
					col1 = [x for x in col1 if random.random() < (1/col_ratio)]
					indices1 = len(col1)
				window["-TABGROUP-"].Widget.select(2)
			except:
				sg.Popup("Alert", "No trajectories found for both files within selected ROI")
				for selverts in all_selverts_copy:
					use_roi(selverts,"orange") 
				return
			
			print ("Balanced trajectory numbers within ROI: {} {}".format(len(col1),len(col2)))
			sel_traj = col1 + col2	

		allpoints = [[trajdict[traj]["points"],minlength,trajdict[traj]["centroid"]] for traj in sel_traj]
		allmetrics = multi(allpoints) # fork these calculations onto all cores
		all_msds = []
		all_diffcoeffs = []

		for num,metrics in enumerate(allmetrics):
			if num%10 == 0:
				bar = 100*num/(len(allmetrics))
				window['-PROGBAR-'].update_bar(bar)
			seldict[num]={}
			points,msds,centroid,diffcoeff,area = metrics
			seldict[num]["points"]=points
			seldict[num]["msds"]=msds
			seldict[num]["area"]=area
			all_msds.append(msds[0])			
			seldict[num]["diffcoeff"]=diffcoeff/(frame_time*3)
			all_diffcoeffs.append(abs(diffcoeff))
			seldict[num]["centroid"]=centroid
		window['-PROGBAR-'].update_bar(0)		
		t2=time.time()	
		print ("Metrics generated for {} trajectories in {} sec".format(len(allmetrics),round(t2-t1,3)))	
		
		# Screen on MSD
		if msd_filter:
			print ("Calculating average MSD...")
			av_msd = np.average(all_msds)
		else:
			av_msd = 10000 # no molecule except in an intergalactic gas cloud has an MSD this big

		filt_indices = []
		for num in seldict:
			if seldict[num]["msds"][0] < av_msd:
				filt_indices.append(num)
				sel_centroids.append(seldict[num]["centroid"])
		print ("{} trajectories passed MSD filter".format(len(filt_indices)))
		
		# Dictionary of all selected points
		sel_points = []
		pointdict = {}
		ct=0
		for idx in filt_indices:
			points = seldict[idx]["points"]
			msds = seldict[idx]["msds"]	
			for point in points:
				pointdict[ct] = {}
				pointdict[ct]["point"] = point # [x,y,t] co-ordinates of point
				pointdict[ct]["traj"]= idx # parent trajectory
				sel_points.append(point)		
				ct+=1
		
		# Determine clustered points
		print ("Clustering selected trajectories...")
		indices = range(len(seldict))
		t1 = time.time()
		squash = epsilon/timewindow		# Convert time window to epsilon, so 3D DBSCAN will work
		pointarray = [[x[0],x[1],x[2]*squash] for x in sel_points]
		labels,clusterlist = dbscan(pointarray,epsilon,minpts)
		pointcount = collections.Counter(labels)
		t2 = time.time()
		print ("{} detections from {} trajectories clustered in {} sec".format(len(pointarray),len(seldict),round(t2-t1,3)))

		# Cluster metrics
		print ("Generating metrics of clustered trajectories...")
		t1 = time.time()
		tempclusterdict = {} # temporary dictionary holding info for each spatial cluster
		clusterdict = {}

		for cluster in clusterlist:
			tempclusterdict[cluster] = {}
			tempclusterdict[cluster]["pointindices"] = [] # indices of clustered points
			tempclusterdict[cluster]["points"] = [] # points co-ordinates
			tempclusterdict[cluster]["indices"] = [] # indices of parent trajectories
				
		for num,label in enumerate(labels):
			tempclusterdict[label]["pointindices"].append(num)
			tempclusterdict[label]["points"].append(pointdict[num]["point"])
			tempclusterdict[label]["indices"].append(pointdict[num]["traj"])
			
		for cluster in clusterlist:
			indices = list(set(tempclusterdict[cluster]["indices"]))
			tempclusterdict[cluster]["indices"] = indices
			tempclusterdict[cluster]["traj_num"] = len(indices)
			tempclusterdict[cluster]["composition"] = len([x for x in indices if x > indices1])/len(indices) # composition 0 = 100% molecule 1, 1 = 100% molecule 2 
			if len(indices) >= minpts and len(tempclusterdict[cluster]["pointindices"]) > 4: # clusters must contain minpts or more trajectories
				clusterdict[cluster] = tempclusterdict[cluster]

		for cluster in clusterdict:
			if cluster > -1:
				msds = [seldict[i]["msds"][0] for i in clusterdict[cluster]["indices"]] # MSDS for all traj in this cluster
				clusterdict[cluster]["av_msd"]= np.average(msds) # Average trajectory MSD in this cluster
				clustertimes = [seldict[i]["centroid"][2] for i in clusterdict[cluster]["indices"]] # all centroid times in this cluster
				clusterdict[cluster]["centroid_times"] = clustertimes
				clusterdict[cluster]["lifetime"] = max(clustertimes) - min(clustertimes) # lifetime of this cluster (sec)
				clusterpoints = [point[:2] for point in clusterdict[cluster]["points"]] # All detection points [x,y] in this cluster
				clusterdict[cluster]["det_num"] = len(clusterpoints) # number of detections in this cluster	
				try:
					ext_x,ext_y,ext_area,int_x,int_y,int_area = double_hull(clusterpoints) # Get external/internal hull area
				except: 
					sg.popup("Alert","Clustering error","Please try different clustering metrics")
					return 
				clusterdict[cluster]["area"] = ext_area # Use external hull area as cluster area (um2)
				clusterdict[cluster]["radius"] = math.sqrt(int_area/math.pi) # radius of cluster (um)
				clusterdict[cluster]["area_xy"] = [int_x,int_y] # area border coordinates	
				clusterdict[cluster]["density"] = clusterdict[cluster]["traj_num"]/int_area # trajectories/um2	
				clusterdict[cluster]["rate"] = clusterdict[cluster]["traj_num"]/(max(clustertimes) - min(clustertimes)) # accumulation rate (trajectories/sec)
				#clustercentroids = [seldict[i]["centroid"] for i in clusterdict[cluster]["indices"]] # Centroids for each trajectory in this cluster
				diffcoeffs = [seldict[i]["diffcoeff"] for i in clusterdict[cluster]["indices"]] # Instantaneous diffusion coefficients for each trajectory in this cluster
				clusterdict[cluster]["av_diffcoeff"]= np.average(diffcoeffs) # average trajectory inst diff coeff in this cluster
				x,y,t = zip(*clusterdict[cluster]["points"])
				xmean = np.average(x)
				ymean = np.average(y)
				tmean = np.average(t)
				clusterdict[cluster]["centroid"] = [xmean,ymean,tmean] # centroid for this cluster
				
			
		# Screen out large and tiny clusters 
		allindices = range(len(seldict))
		clustindices = []
		tempclusterdict = {}
		counter = 1	
		for num in clusterdict:
			if num > -1:
				if clusterdict[num]["radius"] < radius_thresh and len(clusterdict[num]["points"]) > 3:
					tempclusterdict[counter] = clusterdict[num]
					[clustindices.append(i) for i in clusterdict[num]["indices"]]
					counter +=1
		clusterdict = tempclusterdict.copy()	
		
		if len(clusterdict) == 0:
			sg.Popup("Alert","No unique spatiotemporal clusters containing trajectories found in the selected ROI","Please try adjusting the ROI or clustering parameters")
			window.Element("-RESET-").update(disabled=False)
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
		
			# Trajectories appearing in more than one cluster
			trajcount = collections.Counter(clustindices)
			promiscuous = []
			for traj in trajcount:
				if trajcount[traj]>1:
					promiscuous.append(traj)

			window["-TABGROUP-"].Widget.select(3)
			if autoplot and len(clusterdict)>0:
				display_tab(xlims,ylims)
		return

	# DISPLAY CLUSTERED DATA TAB
	def	display_tab(xlims,ylims):
		global buf0,plotflag,plotxmin,plotymin,plotxmax,plotymax,indices1
		print ("Plotting clustered trajectories...")
		xlims = ax0.get_xlim()
		ylims = ax0.get_ylim()

		# User zoom
		if plotxmin !="" and plotxmax !="" and plotymin !="" and plotymax !="":
			xlims = [plotxmin,plotxmax]
			ylims = [plotymin,plotymax]
		
		# Reset zoom	
		if plotxmin ==0.0 and plotxmax ==0.0 and plotymin ==0.0 and plotymax ==0.0:	
			xlims =	[min(min(x_plot1),min(x_plot2)),max(max(x_plot1),max(x_plot2))] 
			ylims =	[min(min(y_plot1),min(y_plot2)),max(max(y_plot1),max(y_plot2))]
		plotxmin,plotxmax,plotymin,plotymax="","","",""			

		ax0.cla()
		ax0.set_facecolor(canvas_color)
		xcent = []
		ycent = []
		# All trajectories
		print ("Plotting trajectories...")
		t1=time.time()
		for num,traj in enumerate(seldict):
			if num%10 == 0:
				bar = 100*num/(len(seldict)-1)
				window['-PROGBAR-'].update_bar(bar)
			centx=seldict[traj]["centroid"][0]
			centy=seldict[traj]["centroid"][1]
			if centx > xlims[0] and centx < xlims[1] and centy > ylims[0] and centy < ylims[1]:
				if plot_trajectories:
					if traj < indices1:
						col = line_color
					else:
						col = line_color2
					x,y,t=zip(*seldict[traj]["points"])
					tr = matplotlib.lines.Line2D(x,y,c=col,alpha=line_alpha,linewidth=line_width)
					ax0.add_artist(tr) 
				if plot_centroids:
					xcent.append(seldict[traj]["centroid"][0])
					ycent.append(seldict[traj]["centroid"][1])	
		ax0.scatter(xcent,ycent,c=centroid_color,alpha=centroid_alpha,s=centroid_size,linewidth=0,zorder=100)
		window['-PROGBAR-'].update_bar(0)
		
		# Clustered trajectories
		print ("Highlighting clusters...")
		
		# Custom colormap
		if cluster_colorby == "composition":
			twmap,twmap_s = custom_colormap([line_color,"orange",line_color2],9)
		else:
			twmap = cmap
	
		for cluster in clusterdict:
			bar = 100*cluster/(len(clusterdict))
			window['-PROGBAR-'].update_bar(bar)
			centx=clusterdict[cluster]["centroid"][0]
			centy=clusterdict[cluster]["centroid"][1]
			if centx > xlims[0] and centx < xlims[1] and centy > ylims[0] and centy < ylims[1]:
				if plot_clusters:
					cx,cy,ct = clusterdict[cluster]["centroid"]
					comp = clusterdict[cluster]["composition"]
					if cluster_colorby == "time":
						col = twmap(ct/float(acq_time))
					else:
						col = twmap(comp)
			
					# Unfilled polygon
					bx,by = clusterdict[cluster]["area_xy"]
					cl = matplotlib.lines.Line2D(bx,by,c=col,alpha=cluster_alpha,linewidth=cluster_width,linestyle=cluster_linetype,zorder=10000-ct)
					ax0.add_artist(cl)
					# Filled polygon
					if cluster_fill:
						vertices = list(zip(*clusterdict[cluster]["area_xy"]))
						cl = plt.Polygon(vertices,facecolor=col,edgecolor=col,alpha=cluster_alpha,zorder=-ct)
						ax0.add_patch(cl) 
		window['-PROGBAR-'].update_bar(0)				
						
		# Hotspots info
		if plot_hotspots:	
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
			print ("Plotting hotspots of overlapping clusters...")	
			for num,overlap in enumerate(overlappers):
				bar = 100*num/len(overlappers)
				window['-PROGBAR-'].update_bar(bar)
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
		
		
		if plot_colorbar:
			print ("Plotting colorbar...")
			xlims = ax0.get_xlim()
			ylims = ax0.get_ylim()
			x_perc = (xlims[1] - xlims[0])/100
			y_perc = (ylims[1] - ylims[0])/100
			ax0.imshow([[0,1], [0,1]], 
			extent = (xlims[0] + x_perc*2,xlims[0] + x_perc*27,ylims[0] + x_perc*2,ylims[0] + x_perc*4),
			cmap = twmap, 
			interpolation = 'bicubic',
			zorder=1000000
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
		global buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, av_msd
		# MSD for clustered and unclustered detections
		if event == "-M1-":
			print ("Plotting MSD curves...")
			t1=time.time()
			fig1 = plt.figure(1,figsize=(4,4))
			ax1 = plt.subplot(111)
			ax1.cla()
			
			clustindices1 = [x for x in clustindices if x < indices1]
			clustindices2 = [x for x in clustindices if x > indices1]
			unclustindices1 = [x for x in unclustindices if x < indices1]
			unclustindices2 = [x for x in unclustindices if x > indices1]			
		
			clust_msds1 = [seldict[x]["msds"] for x in clustindices1]
			unclust_msds1 = [seldict[x]["msds"] for x in unclustindices1]			
			clust_msds2 = [seldict[x]["msds"] for x in clustindices2]
			unclust_msds2 = [seldict[x]["msds"] for x in unclustindices2]
			clust_vals1 = []
			clust_vals2 = []
			unclust_vals1 = []
			unclust_vals2 = []
			for i in range(minlength-1):
				clust_vals1.append([])
				clust_vals2.append([])
				unclust_vals1.append([])
				unclust_vals2.append([])
				[clust_vals1[i].append(x[i]) for x in clust_msds1 if x[i] == x[i]]# don't append NaNs
				[clust_vals2[i].append(x[i]) for x in clust_msds2 if x[i] == x[i]]# don't append NaNs
				[unclust_vals1[i].append(x[i]) for x in unclust_msds1 if x[i] == x[i]]
				[unclust_vals2[i].append(x[i]) for x in unclust_msds2 if x[i] == x[i]]
			clust_av1 = [np.average(x) for x in clust_vals1]	
			clust_sem1 = [np.std(x)/math.sqrt(len(x)) for x in clust_vals1]
			unclust_av1 = [np.average(x) for x in unclust_vals1]	
			unclust_sem1 = [np.std(x)/math.sqrt(len(x)) for x in unclust_vals1]
			
			clust_av2 = [np.average(x) for x in clust_vals2]	
			clust_sem2 = [np.std(x)/math.sqrt(len(x)) for x in clust_vals2]
			unclust_av2 = [np.average(x) for x in unclust_vals2]	
			unclust_sem2 = [np.std(x)/math.sqrt(len(x)) for x in unclust_vals2]	
	
			msd_times = [frame_time*x for x in range(1,minlength,1)]	
			ax1.scatter(msd_times,clust_av1,s=10,c=line_color)
			ax1.errorbar(msd_times,clust_av1,clust_sem1,c=line_color,label="Col 1 Clustered: {}".format(len(clust_msds1)),capsize=5)
			ax1.scatter(msd_times,unclust_av1,s=10,c=line_color)
			ax1.errorbar(msd_times,unclust_av1,unclust_sem1,c=line_color,linestyle="dotted",label="Col 1 Unclustered: {}".format(len(unclust_msds1)),capsize=5)
			
			ax1.scatter(msd_times,clust_av2,s=10,c=line_color2)
			ax1.errorbar(msd_times,clust_av2,clust_sem1,c=line_color2,label="Col 2 Clustered: {}".format(len(clust_msds2)),capsize=5)
			ax1.scatter(msd_times,unclust_av2,s=10,c=line_color2)
			ax1.errorbar(msd_times,unclust_av2,unclust_sem2,c=line_color2,linestyle="dotted",label="Col 2 Unclustered: {}".format(len(unclust_msds2)),capsize=5)
			
			ax1.legend()
			plt.xlabel("Time (s)")
			plt.ylabel(u"MSD (μm²)")
			plt.tight_layout()
			fig1.canvas.manager.set_window_title('MSD Curves')
			plt.show(block=False)
			
			print(reduce(lambda x, y: str(x) + "\t" + str(y), ["TIME (S):"] + msd_times))
			print(reduce(lambda x, y: str(x) + "\t" + str(y), ["COL 1 UNCLUST MSD (um^2):"] + unclust_av1))
			print(reduce(lambda x, y: str(x) + "\t" + str(y), ["COL 1 UNCLUST SEM:"] + unclust_sem1))
			print(reduce(lambda x, y: str(x) + "\t" + str(y), ["COL 1 CLUST MSD (um^2):"] + clust_av1))
			print(reduce(lambda x, y: str(x) + "\t" + str(y), ["COL 1 CLUST SEM:"] + clust_sem1))

			print(reduce(lambda x, y: str(x) + "\t" + str(y), ["TIME (S):"] + msd_times))
			print(reduce(lambda x, y: str(x) + "\t" + str(y), ["COL 2 UNCLUST MSD (um^2):"] + unclust_av2))
			print(reduce(lambda x, y: str(x) + "\t" + str(y), ["COL 2 UNCLUST SEM:"] + unclust_sem2))
			print(reduce(lambda x, y: str(x) + "\t" + str(y), ["COL 2 CLUST MSD (um^2):"] + clust_av2))
			print(reduce(lambda x, y: str(x) + "\t" + str(y), ["COL 2 CLUST SEM:"] + clust_sem2))				
	
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
			#ax5.set_ylim(0,1)
			
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
			metrics_array = []
			col_array = []
			twmap,twmap_s = custom_colormap([line_color,"orange",line_color2],9)
			for num in clusterdict:
				traj_num=clusterdict[num]["traj_num"] # number of trajectories in this cluster
				lifetime = clusterdict[num]["lifetime"]  # lifetime of this cluster (sec)
				av_msd = clusterdict[num]["av_msd"] # Average trajectory MSD in this cluster
				area = clusterdict[num]["area"] # Use internal hull area as cluster area (um2)
				radius = clusterdict[num]["radius"] # cluster radius um
				density = clusterdict[num]["density"] # trajectories/um2
				rate = clusterdict[num]["rate"] # accumulation rate (trajectories/sec)
				composition = clusterdict[num]["composition"] # color composition
				clustarray = [traj_num,lifetime,av_msd,area,radius,density,rate]	
				metrics_array.append(clustarray)
				col_array.append(twmap(composition))
			# Normalise each column	
			metrics_array = list(zip(*metrics_array))
			metrics_array = [normalize(x) for x in metrics_array]
			metrics_array = list(zip(*metrics_array))			
			mapdata = decomposition.TruncatedSVD(n_components=3).fit_transform(metrics_array) 
			#mapdata = manifold.Isomap(len(metrics_array)-1,6).fit_transform(np.array(metrics_array))
			fig3 =plt.figure(3,figsize=(4,4))			
			ax6 = plt.subplot(111,projection='3d')
			ax6.scatter(mapdata[:, 0], mapdata[:, 1],mapdata[:, 2],c=col_array)
			ax6.set_xticks([])
			ax6.set_yticks([])
			ax6.set_zticks([])
			ax6.set_xlabel('Dimension 1')
			ax6.set_ylabel('Dimension 2')
			ax6.set_zlabel('Dimension 3')
			plt.tight_layout()
			fig3.canvas.manager.set_window_title('PCA- all metrics')
			plt.show(block=False)	
			# Pickle
			buf3 = io.BytesIO()
			pickle.dump(ax6, buf3)
			buf3.seek(0)

		# 3D plot
		if event == "-M4-":	
			print ("3D [x,y,t] plot of trajectories...")
			if cluster_colorby == "composition":
				twmap,twmap_s = custom_colormap([line_color,"orange",line_color2],9)
			else:
				twmap = cmap
			
			t1 = time.time()
			fig4 =plt.figure(4,figsize=(8,8))
			ax7 = plt.subplot(111,projection='3d')
			ax7.cla()
			xcent = []
			ycent = []
			tcent = []
			xlims = ax0.get_xlim()
			ylims = ax0.get_ylim()
			for num,traj in enumerate(unclustindices):
				if num%10 == 0:
					bar = 100*num/(len(unclustindices)-1)
					window['-PROGBAR-'].update_bar(bar)
				centx=seldict[traj]["centroid"][0]
				centy=seldict[traj]["centroid"][1]
				centt=seldict[traj]["centroid"][2]
				if centx > xlims[0] and centx < xlims[1] and centy > ylims[0] and centy < ylims[1] and  centt>tmin and centt < tmax:
					x,t,y=zip(*seldict[traj]["points"])
					col = line_color
					if traj > indices1:
						col = line_color2

					tr = art3d.Line3D(x,y,t,c=col,alpha=line_alpha,linewidth=line_width,zorder=acq_time - np.average(y))

					if plot_centroids:
						xcent.append(centx)
						ycent.append(centy)	
						tcent.append(centt)	
					ax7.add_artist(tr) 
					
			for num,traj in enumerate(clustindices):
				if num%10 == 0:
					bar = 100*num/(len(clustindices)-1)
					window['-PROGBAR-'].update_bar(bar)
				centx=seldict[traj]["centroid"][0]
				centy=seldict[traj]["centroid"][1]
				centt = seldict[traj]["centroid"][2]
				if centx > xlims[0] and centx < xlims[1] and centy > ylims[0] and centy < ylims[1] and  centt>tmin and centt < tmax:
					x,t,y=zip(*seldict[traj]["points"])
					col = line_color
					if traj > indices1:
						col = line_color2
						
					tr = art3d.Line3D(x,y,t,c=col,alpha=line_alpha,linewidth=line_width,zorder=acq_time - np.average(y))		

					if plot_centroids:
						xcent.append(centx)
						ycent.append(centy)	
						tcent.append(centt)	
					ax7.add_artist(tr) 	
			ax7.scatter(xcent,tcent,ycent,c=centroid_color,alpha=centroid_alpha,s=centroid_size,linewidth=0)		
					
			if plot_clusters:		
				for cluster in clusterdict:
					bar = 100*cluster/(len(clusterdict))
					window['-PROGBAR-'].update_bar(bar)
					centx=clusterdict[cluster]["centroid"][0]
					centy=clusterdict[cluster]["centroid"][1]
					if centx > xlims[0] and centx < xlims[1] and centy > ylims[0] and centy < ylims[1]:
						cx,cy,ct = clusterdict[cluster]["centroid"]
						comp = clusterdict[cluster]["composition"]
						if cluster_colorby == "time":
							col = twmap(ct/float(acq_time))
						else:
							col = twmap(comp)
						bx,by = clusterdict[cluster]["area_xy"]
						bt = [ct for x in bx]
						cl = art3d.Line3D(bx,bt,by,c=col,alpha=cluster_alpha,linewidth=cluster_width,linestyle=cluster_linetype,zorder=acq_time - ct)
						ax7.add_artist(cl)

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
			#plt.title("3D plot")
			plt.tight_layout()	
			plt.show(block=False)
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
			#ax8 = plt.subplot(111,sharex=ax0,sharey=ax0)
			ax8 = plt.subplot(111)				
			ax8.cla()
			ax8.set_facecolor("k")	
			xlims = ax0.get_xlim()
			ylims = ax0.get_ylim()
			allpoints = [point[:2]  for i in seldict for point in seldict[i]["points"]] # All detection points 
			allpoints = [i for i in allpoints if i[0] > xlims[0] and i[0] < xlims[1] and i[1] > ylims[0] and i[1] < ylims[1]] # Detection points within zoom 
			kde_method = 0.10 # density estimation method. Larger for smaller amounts of data (0.05 - 0.15 should be ok)
			kde_res = 0.7 # resolution of density map (0.5-0.9). Larger = higher resolution
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
			#plt.title("2D KDE")
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
			#ax9 = plt.subplot(111,sharex=ax0,sharey=ax0)	
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
					bar = 100*num/(len(allindices)-1)
					window['-PROGBAR-'].update_bar(bar)
				centx=seldict[traj]["centroid"][0]
				centy=seldict[traj]["centroid"][1]
				if centx > xlims[0] and centx < xlims[1] and centy > ylims[0] and centy < ylims[1]:
					x,y,t=zip(*seldict[traj]["points"])
					diffcoeff  = abs(seldict[traj]["diffcoeff"])
					dcnorm = (math.log(diffcoeff,10)-mindiffcoeff)/dcrange # normalise color 0-1  
					col = cmap3(dcnorm)
					tr = matplotlib.lines.Line2D(x,y,c=col,alpha=0.75,linewidth=line_width,zorder=1-dcnorm)
					ax9.add_artist(tr) 
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
			plt.tight_layout()	
			plt.show(block=False)	

			# DIFF COEFF TIME PLOT		
			fig7 =plt.figure(7,figsize=(6,4))
			ax10 = plt.subplot(311)
			ax11 = plt.subplot(312,sharex=ax10,sharey=ax10)	
			ax12 = plt.subplot(313,sharex=ax10,sharey=ax10)	
			ax10.cla()
			ax10.set_facecolor("k")	
			ax11.cla()
			ax11.set_facecolor("k")		
			ax12.cla()
			ax12.set_facecolor("k")					
			clustcols = []
			diffcols = []
			cols = []
			times = []
			for num,traj in enumerate(clustindices): 
				if num%10 == 0:
					bar = 100*num/(len(clustindices)-1)
					window['-PROGBAR-'].update_bar(bar)
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
					if traj < indices1:
						col = line_color
					else:
						col = line_color2					
					cols.append(col)
					
					
			for num,traj in enumerate(unclustindices): 
				if num%10 == 0:
					bar = 100*num/(len(unclustindices)-1)
					window['-PROGBAR-'].update_bar(bar)
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
					if traj < indices1:
						col = line_color
					else:
						col = line_color2					
					cols.append(col)					

			for i,t in enumerate(times):
				ax10.axvline(t,linewidth=1.5,c=clustcols[i],alpha = 0.75)
				ax11.axvline(t,linewidth=1.5,c=diffcols[i],alpha = 0.75)
				ax12.axvline(t,linewidth=1.5,c=cols[i],alpha = 0.75)

			ax10.set_ylabel("Cluster")
			ax11.set_ylabel("D Coeff")				
			ax12.set_ylabel("Color")				
			ax12.set_xlabel("time (s)")	
			ax10.tick_params(axis = "both",left = False, labelleft = False,bottom=False,labelbottom=False)
			ax11.tick_params(axis = "both",left = False, labelleft = False,bottom=False,labelbottom=False)
			ax12.tick_params(axis = "both",left = False, labelleft = False)			
			plt.tight_layout()	
			plt.show(block=False)	
			
			#plt.title("Diffusion coefficient")			

			t2=time.time()
			# Pickle
			buf6 = io.BytesIO()
			pickle.dump(ax9, buf6)
			buf6.seek(0)
			
			buf7 = io.BytesIO()
			pickle.dump(fig7, buf7)
			buf7.seek(0)			
			print ("Plots completed in {} sec".format(round(t2-t1,3)))	
		
		# 2 color stuff	
		if event == "-M7-":	
			print ("Two color metrics...")		

			t1 = time.time()		
			allcomp = []
			for cluster in clusterdict: 
				comp = clusterdict[cluster]["composition"]
				allcomp.append(comp)

			fig8 =plt.figure(8,figsize=(4,4))
			ax12 = plt.subplot(111)

			twmap,twmap_s = custom_colormap([line_color,"orange",line_color2],9)

			bin_edges = np.histogram_bin_edges(allcomp,bins=10)
			dist,bins =np.histogram(allcomp,bin_edges)
			dist = [float(x)/sum(dist) for x in dist]
			bin_centers = 0.5*(bins[1:]+bins[:-1])
	
			for i in range(len(dist)-1):
				ax12.plot((bin_centers[i],bin_centers[i+1]),(dist[i],dist[i+1]),c = twmap(bin_centers[i]))	
				
			#ax12.plot(bin_centers,dist,c="royalblue")
			plt.ylabel("Frequency")
			plt.xlabel("Proportion of col 2")
			#plt.title("Detection density over time")
			plt.tight_layout()	
			plt.show(block=False)
			t2=time.time()
			
			buf8 = io.BytesIO()
			pickle.dump(fig8, buf8)
			buf8.seek(0)			
			print ("Plot completed in {} sec".format(round(t2-t1,3)))				
			
		# Save metrics	
		if event == "-SAVEANALYSES-":	
			stamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now()) # datestamp
			outpath = os.path.dirname(infilename)
			outdir = outpath + "/" + infilename.split("/")[-1].replace(".trxyt","") + "_" +infilename2.split("/")[-1].replace(".trxyt","_boosh2c_{}".format(stamp))
			os.mkdir(outdir)
			os.chdir(outdir)
			
			outfilename = "{}/metrics.tsv".format(outdir)
			print ("Saving metrics, ROIs and all plots to {}...".format(outdir))
			# Metrics
			with open(outfilename,"w") as outfile:
				outfile.write("BOOSH2C: NANOSCALE SPATIO TEMPORAL DBSCAN CLUSTERING (2 COLOR) - Tristan Wallis t.wallis@uq.edu.au\n") 
				outfile.write("TRAJECTORY FILE 1:\t{}\n".format(infilename))	
				outfile.write("TRAJECTORY FILE 2:\t{}\n".format(infilename2))	
				outfile.write("ANALYSED:\t{}\n".format(stamp))
				outfile.write("TRAJECTORY LENGTH CUTOFFS (steps):\t{} - {}\n".format(minlength,maxlength))	
				outfile.write("EPSILON (s):\t{}\n".format(epsilon))
				outfile.write("MINPTS:\t{}\n".format(minpts))			
				outfile.write("TIME WINDOW (s):\t{}\n".format(timewindow))
				if msd_filter:
					outfile.write("MSD FILTER THRESHOLD (um^2):\t{}\n".format(av_msd))
				else:
					outfile.write("MSD FILTER THRESHOLD (um^2):\tNone\n")
				outfile.write("CLUSTER MAX RADIUS (um):\t{}\n".format(radius_thresh))	
				outfile.write("SELECTION AREA (um^2):\t{}\n".format(sum(all_selareas)))
				outfile.write("SELECTED TRAJECTORIES:\t{}\n".format(len(allindices)))
				outfile.write("CLUSTERED TRAJECTORIES:\t{}\n".format(len(clustindices)))
				outfile.write("UNCLUSTERED TRAJECTORIES:\t{}\n".format(len(unclustindices)))
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
				clustindices1 = [x for x in clustindices if x < indices1]
				clustindices2 = [x for x in clustindices if x > indices1]
				unclustindices1 = [x for x in unclustindices if x < indices1]
				unclustindices2 = [x for x in unclustindices if x > indices1]			
			
				clust_msds1 = [seldict[x]["msds"] for x in clustindices1]
				unclust_msds1 = [seldict[x]["msds"] for x in unclustindices1]			
				clust_msds2 = [seldict[x]["msds"] for x in clustindices2]
				unclust_msds2 = [seldict[x]["msds"] for x in unclustindices2]
				clust_vals1 = []
				clust_vals2 = []
				unclust_vals1 = []
				unclust_vals2 = []
				for i in range(minlength-1):
					clust_vals1.append([])
					clust_vals2.append([])
					unclust_vals1.append([])
					unclust_vals2.append([])
					[clust_vals1[i].append(x[i]) for x in clust_msds1 if x[i] == x[i]]# don't append NaNs
					[clust_vals2[i].append(x[i]) for x in clust_msds2 if x[i] == x[i]]# don't append NaNs
					[unclust_vals1[i].append(x[i]) for x in unclust_msds1 if x[i] == x[i]]
					[unclust_vals2[i].append(x[i]) for x in unclust_msds2 if x[i] == x[i]]
				clust_av1 = [np.average(x) for x in clust_vals1]	
				clust_sem1 = [np.std(x)/math.sqrt(len(x)) for x in clust_vals1]
				unclust_av1 = [np.average(x) for x in unclust_vals1]	
				unclust_sem1 = [np.std(x)/math.sqrt(len(x)) for x in unclust_vals1]
				
				clust_av2 = [np.average(x) for x in clust_vals2]	
				clust_sem2 = [np.std(x)/math.sqrt(len(x)) for x in clust_vals2]
				unclust_av2 = [np.average(x) for x in unclust_vals2]	
				unclust_sem2 = [np.std(x)/math.sqrt(len(x)) for x in unclust_vals2]	
		
				msd_times = [frame_time*x for x in range(1,minlength,1)]	

				outfile.write(reduce(lambda x, y: str(x) + "\t" + str(y), ["TIME (S):"] + msd_times)+"\n")
				outfile.write(reduce(lambda x, y: str(x) + "\t" + str(y), ["COL 1 UNCLUST MSD (um^2):"] + unclust_av1)+"\n")
				outfile.write(reduce(lambda x, y: str(x) + "\t" + str(y), ["COL 1 UNCLUST SEM:"] + unclust_sem1)+"\n")
				outfile.write(reduce(lambda x, y: str(x) + "\t" + str(y), ["COL 1 CLUST MSD (um^2):"] + clust_av1)+"\n")
				outfile.write(reduce(lambda x, y: str(x) + "\t" + str(y), ["COL 1 CLUST SEM:"] + clust_sem1)+"\n")

				outfile.write(reduce(lambda x, y: str(x) + "\t" + str(y), ["COL 2 UNCLUST MSD (um^2):"] + unclust_av2)+"\n")
				outfile.write(reduce(lambda x, y: str(x) + "\t" + str(y), ["COL 2 UNCLUST SEM:"] + unclust_sem2)+"\n")
				outfile.write(reduce(lambda x, y: str(x) + "\t" + str(y), ["COL 2 CLUST MSD (um^2):"] + clust_av2)+"\n")
				outfile.write(reduce(lambda x, y: str(x) + "\t" + str(y), ["COL 2 CLUST SEM:"] + clust_sem2)+"\n")				
				

				# INDIVIDUAL CLUSTER METRICS
				outfile.write("\nINDIVIDUAL CLUSTER METRICS:\n")
				outfile.write("CLUSTER\tMEMBERSHIP\tLIFETIME (s)\tAVG MSD (um^2)\tAREA (um^2)\tRADIUS (um)\tDENSITY (traj/um^2)\tRATE (traj/sec)\tAVG TIME (s)\tCOL 2\n")
				trajnums = []
				lifetimes = []
				times = []
				av_msds = []
				areas = []
				radii = []
				densities = []
				rates = []
				compositions = []
				
				for num in clusterdict:
					traj_num=clusterdict[num]["traj_num"] # number of trajectories in this cluster
					lifetime = clusterdict[num]["lifetime"]  # lifetime of this cluster (sec)
					av_msd = clusterdict[num]["av_msd"] # Average trajectory MSD in this cluster
					area = clusterdict[num]["area"] # Use internal hull area as cluster area (um2)
					radius = clusterdict[num]["radius"] # cluster radius um
					density = clusterdict[num]["density"] # trajectories/um2
					rate = clusterdict[num]["rate"] # accumulation rate (trajectories/sec)
					composition = clusterdict[num]["composition"] # 0 = all col1, 1 = all col2
					clusttime = clusterdict[num]["centroid"][2] # Time centroid of this cluster 
					outarray = [num,traj_num,lifetime,av_msd,area,radius,density,rate,clusttime,composition]
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
					compositions.append(composition)
				# AVERAGE CLUSTER METRICS	
				outarray = ["AVG",np.average(trajnums),np.average(lifetimes),np.average(av_msds),np.average(areas),np.average(radii),np.average(densities),np.average(rates),np.average(times),np.average(compositions)]
				outstring = reduce(lambda x, y: str(x) + "\t" + str(y), outarray)
				outfile.write(outstring + "\n")	
				# SEMS
				outarray = ["SEM",np.std(trajnums)/math.sqrt(len(trajnums)),np.std(lifetimes)/math.sqrt(len(lifetimes)),np.std(av_msds)/math.sqrt(len(av_msds)),np.std(areas)/math.sqrt(len(areas)),np.std(radii)/math.sqrt(len(radii)),np.std(densities)/math.sqrt(len(densities)),np.std(rates)/math.sqrt(len(rates)),np.std(times)/math.sqrt(len(times)),np.std(compositions)/math.sqrt(len(compositions))]
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
					roi_save = "{}/{}_roi_coordinates{}.tsv".format(roi_directory,stamp, roi)
					with open(roi_save,"w") as outfile:
						outfile.write("ROI\tx(um)\ty(um)\n")			
						for coord in selverts:
							outfile.write("{}\t{}\t{}\n".format(roi,coord[0],coord[1]))	
			
			# Plots	
			buf.seek(0)		
			fig10=pickle.load(buf)
			for selverts in all_selverts:			
				vx,vy = list(zip(*selverts))
				plt.plot(vx,vy,linewidth=2,c="orange",alpha=1)
			plt.savefig("{}/raw_acquisition.png".format(outdir),dpi=300)
			plt.close()
			try:
				buf0.seek(0)
				fig10=pickle.load(buf0)
				plt.savefig("{}/main_plot.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass		
			try:
				buf1.seek(0)
				fig10=pickle.load(buf1)
				plt.savefig("{}/MSD.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass
			try:
				buf2.seek(0)
				fig10=pickle.load(buf2)
				plt.savefig("{}/overlap.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass	
			try:
				buf3.seek(0)
				fig10=pickle.load(buf3)
				plt.savefig("{}/pca.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass
			try:
				buf4.seek(0)
				fig10=pickle.load(buf4)
				plt.savefig("{}/3d_trajectories.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass	
			try:
				buf5.seek(0)
				fig10=pickle.load(buf5)
				plt.savefig("{}/KDE.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass	
			try:
				buf6.seek(0)
				fig10=pickle.load(buf6)
				plt.savefig("{}/diffusion_coefficient.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass	
			try:
				buf7.seek(0)
				fig10=pickle.load(buf7)
				plt.savefig("{}/diffusion_coefficient_1d.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass		
			try:
				buf8.seek(0)
				fig10=pickle.load(buf8)
				plt.savefig("{}/2col_proportion.png".format(outdir),dpi=300)
				plt.close()
			except:
				pass	
				
			print ("All data saved")	
		return

	# GET INITIAL VALUES FOR GUI
	cwd = os.path.dirname(os.path.abspath(__file__))
	os.chdir(cwd)
	initialdir = cwd
	if os.path.isfile("boosh2c_gui.defaults"):
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
	tab1_layout = [
		[sg.FileBrowse(tooltip = "Select a TRXYT file to analyse\nEach line must only contain 4 space separated values\nTrajectory X-position Y-position Time",file_types=(("Trajectory Files", "*.trxyt"),),key="-INFILE-",initial_folder=initialdir),sg.Input("Select trajectory TRXYT file 1", key ="-FILENAME-",enable_events=True,size=(55,1))],
		[sg.FileBrowse(tooltip = "Select a second TRXYT file to analyse\nEach line must only contain 4 space separated values\nTrajectory X-position Y-position Time",file_types=(("Trajectory Files", "*.trxyt"),),key="-INFILE2-",initial_folder=initialdir),sg.Input("Select trajectory TRXYT file 2", key ="-FILENAME2-",enable_events=True,size=(55,1))],		
		[sg.T('Minimum trajectory length:',tooltip = "Trajectories must contain at least this many steps"),sg.InputText(minlength,size="50",key="-MINLENGTH-")],
		[sg.T('Maximum trajectory length:',tooltip = "Trajectories must contain fewer steps than this"),sg.InputText(maxlength,size="50",key="-MAXLENGTH-")],
		[sg.T('Probability:',tooltip = "Probability of displaying a trajectory\n1 = all trajectories\nIMPORTANT: only affects display of trajectories,\nundisplayed trajectories can still be selected"),sg.Combo([0.01,0.05,0.1,0.25,0.5,0.75,1.0],default_value=traj_prob,key="-TRAJPROB-")],
		[sg.T('Detection opacity:',tooltip = "Transparency of detection points\n1 = fully opaque"),sg.Combo([0.01,0.05,0.1,0.25,0.5,0.75,1.0],default_value=detection_alpha,key="-DETECTIONALPHA-")],
		[sg.B('PLOT RAW DETECTIONS',size=(25,2),button_color=("white","gray"),key ="-PLOTBUTTON-",disabled=True,tooltip = "Visualise the trajectory detections using the above parameters.\nOnce visualised you may select regions of interest.\nThis button will close any other plot windows.")]
	]

	tab2_layout = [
		[sg.FileBrowse("Load",file_types=(("ROI Files", "roi_coordinates*.tsv *rgn"),),key="-R1-",target="-R2-",disabled=True),sg.In("Load previously defined ROIs",key ="-R2-",enable_events=True, size = (30,1)),sg.T("Pixel(um):", key = '-PIXEL_TEXT-', tooltip = "Please select a conversion factor\nfor converting pixels to um", visible = False), sg.In(pixel, key = '-PIXEL-', visible = False, size = (6,1)),sg.B("Replot ROIs", key = "-REPLOT_ROI-", visible = False)], 
		[sg.B("Save",key="-R8-",disabled=True),sg.T("Save currently defined ROIs"), sg.B("Save Separately", key = "-SEPARATE-", disabled = True), sg.T("Save individual ROI files")],
		[sg.B("Clear",key="-R3-",disabled=True),sg.T("Clear all ROIs")],	
		[sg.B("All",key="-R4-",disabled=True),sg.T("ROI encompassing all detections")],
		[sg.B("Add",key="-R5-",disabled=True),sg.T("Add selected ROI")],
		[sg.B("Remove",key="-R6-",disabled=True),sg.T("Remove last added ROI")],
		[sg.B("Undo",key="-R7-",disabled=True),sg.T("Undo last change"),sg.B("Reset",key="-RESET-",disabled=True),sg.T("Reset to original view with ROI")], 
		[sg.T('Selection density:',tooltip = "Screen out random trajectories to maintain a \nfixed density of selected trajectories (traj/um^2)\n0 = do not adjust density"),sg.InputText(selection_density,size="50",key="-SELECTIONDENSITY-"),sg.T("",key = "-DENSITY-",size=(6,1))],
		[sg.Checkbox("Balance colors",tooltip = "Screen out random trajectories to ensure that \nboth colors have the same number of trajectories",key = "-BALANCE-",default=balance)],
		[sg.B('SELECT DATA IN ROIS',size=(25,2),button_color=("white","gray"),key ="-SELECTBUTTON-",disabled=True,tooltip = "Select trajectories whose detections lie within the yellow ROIs\nOnce selected the ROIs will turn green.\nSelected trajectories may then be clustered."),sg.Checkbox("Cluster immediately",key="-AUTOCLUSTER-",default=autocluster,tooltip="Switch to 'Clustering' tab and begin clustering automatically\nupon selection of data within ROIs")]
	]

	tab3_layout = [
		[sg.T('Acquisition time (s):',tooltip = "Length of the acquisition (s)"),sg.InputText(acq_time,size="50",key="-ACQTIME-")],
		[sg.T('Frame time (s):',tooltip = "Time between frames (s)"),sg.InputText(frame_time,size="50",key="-FRAMETIME-")],
		[sg.T('Epsilon (um):',tooltip = "Spatial radius around each centroid\n to check for other centroids"),sg.InputText(epsilon,size="50",key="-EPSILON-")],	
		[sg.T('MinPts:',tooltip = "Clusters must contain at least this\n many centroids within Epsilon"),sg.InputText(minpts,size="50",key="-MINPTS-")],
		[sg.T('Time window (s):',tooltip = "Temporal radius (s) around each centroid\n to check for other centroids"),sg.InputText(timewindow,size="50",key="-TIMEWINDOW-")],	
		[sg.T('Cluster size screen (um):',tooltip = "Clusters with a radius larger than this (um)are ignored"),sg.InputText(radius_thresh,size="50",key="-RADIUSTHRESH-")],	
			[sg.Checkbox('MSD screen',tooltip = "Don't analyse trajectories with MSD > \nthe average MSD of all trajectories",key = "-MSDFILTER-",default=msd_filter)],
		[sg.B('CLUSTER SELECTED DATA',size=(25,2),button_color=("white","gray"),key ="-CLUSTERBUTTON-",disabled=True, tooltip = "Perform spatiotemporal indexing clustering on the selected trajectories.\nIdentified clusters may then be displayed."),sg.Checkbox("Plot immediately",key="-AUTOPLOT-",default=autoplot,tooltip ="Switch to 'Display' tab and begin plotting automatically\nupon clustering of selected trajectories")],
	]

	trajectory_layout = [
		[sg.T("Width",tooltip = "Width of plotted trajectory lines"),sg.Combo([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],default_value= line_width,key="-LINEWIDTH-")],
		[sg.T("Opacity",tooltip = "Opacity of plotted trajectory lines"),sg.Combo([0.01,0.05,0.1,0.25,0.5,0.75,1.0],default_value= line_alpha,key="-LINEALPHA-")],
		[sg.T("Color 1",tooltip = "Trajectory color 1"),sg.ColorChooserButton("Choose",key="-LINECOLORCHOOSE-",target="-LINECOLOR-",button_color=("gray",line_color),disabled=True),sg.Input(line_color,key ="-LINECOLOR-",enable_events=True,visible=False)],
		[sg.T("Color 2",tooltip = "Trajectory color 2"),sg.ColorChooserButton("Choose",key="-LINECOLORCHOOSE2-",target="-LINECOLOR2-",button_color=("gray",line_color2),disabled=True),sg.Input(line_color2,key ="-LINECOLOR2-",enable_events=True,visible=False)]	
	]

	centroid_layout = [
		[sg.T("Size",tooltip = "Size of plotted trajectory centroids"),sg.Combo([1,2,5,10,20,50],default_value= centroid_size,key="-CENTROIDSIZE-")],
		[sg.T("Opacity",tooltip = "Opacity of plotted trajectory lines"),sg.Combo([0.01,0.05,0.1,0.25,0.5,0.75,1.0],default_value= centroid_alpha,key="-CENTROIDALPHA-")],
		[sg.T("Color",tooltip = "Trajectory color"),sg.ColorChooserButton("Choose",key="-CENTROIDCOLORCHOOSE-",target="-CENTROIDCOLOR-",button_color=("gray",centroid_color),disabled=True),sg.Input(centroid_color,key ="-CENTROIDCOLOR-",enable_events=True,visible=False)]
	]

	cluster_layout = [	
		[sg.T("Color by",tooltip = "Color clusters by their average time\nor by the proportion of each molecule"),sg.Combo(["time","composition"],default_value= cluster_colorby,key="-CLUSTERCOLORBY-")],
		[sg.T("Opacity",tooltip = "Opacity of plotted clusters"),sg.Combo([0.1,0.25,0.5,0.75,1.0],default_value= cluster_alpha,key="-CLUSTERALPHA-"),sg.Checkbox('Filled',tooltip = "Display clusters as filled polygons",key = "-CLUSTERFILL-",default=cluster_fill)],
		[sg.T("Line width",tooltip = "Width of plotted cluster lines"),sg.Combo([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],default_value= cluster_width,key="-CLUSTERWIDTH-")],
		[sg.T("Line type",tooltip = "Cluster line type"),sg.Combo(["solid","dashed","dotted"],default_value =cluster_linetype,key="-CLUSTERLINETYPE-")]
	]
	
	hotspot_layout = [	
		[sg.T("Radius",tooltip = "Clusters within this multiple of the \naverage cluster radius"),sg.Combo([0.1,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0],default_value= hotspot_radius,key="-HOTSPOTRADIUS-")],
		[sg.T("Opacity",tooltip = "Opacity of plotted hotspots"),sg.Combo([0.1,0.25,0.5,0.75,1.0],default_value= hotspot_alpha,key="-HOTSPOTALPHA-")],
		[sg.T("Line width",tooltip = "Width of plotted hotspot lines"),sg.Combo([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],default_value= hotspot_width,key="-HOTSPOTWIDTH-")],
		[sg.T("Line type",tooltip = "Hotspot line type"),sg.Combo(["solid","dashed","dotted"],default_value =hotspot_linetype,key="-HOTSPOTLINETYPE-")],
		[sg.T("Color",tooltip = "Hotspot color"),sg.ColorChooserButton("Choose",key="-HOTSPOTCOLORCHOOSE-",target="-HOTSPOTCOLOR-",button_color=("gray",hotspot_color),disabled=True),sg.Input(hotspot_color,key ="-HOTSPOTCOLOR-",enable_events=True,visible=False)]
	]	

	export_layout = [
		[sg.T("Format",tooltip = "Format of saved figure"),sg.Combo(["eps","pdf","png","ps","svg"],default_value= saveformat,key="-SAVEFORMAT-"),sg.Checkbox('Transparent background',tooltip = "Useful for making figures",key = "-SAVETRANSPARENCY-",default=False)],
		[sg.T("DPI",tooltip = "Resolution of saved figure"),sg.Combo([50,100,300,600,1200],default_value=savedpi,key="-SAVEDPI-")],
		[sg.T("Directory",tooltip = "Directory for saved figure"),sg.FolderBrowse("Choose",key="-SAVEFOLDERCHOOSE-",target="-SAVEFOLDER-"),sg.Input(key="-SAVEFOLDER-",enable_events=True,size=(43,1))]
	]

	tab4_layout = [
		[sg.T('Canvas',tooltip = "Background color of plotted data"),sg.Input(canvas_color,key ="-CANVASCOLOR-",enable_events=True,visible=False),sg.ColorChooserButton("Choose",button_color=("gray",canvas_color),target="-CANVASCOLOR-",key="-CANVASCOLORCHOOSE-",disabled=True),sg.Checkbox('Traj.',tooltip = "Plot trajectories",key = "-TRAJECTORIES-",default=plot_trajectories),sg.Checkbox('Centr.',tooltip = "Plot trajectory centroids",key = "-CENTROIDS-",default=plot_centroids),sg.Checkbox('Clust.',tooltip = "Plot cluster boundaries",key = "-CLUSTERS-",default=plot_clusters),sg.Checkbox('Hotsp.',tooltip = "Plot cluster hotspots",key = "-HOTSPOTS-",default=plot_hotspots),sg.Checkbox('Col.bar',tooltip = "Plot colorbar for cluster times\nBlue = 0 sec --> green = full acquisition time\nHit 'Plot clustered data' button to refresh colorbar after a zoom",key = "-COLORBAR-",default=plot_colorbar)],
		[sg.TabGroup([
			[sg.Tab("Trajectory",trajectory_layout)],
			[sg.Tab("Centroid",centroid_layout)],
			[sg.Tab("Cluster",cluster_layout)],
			[sg.Tab("Hotspot",hotspot_layout)],
			[sg.Tab("Export",export_layout)]
			])
		],
		[sg.B('PLOT CLUSTERED DATA',size=(25,2),button_color=("white","gray"),key ="-DISPLAYBUTTON-",disabled=True,tooltip="Plot clustered data using the above parameters.\nHit button again after changing parameters, to replot"),sg.B('SAVE PLOT',size=(25,2),button_color=("white","gray"),key ="-SAVEBUTTON-",disabled=True,tooltip = "Save plot using the above parameters in 'Export options'.\nEach time this button is pressed a new datastamped image will be saved.")],
		[sg.T("Xmin"),sg.InputText(plotxmin,size="3",key="-PLOTXMIN-"),sg.T("Xmax"),sg.InputText(plotxmax,size="3",key="-PLOTXMAX-"),sg.T("Ymin"),sg.InputText(plotymin,size="3",key="-PLOTYMIN-"),sg.T("Ymax"),sg.InputText(plotymax,size="3",key="-PLOTYMAX-"),sg.Checkbox("Metrics immediately",key="-AUTOMETRIC-",default=auto_metric,tooltip ="Switch to 'Metrics' tab after plotting of clustered trajectories")]
	]

	tab5_layout = [
		[sg.B("MSD",key="-M1-",disabled=True),sg.T("Plot clustered vs unclustered MSDs")],
		[sg.B("Hotspot",key="-M2-",disabled=True),sg.T("Plot cluster overlap data")],
		[sg.B("PCA",key="-M3-",disabled=True),sg.T("Multidimensional analysis of cluster metrics")],
		[sg.B("3D",key="-M4-",disabled=True),sg.T("X,Y,T plot of trajectories"),sg.T("Tmin:"),sg.InputText(tmin,size="4",key="-TMIN-",tooltip = "Only plot trajectories whose time centroid is greater than this"),sg.T("Tmax"),sg.InputText(tmax,size="4",key="-TMAX-",tooltip = "Only plot trajectories whose time centroid is less than this"),sg.Checkbox('Axes',tooltip = "Plot axes and grid",key = "-AXES3D-",default=axes_3d)],
		[sg.B("KDE",key="-M5-",disabled=True),sg.T("2D kernel density estimation of all detections (very slow)")],	
		[sg.B("Diffusion coefficient",key="-M6-",disabled=True),sg.T("Instantaneous diffusion coefficient plot of trajectories")],	
		[sg.B("2 color metrics",key="-M7-",disabled=True),sg.T("Specific 2 color clustering metrics")],			
		[sg.B("SAVE ANALYSES",key="-SAVEANALYSES-",size=(25,2),button_color=("white","gray"),disabled=True,tooltip = "Save all analysis metrics, ROIs and plots")]	
	]

	menu_def = [
		['&File', ['&Load settings', '&Save settings','&Default settings','&Exit']],
		['&Info', ['&About', '&Help','&Licence' ]],
	]

	layout = [
		[sg.Menu(menu_def)],
		[sg.T('BOOSH2C',font="Any 20")],
		[sg.TabGroup([
			[sg.Tab("File",tab1_layout)],
			[sg.Tab("ROI",tab2_layout)],
			[sg.Tab("Clustering",tab3_layout)],
			[sg.Tab("Display",tab4_layout)],
			[sg.Tab("Metrics",tab5_layout)]
			],key="-TABGROUP-")
		],
		[sg.ProgressBar(100, orientation='h',size=(53,20),key='-PROGBAR-')],
		#[sg.Output(size=(63,10))]	
	]
	window = sg.Window('boosh2c v{}'.format(last_changed), layout)
	popup.close()

	# VARS
	cmap = matplotlib.cm.get_cmap('brg') # colormap for conditional coloring of clusters based on their average acquisition time
	all_selverts = [] # all ROI vertices
	all_selareas = [] # all ROI areas
	roi_list = [] # ROI artists
	trajdict = {} # Dictionary holding raw trajectory info
	sel_traj = [] # Selected trajectory indices
	lastfile = "" # Force the program to load a fresh TRXYT
	lastfile2 = "" # Force the program to load a fresh TRXYT	
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
		infilename2 = values["-INFILE2-"]			
		minlength = values["-MINLENGTH-"]
		maxlength = values["-MAXLENGTH-"]
		traj_prob = values["-TRAJPROB-"]
		selection_density = values["-SELECTIONDENSITY-"]
		balance = values["-BALANCE-"]
		roi_file = values["-R2-"]
		detection_alpha = values["-DETECTIONALPHA-"]
		acq_time = values["-ACQTIME-"]
		frame_time = values["-FRAMETIME-"]		
		epsilon = values["-EPSILON-"]
		minpts = values["-MINPTS-"]
		timewindow = values["-TIMEWINDOW-"]
		canvas_color = values["-CANVASCOLOR-"]
		plot_trajectories = values["-TRAJECTORIES-"]
		plot_centroids = values["-CENTROIDS-"]
		plot_clusters = values["-CLUSTERS-"]
		plot_hotspots = values["-HOTSPOTS-"]		
		plot_colorbar = values["-COLORBAR-"]	
		line_width = values["-LINEWIDTH-"]
		line_alpha = values["-LINEALPHA-"]
		line_color = values["-LINECOLOR-"]
		line_color2 = values["-LINECOLOR2-"]	
		cluster_colorby = values["-CLUSTERCOLORBY-"]		
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
		autoplot = values["-AUTOPLOT-"]
		autocluster = values["-AUTOCLUSTER-"]
		radius_thresh=values['-RADIUSTHRESH-']
		cluster_fill = values['-CLUSTERFILL-']
		auto_metric = values['-AUTOMETRIC-']
		plotxmin = values['-PLOTXMIN-']
		plotxmax = values['-PLOTXMAX-']
		plotymin = values['-PLOTYMIN-']
		plotymax = values['-PLOTYMAX-']	
		msd_filter = values['-MSDFILTER-']	
		tmin = values['-TMIN-']	
		tmax = values['-TMAX-']	
		hotspot_width = values["-HOTSPOTWIDTH-"]
		hotspot_alpha = values["-HOTSPOTALPHA-"]
		hotspot_linetype = values["-HOTSPOTLINETYPE-"]		
		hotspot_color = values["-HOTSPOTCOLOR-"]
		hotspot_radius = values["-HOTSPOTRADIUS-"]
		axes_3d = values["-AXES3D-"]	
		pixel = values['-PIXEL-']
		
		# Check variables
		check_variables()

		# Exit	
		if event in (sg.WIN_CLOSED, 'Exit'):
			break
			
		# If main display window is closed
		fignums = [x.num for x in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
		if 0 not in fignums:
			sg.popup("Main display window closed!","Reinitialising new window","Please restart your analysis")
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
			lastfile2 = "" # Force the program to load a fresh TRXYT
			seldict = {} # Selected trajectories and metrics
			clusterdict = {} # Cluster information
			
			# Close any other windows
			for i in [1,2,3,4,5,6,7,8,9,10]:
				try:
					plt.close(i)
				except:
					pass

			# Unshare any shared axes

			try:			
				shared = [ax0,ax8]
				shax = shared.get_shared_x_axes()
				shay = shared.get_shared_y_axes()
				shax.remove(shared)
				shay.remove(shared)
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
				"A full helpfile will be added once the program is complete",
				"All buttons have popup tooltips in the mean time!", 
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

		# Read and plot input file	
		if event == '-PLOTBUTTON-':
			trxyt_tab()

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
			all_selverts_copy = [x for x in all_selverts] 
			all_selverts = [] 
			for roi in roi_list: 
				roi.remove()
			roi_list = []
			cluster_tab()

		# Display
		if event ==	"-DISPLAYBUTTON-" and len(clusterdict)>0:
			display_tab(xlims,ylims)
			
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
			
	print ("Exiting...")		
	plt.close('all')				
	window.close()
	quit()