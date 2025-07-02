# -*- coding: utf-8 -*-
'''
BOOSH_CLI
COMMAND LINE (CLI) VERSION FOR SPATIOTEMPORAL CLUSTERING OF MOLECULAR TRAJECTORY DATA USING 3D DBSCAN. TIME CONVERTED TO Z 
THIS VERSION CLUSTERS THE INDIVIDUAL DETECTIONS RATHER THAN TRAJECTORY CENTROIDS

Design and coding: Tristan Wallis
Additional coding: Kyle Young, Alex McCann
Debugging: Sophie Huiyi Hou, Kye Kudo, Alex McCann
Queensland Brain Institute
The University of Queensland
Fred Meunier: f.meunier@uq.edu.au

REQUIRED:
Python 3.8 or greater
python -m pip install scipy numpy scikit-learn 

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

USAGE:
Parameters are adjusted in the # PARAMETERS section below
python boosh_cli.py inputfilename.trxyt

NOTES:
This script has been tested and will run as intended on Windows 7/10/11, Linux, and MacOS.
The script is single threaded and should run on virtual CPUs without issues.
Feedback, suggestions and improvements are welcome. Sanctimonious critiques on the pythonic inelegance of the coding are not.

CHECK FOR UPDATES:
https://github.com/tristanwallis/smlm_clustering/releases
'''

lastchanged = "20250701"

# LOAD MODULES
from argparse import ArgumentParser
import os
from scipy.spatial import ConvexHull
from scipy.stats import variation
from sklearn.cluster import DBSCAN
import numpy as np
import math
from math import dist
import time
import datetime
from functools import reduce
import warnings
warnings.filterwarnings("ignore")

# PARAMETERS
# Trajectory parameters
minlength = 8 # Trajectories must be longer than this to be considered (default 8; value must be >= 5)
maxlength = 1000 # Trajectories must be shorter than this to be considered(default 1000; value must be >= minlength)
frame_time = 0.02 # Time between frames (sec) in original acquisition (default 0.02; value must be >0) 
acq_time = 320 # length of acquisition (sec) (default 320)

# Clustering parameters
epsilon = 0.05 # Radius (um) around each detection to check for detections from other trajectories (default 0.05 um for sptPALM)
timewindow = 10 # (sec) Detections must be within this many seconds of another detection as well as overlapping in space (default 10)
minpts = 3 # This many detections from different trajectories will be considered a cluster (default 3)
radius_thresh = 0.2 # um  - clusters will be excluded if bigger than this. Set to a large number to include all clusters (default 0.2)
msd_filter =True # Trajectories whose MSD at time point 1 are greater than the average will be excluded (default True)

# MSD
all_msds = []
def getmsds(points): 
	msdarray = []
	for i in range(1,minlength,1):
		all_diff_sq = []
		for j in range(0,i):
			msdpoints = points[j::i]
			diff = [dist(msdpoints[k][:2],msdpoints[k-1][:2]) for k in range(1,len(msdpoints))] # displacement 
			diff_sq = np.array(diff)**2 # square displacement
			[all_diff_sq.append(x) for x in diff_sq]
		msd = np.average(all_diff_sq)
		msdarray.append(msd)
	all_msds.append(msdarray[0])
	diffcoeff = (msdarray[3]-msdarray[0]) /(frame_time*3)	
	return msdarray,diffcoeff	

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


# DBSCAN
def dbscan(points,epsilon,minpts):
	db = DBSCAN(eps=epsilon, min_samples=minpts).fit(points)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_ # which sample belongs to which cluster
	clusterlist = list(set(labels)) # list of clusters
	return labels,clusterlist


# MAIN PROGRAM

# Pass infilename from console to python
parser = ArgumentParser()
parser.add_argument("infilename")
args = parser.parse_args()
infilename = args.infilename

# Initial directory
cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd)
initialdir = cwd

# Header (console)
os.system('cls' if os.name == 'nt' else 'clear')
print ("BOOSH CLI - Tristan Wallis {}\n-----------------------------------------------------".format(lastchanged))
print("\nInput file selected: {}\n".format(infilename))

# Variables
rawtrajdict = {}
ct = 99999
x0 = -10000
y0 = -10000

# Reading input file
print ("Reading data into dictionary...")
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
	print("\nAlert: No trajectory information found")
else:
	# Screen trajectories by length	
	print ("Filtering trajectories by length...")
	filttrajdict = {}
	trajdict = {}
	for traj in rawtrajdict:
		points = rawtrajdict[traj]["points"]
		x,y,t = zip(*points)
		if len(points) >=minlength and len(points) <=maxlength and variation(x) > 0.0001 and variation(y) > 0.0001:
			filttrajdict[traj] = rawtrajdict[traj] 
	if len(filttrajdict) == 0:
		print("0 remaining trajectories")
		print("\nAlert: No trajectories remaining after length filtering with step lengths of >{} and <{}".format(minlength,maxlength)) 
		print("\nAlert: Not enough trajectories for clustering")
	elif len(filttrajdict) == 1:
		print("{} raw trajectories, 1 remaining trajectory with step lengths of >{} and <{}".format(len(rawtrajdict),minlength,maxlength)) 
		print("\nAlert: Not enough trajectories for clustering")
	elif len(filttrajdict) >1:
		print ("{} raw trajectories, {} remaining trajectories with step lengths of >{} and <{}".format(len(rawtrajdict),len(filttrajdict),minlength,maxlength))		
			
		
		trajdict = filttrajdict

		# Generate centroids and MSDs
		print ("Calculating trajectory centroids and MSDs...")
		all_x= []
		all_y= []
		for traj in trajdict:
			points = trajdict[traj]["points"]
			x,y,t=list(zip(*points))
			xmean = np.average(x)
			ymean = np.average(y)	
			tmean = np.average(t)
			centroid = [xmean,ymean,tmean]
			trajdict[traj]["centroid"] = centroid
			msds,diffcoeff = getmsds(points)	
			trajdict[traj]["msds"] = msds
			trajdict[traj]["diffcoeff"] = diffcoeff
			[all_x.append(val) for val in x] # use to calculate total area later
			[all_y.append(val) for val in y] # """"

		# Selection area based on extent of selected trajectory detections. Will be um^2 if TRXYT values are um
		selarea = (max(all_x) - min(all_x))*(max(all_y) - min(all_y))
		print ("Area containing selected trajectory detections: {} um^2".format(selarea))

		# MSD filter	
		if msd_filter:
			av_msd = np.average(all_msds)	
		else:
			av_msd =10000

		filt_indices = []
		unfilt_indices = []
		for num in trajdict:
			unfilt_indices.append(num)
			if trajdict[num]["msds"][0] < av_msd:
				filt_indices.append(num)
		print ("{} trajectories passed MSD filter".format(len(filt_indices)))

		# Dictionary of all selected points
		sel_points = []
		pointdict = {}
		ct=0
		for idx in filt_indices:
			points = trajdict[idx]["points"]
			msds = trajdict[idx]["msds"]	
			for point in points:
				pointdict[ct] = {}
				pointdict[ct]["point"] = point # [x,y,t] co-ordinates of point
				pointdict[ct]["traj"]= idx # parent trajectory
				sel_points.append(point)		
				ct+=1

		# Determine clustered points
		print ("Clustering selected trajectories...")
		squash = epsilon/timewindow		# Convert time window to epsilon, so 3D DBSCAN will work
		pointarray = [[x[0],x[1],x[2]*squash] for x in sel_points]
		labels,clusterlist = dbscan(pointarray,epsilon,minpts)
		print ("{} detections from {} trajectories clustered".format(len(pointarray),len(filt_indices)))


		# Generate cluster metrics	
		print ("Generating cluster metrics...")
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
			if len(indices) >= minpts and len(tempclusterdict[cluster]["pointindices"]) > 4: # clusters must contain minpts or more trajectories
				clusterdict[cluster] = tempclusterdict[cluster]

		for cluster in clusterdict:
			if cluster > -1:
				msds = [trajdict[i]["msds"][0] for i in clusterdict[cluster]["indices"]] # MSDS for all traj in this cluster
				clusterdict[cluster]["av_msd"]= np.average(msds) # Average trajectory MSD in this cluster
				clustertimes = [trajdict[i]["centroid"][2] for i in clusterdict[cluster]["indices"]] # all centroid times in this cluster
				clusterdict[cluster]["centroid_times"] = clustertimes
				clusterdict[cluster]["lifetime"] = max(clustertimes) - min(clustertimes) # lifetime of this cluster (sec)
				clusterpoints = [point[:2]  for point in clusterdict[cluster]["points"]] # All detection points [x,y] in this cluster
				clusterdict[cluster]["det_num"] = len(clusterpoints) # number of detections in this cluster	
				ext_x,ext_y,ext_area,int_x,int_y,int_area = double_hull(clusterpoints) # Get external/internal hull area
				clusterdict[cluster]["area"] = ext_area # Use external hull area as cluster area (um2)
				clusterdict[cluster]["radius"] = math.sqrt(int_area/math.pi) # radius of cluster (um)
				clusterdict[cluster]["area_xy"] = [int_x,int_y] # area border coordinates	
				clusterdict[cluster]["density"] = clusterdict[cluster]["traj_num"]/int_area # trajectories/um2	
				clusterdict[cluster]["rate"] = clusterdict[cluster]["traj_num"]/(max(clustertimes) - min(clustertimes)) # accumulation rate (trajectories/sec)
				#clustercentroids = [trajdict[i]["centroid"] for i in clusterdict[cluster]["indices"]] # Centroids for each trajectory in this cluster
				diffcoeffs = [trajdict[i]["diffcoeff"] for i in clusterdict[cluster]["indices"]] # Instantaneous diffusion coefficients for each trajectory in this cluster
				clusterdict[cluster]["av_diffcoeff"]= np.average(diffcoeffs) # average trajectory inst diff coeff in this cluster
				x,y,t = zip(*clusterdict[cluster]["points"])
				xmean = np.average(x)
				ymean = np.average(y)
				tmean = np.average(t)
				clusterdict[cluster]["centroid"] = [xmean,ymean,tmean] # centroid for this cluster

		# Screen out large and tiny clusters 
		allindices = unfilt_indices
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
		unclustindices = [idx for idx in allindices if idx not in clustindices]

		if len(clusterdict) == 0:
			print("\nAlert: No clusters found")
		else:
			print ("{} clusters containing {} trajectories".format(len(clusterdict),len(clustindices)))
			
			# Save metrics
			stamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now()) # datestamp
			outpath = os.getcwd()
			outpath = outpath.split("\\")
			outpath = "/".join(outpath)
			inpath = "{}/{}".format(outpath,infilename) 
			outdir = outpath + "/" + infilename.split("/")[-1].replace(".trxyt","_BOOSH_{}".format(stamp))
			os.mkdir(outdir)
			os.chdir(outdir)
			outfilename = "{}/metrics.tsv".format(outdir)
			print ("Saving metrics...")
			with open(outfilename,"w") as outfile:
				outfile.write("BOOSH: NANOSCALE SPATIO TEMPORAL DBSCAN CLUSTERING - Tristan Wallis t.wallis@uq.edu.au\n") 
				outfile.write("TRAJECTORY FILE:\t{}\n".format(inpath)) 	
				outfile.write("ANALYSED:\t{}\n".format(stamp))
				outfile.write("TRAJECTORY LENGTH CUTOFFS (steps):\t{} - {}\n".format(minlength,maxlength))	
				outfile.write("ACQUISITION TIME (s):\t{}\n".format(acq_time))
				outfile.write("FRAME TIME (s):\t{}\n".format(frame_time))
				outfile.write("EPSILON (um):\t{}\n".format(epsilon))
				outfile.write("MINPTS:\t{}\n".format(minpts))
				outfile.write("TIME WINDOW (s):\t{}\n".format(timewindow))
				outfile.write("CLUSTER MAX RADIUS (um):\t{}\n".format(radius_thresh))
				try:
					if msd_filter:
						outfile.write("MSD FILTER THRESHOLD (um^2):\t{}\n".format(av_msd))
					else:
						outfile.write("MSD FILTER THRESHOLD (um^2):\tNone\n")
				except:
					pass
				outfile.write("SELECTION AREA (um^2):\t{}\n".format(selarea))
				outfile.write("SELECTED TRAJECTORIES:\t{}\n".format(len(allindices)))
				outfile.write("CLUSTERED TRAJECTORIES:\t{}\n".format(len(clustindices)))
				outfile.write("UNCLUSTERED TRAJECTORIES:\t{}\n".format(len(unclustindices)))
				outfile.write("TOTAL CLUSTERS:\t{}\n".format(len(clusterdict)))
				
				# INSTANTANEOUS DIFFUSION COEFFICIENT (1ST 4 POINTS)
				clustdiffcoeffs = []
				for i in clustindices:
					clustdiffcoeffs.append(trajdict[i]["diffcoeff"])
				outfile.write("CLUSTERED TRAJECTORIES AVERAGE INSTANTANEOUS DIFFUSION COEFFICIENT (um^2/s):\t{}\n".format(np.average(clustdiffcoeffs)))
				unclustdiffcoeffs = []
				for i in unclustindices:
					unclustdiffcoeffs.append(trajdict[i]["diffcoeff"])
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
				clust_msds = [trajdict[x]["msds"] for x in clustindices]
				unclust_msds = [trajdict[x]["msds"] for x in unclustindices]
				all_msds = [trajdict[x]["msds"] for x in allindices]
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
				
			print("Metrics saved to: {}\n".format(outfilename))
			print("\nDone!")