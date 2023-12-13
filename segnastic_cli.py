# -*- coding: utf-8 -*-
'''
SEGNASTIC_CLI
COMMAND LINE (CLI) VERSION FOR SPATIOTEMPORAL INDEXING CLUSTERING OF MOLECULAR TRAJECTORY SEGMENT DATA

Design and coding: Tristan Wallis
Additional coding: Kyle Young, Alex McCann
Debugging: Sophie Huiyi Hou, Kye Kudo, Alex McCann
Queensland Brain Institute
The University of Queensland
Fred Meunier: f.meunier@uq.edu.au

REQUIRED:
Python 3.8 or greater
python -m pip install scipy numpy rtree scikit-learn 

INPUT:
TRXYT trajectory files
Space separated: Trajectory X(um) Y(um) T(sec)  
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
python segnastic_cli.py inputfilename.trxyt

NOTES:
This script has been tested and will run as intended on Windows 7/10/11, Linux, and MacOS.
The script is single threaded and should run on virtual CPUs without issues.
Feedback, suggestions and improvements are welcome. Sanctimonious critiques on the pythonic inelegance of the coding are not.

CHECK FOR UPDATES:
https://github.com/tristanwallis/smlm_clustering/releases
'''

lastchanged = "20231212"

# LOAD MODULES
from argparse import ArgumentParser
import os
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import numpy as np
from rtree import index
import math
from math import dist
import datetime
from functools import reduce
import warnings
warnings.filterwarnings("ignore")

# PARAMETERS 

# Trajectory parameters
minlength = 8 # Trajectories must be longer than this to be considered (default 8; value must be >=5)
maxlength = 100	# Trajectories must be shorter than this to be considered(default 100; value must be >= minlength)
frame_time = 0.02 # Time between frames (sec) in original acquisition (default 0.02; value must be >0) 
acq_time = 320 # length of acquisition (sec)

# Clustering parameters
time_threshold = 20 # Trajectories must be within this many seconds of another trajectory as well as overlapping in space (default 20)
segment_threshold = 1 # Trajectories must contain at least this many segments which overlap with other trajectory segments
overlap_override = 1 # Float. Multiplier of the number of overlaps for a segment to be considered as potentially clustered. 1 = use average of all segment overlaps as threshold. 2 = use double the average overlap as a threshold
radius_thresh = 0.25 # um  - clusters will be excluded if bigger than this. Set to a large number to include all clusters (default 0.25)
msd_filter = True # Trajectories whose MSD at time point 1 are greater than the average will be excluded (default True)

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

# FIND SEGMENTS WHOSE BOUNDING BOXES OVERLAP IN SPACE AND TIME	
def segment_overlap(segdict,time_threshold,av_msd):
	# Create and populate 3D r-tree
	p = index.Property()
	p.dimension=3
	idx_3d = index.Index(properties=p)
	intree = []
	indices = segdict.keys()
	for idx in indices:
		if segdict[idx]["msds"][0] < av_msd: # potentially screen by MSD of parent traj
			idx_3d.insert(idx,segdict[idx]["bbox"])
			intree.append(idx)
	# Query the r-tree
	overlappers = []
	if len(intree) == 0:
		return
	else: 
		for idx in intree:
			if idx%10 == 0:
				try: 
					bar = 100*idx/(len(intree)-10)
					window['-PROGBAR-'].update_bar(bar)
				except:
					pass
			bbox = segdict[idx]["bbox"]
			left,bottom,early,right,top,late = bbox[0],bbox[1],bbox[2]-time_threshold/2,bbox[3],bbox[4],bbox[5]+time_threshold/2
			intersect = list(idx_3d.intersection([left,bottom,early,right,top,late]))
			# Remove overlap with segments from same trajectory
			segtraj = segdict[idx]["traj"]
			intersect = [x for x in intersect if segdict[x]["traj"] != segtraj]
			if len(intersect) > 0:
				# Update overlap count for each segment
				for x in intersect:
					segdict[x]["overlap"] +=1 
				# Add to the list of lists of overlapping segments
				overlappers.append(intersect)
		return overlappers

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
	for idx in indices:
		if seldict[idx]["msds"][0] < av_msd:
			idx_3d.insert(idx,seldict[idx]["bounding_box"])
			intree.append(idx)
	# Query the r-tree
	overlappers = []
	for idx in intree:
		bbox = seldict[idx]["bounding_box"]
		left,bottom,early,right,top,late = bbox[0],bbox[1],bbox[2]-time_threshold/2,bbox[3],bbox[4],bbox[5]+time_threshold/2
		intersect = list(idx_3d.intersection([left,bottom,early,right,top,late]))
		overlap = [int(x) for x in intersect]
		overlappers.append(overlap)
	# Distill the list	
	overlappers =  distill_list(overlappers)
	overlappers = [x for x in overlappers if len(x) >= cluster_threshold]
	return overlappers		

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
print ("SEGNASTIC CLI - Tristan Wallis {}\n-----------------------------------------------------".format(lastchanged))
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
		
# Length filtering	
print ("Filtering based on trajectory length...")
trajdict = {}
for traj in rawtrajdict:
	points = rawtrajdict[traj]["points"]
	if len(points) >=minlength and len(points) <=maxlength:
		trajdict[traj] = rawtrajdict[traj]
print ("{} raw trajectories, {} trajectories with step >{} and <{}".format(len(rawtrajdict),len(trajdict),minlength,maxlength))		

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
	trajdict[traj]["overlapsegs"] = 0 # how many segments in this traj are greater than the overlap threshold
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

# Dictionary of all segments
print ("Generating trajectory segment bounding boxes...")
segdict = {}
ct=0
for traj in trajdict:
	points = trajdict[traj]["points"]
	msds = trajdict[traj]["msds"]	
	for i in range(1,len(points),1):
		segment = [points[i-1],points[i]]
		segdict[ct] = {}
		segdict[ct]["traj"]=traj
		segdict[ct]["segment"] = segment
		segdict[ct]["overlap"] = 1
		segdict[ct]["centroid"] = np.average(segment,axis=0)
		segdict[ct]["msds"] = msds # MSDs for parent trajectory									
		left = min(points[i-1][0],points[i][0])
		right = max(points[i-1][0],points[i][0])
		top = max(points[i-1][1],points[i][1])
		bottom = min(points[i-1][1],points[i][1])
		early = min(points[i-1][2],points[i][2])
		late = max(points[i-1][2],points[i][2])
		segdict[ct]["bbox"] = [left,bottom,early,right,top,late]
		ct+=1

# Determine overlapping segments
print ("Total segment overlap...")
segment_overlap(segdict,time_threshold,av_msd) # list of lists of overlapping segments
all_overlaps = [segdict[seg]["overlap"] for seg in segdict]
overlap_threshold = np.average(all_overlaps)
#overlap_threshold = overlap_threshold * overlap_override
print ("{} segments analysed. Average segment overlap (threshold): {}".format(len(segdict),round(overlap_threshold,3)))

print ("Clustering thresholded segments...")
thresh_segdict = {}
for seg in segdict:
	if segdict[seg]["overlap"] > overlap_threshold:
		thresh_segdict[seg]=segdict[seg]
raw_seg_clusters =  segment_overlap(thresh_segdict,time_threshold,av_msd)
seg_clusters = distill_list(raw_seg_clusters)
	
# For each trajectory determine how many of its segments are greater than the overlap threshold
print ("Screening trajectories...")
for cluster in seg_clusters:
	for seg in cluster:
		traj = thresh_segdict[seg]["traj"]
		trajdict[traj]["overlapsegs"] += 1
		
# Now screen each cluster to remove segments belonging to trajectories with fewer than segment_threshold 		
screened_seg_clusters = []
for cluster in seg_clusters:
	screened_cluster = []
	for seg in cluster:	
		traj = thresh_segdict[seg]["traj"]
		if trajdict[traj]["overlapsegs"] >= segment_threshold:
			screened_cluster.append(seg)
	if len(screened_cluster)>= 3: # A cluster must contain segments from at least 3 trajectories 		
		screened_seg_clusters.append(screened_cluster)

seg_clusters = screened_seg_clusters
all_overlaps = [thresh_segdict[seg]["overlap"] for seg in thresh_segdict]
av_overlap = np.average(all_overlaps)
max_overlap = max(all_overlaps)	
	
print ("{} clusters of {} thresholded segments. Average segment overlap: {}".format(len(seg_clusters),len(thresh_segdict),round(av_overlap,3)))	

# Generate cluster metrics	
print ("Generating cluster metrics...")
clusterdict = {} # dictionary holding info for each spatial cluster
for num,cluster in enumerate(seg_clusters):
	clusterdict[num] = {"indices":cluster} # indices of segments in this cluster
	clusterdict[num]["seg_num"] = len(cluster) # number of segments in this cluster
	traj_list = list(set([segdict[x]["traj"] for x in cluster]))
	clusterdict[num]["traj_list"] = traj_list # indices of trajectories in this cluster
	clusterdict[num]["traj_num"] = len(traj_list) # number of trajectories in this cluster
	clustertimes = [trajdict[i]["centroid"][2] for i in traj_list] # all traj centroid times in this cluster
	clusterdict[num]["centroid_times"] = clustertimes
	clusterdict[num]["lifetime"] = max(clustertimes) - min(clustertimes) # lifetime of this cluster (sec)
	msds = [trajdict[i]["msds"][0] for i in traj_list] # MSDs for each trajectory in this cluster
	clusterdict[num]["av_msd"]= np.average(msds) # average trajectory MSD in this cluster
	diffcoeffs = [trajdict[i]["diffcoeff"] for i in traj_list] # Instantaneous diffusion coefficients for each trajectory in this cluster
	clusterdict[num]["av_diffcoeff"]= np.average(diffcoeffs) # average trajectory inst diff coeff in this cluster				
	clusterpoints = [point[:2]  for i in cluster for point in segdict[i]["segment"]] # All segment points [x,y] in this cluster
	ext_x,ext_y,ext_area,int_x,int_y,int_area = double_hull(clusterpoints) # Get external/internal hull area
	clusterdict[num]["area"] = int_area # internal hull area as cluster area (um2)
	clusterdict[num]["radius"] = math.sqrt(int_area/math.pi) # radius of cluster (um)
	clusterdict[num]["area_xy"] = [int_x,int_y] # area border coordinates	
	clusterdict[num]["density"] = len(traj_list)/int_area # trajectories/um2
	clusterdict[num]["rate"] = len(traj_list)/(max(clustertimes) - min(clustertimes)) # accumulation rate (trajectories/sec)
	clustercentroids = [trajdict[i]["centroid"] for i in traj_list]
	x,y,t = zip(*clustercentroids)
	xmean = np.average(x)
	ymean = np.average(y)
	tmean = np.average(t)
	clusterdict[num]["centroid"] = [xmean,ymean,tmean] # centroid for this cluster

# Screen out large clusters
tempclusterdict = {}
counter = 0	
for num in clusterdict:
	if clusterdict[num]["radius"] < radius_thresh:
		tempclusterdict[counter] = clusterdict[num]
		counter +=1			
clusterdict = tempclusterdict.copy()	

allindices = list(trajdict.keys())
clustindices = [y for x in clusterdict for y in clusterdict[x]["traj_list"]]
unclustindices = [idx for idx in allindices if idx not in clustindices]

print ("{} clusters containing {} trajectories".format(len(clusterdict),len(clustindices)))
	
# Save metrics
stamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now()) # datestamp
outpath = os.getcwd()
outpath = outpath.split("\\")
outpath = "/".join(outpath)
outdir = outpath + "/" + infilename.split("/")[-1].replace(".trxyt","_SEGNASTIC_{}".format(stamp))
os.mkdir(outdir)
os.chdir(outdir)
outfilename = "{}/metrics.tsv".format(outdir)
print ("Saving metrics t0 {}...".format(outdir))
with open(outfilename,"w") as outfile:
	outfile.write("SEGNASTIC: SEGMENT NANOSCALE SPATIO TEMPORAL INDEXING CLUSTERING - Tristan Wallis t.wallis@uq.edu.au\n") 
	outfile.write("TRAJECTORY FILE:\t{}\n".format(infilename))	
	outfile.write("ANALYSED:\t{}\n".format(stamp))
	outfile.write("ACQUISITION TIME (s):\t{}\n".format(acq_time))
	outfile.write("FRAME TIME (s):\t{}\n".format(frame_time))
	outfile.write("TRAJECTORY LENGTH CUTOFFS (steps):\t{} - {}\n".format(minlength,maxlength))	
	outfile.write("TIME THRESHOLD (s):\t{}\n".format(time_threshold))
	outfile.write("SEGMENT THRESHOLD:\t{}\n".format(segment_threshold))	
	outfile.write("OVERLAP THRESHOLD:\t{}\n".format(overlap_threshold))	
	if msd_filter:
		outfile.write("MSD FILTER THRESHOLD (um^2):\t{}\n".format(av_msd))
	else:
		outfile.write("MSD FILTER THRESHOLD (um^2):\tNone\n")
	outfile.write("CLUSTER MAX RADIUS (um):\t{}\n".format(radius_thresh))
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
			times = [clusterdict[i]["centroid"][2] for i in clusters]
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
	
print ("Done. Metrics saved to: {}.".format(outfilename))

