# -*- coding: utf-8 -*-
'''
DBSCAN_WRANGLER_GUI
PYSIMPLEGUI BASED GUI TO PERFROM META ANALYSIS OF METRICS PRODUCED BY DBSCAN AND VORONOI CLUSTERING

Design and coding: Tristan Wallis and Sophie Hou
Queensland Brain Institute
The University of Queensland
Fred Meunier: f.meunier@uq.edu.au

REQUIRED:
Python 3.8 or greater
python -m pip install colorama matplotlib numpy pandas scipy Pillow pysimplegui seaborn scikit-learn

INPUT:
metrics.tsv files inside directories produced by DBSCAN and Voronoi

CHECK FOR UPDATES:
https://github.com/tristanwallis/smlm_clustering/releases
'''

last_changed = "20231212"

import PySimpleGUI as sg
sg.theme('DARKGREY11')
popup = sg.Window("Initialising...",[[sg.T("DBSCAN Wrangler initialising\nLots of modules...",font=("Arial bold",18))]],finalize=True,no_titlebar = True,alpha_channel=0.9)

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
from scipy.stats import ttest_ind, ttest_ind_from_stats
import math
from sklearn import manifold, datasets, decomposition, ensemble, random_projection
from functools import reduce
import datetime

colorama_init()
os.system('cls' if os.name == 'nt' else 'clear')	
print(f'{Fore.GREEN}=================================================={Style.RESET_ALL}')
print(f'{Fore.GREEN}DBSCAN Wrangler {last_changed} initialising...{Style.RESET_ALL}')
print(f'{Fore.GREEN}=================================================={Style.RESET_ALL}')

# INITIAL SETUP
cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd)
initialdir = cwd
plt.rcdefaults() 
font = {"family" : "Arial","size": 14} 
matplotlib.rc('font', **font)

#VALS
cond_dict_1 = {}
cond_dict_2 = {}
outlist = [] 
stamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now()) # datestamp

# READ METRICS
def read_metrics(infile):
	metric_dict = {"CLUSTERS":{}}
	with open (infile,"r") as infile:
		for line in infile:
			line = line.strip()
			spl = line.split("\t")
			try:
				metric_dict["CLUSTERS"][int(spl[0])] = spl[1:]
			except:
				metric_dict[spl[0].replace(":","")] = spl[1:]
	return metric_dict		

# COMPARATIVE BAR PLOTS WITH STATS
def barplot(num,cond1,cond2,title,ylabel,swarm):
	ax = plt.subplot(3,4,num)
	avg_cond1 = np.average(cond1)
	avg_cond1_sem = np.std(cond1)/math.sqrt(len(cond1))	
	avg_cond2 = np.average(cond2)
	avg_cond2_sem = np.std(cond2)/math.sqrt(len(cond2))
	bars = [shortname1,shortname2]
	avgs = [avg_cond1,avg_cond2]
	sems = [avg_cond1_sem,avg_cond2_sem]
	color=["orange","royalblue"]
	ax.bar(bars, avgs, yerr=sems, align='center',color=color,edgecolor=color,linewidth=1.5, alpha=1,error_kw=dict(ecolor="k",elinewidth=1.5,antialiased=True,capsize=5,capthick=1.5,zorder=1000))
	if swarm:
		rows = []		
		for val in cond1:
			rows.append({"condition":shortname1,"val":val})
		for val in cond2:
			rows.append({"condition":shortname2,"val":val})
		df = pd.DataFrame(rows)
		ax = sns.swarmplot(x="condition", y="val", data=df,alpha=0.9,size=5,order=bars,palette=["k","k"])
	
	#plt.title(title)
	plt.ylabel(ylabel)
	plt.xlabel("")
	t, p = ttest_ind(cond2,cond1, equal_var=False)
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
		ax.text(0.5, 0.90, star, ha='center', va='bottom', color="k",transform=ax.transAxes)
	else:
		ax.text(0.5, 0.92, star, ha='center', va='bottom', color="k",transform=ax.transAxes)
	ax.plot([0.25,0.25, 0.75, 0.75], [0.9, 0.92, 0.92, 0.9], lw=1.5,c="k",transform=ax.transAxes)
	ylim = ax.get_ylim()[1]
	plt.ylim(0,ylim*1.1)
	plt.tight_layout()
	
	outlist.append([ylabel,shortname1,avgs[0],sems[0],p,len(cond1),reduce(lambda x, y: str(x) + "\t" + str(y), cond1)])
	outlist.append([ylabel,shortname2,avgs[1],sems[1],p,len(cond2),reduce(lambda x, y: str(x) + "\t" + str(y), cond2)])		

# NORMALIZE
def normalize(lst):
    s = sum(lst)
    return map(lambda x: float(x)/s, lst)	
	

# LAYOUT
sg.theme('DARKGREY11')
appFont = ("Any 12")
sg.set_options(font=appFont)
layout = [
[sg.T("DBSCAN Wrangler",font=("Any",24))],
[sg.FolderBrowse("Browse",key="-B1-",target="-H1-",initial_folder=initialdir),sg.T("Choose directory 1",key = "-T1-",size=(20,1)),sg.T("Short:"),sg.Input("",key="-I1-",size=(10,1)),sg.T("Exclude"),sg.Combo([""] ,key="-C1-"),sg.Input("",key="-H1-",visible=False,enable_events=True)],
[sg.FolderBrowse("Browse",key="-B2-",target="-H2-",initial_folder=initialdir),sg.T("Choose directory 2",key = "-T2-",size=(20,1)),sg.T("Short:"),sg.Input("",key="-I2-",size=(10,1)),sg.T("Exclude"),sg.Combo([""],key="-C2-"),sg.Input("",key="-H2-",visible=False,enable_events=True)],
[sg.B("LOAD DATA",key = "-B3-",size=(10,2),button_color=("white","gray"),disabled=True),sg.B("PLOT DATA",key = "-B4-",size=(10,2),button_color=("white","gray"),disabled=True),sg.B("HELP",key = "-B6-",size=(10,2),button_color=("white","dodgerblue")),sg.B("EXIT",key = "-B5-",size=(10,2),button_color=("white","red"))],
#[sg.Output(size=(60,10))]
]
window = sg.Window("DBSCAN Wrangler {}".format(last_changed),layout)
popup.close()

# MAIN LOOP
while True:
	#Read events and values
	event, values = window.read(timeout=250)
	dir1 = values["-B1-"]
	dir2 = values["-B2-"]
	shortname1 = values["-I1-"]
	shortname2 = values["-I2-"]
	exclude1 = values["-C1-"]
	exclude2 = values["-C2-"]	
	
	# Directory stuff
	if event == "-H1-":
		shortdir1 = dir1.split("/")[-1]
		if shortdir1 !="":
			window.Element("-T1-").update(shortdir1)
			cond1files = glob.glob(dir1 + '/**/metrics.tsv')
			combolist1 = [""]+[x for x in range(1,len(cond1files)+1)]
			window.Element("-C1-").update(values=combolist1)
	if event == "-H2-":		
		shortdir2 = dir2.split("/")[-1]
		if shortdir2 !="":
			window.Element("-T2-").update(shortdir2)
			cond2files = glob.glob(dir2 + '/**/metrics.tsv')	
			combolist2 = [""]+[x for x in range(1,len(cond2files)+1)]
			window.Element("-C2-").update(values=combolist2)
			
	# Load
	if dir1 != "" and dir2 != "" and shortname1 != "" and shortname2 != "":
		if len(cond1files) > 0 and len(cond2files) > 0: 
			window.Element("-B3-").update(disabled=False,button_color=("white","green"))
	if event == "-B3-":
		# Close previous plots if needed	
		try:
			plt.close(1)
			plt.close(2)
			plt.close(3)
		except:
			pass
	
		print ("Analysing files in",shortname1)
		cond_dict_1 = {}
		nums1 = []
		for num,infile in enumerate(cond1files,start=1):
			if num != exclude1:
				infile=infile.replace("\\","/")
				metric_dict = read_metrics(infile)
				print(num,metric_dict["TRAJECTORY FILE"][0])
				cond_dict_1[num] = metric_dict
				nums1.append(num)
		print ("Analysing files in",shortname2)
		cond_dict_2 = {}
		nums2 = []
		for num,infile in enumerate(cond2files,start=1):
			if num != exclude2:
				infile=infile.replace("\\","/")
				metric_dict = read_metrics(infile)
				print(num,metric_dict["TRAJECTORY FILE"][0])
				cond_dict_2[num] = metric_dict
				nums2.append(num)
	
	# Plot
	if cond_dict_1 != {} and cond_dict_2 != {}:
		window.Element("-B4-").update(disabled=False,button_color=("white","orange"))
	if event == "-B4-":
		# WRITE OUTPUT FILE
		outdir = "dbscan_wrangler_output_{}".format(stamp)
		os.mkdir(outdir)
		os.chdir(outdir)
		
		with open("processed_metrics.tsv","w",encoding = 'utf') as outfile:
			outfile.write("DBSCAN / VORONOI WRANGLER - Tristan Wallis t.wallis@uq.edu.au\n")
			outfile.write("ANALYSED:\t{}\n".format(stamp))
			outfile.write("{}:\t{}\n".format(shortname1,dir1))
			outfile.write("{}:\t{}\n".format(shortname2,dir2))	
	

			print ("Plotting aggregate data for all samples")
			outlist.append(["\nAGGREGATE CLUSTER DATA"])
			outlist.append(["METRIC\tCONDITION\tAVG\tSEM\tT-TEST P\tN\tDATAPOINTS"])
			
			# Aggregate cluster data for all samples	
			aggregate_1 = [[],[],[],[],[]]
			for sample in cond_dict_1:
				clusters = cond_dict_1[sample]["CLUSTERS"]
				for cluster in clusters:
					for num,val in enumerate(clusters[cluster]):
						aggregate_1[num].append(float(val))
			aggregate_2 = [[],[],[],[],[]]
			for sample in cond_dict_2:
				clusters = cond_dict_2[sample]["CLUSTERS"]
				for cluster in clusters:
					for num,val in enumerate(clusters[cluster]):
						aggregate_2[num].append(float(val))
			memb1,msd1,area1,radius1,density1 = aggregate_1	
			memb2,msd2,area2,radius2,density2 = aggregate_2			
			fig1 = plt.figure(1,figsize=(10,10))
			barplot(1,memb1,memb2,"Membership",u"Membership \n(traj/cluster)",False)		
			barplot(2,msd1,msd2,"MSD",u"Cluster avg. MSD \n(μm²)",False)
			barplot(3,area1,area2,"Area",u"Cluster area \n(μm²)",False)
			barplot(4,radius1,radius2,"Radius",u"Cluster radius \n(μm)",False)
			barplot(5,density1,density2,"Density",u"Density in clusters \n(traj/μm²)",False)
			fig1.canvas.manager.set_window_title('Aggregate data')
			plt.show(block=False)
			plt.savefig("aggregate_plots.png")
			
			print ("Plotting average data for all samples")
			outlist.append(["\nAGGREGATE CLUSTER DATA"])
			outlist.append(["METRIC\tCONDITION\tAVG\tSEM\tT-TEST P\tN\tDATAPOINTS"])
			
			# Average cluster data for all samples
			average_1 = [[],[],[],[],[],[],[]]
			for sample in cond_dict_1:
				averages = cond_dict_1[sample]["AVG"]
				clust_traj = float(cond_dict_1[sample]["CLUSTERED TRAJECTORIES"][0])
				sel_traj = float(cond_dict_1[sample]["SELECTED TRAJECTORIES"][0])
				clust_num = float(cond_dict_1[sample]["TOTAL CLUSTERS"][0])
				sel_area = float(cond_dict_1[sample]["SELECTION AREA (um^2)"][0])	
				for num,val in enumerate(averages):
					average_1[num].append(float(val))
				perc_clust = 100*clust_traj/sel_traj
				clust_density = clust_num/sel_area	
				average_1[5].append(perc_clust)
				average_1[6].append(clust_density)
			average_2 = [[],[],[],[],[],[],[]]
			for sample in cond_dict_2:
				averages = cond_dict_2[sample]["AVG"]
				clust_traj = float(cond_dict_2[sample]["CLUSTERED TRAJECTORIES"][0])
				sel_traj = float(cond_dict_2[sample]["SELECTED TRAJECTORIES"][0])
				clust_num = float(cond_dict_2[sample]["TOTAL CLUSTERS"][0])
				sel_area = float(cond_dict_2[sample]["SELECTION AREA (um^2)"][0])			
				for num,val in enumerate(averages):
					average_2[num].append(float(val))
				perc_clust = 100*clust_traj/sel_traj
				clust_density = clust_num/sel_area	
				average_2[5].append(perc_clust)
				average_2[6].append(clust_density)				
					
			memb1,msd1,area1,radius1,density1,perc1,cldensity1 = average_1	
			memb2,msd2,area2,radius2,density2,perc2,cldensity2 = average_2			
			fig2 = plt.figure(2,figsize=(10,10))
			barplot(1,memb1,memb2,"Membership",u"Membership \n(traj/cluster)",True)		
			barplot(2,msd1,msd2,"MSD",u"Cluster avg. MSD \n(μm²)",True)
			barplot(3,area1,area2,"Area",u"Cluster area \n(μm²)",True)
			barplot(4,radius1,radius2,"Radius",u"Cluster radius \n(μm)",True)
			barplot(5,density1,density2,"Density",u"Density in clusters\n(traj/μm²)",True)
			barplot(6,perc1,perc2,"Percentage",u"Clustered trajectories \n(%)",True)
			barplot(7,cldensity1,cldensity2,"Cluster density",u"Cluster density \n(clusters/μm²)",True)				
			fig2.canvas.manager.set_window_title('Average data')
			plt.show(block=False)
			plt.savefig("average_plots.png")
			
			# PCA
			print ("Plotting PCA of all average metrics")
			avnorm1 = [normalize(x) for x in average_1]
			avnorm2 = [normalize(x) for x in average_2]
			average_1 = list(zip(*avnorm1))
			average_2 = list(zip(*avnorm2))
			alldata_temp = average_1+average_2
			alldata = []
			for row in alldata_temp:
				temprow = []
				for val in row:
					if val == val:
						temprow.append(val)
					else:
						temprow.append(0)
				alldata.append(temprow)
			colors = []
			[colors.append("orange")for x in cond_dict_1]
			[colors.append("royalblue")for x in cond_dict_2]
			
			names = []
			ct1 = 1
			for ct, infile in enumerate(cond1files,start=1):
				names.append("1.{}".format(ct1))
				ct1+=1
			ct2 = 1
			for ct, infile in enumerate(cond2files,start=1):
				names.append("2.{}".format(ct2))
				ct2+=1
			mapdata = decomposition.TruncatedSVD(n_components=2).fit_transform(alldata) 
			fig3 = plt.figure(3,figsize=(4,4))
			ax0 = plt.subplot(111)
			ax0.scatter(mapdata[:, 0], mapdata[:, 1],c=colors)
			for i in range(mapdata.shape[0]):
				ax0.text(mapdata[i, 0], mapdata[i, 1],names[i],alpha=0.5)
			ax0.set_xticks([])
			ax0.set_yticks([])
			ax0.set_xlabel('Dimension 1')
			ax0.set_ylabel('Dimension 2')
			plt.tight_layout()
			fig3.canvas.manager.set_window_title('PCA - all metrics')
			plt.show(block=False)
			plt.savefig("PCA_plot.png")

			for line in outlist:
				outstring = reduce(lambda x, y: str(x) + "\t" + str(y), line)
				outstring = outstring.replace(u"μ","u").replace(u"²","^2").replace("\n","")
				outfile.write(outstring + "\n")	
	# Help
	if event in ('-B6-'): 
		sg.Popup("DBSCAN WRANGLER HELP",
		"This program allows visualising of the metrics.tsv files produced by DBSCAN and Voronoi Clustering. Comparison bar plots for each metric and statisitical significance (t-test) are shown.",
		"Aggregate data: aggregates individual cluster metrics across all samples. N = total number of clusters",
		"Average data: aggregrates the averaged cluster metrics for each sample. N = number of samples"	,
		"3D PCA analysis allows you to determine the overall relationships between samples",  	
		"For each of the two experimental conditions, select a directory, and the program will recursively search it for metrics.tsv files",
		"Give each experimental condition a short name which will appear in subsequent plots",
		"You can optionally exclude one file from each experimental condition",
		"LOAD DATA: load the files and extract information\n"
		"PLOT DATA: plot aggregate, average and PCA. Also saves a datestamped TSV of raw data used for plots",
		"Tristan Wallis, Sophie Hou {}".format(last_changed))
			
	# Exit	
	if event in (sg.WIN_CLOSED, '-B5-'):   
		break
	
exit()