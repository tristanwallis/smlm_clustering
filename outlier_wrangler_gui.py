# -*- coding: utf-8 -*-
'''
OUTLIER_WRANGLER_GUI

PYSIMPLEGUI BASED GUI FOR REMOVING OUTLIERS USING MEDIAN FILTERING.

Design and coding: Tristan Wallis
Additional coding: Alex McCann
Queensland Brain Institute
The University of Queensland
Fred Meunier: f.meunier@uq.edu.au

REQUIRED:
Python 3.8 or greater
python -m pip install numpy matplotlib pysimplegui seaborn colorama scikit-learn

INPUT:
CSV file (Excel)
Comma separated: One column per dataset - all values must be numbers
No headers

NOTES:
Outliers are removed from data using median filtering, which works by calculating the mean for a given set of data, establishing how far each value in the data deviates from the mean, and then dividing each deviation by the mean of all the deviations. Those values whose deviation/mean of deviation exceed a threshold (default 2) are excluded. Median filtering is a very commonly used technique for removing "spikes" in datasets.

CHECK FOR UPDATES:
https://github.com/tristanwallis/smlm_clustering/releases
'''

last_changed = "20240215"

# MAIN PROG AND FUNCTIONS
if __name__ == "__main__":
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
	print(f'{Fore.GREEN}OUTLIER WRANGLER {last_changed} initialising...{Style.RESET_ALL}')
	print(f'{Fore.GREEN}=================================================={Style.RESET_ALL}')
	popup = sg.Window("Initialising...",[[sg.T("OUTLIER WRANGLER initialising...",font=("Arial bold",18))]],finalize=True,no_titlebar = True,alpha_channel=0.9)
	
	import random
	from scipy.spatial import ConvexHull
	from sklearn.cluster import DBSCAN
	import numpy as np
	import pandas as pd
	import seaborn as sns
	import matplotlib
	matplotlib.use('TkAgg') # prevents Matplotlib related crashes --> self.tk.call('image', 'delete', self.name)
	import matplotlib.pyplot as plt
	import datetime
	import pickle
	import io
	from functools import reduce 
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
		xmin =-180
		xmax=180
		ymin=-70
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
		graph.DrawText("OUTLIER WRANGLER v{}".format(last_changed),(0,70),color="white",font=("Any",16),text_location="center")
		graph.DrawText("Design and coding: Tristan Wallis",(0,45),color="white",font=("Any",10),text_location="center")
		graph.DrawText("Additional coding: Alex McCann",(0,30),color="white",font=("Any",10),text_location="center")
		graph.DrawText("Queensland Brain Institute",(0,15),color="white",font=("Any",10),text_location="center")	
		graph.DrawText("The University of Queensland",(0,0),color="white",font=("Any",10),text_location="center")	
		graph.DrawText("Fred Meunier f.meunier@uq.edu.au",(0,-15),color="white",font=("Any",10),text_location="center")	
		graph.DrawText("PySimpleGUI: https://pypi.org/project/PySimpleGUI/",(0,-40),color="white",font=("Any",10),text_location="center")	
		while True:
			# READ AND UPDATE VALUES
			event, values = splash.read(timeout=timeout) 
			ct += timeout
			# Exit	
			if event in (sg.WIN_CLOSED, '-OK-'): 
				break
			# UPDATE EACH PARTICLE
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
		
	# OUTLIER REJECTION
	def reject_outliers(data,stringency):
		d = np.abs(data - np.median(data))
		mdev = np.median(d)
		s = d/(mdev if mdev else 1.)
		return data[s<stringency]

	# READ DATA
	def read_file(infilename,stringency):
		print ("Reading {}...".format(infilename))
		rawdata = []
		with open (infilename,"r") as infile:
			for line in infile:
				line = line.replace("\n","").replace("\r","")
				spl = line.split(",")
				rawdata.append(spl)
		try:		
			rows = len(rawdata)		
			columns = list(zip(*rawdata))
			print ("Number of data columns: {}".format(len(columns)))
			print ("Filtering data with stringency {}...".format(stringency))	
			outarray = []
			total=0
			removed = 0
			for column in columns:
				data = [float(x) for x in column if x !=""] # ignore empy values
				filtered = list(reject_outliers(np.array(data),stringency))
				data_or = []
				for x in data:
					total +=1
					if x in filtered:
						data_or.append(x)
					else:
						data_or.append("-")
						removed +=1
				# Replace empty values
				missing = rows - len(data)
				[data_or.append("") for x in range(missing)]			
				outarray.append(data_or)
			print ("Median filtering removed {} values from {} total values".format(removed,total))	
			outarray = list(zip(*outarray))
			return columns,outarray
		except:
			print ("Unable to load data, please make sure it is in correct CSV format with data in columns")
			outarray = []
			columns = []
			return columns,outarray

	# WRITE FILTERED DATA AND SAVE PLOT	
	def write_data (infilename,stringency,outarray):
		stamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now())
		outpath = os.path.dirname(infilename)
		outdir = outpath + "/OUTLIER_WRANGLER_{}".format(stamp)
		outfilename = outdir + "/" + infilename.split("/")[-1].replace(".csv","median_filtered.csv")
		try:
			os.mkdir(outdir)
			output_directory = outdir
			os.makedirs(output_directory,exist_ok = True)
			os.chdir(output_directory)
			# Write and save filtered data
			print ("Writing filtered data...")
			with open(outfilename,"w") as outfile:
				outfile.write("# Outlier Wrangler - Tristan Wallis\n")
				outfile.write("# Input file: {}\n".format(infilename))
				outfile.write("# Processed: {}\n".format(stamp))
				outfile.write("# Median filtering stringency: {}\n".format(stringency))
				outfile.write("###\n")
				for row in outarray:
					outfile.write(reduce(lambda x, y: str(x) + "," + str(y), row) + "\n")
			print("Filtered data saved as {}".format(outfilename))
			# Save filtered data plot
			print("Saving filtered data plot...")
			buf1.seek(0)
			fig100=pickle.load(buf1)
			plt.savefig("{}/stringency_plot.png".format(outdir),dpi=300)
			plt.close()
			print("Filtered data plot saved as {}/stringency_plot.png".format(outdir))
			sg.Popup("Filtered data and plot has been saved")			
		except:
			print("ALERT: Error saving data. Please check whether data has already been saved.")
			sg.Popup("Alert", "Error with saving data", "Please check whether data has already been saved")

	# GET INITIAL VALUES FOR GUI
	cwd = os.path.dirname(os.path.abspath(__file__))
	os.chdir(cwd)
	initialdir = cwd
	
	# GUI LAYOUT
	appFont = ("Any 12")
	sg.set_options(font=appFont)
	sg.theme('DARKGREY11')
	
	menu_def = [
		['&File', ['&Exit']],
		['&Info', ['&About', '&Help','&Licence','&Updates' ]],
	]
	
	layout = [
		[sg.Menu(menu_def)],
		[sg.T("OUTLIER WRANGLER",font=("Any 20"))],
		[sg.FileBrowse("Browse",file_types=(("CSV Files", "*.csv"),),key="-FILE-",target="-HIDDEN-",initial_folder=initialdir,tooltip="Select a CSV file (comma separated values) to remove outliers from\nNo headers\nOne dataset per column (numbers only)"),sg.Input("Select CSV file", key = "-HIDDEN-",visible=True,size=(40,1))],
		[sg.T("Stringency",tooltip="Stringency of median filtering outlier removal.\n1=remove a lot, 5 = remove very few\nDefault = 2"),sg.Combo([1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0],default_value=2.0,key="-STRINGENCY-", enable_events = True)],		
		[sg.B("FILTER DATA",key="-FILTER-",size=(23,2),button_color=("white","gray"),disabled=True),sg.B("SAVE FILTERED DATA",key="-SAVE-",size=(23,2),button_color=("white","gray"),disabled=True)],
	]

	window = sg.Window('Outlier Wrangler v{}'.format(last_changed), layout)
	popup.close()
	
	# SET UP PLOTS
	plt.rcdefaults() 
	font = {"family" : "Arial","size": 12} 
	matplotlib.rc('font', **font)
	fig0 = plt.figure(1,figsize=(8,8))
	ax0 = plt.subplot(111)
	
	# MAIN LOOP
	outarray = []
	while True:
		#READ EVENTS AND VALUES
		event, values = window.read(timeout=1000)
		infilename = values['-HIDDEN-']
		stringency = values['-STRINGENCY-']
		
		# Exit
		if event in (sg.WIN_CLOSED, 'Exit'):  
			break
		
		# About
		if event == 'About':
			splash = create_splash()	
			splash.close()
		
		# Help	
		if event == 'Help':
			sg.Popup(
				"Help",
				"This program will allow you to remove outliers from data in column format, using Median Filtering.",
				"This works by calculating the mean for a given set of data, establishing how far each value in the data deviates from the mean, and then dividing each deviation by the mean of all the deviations. Those values whose deviation/mean of deviation exceed a threshold (default 2) are excluded. Median filtering is a very commonly used technique for removing 'spikes' in datasets.",
				"Input data: Save the data from your EXCEL sheet in csv (comma separated variable) format. Make sure there are no headers, just columns of data. There can be as many or as few columns as you like. Median filtering is applied to each column separately. ALL VALUES MUST BE NUMBERS.",
				"Parameters: select an input file, and a stringency setting. 2 works well as the default. >2: less data removed, <2: more data removed.",
				"FILTER DATA: Load the file and perform median filtering using your selected stringency. You can adjust stringency and re press this button. A plot will be generated showing the distribution of your raw and filtered data.",
				"SAVE FILTERED DATA: Save the filtered data with a 'median_filtered.csv' suffix and the stringency plot as 'stringency_plot.png' in a datestamped folder generated in the same location as the input data. Outlier values in the median_filtered.csv file are replaced with a '-'.",
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
		
		# Update buttons
		if infilename.endswith(".csv") and len(infilename)>5:
			window.Element("-FILTER-").update(button_color=("white","#111111"),disabled=False)
		
		if event == '-STRINGENCY-':
			window.Element("-SAVE-").update(button_color=("white","gray"),disabled=True)
			filter_check = "False"
			
		if len(outarray) > 0 and filter_check == "True":
			window.Element("-SAVE-").update(button_color=("white","#111111"),disabled=False)		
			
		# Filter data	
		if event == '-FILTER-':  
			filter_check = "True"
			# Plot
			inarray,outarray = read_file(infilename,stringency)
			print("Plotting raw vs filtered data...")
			if len(inarray)> 1 and len(outarray) > 1:
				datadict = {"Column":[],"Val":[],"Treat":[]}
				columns = range(1,100)	
				for n,l in enumerate(inarray):
					col = columns[n]
					for val in l:
						try:
							val = float(val)
							datadict["Column"].append(col)
							datadict["Val"].append(val)
							datadict["Treat"].append("Raw")
						except:
							pass
				for n,l in enumerate(zip(*outarray)):
					col = columns[n]
					for val in l:
						try:
							val = float(val)
							datadict["Column"].append(col)
							datadict["Val"].append(val)
							datadict["Treat"].append("Filt")
						except:
							pass
				df = pd.DataFrame(datadict)
				try:
					plt.close('all')
				except:
					pass
				plotcols = len(inarray)
				if plotcols > 10:
					plotcols=10
				fig1 = plt.figure(1,figsize=(1.5*plotcols,8))
				fig1.suptitle("Stringency = {}".format(stringency))
				ax1 = plt.subplot(211)
				ax1.cla()				
				sns.violinplot(data=df,x="Column",y="Val",hue="Treat",dodge=True)
				ax2 = plt.subplot(212) 
				ax2.cla()
				sns.stripplot(data=df,x="Column",y="Val",hue="Treat",dodge=True,size=3.5)
				plt.tight_layout()
				fig1.canvas.manager.set_window_title('Stringency = {}'.format(stringency))
				
				plt.show(block=False)
				# Pickle
				buf1 = io.BytesIO()
				pickle.dump(fig1,buf1)
				buf1.seek(0)
				print("Raw vs filtered Data plotted")
				
		# Save filtered data	
		if event == '-SAVE-': 	
			write_data(infilename,stringency,outarray)		
			
	print ("Exiting...")		
	plt.close('all')				
	window.close()
	quit()