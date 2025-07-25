# -*- coding: utf-8 -*-
'''
SUPER_RES_DATA_WRANGLER_GUI
FREESIMPLEGUI BASED GUI FOR THE CONVERSION OF TRAJECTORY FILES
CONVERT BETWEEN MUTLIPLE FILE FORMATS FROM THE PALMTRACER (METAMORPH PLUGIN)/TRACKMATE (IMAGEJ PLUGIN)--> SHARP VISU --> NASTIC/segNASTIC/BOOSH --> NASTIC WRANGLER SUPER RESOLUTION DATA PROCESSING PIPELINE

Design and coding: Tristan Wallis and Alex McCann
Queensland Brain Institute
The University of Queensland
Fred Meunier: f.meunier@uq.edu.au


REQUIRED:
Python 3.8 or greater
python -m pip install freesimplegui colorama scipy numpy scikit-learn


INPUT: 
Any of the below formats can be interchanged with any other:
.txt (PalmTracer)
.trc (PalmTracer)
.trxyt (NASTIC/segNASTIC/BOOSH)
.ascii (before drift correction; SharpViSu)
.ascii (drift corrected; SharpViSu) 

Additionally, the below format can be converted into any of the above filetypes:
.csv (TrackMate)

Futhermore, other filetypes not mentioned above can be converted into the above filetypes by manually selecting conversion parameters:
other(manual input)


NOTES:

Converting to/from the ascii file format:
When writing .ascii files, a trajectory .id file containing the Trajectory ID (Trajectory#) of each .ascii data line (row) is generated (with the same name and in the same folder as the .ascii), which is then used to convert back from the .ascii file format to other formats. 

Intensity information: 
Depending on the file conversion, intensity information is lost. This intensity information is not relevant to subsequent analyses, so no big deal.

Converting between different X,Y units:
Internally the system works in microns, so .trc, .txt and .ascii formats need to be converted from pixels to microns using the Pixel size parameter as a conversion factor (default = 0.106um/px). Drift corrected .ascii files are in nanometers whereas uncorrected .ascii files are in pixels - this difference is accounted for in the file converter by selecting 'ascii(drift corrected)' for files in nm, and 'ascii' for files in px. The X,Y units for TrackMate .csv files depends on how FIJI interprets the microscope data, and can be manually selected in the 'Set parameters' tab. The X,Y units of other filetypes (using the 'other(manual input)' option) can also be manually selected in the 'Set parameters' tab. If conversion between pixels and either microns or nanometers is required, the 'Pixel size(um/px)' parameter will appear as an option in the 'Set parameters' tab.

Converting between different Time units:
The time information for each acquired frame in .trxyt files is in seconds, and needs to be converted to Frame# when converting to other filetypes. This is done using the Acquisition frequency (Hz) parameter as a conversion factor (default = 50Hz). To find the Acquisition frequency of a file, divide 1 by the Frame time (seconds): e.g., for a file where a frame is acquired every 0.02 seconds, 1/0.02 = 50Hz. The Time units for TrackMate .csv files depends on how FIJI interprets the microscope data, and can be manually selected in the 'Set parameters' tab. The Time units of other filetypes (using the 'other(manual input)' option) can also be manually selected in the 'Set parameters' tab. If conversion between Frame# and seconds is required, the 'Acquisition Frequency(Hz)' parameter will appear as an option in the 'Set parameters' tab.

Starting frame parameter:
If the Time units for an input file is set to 'Frames' (each timepoint is given a Frame number in ascending order), the 'Starting frame' parameter will appear. This parameter corresponds to whether frame numbering starts at 0 or 1. 


USAGE: 
1 - Run the script (either by double clicking or navigating to the location of the script in the terminal and using the command 'python super_res_data_wrangler_GUI.py')

'Select filetypes' tab:
2 - Specify the filetype you want to convert FROM using the 'Input filetype' dropdown box
3 - Specify the filetype you want to convert TO using the 'Output filetype' dropdown box
4 - Press 'CONFIRM FILETYPES' to confirm filetype selection and swap to the next tab

'Load files' tab:
5 - Select whether you wish to load a single file or multiple files (by clicking on the 'Browse file' and 'Browse folder (directory)' radio option respectively)
6 - Browse for the trajectory file/folder containing trajectory files that you wish to convert:
	
	'Browse file' option:
	6.1 - Press the 'Browse file' button on the left of the 'Select input file' textbox to browse for and select the trajectory file that you wish to convert
	
	'Browse folder' option:
	6.1 - Press the 'Browse folder' button on the left of the 'Select directory containing input files' textbox to browse for a select a folder containing the trajectory files that you wish to convert
	6.2 - Press the 'Find files' button to populate the below table with files from the selected folder
	6.3 - Optional: To only find files that contain a certain phrase in the filename, enter that phrase in the '(Optional) Filenames contain:' text box. Leave this text box blank to find all files.
	6.4 - Optional (other(manual input) files only): To only find files of a particular filetype, enter the extension in the '(Optional) Filenames end with:' text box. Leave this text box blank to find all filetypes.  
	6.5 - Tick/untick files that you wish to include/exclude from conversion

7 - Press the 'LOAD DATA' button to confirm the selected input files and swap to the next tab  

'Set parameters' tab:
8 - Select input file parameters (parameters will appear/become enabled if input is required):
	
	'Define spatial and temporal information' parameter inputs:
	8.1 - X,Y units dropdown box (px, um, nm) - spatial units for trajectory data
	8.2 - Pixel size (um/px) - used to convert bewteen different X,Y units (will appear if input is required)
	8.3 - Time units (Frames, s) - temporal units for trajectory data
	8.4 - Acquisition frequency (Hz) - used to convert between different Time units (will appear if input is required)
	
	'Define file structure' parameter inputs:
	8.5 - Number of headers (1-10) - number of rows before data rows start
	8.6 - Delimiter (tab, comma, 1 space, 2 spaces, 3 spaces, 4 spaces, semicolon) - what separates the data into columns
	
	'Select which columns contain trajectory data' parameter inputs:
	8.7 - Trajectory col (1-20) - column number that contains trajectory# information
	8.8 - X-position col (1-20) - column number that contains X-position information
	8.9 - Y-position col (1-20) - column number that contains Y-position information
	8.10 - Frame/Time col (1-20) - column number that contains Time information (either Frame# or time in s)
	8.11 - Starting frame (0,1) - whether the first frame starts at 0 or 1 (will appear if Time units are in Frames)
	
9 - Press the "CONVERT FILE" button - files will be converted and saved to the same place as the original file, with the appropriate suffix and a date stamp

CHECK FOR UPDATES:
https://github.com/tristanwallis/smlm_clustering/releases
'''

last_changed = "20250724"

# LOAD MODULES (Functions)
import random
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from functools import reduce
import datetime
import glob
from io import BytesIO
from PIL import Image, ImageDraw
import webbrowser
import warnings

warnings.filterwarnings("ignore")

# MAIN PROG AND FUNCTIONS
if __name__ == "__main__": # has to be called this way for multiprocessing to work
	
	# LOAD MODULES (GUI and console)
	import FreeSimpleGUI as sg
	import os
	from colorama import init as colorama_init
	from colorama import Fore
	from colorama import Style
	
	#sg.set_options(dpi_awareness=True) # turns on DPI awareness (Windows only)
	sg.theme('DARKGREY11')
	colorama_init()
	os.system('cls' if os.name == 'nt' else 'clear')
	print(f'{Fore.GREEN}======================================================={Style.RESET_ALL}')
	print(f'{Fore.GREEN}SUPER RES DATA WRANGLER {last_changed} initialising...{Style.RESET_ALL}')
	print(f'{Fore.GREEN}======================================================={Style.RESET_ALL}')
	popup = sg.Window("Initialising...",[[sg.T("SUPER RES DATA WRANGLER initialising...",font=("Arial bold",18))]],finalize=True,no_titlebar = True,alpha_channel=0.9)
	
	# VARS
	load_data_button = False # toggles 'LOAD DATA' button
	find_files = False # toggles 'Find Files' button 
	tree_icon_dict = {} # dictionary keeping track of which files have been ticked/unticked
	
	# FUNCTIONS
	
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
		xmin =-350
		xmax=350
		ymin=-175
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
		xmin =-350
		xmax=350
		ymin=-175
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
		graph.DrawText("SUPER RES DATA WRANGLER v{}".format(last_changed),(0,70),color="white",font=("Any",16),text_location="center")
		graph.DrawText("Design and coding: Tristan Wallis and Alex McCann",(0,10),color="white",font=("Any",10),text_location="center")
		graph.DrawText("Queensland Brain Institute",(0,-40),color="white",font=("Any",10),text_location="center")	
		graph.DrawText("The University of Queensland",(0,-70),color="white",font=("Any",10),text_location="center")	
		graph.DrawText("Fred Meunier f.meunier@uq.edu.au",(0,-110),color="white",font=("Any",10),text_location="center")	
		graph.DrawText("FreeSimpleGUI: https://pypi.org/project/FreeSimpleGUI/",(0,-150),color="white",font=("Any",10),text_location="center")	

		while True:
			# READ AND UPDATE VALUES
			event, values = splash.read(timeout=timeout) 
			ct += timeout
			# Exit	
			if event in (sg.WIN_CLOSED, 'Exit', '-OK-'): 
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
		
	# SIMPLE CONVEX HULL AROUND SPLASH CLUSTERS
	def hull(points):
		points = np.array(points)
		hull = ConvexHull(points)
		hullarea = hull.volume
		vertices = hull.vertices
		vertices = np.append(vertices,vertices[0])
		hullpoints = np.array(points[hull.vertices])
		return hullpoints,hullarea
	
	# USE HARD CODED DEFAULTS
	def reset_defaults():
		print ("\nUsing default GUI settings...")
		global convertfrom, convertto, files_contain, files_ext, xy_units_input, pix2um_input, time_units_input, acqfreq_input, header_num_input, delim_input, tr_col_input, x_col_input, y_col_input, t_col_input, starting_frame_input
		convertfrom = "txt"
		convertto = "trxyt"
		files_contain = ""
		files_ext = ""
		pix2um_input = 0.106
		acqfreq_input = 50.0
		xy_units_input = "px"
		time_units_input = "Frames"
		header_num_input = 3
		delim_input = "tab"
		tr_col_input = 1
		x_col_input = 3
		y_col_input = 4
		t_col_input = 2
		starting_frame_input = 1
		return 
	
	# SAVE SETTINGS
	def save_defaults():
		print ("\nSaving GUI settings to super_res_data_wrangler_gui.defaults...")
		with open("super_res_data_wrangler_gui.defaults","w") as outfile:
			outfile.write("{}\t{}\n".format("Convert from filetype",convertfrom))
			outfile.write("{}\t{}\n".format("Convert to filetype",convertto))
			outfile.write("{}\t{}\n".format("Only find files with filenames that contain",files_contain))
			outfile.write("{}\t{}\n".format("Only find files with filenames that end with",files_ext))
			outfile.write("{}\t{}\n".format("XY units", xy_units_input))
			outfile.write("{}\t{}\n".format("Pixel size (um/px)",pix2um_input))
			outfile.write("{}\t{}\n".format("Time units", time_units_input))
			outfile.write("{}\t{}\n".format("Acquisition frequency (Hz)",acqfreq_input))	
			outfile.write("{}\t{}\n".format("Number of headers",header_num_input))	
			outfile.write("{}\t{}\n".format("Delimiter separating columns", delim_input))
			outfile.write("{}\t{}\n".format("Column containing Trajectory# data", tr_col_input))
			outfile.write("{}\t{}\n".format("Column containing X-position data", x_col_input))
			outfile.write("{}\t{}\n".format("Column containing Y-position data", y_col_input))
			outfile.write("{}\t{}\n".format("Column containing Time data", t_col_input))
			outfile.write("{}\t{}\n".format("Starting frame#", starting_frame_input))
		return
	
	# LOAD DEFAULTS
	def load_defaults():
		global convertfrom, convertto, files_contain, files_ext, xy_units_input, pix2um_input, time_units_input, acqfreq_input, header_num_input, delim_input, tr_col_input, x_col_input, y_col_input, t_col_input, starting_frame_input
		try:
			with open ("super_res_data_wrangler_gui.defaults","r") as infile:
				print ("\nLoading GUI settings from super_res_data_wrangler_gui.defaults...")
				defaultdict = {}
				for line in infile:
					spl = line.split("\t")
					defaultdict[spl[0]] = spl[1].strip()
			convertfrom = (defaultdict["Convert from filetype"])
			convertto = (defaultdict["Convert to filetype"])
			files_contain = (defaultdict["Only find files with filenames that contain"])
			files_ext = (defaultdict["Only find files with filenames that end with"])
			xy_units_input = (defaultdict["XY units"])
			pix2um_input = float(defaultdict["Pixel size (um/px)"])
			time_units_input = (defaultdict["Time units"])
			acqfreq_input = float(defaultdict["Acquisition frequency (Hz)"])
			header_num_input = int(defaultdict["Number of headers"])
			delim_input = (defaultdict["Delimiter separating columns"])
			tr_col_input = int(defaultdict["Column containing Trajectory# data"])
			x_col_input = int(defaultdict["Column containing X-position data"])
			y_col_input = int(defaultdict["Column containing Y-position data"])
			t_col_input = int(defaultdict["Column containing Time data"])
			starting_frame_input = int(defaultdict["Starting frame#"])
		except:
			print ("\nSettings could not be loaded")
		return
		
	# CHECK VARIABLES
	def check_variables():
		global xy_units_input, pix2um_input, time_units_input, acqfreq_input, header_num_input, delim_input, tr_col_input, x_col_input, y_col_input, t_col_input, starting_frame_input
		# XY Units
		try: 
			if xy_units_input not in ["px","um","nm"]:
				xy_units_input = "px"
		except:
			xy_units_input = "px"
		# Pixel size
		try:
			pix2um_input = float(pix2um_input)
			if pix2um_input <= 0:
				pix2um_input = 0.106
		except:
			pix2um_input = 0.106	
		# Time Units
		if time_units_input not in ["Frames","s"]:
			time_units_input = "Frames"
		# Acquisition frequency
		try:
			acqfreq_input = float(acqfreq_input)
			if acqfreq_input < 1:
				acqfreq_input = 1.0
		except:
			acqfreq_input = 50.0
		# Header number
		try:
			int(header_num_input)
		except:
			header_num_input = 0
		if header_num_input <0:
			header_num_input = 0
		# Delimiter
		if delim_input not in ["tab","comma","1 space", "2 spaces", "3 spaces", "4 spaces","semicolon"]:
			delim_input = "tab"
		# Trajectory# column
		try:
			int(tr_col_input)
		except:
			tr_col_input = 1
		if tr_col_input <1:
			tr_col_input = 1
		# X-position column
		try:
			int(x_col_input)
		except:
			x_col_input = 2
		if x_col_input <1:
			x_col_input = 1
		# Y-position column
		try:
			int(y_col_input)
		except:
			y_col_input = 3
		if y_col_input <1:
			y_col_input = 1
		# Time column
		try:
			int(t_col_input)
		except:
			t_col_input = 4
		if t_col_input <1:
			t_col_input = 1
		# Starting frame
		try:
			int(starting_frame_input)
		except:
			starting_frame_input = 1
		if starting_frame_input <0:
			starting_frame_input = 0
		return
		
	# Update 
	def update_buttons():
		
		# Tab 1 (update Input filetype and Output filetype dropdowns)
		window.Element('-SELECT_INPUT-').update(convertfrom)
		window.Element('-SELECT_OUTPUT-').update(convertto)
		window.Element('-FILES_CONTAIN-').update(files_contain)
		window.Element('-FILES_EXT-').update(files_ext)
		
		# TAB 1, 2 CONVERT FROM TXT
		if convertfrom == "txt":
			# Tab 1 (swap to txt input filetype subtab)
			window.Element('-TXT_INPUT_TAB-').update(disabled = False)
			window['-SELECT_INPUT_TAB-'].Widget.select(0)
			window.Element('-TRC_INPUT_TAB-').update(disabled = True)
			window.Element('-TRXYT_INPUT_TAB-').update(disabled = True)
			window.Element('-ASCII_INPUT_TAB-').update(disabled = True)
			window.Element('-DCASCII_INPUT_TAB-').update(disabled = True)
			window.Element('-CSV_INPUT_TAB-').update(disabled = True)
			window.Element('-OTHER_INPUT_TAB-').update(disabled = True)
			# Tab 2 (swap to txt specific browse file option)
			window.Element('-INPUT_FILE-').update(visible = False)
			window.Element('-BROWSE_FILE_TRC-').update(visible = False)
			window.Element('-BROWSE_FILE_TRXYT-').update(visible = False)
			window.Element('-BROWSE_FILE_ASCII-').update(visible = False)
			window.Element('-BROWSE_FILE_CSV-').update(visible = False)
			window.Element('-BROWSE_FILE_OTHER-').update(visible = False)
			window.Element('-BROWSE_FILE_TXT-').update(visible = True)
			window.Element('-INPUT_FILE-').update(visible = True)
		
		# TAB 1, 2 CONVERT FROM TRC
		elif convertfrom == 'trc':
			# Tab 1 (swap to trc input filetype subtab)
			window.Element('-TRC_INPUT_TAB-').update(disabled = False)
			window['-SELECT_INPUT_TAB-'].Widget.select(1)
			window.Element('-TXT_INPUT_TAB-').update(disabled = True)
			window.Element('-TRXYT_INPUT_TAB-').update(disabled = True)
			window.Element('-ASCII_INPUT_TAB-').update(disabled = True)
			window.Element('-DCASCII_INPUT_TAB-').update(disabled = True)
			window.Element('-CSV_INPUT_TAB-').update(disabled = True)
			window.Element('-OTHER_INPUT_TAB-').update(disabled = True)
			# Tab 2 (swap to trc specific browse file option)
			window.Element('-INPUT_FILE-').update(visible = False)
			window.Element('-BROWSE_FILE_TXT-').update(visible = False)
			window.Element('-BROWSE_FILE_TRXYT-').update(visible = False)
			window.Element('-BROWSE_FILE_ASCII-').update(visible = False)
			window.Element('-BROWSE_FILE_CSV-').update(visible = False)
			window.Element('-BROWSE_FILE_OTHER-').update(visible = False)
			window.Element('-BROWSE_FILE_TRC-').update(visible = True)
			window.Element('-INPUT_FILE-').update(visible = True)
		
		# TAB 1, 2 CONVERT FROM TRXYT	
		elif convertfrom == 'trxyt':
			# Tab 1 (swap to trxyt input filetype subtab)
			window.Element('-TRXYT_INPUT_TAB-').update(disabled = False)
			window['-SELECT_INPUT_TAB-'].Widget.select(2)
			window.Element('-TXT_INPUT_TAB-').update(disabled = True)
			window.Element('-TRC_INPUT_TAB-').update(disabled = True)
			window.Element('-ASCII_INPUT_TAB-').update(disabled = True)
			window.Element('-DCASCII_INPUT_TAB-').update(disabled = True)
			window.Element('-CSV_INPUT_TAB-').update(disabled = True)
			window.Element('-OTHER_INPUT_TAB-').update(disabled = True)	
			# Tab 2 (swap to trxyt specific browse file option)
			window.Element('-INPUT_FILE-').update(visible = False)
			window.Element('-BROWSE_FILE_TXT-').update(visible = False)
			window.Element('-BROWSE_FILE_TRC-').update(visible = False)
			window.Element('-BROWSE_FILE_ASCII-').update(visible = False)
			window.Element('-BROWSE_FILE_CSV-').update(visible = False)
			window.Element('-BROWSE_FILE_OTHER-').update(visible = False)
			window.Element('-BROWSE_FILE_TRXYT-').update(visible = True)
			window.Element('-INPUT_FILE-').update(visible = True)			
		
		# TAB 1, 2 CONVERT FROM ASCII
		elif convertfrom == 'ascii':
			# Tab 1 (swap to ascii input filetype subtab)
			window.Element('-ASCII_INPUT_TAB-').update(disabled = False)
			window['-SELECT_INPUT_TAB-'].Widget.select(3)
			window.Element('-TXT_INPUT_TAB-').update(disabled = True)
			window.Element('-TRC_INPUT_TAB-').update(disabled = True)
			window.Element('-TRXYT_INPUT_TAB-').update(disabled = True)
			window.Element('-DCASCII_INPUT_TAB-').update(disabled = True)
			window.Element('-CSV_INPUT_TAB-').update(disabled = True)
			window.Element('-OTHER_INPUT_TAB-').update(disabled = True)
			# Tab 2 (swap to ascii specific browse file option)
			window.Element('-INPUT_FILE-').update(visible = False)
			window.Element('-BROWSE_FILE_TXT-').update(visible = False)
			window.Element('-BROWSE_FILE_TRC-').update(visible = False)
			window.Element('-BROWSE_FILE_TRXYT-').update(visible = False)
			window.Element('-BROWSE_FILE_CSV-').update(visible = False)
			window.Element('-BROWSE_FILE_OTHER-').update(visible = False)
			window.Element('-BROWSE_FILE_ASCII-').update(visible = True)
			window.Element('-INPUT_FILE-').update(visible = True)
		
		# TAB 1, 2 CONVERT FROM ASCII(DRIFT CORRECTED)
		elif convertfrom == 'ascii(drift corrected)':
			# Tab 1 (swap to ascii(drift corrected) input filetype subtab)
			window.Element('-DCASCII_INPUT_TAB-').update(disabled = False)
			window['-SELECT_INPUT_TAB-'].Widget.select(4)
			window.Element('-TXT_INPUT_TAB-').update(disabled = True)
			window.Element('-TRC_INPUT_TAB-').update(disabled = True)
			window.Element('-TRXYT_INPUT_TAB-').update(disabled = True)
			window.Element('-ASCII_INPUT_TAB-').update(disabled = True)
			window.Element('-CSV_INPUT_TAB-').update(disabled = True)
			window.Element('-OTHER_INPUT_TAB-').update(disabled = True)
			# Tab 2 (swap to ascii specific browse file option)
			window.Element('-INPUT_FILE-').update(visible = False)
			window.Element('-BROWSE_FILE_TXT-').update(visible = False)
			window.Element('-BROWSE_FILE_TRC-').update(visible = False)
			window.Element('-BROWSE_FILE_TRXYT-').update(visible = False)
			window.Element('-BROWSE_FILE_CSV-').update(visible = False)
			window.Element('-BROWSE_FILE_OTHER-').update(visible = False)
			window.Element('-BROWSE_FILE_ASCII-').update(visible = True)
			window.Element('-INPUT_FILE-').update(visible = True)
		
		# TAB 1, 2 CONVERT FROM CSV
		elif convertfrom == 'csv(TrackMate)':
			# Tab 1 (swap to csv(TrackMate) input filetype subtab)
			window.Element('-CSV_INPUT_TAB-').update(disabled = False)
			window['-SELECT_INPUT_TAB-'].Widget.select(5)
			window.Element('-TXT_INPUT_TAB-').update(disabled = True)
			window.Element('-TRC_INPUT_TAB-').update(disabled = True)
			window.Element('-TRXYT_INPUT_TAB-').update(disabled = True)
			window.Element('-ASCII_INPUT_TAB-').update(disabled = True)
			window.Element('-OTHER_INPUT_TAB-').update(disabled = True)
			# Tab 2 (swap to csv specific browse file option)
			window.Element('-INPUT_FILE-').update(visible = False)
			window.Element('-BROWSE_FILE_TXT-').update(visible = False)
			window.Element('-BROWSE_FILE_TRC-').update(visible = False)
			window.Element('-BROWSE_FILE_TRXYT-').update(visible = False)
			window.Element('-BROWSE_FILE_ASCII-').update(visible = False)
			window.Element('-BROWSE_FILE_OTHER-').update(visible = False)
			window.Element('-BROWSE_FILE_CSV-').update(visible = True)
			window.Element('-INPUT_FILE-').update(visible = True)
		
		# TAB 1, 2 CONVERT FROM OTHER
		elif convertfrom == 'other(manual input)':
			# Tab 1 (swap to other(manual input) input filetype subtab)
			window.Element('-OTHER_INPUT_TAB-').update(disabled = False)
			window['-SELECT_INPUT_TAB-'].Widget.select(6)
			window.Element('-TXT_INPUT_TAB-').update(disabled = True)
			window.Element('-TRC_INPUT_TAB-').update(disabled = True)
			window.Element('-TRXYT_INPUT_TAB-').update(disabled = True)
			window.Element('-ASCII_INPUT_TAB-').update(disabled = True)
			window.Element('-CSV_INPUT_TAB-').update(disabled = True)
			# Tab 2 (swap to other specific browse file option)
			window.Element('-INPUT_FILE-').update(visible = False)
			window.Element('-BROWSE_FILE_TXT-').update(visible = False)
			window.Element('-BROWSE_FILE_TRC-').update(visible = False)
			window.Element('-BROWSE_FILE_TRXYT-').update(visible = False)
			window.Element('-BROWSE_FILE_ASCII-').update(visible = False)
			window.Element('-BROWSE_FILE_CSV-').update(visible = False)
			window.Element('-BROWSE_FILE_OTHER-').update(visible = True)
			window.Element('-INPUT_FILE-').update(visible = True)
		
		# TAB 1 CONVERT TO TXT
		if convertto == 'txt':
			# Tab 1 (swap to txt output filetype subtab)
			window.Element('-TXT_OUTPUT_TAB-').update(disabled = False)
			window['-SELECT_OUTPUT_TAB-'].Widget.select(0)
			window.Element('-TRC_OUTPUT_TAB-').update(disabled = True)
			window.Element('-TRXYT_OUTPUT_TAB-').update(disabled = True)
			window.Element('-ASCII_OUTPUT_TAB-').update(disabled = True)
			window.Element('-DCASCII_OUTPUT_TAB-').update(disabled = True)
			
		# TAB 1 CONVERT TO TRC
		elif convertto == 'trc':
			# Tab 1 (swap to trc output filetype subtab)
			window.Element('-TRC_OUTPUT_TAB-').update(disabled = False)
			window['-SELECT_OUTPUT_TAB-'].Widget.select(1)
			window.Element('-TXT_OUTPUT_TAB-').update(disabled = True)
			window.Element('-TRXYT_OUTPUT_TAB-').update(disabled = True)
			window.Element('-ASCII_OUTPUT_TAB-').update(disabled = True)
			window.Element('-DCASCII_OUTPUT_TAB-').update(disabled = True)
		
		# TAB 1 CONVERT TO TRXYT
		elif convertto == 'trxyt':
			# Tab 1 (swap to trxyt output filetype subtab)
			window.Element('-TRXYT_OUTPUT_TAB-').update(disabled = False)
			window['-SELECT_OUTPUT_TAB-'].Widget.select(2)
			window.Element('-TXT_OUTPUT_TAB-').update(disabled = True)
			window.Element('-TRC_OUTPUT_TAB-').update(disabled = True)
			window.Element('-ASCII_OUTPUT_TAB-').update(disabled = True)
			window.Element('-DCASCII_OUTPUT_TAB-').update(disabled = True)
		
		# TAB 1 CONVERT TO ASCII	
		elif convertto == 'ascii':
			# Tab 1 (swap to ascii output filetype subtab)
			window.Element('-ASCII_OUTPUT_TAB-').update(disabled = False)
			window['-SELECT_OUTPUT_TAB-'].Widget.select(3)
			window.Element('-TXT_OUTPUT_TAB-').update(disabled = True)
			window.Element('-TRC_OUTPUT_TAB-').update(disabled = True)
			window.Element('-TRXYT_OUTPUT_TAB-').update(disabled = True)
			window.Element('-DCASCII_OUTPUT_TAB-').update(disabled = True)
		
		# TAB 1 CONVERT TO ASCII(DRIFT CORRECTED)	
		elif convertto == 'ascii(drift corrected)':
			# Tab 1 (swap to ascii(drift corrected) output filetype subtab)
			window.Element('-DCASCII_OUTPUT_TAB-').update(disabled = False)
			window['-SELECT_OUTPUT_TAB-'].Widget.select(4)
			window.Element('-TXT_OUTPUT_TAB-').update(disabled = True)
			window.Element('-TRC_OUTPUT_TAB-').update(disabled = True)
			window.Element('-TRXYT_OUTPUT_TAB-').update(disabled = True)
			window.Element('-ASCII_OUTPUT_TAB-').update(disabled = True)
		
		# Tab 2 (Toggle Browse file/Browse folder options)
		if browse_file == True:
			window.Element('-BROWSE_FILE_TXT-').update(disabled = False)
			window.Element('-BROWSE_FILE_TRC-').update(disabled = False)
			window.Element('-BROWSE_FILE_TRXYT-').update(disabled = False)
			window.Element('-BROWSE_FILE_ASCII-').update(disabled = False)
			window.Element('-BROWSE_FILE_CSV-').update(disabled = False)
			window.Element('-BROWSE_FILE_OTHER-').update(disabled = False)
			window.Element('-INPUT_FILE-').update(disabled = False, text_color = "#cccdcf")
			window.Element('-BROWSE_FOLDER-').update(disabled = True)
			window.Element('-INPUT_FOLDER-').update(disabled = True, text_color = "grey", background_color = "#313641")
			window.Element('-FIND_FILES_TEXT-').update(visible = False)
			window.Element('-FIND_FILES-').update(visible = False)
			window.Element('-FILES_CONTAIN_TEXT-').update(visible = False)
			window.Element('-FILES_CONTAIN-').update(visible = False)
			window.Element('-FILES_EXT_TEXT-').update(visible = False)
			window.Element('-FILES_EXT-').update(visible = False)
			use_tree = False
		else:
			window.Element('-BROWSE_FILE_TXT-').update(disabled = True)
			window.Element('-BROWSE_FILE_TRC-').update(disabled = True)
			window.Element('-BROWSE_FILE_TRXYT-').update(disabled = True)
			window.Element('-BROWSE_FILE_ASCII-').update(disabled = True)
			window.Element('-BROWSE_FILE_CSV-').update(disabled = True)
			window.Element('-BROWSE_FILE_OTHER-').update(disabled = True)
			window.Element('-INPUT_FILE-').update(disabled = True, text_color = "grey")
			window.Element('-BROWSE_FOLDER-').update(disabled = False)
			window.Element('-INPUT_FOLDER-').update(disabled = False,  text_color = "#cccdcf")
			window.Element('-FIND_FILES_TEXT-').update(visible = True)
			window.Element('-FIND_FILES-').update(visible = True)
			window.Element('-FILES_CONTAIN_TEXT-').update(visible = True)
			window.Element('-FILES_CONTAIN-').update(visible = True)
			window.Element('-FILES_EXT_TEXT-').update(visible = True)
			window.Element('-FILES_EXT-').update(visible = True)
			use_tree = True
			if find_files == True:
				window.Element('-FIND_FILES-').update(disabled = False)
				window.Element('-FIND_FILES_TEXT-').update(text_color = "white")
			else:
				window.Element('-FIND_FILES-').update(disabled = True)
				window.Element('-FIND_FILES_TEXT-').update(text_color = "grey")
			if convertfrom == "other(manual input)":
				window.Element('-FILES_EXT_TEXT-').update(text_color = "white")
				window.Element('-FILES_EXT-').update(disabled = False)
			else:
				window.Element('-FILES_EXT_TEXT-').update(visible = False)
				window.Element('-FILES_EXT-').update("",visible = False)
			
		# Tab 3 (update input file parameters)
		window.Element('-XY_UNITS_INPUT-').update(xy_units_input)
		window.Element('-PIXEL_SIZE_INPUT-').update(pix2um_input)
		window.Element('-TIME_UNITS_INPUT-').update(time_units_input)
		window.Element('-ACQ_FREQ_INPUT-').update(acqfreq_input)
		window.Element('-HEADER_NUM_INPUT-').update(header_num_input)
		window.Element('-DELIM_INPUT-').update(delim_input)
		window.Element('-TR_COL_INPUT-').update(tr_col_input)
		window.Element('-X_COL_INPUT-').update(x_col_input)
		window.Element('-Y_COL_INPUT-').update(y_col_input)
		window.Element('-T_COL_INPUT-').update(t_col_input)
		window.Element('-STARTING_FRAME_INPUT-').update(starting_frame_input)
		
		# Tab 2 (Toggle 'Find files' button) 
		if find_files == True:
			window.Element('-FIND_FILES-').update(disabled = False)
			window.Element('-FILES_CONTAIN_TEXT-').update(text_color = "white")
			window.Element('-FILES_CONTAIN-').update(disabled = False, text_color = "white")
			window.Element('-FILES_EXT_TEXT-').update(text_color = "white")
			window.Element('-FILES_EXT-').update(disabled = False, text_color = "white")
		else:	
			window.Element('-FIND_FILES-').update(disabled = True)
			window.Element('-FILES_CONTAIN_TEXT-').update(text_color = "grey")
			window.Element('-FILES_CONTAIN-').update(disabled = True, text_color = "grey")
			window.Element('-FILES_EXT_TEXT-').update(text_color = "grey")
			window.Element('-FILES_EXT-').update(disabled = True, text_color = "grey")
		
		# Tab 2 (Toggle 'LOAD DATA' button)
		if load_data_button == True:
			window.Element('-LOAD-').update(disabled = False)
		else:
			window.Element('-LOAD-').update(disabled = True)
		
		# Tab 2 (Toggle Tree)
		if use_tree == True:
			window.Element("-TREE-").update(visible = True)
		else:
			window.Element("-TREE-").update(visible = False)
			pass
		
		# TAB 3 CONVERT TO TXT
		if convertto == "txt":
			# Tab 3 (Change output file parameters to txt parameters)
			window.Element('-XY_UNITS_OUTPUT-').update("px")
			window.Element('-TIME_UNITS_OUTPUT-').update("Frames")
			window.Element('-HEADER_NUM_OUTPUT-').update(3)
			window.Element('-DELIM_OUTPUT-').update("tab")
			window.Element('-TR_COL_OUTPUT-').update(1)
			window.Element('-X_COL_OUTPUT-').update(3)
			window.Element('-Y_COL_OUTPUT-').update(4)
			window.Element('-T_COL_OUTPUT-').update(2)	
			window.Element('-STARTING_FRAME_OUTPUT-').update(1)
			window.Element('-T_COL_TEXT_OUTPUT-').update("          Frame col:")				
		
		# TAB 3 CONVERT TO TRC
		elif convertto == "trc":
			# Tab 3 (Change output file parameters to trc parameters)
			window.Element('-XY_UNITS_OUTPUT-').update("px")
			window.Element('-TIME_UNITS_OUTPUT-').update("Frames")
			window.Element('-HEADER_NUM_OUTPUT-').update(0)
			window.Element('-DELIM_OUTPUT-').update("tab")
			window.Element('-TR_COL_OUTPUT-').update(1)
			window.Element('-X_COL_OUTPUT-').update(3)
			window.Element('-Y_COL_OUTPUT-').update(4)
			window.Element('-T_COL_OUTPUT-').update(2)
			window.Element('-STARTING_FRAME_OUTPUT-').update(1)
			window.Element('-T_COL_TEXT_OUTPUT-').update("          Frame col:")			
		
		# TAB 3 CONVERT TO TRXYT
		elif convertto == "trxyt":
			# Tab 3 (Change output file parameters to trxyt parameters)
			window.Element('-XY_UNITS_OUTPUT-').update("um")
			window.Element('-TIME_UNITS_OUTPUT-').update("s")
			window.Element('-HEADER_NUM_OUTPUT-').update(0)
			window.Element('-DELIM_OUTPUT-').update("1 space")
			window.Element('-TR_COL_OUTPUT-').update(1)
			window.Element('-X_COL_OUTPUT-').update(2)
			window.Element('-Y_COL_OUTPUT-').update(3)
			window.Element('-T_COL_OUTPUT-').update(4)
			window.Element('-STARTING_FRAME_OUTPUT-').update(1)
			window.Element('-T_COL_TEXT_OUTPUT-').update("          Time col:")				
		
		# TAB 3 CONVERT TO ASCII
		elif convertto == "ascii":
			# Tab 3 (Change output file parameters to ascii parameters)
			window.Element('-XY_UNITS_OUTPUT-').update("px")
			window.Element('-TIME_UNITS_OUTPUT-').update("Frames")
			window.Element('-HEADER_NUM_OUTPUT-').update(0)
			window.Element('-DELIM_OUTPUT-').update("comma")
			window.Element('-TR_COL_OUTPUT-').update("ID")
			window.Element('-X_COL_OUTPUT-').update(3)
			window.Element('-Y_COL_OUTPUT-').update(4)
			window.Element('-T_COL_OUTPUT-').update(2)
			window.Element('-STARTING_FRAME_OUTPUT-').update(0)
			window.Element('-T_COL_TEXT_OUTPUT-').update("          Frame col:")	
		
		#TAB 3 CONVERT TO ASCII(DRIFT CORRECTED)
		elif convertto == "ascii(drift corrected)":
			# Tab 3 (Change output file parameters to ascii(drift corrected) parameters)
			window.Element('-XY_UNITS_OUTPUT-').update("nm")
			window.Element('-TIME_UNITS_OUTPUT-').update("Frames")
			window.Element('-HEADER_NUM_OUTPUT-').update(0)
			window.Element('-DELIM_OUTPUT-').update("comma")
			window.Element('-TR_COL_OUTPUT-').update("ID")
			window.Element('-X_COL_OUTPUT-').update(3)
			window.Element('-Y_COL_OUTPUT-').update(4)
			window.Element('-T_COL_OUTPUT-').update(2)
			window.Element('-STARTING_FRAME_OUTPUT-').update(0)
			window.Element('-T_COL_TEXT_OUTPUT-').update("          Frame col:")				
		
		if convertfrom in ["txt", "trc", "trxyt", "ascii", "ascii(drift corrected)", "csv(TrackMate)"]:
			# Tab 3 (Change title of input file frame and output file frame) 
			window.Element('-INPUT_FILE_FRAME-').update("Input {} parameters".format(convertfrom))
			window.Element('-OUTPUT_FILE_FRAME-').update("Output {} parameters".format(convertto))
			
			# Tab 3 (Disable input file parameters) 
			window.Element('-DEFINE_SPACE_TIME_INPUT-').update(text_color = "grey")
			window.Element('-XY_UNITS_TEXT_INPUT-').update(text_color = "grey")
			window.Element('-XY_UNITS_INPUT-').update(disabled = True)
			window.Element('-TIME_UNITS_TEXT_INPUT-').update(text_color = "grey")
			window.Element('-TIME_UNITS_INPUT-').update(disabled = True)
			window.Element('-DEFINE_FILE_STRUCTURE_INPUT-').update(text_color = "grey")
			window.Element('-HEADER_NUM_TEXT_INPUT-').update(text_color = "grey")
			window.Element('-HEADER_NUM_INPUT-').update(disabled = True)
			window.Element('-DELIM_TEXT_INPUT-').update(text_color = "grey")
			window.Element('-DELIM_INPUT-').update(disabled = True)
			window.Element('-TRAJ_DATA_COLS_INPUT-').update(text_color = "grey")
			window.Element('-TR_COL_TEXT_INPUT-').update(text_color = "grey")
			window.Element('-TR_COL_INPUT-').update(disabled = True)
			window.Element('-X_COL_TEXT_INPUT-').update(text_color = "grey")
			window.Element('-X_COL_INPUT-').update(disabled = True)
			window.Element('-Y_COL_TEXT_INPUT-').update(text_color = "grey")
			window.Element('-Y_COL_INPUT-').update(disabled = True)
			window.Element('-T_COL_TEXT_INPUT-').update(text_color = "grey")
			window.Element('-T_COL_INPUT-').update(disabled = True)
			window.Element('-STARTING_FRAME_TEXT_INPUT-').update(text_color = "grey")
			window.Element('-STARTING_FRAME_INPUT-').update(disabled = True)
			
			# TAB 3 CONVERT FROM TXT
			if convertfrom == "txt":
				# Tab 3 (Change input file parameters to txt parameters)
				window.Element('-XY_UNITS_INPUT-').update("px")
				window.Element('-TIME_UNITS_INPUT-').update("Frames")
				window.Element('-HEADER_NUM_INPUT-').update(3)
				window.Element('-DELIM_INPUT-').update("tab")
				window.Element('-TR_COL_INPUT-').update(1)
				window.Element('-X_COL_INPUT-').update(3)
				window.Element('-Y_COL_INPUT-').update(4)
				window.Element('-T_COL_INPUT-').update(2)
				window.Element('-STARTING_FRAME_TEXT_INPUT-').update(visible = True)
				window.Element('-STARTING_FRAME_INPUT-').update(1, visible = True)
				window.Element('-T_COL_TEXT_INPUT-').update("          Frame col:")	
				
				if convertto == "trc":
					# Tab 3 (Change input txt parameters based on trc output file parameters)
					window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = False)
					window.Element('-PIXEL_SIZE_INPUT-').update(visible = False)
					window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = False)
					window.Element('-ACQ_FREQ_INPUT-').update(visible = False)
					window.Element('-PARAMETER_MESSAGE-').update("No parameter selection required")
				
				elif convertto == "trxyt":
					# Tab 3 (Change input txt parameters based on trxyt output file parameters)
					window.Element('-DEFINE_SPACE_TIME_INPUT-').update(text_color = "white")
					window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = True)
					window.Element('-PIXEL_SIZE_INPUT-').update(visible = True)
					window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = True)
					window.Element('-ACQ_FREQ_INPUT-').update(visible = True)
					window.Element('-PARAMETER_MESSAGE-').update("Please enter pixel size (um/px) and acquisition frequency (Hz) for conversion from px to um and Frames to s")
				
				elif convertto == "ascii":
					# Tab 3 (Change input txt parameters based on ascii output file parameters)
					window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = False)
					window.Element('-PIXEL_SIZE_INPUT-').update(visible = False)
					window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = False)
					window.Element('-ACQ_FREQ_INPUT-').update(visible = False)
					window.Element('-PARAMETER_MESSAGE-').update("No parameter selection required")
				
				elif convertto == "ascii(drift corrected)":
					# Tab 3 (Change input txt parameters based on ascii(drift corrected) output file parameters)
					window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = True)
					window.Element('-PIXEL_SIZE_INPUT-').update(visible = True)
					window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = False)
					window.Element('-ACQ_FREQ_INPUT-').update(visible = False)
					window.Element('-PARAMETER_MESSAGE-').update("Please enter pixel size (um/px) for conversion from px to nm")
			
			# TAB 3 CONVERT FROM TRC
			elif convertfrom == "trc":
				# Tab 3 (Change input file parameters to trc parameters)
				window.Element('-XY_UNITS_INPUT-').update("px")
				window.Element('-TIME_UNITS_INPUT-').update("Frames")
				window.Element('-HEADER_NUM_INPUT-').update(0)
				window.Element('-DELIM_INPUT-').update("tab")
				window.Element('-TR_COL_INPUT-').update(1)
				window.Element('-X_COL_INPUT-').update(3)
				window.Element('-Y_COL_INPUT-').update(4)
				window.Element('-T_COL_INPUT-').update(2)
				window.Element('-STARTING_FRAME_TEXT_INPUT-').update(visible = True)
				window.Element('-STARTING_FRAME_INPUT-').update(1, visible = True)
				window.Element('-T_COL_TEXT_INPUT-').update("          Frame col:")
				
				if convertto == "txt":
					# Tab 3 (Change input trc parameters based on output txt parameters)
					window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = False)
					window.Element('-PIXEL_SIZE_INPUT-').update(visible = False)
					window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = False)
					window.Element('-ACQ_FREQ_INPUT-').update(visible = False)
					window.Element('-PARAMETER_MESSAGE-').update("No parameter selection required")
					
				elif convertto == "trxyt":
					# Tab 3 (Change input trc parameters based on output trxyt parameters)
					window.Element('-DEFINE_SPACE_TIME_INPUT-').update(text_color = "white")
					window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = True)
					window.Element('-PIXEL_SIZE_INPUT-').update(visible = True)
					window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = True)
					window.Element('-ACQ_FREQ_INPUT-').update(visible = True)
					window.Element('-PARAMETER_MESSAGE-').update("Please enter pixel size (um/px) and acquisition frequency (Hz) for conversion from px to um and Frames to s")
					
				elif convertto == "ascii":
					# Tab 3 (Change input trc parameters based on output ascii parameters)
					window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = False)
					window.Element('-PIXEL_SIZE_INPUT-').update(visible = False)
					window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = False)
					window.Element('-ACQ_FREQ_INPUT-').update(visible = False)
					window.Element('-PARAMETER_MESSAGE-').update("No parameter selection required")
					
				elif convertto == "ascii(drift corrected)":
					# Tab 3 (Change input trc parameters based on output ascii/ascii(drift corrected) parameters)
					window.Element('-DEFINE_SPACE_TIME_INPUT-').update(text_color = "white")
					window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = False)
					window.Element('-ACQ_FREQ_INPUT-').update(visible = False)
					window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = True)
					window.Element('-PIXEL_SIZE_INPUT-').update(visible = True)
					window.Element('-PARAMETER_MESSAGE-').update("Please enter pixel size (um/px) for conversion from px to nm")
			
			# TAB 3 CONVERT FROM TRXYT		
			elif convertfrom == "trxyt":
				# Tab 3 (Change input file parameters to trxyt parameters)
				window.Element('-XY_UNITS_INPUT-').update("um")
				window.Element('-TIME_UNITS_INPUT-').update("s")
				window.Element('-HEADER_NUM_INPUT-').update(0)
				window.Element('-DELIM_INPUT-').update("1 space")
				window.Element('-TR_COL_INPUT-').update(1)
				window.Element('-X_COL_INPUT-').update(2)
				window.Element('-Y_COL_INPUT-').update(3)
				window.Element('-T_COL_INPUT-').update(4)
				window.Element('-STARTING_FRAME_TEXT_INPUT-').update(visible = True)
				window.Element('-STARTING_FRAME_INPUT-').update(1, visible = True)
				window.Element('-T_COL_TEXT_INPUT-').update("          Time col:")
				
				if convertto in ["txt", "trc", "ascii"]:
					# Tab 3 (Change input trxyt parameters based on output txt/trc/ascii parameters)
					window.Element('-DEFINE_SPACE_TIME_INPUT-').update(text_color = "white")
					window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = True)
					window.Element('-PIXEL_SIZE_INPUT-').update(visible = True)
					window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = True)
					window.Element('-ACQ_FREQ_INPUT-').update(visible = True)
					window.Element('-PARAMETER_MESSAGE-').update("Please enter pixel size (um/px) and acquisition frequency (Hz) for conversion from um to px and s to Frames")
				
				elif convertto == "ascii(drift corrected)":
					# Tab 3 (Change input trxyt parameters based on output ascii(drift corrected) parameters)
					window.Element('-DEFINE_SPACE_TIME_INPUT-').update(text_color = "white")
					window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = False)
					window.Element('-PIXEL_SIZE_INPUT-').update(visible = False)
					window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = True)
					window.Element('-ACQ_FREQ_INPUT-').update(visible = True)
					window.Element('-PARAMETER_MESSAGE-').update("Please select acquisition frequency (Hz) for conversion from s to Frames")
			
			# TAB 3 CONVERT FROM ASCII
			elif convertfrom == "ascii":				
				# Tab 3 (Change input file parameters to ascii parameters)	
				window.Element('-XY_UNITS_INPUT-').update("px")
				window.Element('-TIME_UNITS_INPUT-').update("Frames")
				window.Element('-HEADER_NUM_INPUT-').update(0)
				window.Element('-DELIM_INPUT-').update("comma")
				window.Element('-TR_COL_INPUT-').update("ID")
				window.Element('-X_COL_INPUT-').update(3)
				window.Element('-Y_COL_INPUT-').update(4)
				window.Element('-T_COL_INPUT-').update(2)
				window.Element('-STARTING_FRAME_TEXT_INPUT-').update(visible = True)
				window.Element('-STARTING_FRAME_INPUT-').update(0, visible = True)
				window.Element('-T_COL_TEXT_INPUT-').update("          Frame col:")
							
				if convertto in ["txt","trc"]:
					# Tab 3 (Change input ascii parameters based on output txt/trc parameters)	
					window.Element('-DEFINE_SPACE_TIME_INPUT-').update(text_color = "white")
					window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = False)
					window.Element('-ACQ_FREQ_INPUT-').update(visible = False)
					window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = False)
					window.Element('-PIXEL_SIZE_INPUT-').update(visible = False)
					window.Element('-PARAMETER_MESSAGE-').update("No parameter selection required")
					
				elif convertto == "trxyt":
					# Tab 3 (Change input ascii parameters based on output trxyt parameters)
					window.Element('-DEFINE_SPACE_TIME_INPUT-').update(text_color = "white")
					window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = True)
					window.Element('-PIXEL_SIZE_INPUT-').update(visible = True)
					window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = True)
					window.Element('-ACQ_FREQ_INPUT-').update(visible = True)
					window.Element('-PARAMETER_MESSAGE-').update("Please enter pixel size (um/px) and acquisition frequency (Hz) for conversion from px to um and Frames to s")
				
				elif convertto == "ascii(drift corrected)":
					# Tab 3 (Change input ascii parameters based on output ascii(drift corrected) parameters)
					window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = True)
					window.Element('-PIXEL_SIZE_INPUT-').update(visible = True)
					window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = False)
					window.Element('-ACQ_FREQ_INPUT-').update(visible = False)
					window.Element('-PARAMETER_MESSAGE-').update("Please enter pixel size (um/px) for conversion from px to nm")
			
			# TAB 3 CONVERT FROM ASCII(DRIFT CORRECTED)
			elif convertfrom == "ascii(drift corrected)":
				# Tab 3 (Change input file parameters to ascii(drift corrected) parameters)
				window.Element('-XY_UNITS_INPUT-').update("nm")
				window.Element('-TIME_UNITS_INPUT-').update("Frames")
				window.Element('-HEADER_NUM_INPUT-').update(0)
				window.Element('-DELIM_INPUT-').update("comma")
				window.Element('-TR_COL_INPUT-').update("ID")
				window.Element('-X_COL_INPUT-').update(3)
				window.Element('-Y_COL_INPUT-').update(4)
				window.Element('-T_COL_INPUT-').update(2)
				window.Element('-STARTING_FRAME_TEXT_INPUT-').update(visible = True)
				window.Element('-STARTING_FRAME_INPUT-').update(0, visible = True)
				window.Element('-T_COL_TEXT_INPUT-').update("          Frame col:")
				
				if convertto in ["txt","trc","ascii"]:
					# Tab 3 (Change input ascii(drift corrected) parameters based on output txt/trc/ascii parameters)
					window.Element('-DEFINE_SPACE_TIME_INPUT-').update(text_color = "white")
					window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = False)
					window.Element('-ACQ_FREQ_INPUT-').update(visible = False)
					window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = True)
					window.Element('-PIXEL_SIZE_INPUT-').update(visible = True)
					window.Element('-PARAMETER_MESSAGE-').update("Please enter pixel size (um/px) for conversion from nm to px")
				
				elif convertto == "trxyt":
					# Tab 3 (Change input ascii(drift corrected) parameters based on output trxyt parameters)
					window.Element('-DEFINE_SPACE_TIME_INPUT-').update(text_color = "white")
					window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = False)
					window.Element('-PIXEL_SIZE_INPUT-').update(visible = False)
					window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = True)
					window.Element('-ACQ_FREQ_INPUT-').update(visible = True)
					window.Element('-PARAMETER_MESSAGE-').update("Please enter acquisition frequency (Hz) for conversion from Frames to s")	
			
			# TAB 3 CONVERT FROM CSV
			elif convertfrom == "csv(TrackMate)":
				# Tab 3 (Change input file parameters to csv(TrackMate) parameters)	
				window.Element('-DEFINE_SPACE_TIME_INPUT-').update(text_color = "white")
				window.Element('-XY_UNITS_TEXT_INPUT-').update(text_color = "white")
				window.Element('-XY_UNITS_INPUT-').update(disabled = False)
				window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(text_color = "white")
				window.Element('-PIXEL_SIZE_INPUT-').update(disabled = False, text_color = "white")
				window.Element('-TIME_UNITS_TEXT_INPUT-').update(text_color = "white")
				window.Element('-TIME_UNITS_INPUT-').update(disabled = False)
				window.Element('-ACQ_FREQ_TEXT_INPUT-').update(text_color = "white")
				window.Element('-ACQ_FREQ_INPUT-').update(disabled = False, text_color = "white")
				window.Element('-DEFINE_FILE_STRUCTURE_INPUT-').update(text_color = "white")
				window.Element('-HEADER_NUM_TEXT_INPUT-').update(text_color = "white")
				window.Element('-HEADER_NUM_INPUT-').update(disabled = False)
				window.Element('-DELIM_INPUT-').update("comma")
				window.Element('-TRAJ_DATA_COLS_INPUT-').update(text_color = "white")
				window.Element('-TR_COL_TEXT_INPUT-').update(text_color = "white")
				window.Element('-TR_COL_INPUT-').update(disabled = False)
				window.Element('-X_COL_TEXT_INPUT-').update(text_color = "white")
				window.Element('-X_COL_INPUT-').update(disabled = False)
				window.Element('-Y_COL_TEXT_INPUT-').update(text_color = "white")
				window.Element('-Y_COL_INPUT-').update(disabled = False)
				window.Element('-T_COL_TEXT_INPUT-').update(text_color = "white")
				window.Element('-T_COL_INPUT-').update(disabled = False)
				window.Element('-STARTING_FRAME_TEXT_INPUT-').update(text_color = "white")
				window.Element('-STARTING_FRAME_INPUT-').update(disabled = False)
				window.Element('-PARAMETER_MESSAGE-').update("Please enter input file parameters")
				
				# Tab 3 (Toggle Pixel size parameter based on XY Unit selection)
				if xy_units_input == xy_units_output or xy_units_input in ["um","nm"] and xy_units_output in ["um","nm"]:
					window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = False)
					window.Element('-PIXEL_SIZE_INPUT-').update(visible = False)
				else:
					window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = True)
					window.Element('-PIXEL_SIZE_INPUT-').update(visible = True)
				
				# Tab 3 (Change name of Time column based on Time unit selection)	
				if time_units_input == "Frames":
					window.Element('-T_COL_TEXT_INPUT-').update("          Frame col:")
					window.Element('-STARTING_FRAME_TEXT_INPUT-').update(visible = True)
					window.Element('-STARTING_FRAME_INPUT-').update(visible = True)
				elif time_units_input == "s":
					window.Element('-T_COL_TEXT_INPUT-').update("          Time col:")
					window.Element('-STARTING_FRAME_TEXT_INPUT-').update(visible = False)
					window.Element('-STARTING_FRAME_INPUT-').update(visible = False)					
				if time_units_input != time_units_output:
					window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = True)
					window.Element('-ACQ_FREQ_INPUT-').update(visible = True)
				else:
					window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = False)
					window.Element('-ACQ_FREQ_INPUT-').update(visible = False)
		
		# TAB 3 CONVERT FROM OTHER				
		elif convertfrom == "other(manual input)":
			# Tab 3 (Change title of input file frame to include file extension of selected input files) 
			window.Element('-OUTPUT_FILE_FRAME-').update("Output {} parameters".format(convertto))
			
			# Tab 3 (Enable all input parameters)
			window.Element('-DEFINE_SPACE_TIME_INPUT-').update(text_color = "white")
			window.Element('-XY_UNITS_TEXT_INPUT-').update(text_color = "white")
			window.Element('-XY_UNITS_INPUT-').update(disabled = False)
			window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(text_color = "white")
			window.Element('-PIXEL_SIZE_INPUT-').update(disabled = False, text_color = "white")
			window.Element('-ACQ_FREQ_TEXT_INPUT-').update(text_color = "white")
			window.Element('-ACQ_FREQ_INPUT-').update(disabled = False, text_color = "white")
			window.Element('-TIME_UNITS_TEXT_INPUT-').update(text_color = "white")
			window.Element('-TIME_UNITS_INPUT-').update(disabled = False)
			window.Element('-DEFINE_FILE_STRUCTURE_INPUT-').update(text_color = "white")
			window.Element('-HEADER_NUM_TEXT_INPUT-').update(text_color = "white")
			window.Element('-HEADER_NUM_INPUT-').update(disabled = False)
			window.Element('-DELIM_TEXT_INPUT-').update(text_color = "white")
			window.Element('-DELIM_INPUT-').update(disabled = False)
			window.Element('-TRAJ_DATA_COLS_INPUT-').update(text_color = "white")
			window.Element('-TR_COL_TEXT_INPUT-').update(text_color = "white")
			window.Element('-TR_COL_INPUT-').update(disabled = False)
			window.Element('-X_COL_TEXT_INPUT-').update(text_color = "white")
			window.Element('-X_COL_INPUT-').update(disabled = False)
			window.Element('-Y_COL_TEXT_INPUT-').update(text_color = "white")
			window.Element('-Y_COL_INPUT-').update(disabled = False)
			window.Element('-T_COL_TEXT_INPUT-').update(text_color = "white")
			window.Element('-T_COL_INPUT-').update(disabled = False)
			window.Element('-STARTING_FRAME_TEXT_INPUT-').update(text_color = "white")
			window.Element('-STARTING_FRAME_INPUT-').update(disabled = False)
			window.Element('-PARAMETER_MESSAGE-').update("Please enter input file parameters")
			
			# Tab 3 (Toggle Pixel size parameter based on XY unit selection)
			if xy_units_input == xy_units_output:
				window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = False)
				window.Element('-PIXEL_SIZE_INPUT-').update(visible = False)
			elif xy_units_input in ["um","nm"] and xy_units_output in ["um","nm"]:
				window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = False)
				window.Element('-PIXEL_SIZE_INPUT-').update(visible = False)
			else:
				window.Element('-PIXEL_SIZE_TEXT_INPUT-').update(visible = True)
				window.Element('-PIXEL_SIZE_INPUT-').update(visible = True)
			
			# Tab 3 (Toggle Acquisition frequency parameter based on Time unit selection)
			if time_units_input == time_units_output:
				window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = False)
				window.Element('-ACQ_FREQ_INPUT-').update(visible = False)
			else:
				window.Element('-ACQ_FREQ_TEXT_INPUT-').update(visible = True)
				window.Element('-ACQ_FREQ_INPUT-').update(visible = True)
			
			# Tab 3 (Change name of Time column parameter based on Time unit selection)
			if time_units_input == "Frames":
				window.Element('-T_COL_TEXT_INPUT-').update("          Frame col:")
				window.Element('-STARTING_FRAME_TEXT_INPUT-').update(visible = True)
				window.Element('-STARTING_FRAME_INPUT-').update(visible = True)
			else:
				window.Element('-T_COL_TEXT_INPUT-').update("          Time col:")
				window.Element('-STARTING_FRAME_TEXT_INPUT-').update(visible = False)
				window.Element('-STARTING_FRAME_INPUT-').update(visible = False)
			if time_units_output == "Frames":
				window.Element('-T_COL_TEXT_OUTPUT-').update("          Frame col:")
			else:
				window.Element('-T_COL_TEXT_OUTPUT-').update("          Time col:")		
		
		return
	
	# Obtain prefix of input file
	def file_name(infilename):
		filesplit = infilename.split(".")
		prefix = filesplit[:-1]
		prefix = reduce(lambda x, y: str(x) + "." + str(y), prefix)
		return prefix
	
	# Read palmtracer txt
	def read_txt(infilename):
		'''
		Measurements are in pixels
		
		Width	Height	nb_Planes	nb_Tracks	Pixel_Size(um)	Frame_Duration(s)	Gaussian_Fit	Spectral
		329	158	8000	8081	0.106	0.02	None	False
		Track	Plane	CentroidX(px)	CentroidY(px)	CentroidZ(um)	Integrated_Intensity	id	Pair_Distance(px)
		 1	 1	 276.269572830706	 47.0805395577243	 0	 13089.1528320313	 43	 0
		 1	 2	 274.303133516352	 48.2782523520935	 0	 7672.34887695313	 76	 0
		 1	 3	 274.891970316922	 47.5279810094224	 0	 9931.11743164063	 146	 0
		 1	 4	 273.669614996879	 47.8070693296009	 0	 5767.41088867188	 236	 0
		 1	 5	 275.414575953906	 47.397724656005	 0	 7558.37377929688	 291	 0
		 1	 6	 275.93406753494	 46.6645400960479	 0	 6259.39013671875	 308	 0
		 1	 7	 275.840465835188	 46.1239353282329	 0	 3471.96459960938	 416	 0
		 1	 8	 275.438006269573	 46.2149241717978	 0	 2029.81518554688	 440	 0
		 1	 9	 274.791478789354	 45.8095767231888	 0	 9122.86352539063	 524	 0
		 etc
		'''
		print ("\nReading {}...".format(infilename))
		rawdata = []
		ct = 0
		with open(infilename,"r") as infile:
			for line in infile:
				ct += 1
				if ct > 3 and len(line) > 10: # ignore first three header lines
					spl = [float(j) for j in line.split("\t")]
					if len(spl) != 8:
						print ("Data should contain 8 tab separated columns. eg:")
						print (" 1	 1	 276.269572830706	 47.0805395577243	 0	 13089.1528320313	 43	 0")
						print ("Your data:")
						print(reduce(lambda x, y: str(x) + "\t" + str(y),spl))
						quit()
					tr = spl[0]
					fr = spl[1]
					x = spl[2]*pix2um_input # convert pixels to microns
					y = spl[3]*pix2um_input
					i = spl[5]
					rawdata.append([tr,fr,x,y,i])
		if len(rawdata) != 0:
			print ("{} lines read".format(ct-3))				
			return rawdata		
		elif len(rawdata) == 0:
			print("0 lines read")
			sg.popup("0 lines read.\n\nPlease make sure the file is not empty.\n", keep_on_top=True)
			
	# Read palmtracer trc
	def read_trc(infilename):
		'''
		Measurements are in pixels
		
		1	1	276.269572723821	47.080539615183	-1	13089.1525878906
		1	2	274.303133599286	48.2782523061602	-1	7672.34838867188
		1	3	274.891970360777	47.5279809124645	-1	9931.11694335938
		1	4	273.669614968534	47.8070693377678	-1	5767.4111328125
		1	5	275.414575972816	47.3977246431582	-1	7558.3740234375
		1	6	275.93406753494	46.664540057044	-1	6259.39013671875
		1	7	275.840465753652	46.12393526663	-1	3471.96435546875
		1	8	275.43800630337	46.2149242190112	-1	2029.81530761719
		1	9	274.791478757012	45.809576798377	-1	9122.86328125
		1	1	275.824958927795	46.5723829447615	-1	8954.314453125
		2	2	58.6367908782501	101.742870799136	-1	12686.4106445313
		2	3	58.361849347087	102.198444512936	-1	10684.8942871094
		2	4	58.3014790906691	102.042988140333	-1	16990.9973144531
		2	5	58.3724221575284	101.936737912003	-1	15367.7707519531
		 etc
		'''
		print ("\nReading {}...".format(infilename))
		ct = 0
		rawdata = []
		with open(infilename,"r") as infile:
			for line in infile:
				ct+1
				if len(line) > 0:
					spl = [float(j) for j in line.split("\t")]
					if len(spl) != 6:
						print ("Data should contain 6 tab separated columns. eg:")
						print ("1	1	276.269572723821	47.080539615183	-1	13089.1525878906")
						print ("Your data:")
						print(reduce(lambda x, y: str(x) + "\t" + str(y),spl))
						quit()
					tr = spl[0]
					fr = spl[1]
					x = spl[2]*pix2um_input # convert pixels to microns
					y = spl[3]*pix2um_input
					i = spl[5]
					rawdata.append([tr,fr,x,y,i])
					ct += 1
		if len(rawdata) != 0:
			print ("{} lines read".format(ct))			
			return rawdata
		elif len(rawdata) == 0:
			print("0 lines read")
			sg.popup("0 lines read.\n\nPlease make sure the file is not empty.\n", keep_on_top=True)
			
	# Read NASTIC/segNASTIC trxyt	
	def read_trxyt(infilename):
		'''
		Measurements are in microns
		
		1 29.284574708725025 4.990537199209398 0.02
		1 29.076132161524317 5.117494744452981 0.04
		1 29.138548858242363 5.037965976721237 0.06
		1 29.008979186664604 5.067549349803387 0.08
		1 29.193945053118497 5.024158812174769 0.1
		1 29.24901115870364 4.946441246046664 0.12
		1 29.23908936988711 4.88913713826278 0.14
		1 29.19642866815722 4.8987819672151875 0.16
		1 29.12789674824327 4.855815140627962 0.18
		1 29.23744564634627 4.936672592144719 0.2
		2 6.215499833094511 10.784744304708415 0.02
		2 6.186356030791222 10.833035118371216 0.04
		2 6.179956783610924 10.816556742875298 0.06
		2 6.18747674869801 10.805294218672318 0.08
		 etc
		'''
		print ("\nReading {}...".format(infilename))	
		ct = 0
		rawdata = []
		with open(infilename,"r") as infile:
			for line in infile:
				if len(line) > 10:
					spl = [float(j) for j in line.split(" ")]
					if len(spl) != 4:
						print ("Data should contain 4 space separated columns. eg:")
						print ("1 29.284574708725025 4.990537199209398 0.02")
						print ("Your data:")
						print(reduce(lambda x, y: str(x) + " " + str(y),spl))
						quit()
					tr = spl[0]
					fr = spl[3]*acqfreq_input # convert seconds to frames
					x = spl[1]
					y = spl[2]
					i = -1
					rawdata.append([tr,fr,x,y,i])
					ct += 1
		if len(rawdata) != 0:
			print ("{} lines read".format(ct))			
			return rawdata	
		elif len(rawdata) == 0:
			print("0 lines read")
			sg.popup("0 lines read.\n\nPlease make sure the file is not empty.\n", keep_on_top=True)
			
	# Read SharpViSu ascii (not drift corrected) file
	def read_ascii(infilename,ids):
		'''
		Measurements in pixels
		
		1.00000,0,1, 209.045592443744, 142.834293217347, 50870.0478515625,0.00000,0.00000,0.00000
		1.00000,0,2, 208.435743897142, 63.6668300319305, 10899.9936523438,0.00000,0.00000,0.00000
		1.00000,0,3, 240.230942584511, 64.2850805918474, 8377.55151367188,0.00000,0.00000,0.00000
		1.00000,0,4, 282.032840365532, 57.7746372087263, 6812.4140625,0.00000,0.00000,0.00000
		1.00000,0,5, 127.394518568774, 86.3333469009349, 4333.64733886719,0.00000,0.00000,0.00000
		1.00000,0,6, 286.850200721154, 38.867410435638, 2464.98876953125,0.00000,0.00000,0.00000
		1.00000,1,1, 209.045196432015, 142.698726294003, 43768.1682128906,0.00000,0.00000,0.00000
		1.00000,1,2, 208.493705895093, 63.1041156726907, 4010.64306640625,0.00000,0.00000,0.00000
		1.00000,1,3, 240.629923029567, 64.1068561613844, 9554.67211914063,0.00000,0.00000,0.00000
		1.00000,1,4, 281.878790360555, 57.7497140474961, 6441.98608398438,0.00000,0.00000,0.00000
		1.00000,1,5, 125.282625184993, 89.0234809920917, 2805.40869140625,0.00000,0.00000,0.00000
		1.00000,1,6, 283.664077863468, 38.4132870736027, 3613.64709472656,0.00000,0.00000,0.00000
		1.00000,1,7, 58.1500695109681, 53.4211913628692, 9916.34057617188,0.00000,0.00000,0.00000
		 etc
		'''
		print ("\nReading {}...".format(infilename))
		rawdata = []
		ct = 0
		with open(infilename,"r") as infile:
			for line in infile:
				if len(line) > 10:
					spl = [float(j) for j in line.split(",")]
					if len(spl) != 9:
						print ("Data should contain 9 comma separated columns. eg:")
						print ("1.00000,0,1, 209.045592443744, 142.834293217347, 50870.0478515625,0.00000,0.00000,0.00000")
						print ("Your data:")
						print(reduce(lambda x, y: str(x) + "," + str(y),spl))
						quit()
					tr = ids[ct]
					fr = spl[1] + 1
					x = spl[3]*pix2um_input # convert pixels to microns
					y = spl[4]*pix2um_input 
					i = spl[5]
					rawdata.append([tr,fr,x,y,i])
					ct +=1
		if len(rawdata) != 0:
			print ("{} lines read".format(ct))			
			return rawdata
		elif len(rawdata) == 0:
			print("0 lines read")
			sg.popup("0 lines read.\n\nPlease make sure the file is not empty.\n", keep_on_top=True)
			
	# Read SharpViSu ascii(drift corrected) file
	def read_dcascii(infilename,ids):
		'''
		Measurements in nanometres
		
		1.00000,0,1, 209045.592443744, 142834.293217347, 50870.0478515625,0.00000,0.00000,0.00000
		1.00000,0,2, 208435.743897142, 63666.8300319305, 10899.9936523438,0.00000,0.00000,0.00000
		1.00000,0,3, 240230.942584511, 64285.0805918474, 8377.55151367188,0.00000,0.00000,0.00000
		1.00000,0,4, 282032.840365532, 57774.6372087263, 6812.4140625,0.00000,0.00000,0.00000
		1.00000,0,5, 127394.518568774, 86333.3469009349, 4333.64733886719,0.00000,0.00000,0.00000
		1.00000,0,6, 286850.200721154, 38867.410435638, 2464.98876953125,0.00000,0.00000,0.00000
		1.00000,1,1, 209045.196432015, 142698.726294003, 43768.1682128906,0.00000,0.00000,0.00000
		1.00000,1,2, 208493.705895093, 63104.1156726907, 4010.64306640625,0.00000,0.00000,0.00000
		1.00000,1,3, 240629.923029567, 64106.8561613844, 9554.67211914063,0.00000,0.00000,0.00000
		1.00000,1,4, 281878.790360555, 57749.7140474961, 6441.98608398438,0.00000,0.00000,0.00000
		1.00000,1,5, 125282.625184993, 89023.4809920917, 2805.40869140625,0.00000,0.00000,0.00000
		1.00000,1,6, 283664.077863468, 38413.2870736027, 3613.64709472656,0.00000,0.00000,0.00000
		1.00000,1,7, 58150.0695109681, 53421.1913628692, 9916.34057617188,0.00000,0.00000,0.00000
		 etc
		'''
		print ("\nReading {}...".format(infilename))
		rawdata = []
		ct = 0
		with open(infilename,"r") as infile:
			for line in infile:
				if len(line) > 10:
					spl = [float(j) for j in line.split(",")]
					if len(spl) != 9:
						print ("Data should contain 9 comma separated columns. eg:")
						print ("1.00000,0,1, 209045.592443744, 142834.293217347, 50870.0478515625,0.00000,0.00000,0.00000")
						print ("Your data:")
						print(reduce(lambda x, y: str(x) + "," + str(y),spl))
						quit()
					tr = ids[ct]
					fr = spl[1] + 1
					x = spl[3]/1000 # convert nanometres to microns
					y = spl[4]/1000
					i = spl[6]
					rawdata.append([tr,fr,x,y,i])
					ct +=1
		if len(rawdata) != 0:
			print ("{} lines read".format(ct))			
			return rawdata
		elif len(rawdata) == 0:
			print("0 lines read")
			sg.popup("0 lines read.\n\nPlease make sure the file is not empty.\n", keep_on_top=True)
			
	# Read idfile (contains trajectory IDs of ascii file)
	def read_ids(infilename):
		'''
		1,
		2,
		3,
		4,
		1,
		2,
		 etc
		'''
		prefix = file_name(infilename)
		infilename = prefix + ".id"
		print ("\nReading {}...".format(infilename))
		ct = 0
		ids = []
		with open(infilename,"r") as infile:
			for line in infile:
				ids.append(float(line))
				ct += 1
		if len(ids) != 0:
			print ("{} lines read".format(ct))			
			return ids		
		elif len(ids) == 0:
			print("0 lines read")
			sg.popup("0 lines read.\n\nPlease make sure the file is not empty.\n", keep_on_top=True)
			
	# Read TrackMate .csv
	def read_csv(infilename):
		'''
		Number of headers: variable (requires manual selection)
		Number of columns: variable (requires manual selection)
		X,Y Units: variable (requires manual selection)
		Time Units: variable (requires manual selection)
		LABEL,ID,TRACK_ID,QUALITY,POSITION_X,POSITION_Y,POSITION_Z,POSITION_T,FRAME,RADIUS,VISIBILITY,MANUAL_SPOT_COLOR,MEAN_INTENSITY_CH1,MEDIAN_INTENSITY_CH1,MIN_INTENSITY_CH1,MAX_INTENSITY_CH1,TOTAL_INTENSITY_CH1	STD_INTENSITY_CH1,CONTRAST_CH1,SNR_CH1
		Label,Spot ID,Track ID,Quality,X,Y,Z,T,Frame,Radius,Visibility,Manual spot color,Mean intensity ch1,Median intensity ch1,Min intensity ch1,Max intensity ch1,Sum intensity ch1,Std intensity ch1,Contrast ch1,Signal/Noise ratio ch1
		Label,Spot ID,Track ID,Quality,X,Y,Z,T,Frame,R,Visibility,Spot color,Mean ch1,Median ch1,Min ch1,Max ch1,Sum ch1,Std ch1,Ctrst ch1,SNR ch1
		 , , ,(quality),(micron),(micron),(micron),(sec),	,(micron), , ,(counts),(counts),(counts),(counts),(counts),(counts), , 		
		ID282628,282628,2,298.0895081,21.76864672,36.30233107,0,55.17871248,1723,0.4,1, ,3446.819672,3232,1872,6544,210256,1033.032905,0.184146466,1.037749106
		ID200706,200706,2,609.3094482,21.77325704,36.38438245,0,7.621899925,238,0.4,1, ,5920,5936,2656,8880,361120,1541.432061,0.262635782,1.597728995
		ID192512,192512,2,640.4356079,21.76171236,36.40632485,0,4.419420965,138,0.4,1, ,6025.704918,5904,3376,9968,367568,1565.816926,0.267932196,1.626393935
		ID229377,229377,2,579.6768188,21.76205112,36.42347658,0,20.91218761,653,0.4,1, ,5104.262295,4896,2736,9328,311360,1571.488614,0.263986861,1.356723956
		ID270343,270343,2,181.9257202,21.73377113,36.28705447,0,45.41115166,1418,0.4,1, ,3099.803279,2880,2032,5664,189088,692.8135588,0.115954163,0.929796138
		ID286722,286722,2,156.4457703,21.76040318,36.35765508,0,58.63738976,1831,0.4,1, ,2732.327869,2688,1712,4448,166672,573.9474053,0.107216306,0.921974859
		 ect
		'''
		print ("\nReading {}...".format(infilename))
		rawdata = []
		ct = 0
		with open(infilename,"r") as infile:
			for line in infile:
				ct+=1
				if ct > header_num_input: # ignore header lines
					spl = [j for j in line.split(",")]
					tr = spl[tr_col_input-1]
					tr = float(tr)
					if time_units_input == "Frames":
						if starting_frame_input == 0:
							fr = float(spl[t_col_input-1])+1 # start Frame# from 0
						else:
							fr = float(spl[t_col_input-1]) # start Frame# from 1
					elif time_units_input == "s":
						fr = float(spl[t_col_input-1])*float(acqfreq_input) # convert from seconds to frames
					if xy_units_input == "um":
						x = float(spl[x_col_input-1])
						y = float(spl[y_col_input-1])
					elif xy_units_input == "nm":
						x = float(spl[x_col_input-1])/1000 # convert nanometers to microns
						y = float(spl[y_col_input-1])/1000 
					elif xy_units_input == "px":
						x = (float(spl[x_col_input-1]))*(float(pix2um_input)) # convert pixels to microns
						y = (float(spl[y_col_input-1]))*(float(pix2um_input))
					i = -1
					rawdata.append([tr,fr,x,y,i])
		if len(rawdata) != 0:
			print ("{} lines read".format(ct-header_num_input))				
			return rawdata		
		elif len(rawdata) == 0:
			print("0 lines read")
			sg.popup("0 lines read.\n\nPlease make sure the file is not empty.\n", keep_on_top=True)
			
	# read other(manual input)
	def read_other(infilename):
		print ("\nReading {}...".format(infilename))
		# Set input file delimiter
		if delim_input == "comma":
			separator_input = ","
		elif delim_input == "tab":
			separator_input = "\t"
		elif delim_input == "1 space":
			separator_input = " "
		elif delim_input == "2 spaces":
			separator_input = "  "
		elif delim_input == "3 spaces":
			separator_input = "   "
		elif delim_input == "4 spaces":
			separator_input = "    "
		elif delim_input == "semicolon":
			separator_input = ";"
		rawdata = []
		ct = 0
		with open(infilename,"r") as infile:
			for line in infile:
				ct+=1
				if ct > header_num_input: # ignore header lines
					spl = [j for j in line.split(separator_input)]
					tr = spl[tr_col_input-1]
					tr = float(tr)
					if time_units_input == "Frames":
						if starting_frame_input == 0:
							fr = float(spl[t_col_input-1]) + 1 # start Frame# from 0
						else:
							fr = float(spl[t_col_input-1]) # start Frame# from 1
					elif time_units_input == "s":
						fr = float(spl[t_col_input-1])*acqfreq_input # convert from seconds to Frames
					if xy_units_input == "um":
						x = float(spl[x_col_input-1])
						y = float(spl[y_col_input-1])
					elif xy_units_input == "nm":
						x = float(spl[x_col_input-1])/1000 # convert from nanometers to microns
						y = float(spl[y_col_input-1])/1000 
					elif xy_units_input == "px":
						x = (float(spl[x_col_input-1]))*pix2um_input # convert from pixels to microns
						y = (float(spl[y_col_input-1]))*pix2um_input
					i = -1
					rawdata.append([tr,fr,x,y,i])
		if len(rawdata) != 0:
			print ("{} lines read".format(ct-header_num_input))	
			return rawdata		
		elif len(rawdata) == 0:
			print("0 lines read")
			sg.popup("0 lines read.\n\nPlease make sure the file is not empty and that input file parameters are correct.\n", keep_on_top=True)
	
	# Write palmtracer txt
	def write_txt(rawdata,infilename):
		prefix = file_name(infilename)
		outfilename = prefix + "-" + stamp + ".txt"
		print ("\nWriting {}...".format(outfilename))
		ct = 0
		with open(outfilename,"w") as outfile:
			outfile.write("DUMMY HEADER LINE 1\n")
			outfile.write("DUMMY HEADER LINE 2\n")
			outfile.write("Track\tPlane\tCentroidX(px)\tCentroidY(px)\tCentroidZ(um)\tIntegrated_Intensity\tid\tPair_Distance(px)\n")
			for det in rawdata:
				tr = int(round(det[0],0)) 
				fr = int(round(det[1],0))
				x = det[2]/pix2um_input # convert microns to pixels
				y = det[3]/pix2um_input
				i = det[4]
				outstring =reduce(lambda x, y: str(x) + "\t" + str(y), [tr,fr,x,y,0,i,0,0])
				outfile.write(outstring + "\n")
				ct+=1
		print ("{} lines written".format(ct))
		return
		
	# Write palmtracer trc
	def write_trc(rawdata,infilename):
		prefix = file_name(infilename)
		outfilename = prefix  + "-" + stamp + ".trc"
		print ("\nWriting {}...".format(outfilename))
		ct = 0
		with open(outfilename,"w") as outfile:
			for det in rawdata:
				tr = int(round(det[0],0)) 
				fr = int(round(det[1],0))
				x = det[2]/pix2um_input # convert microns to pixels
				y = det[3]/pix2um_input
				i = det[4]
				outstring =reduce(lambda x, y: str(x) + "\t" + str(y), [tr,fr,x,y,-1,i])
				outfile.write(outstring + "\n")
				ct+=1
		print ("{} lines written".format(ct))	
		return
		
	# Write NASTIC/segNASTIC trxyt
	def write_trxyt(rawdata,infilename):
		prefix = file_name(infilename)
		outfilename = prefix  + "-" + stamp + ".trxyt"
		print ("\nWriting {}...".format(outfilename))
		ct = 0
		with open(outfilename,"w") as outfile:
			for det in rawdata:
				tr = int(round(det[0],0))
				t = det[1]/float(acqfreq_input) # Convert Frames to seconds
				x = det[2]
				y = det[3]
				outstring =reduce(lambda x, y: str(x) + " " + str(y),[tr,x,y,t])
				outfile.write(outstring + "\n")	
				ct+=1
		print ("{} lines written".format(ct))			
		return
		
	# Write SharpViSu ascii (not drift corrected)
	def write_ascii(rawdata,infilename):
		prefix = file_name(infilename)
		outfilename = prefix  + "-" + stamp + ".ascii"
		print ("\nWriting {}...".format(outfilename))
		ids = []
		asciidict = {}
		for det in rawdata:
			tr,fr,x,y,i = det	
			try:
				asciidict[fr].append(det)
			except:
				asciidict[fr] = [det] 
		with open(outfilename,"w") as outfile:
			ct = 0
			for line in asciidict:
				linedata = asciidict[line]
				for n,det in enumerate(linedata, start=1):
					frame = int(round(line,0))
					tr = int(round(det[0],0))
					x = det[2]/pix2um_input # convert microns to pixels
					y = det[3]/pix2um_input
					i = det[4]
					outstring = "1,{},{},{},{},{},0,0,0\n".format(frame-1,n,x,y,i)
					outfile.write(outstring)
					ids.append(tr)
					ct+=1
		print ("{} lines written".format(ct))					
		return ids	

	# Write SharpViSu ascii(drift corrected)
	def write_dcascii(rawdata,infilename):
		prefix = file_name(infilename)
		outfilename = prefix  + "-" + stamp + ".ascii"
		print ("\nWriting {}...".format(outfilename))
		ids = []
		asciidict = {}
		for det in rawdata:
			tr,fr,x,y,i = det	
			try:
				asciidict[fr].append(det)
			except:
				asciidict[fr] = [det] 
		with open(outfilename,"w") as outfile:
			ct = 0
			for line in asciidict:
				linedata = asciidict[line]
				for n,det in enumerate(linedata, start=1):
					frame = int(round(line,0))
					tr = int(round(det[0],0))
					x = det[2] * 1000 # convert microns to nanometers
					y = det[3] * 1000
					i = det[4]
					outstring = "1,{},{},{},{},0,{},0,0\n".format(frame-1,n,x,y,i)
					outfile.write(outstring)
					ids.append(tr)
					ct+=1
		print ("{} lines written".format(ct))					
		return ids		
				
	# Write idfile (contains trajectoryIDs of ascii file)
	def write_ids(ids,infilename):				
		prefix = file_name(infilename)
		outfilename = prefix  + "-" + stamp + ".id"
		print ("\nWriting {}...".format(outfilename))
		with open(outfilename,"w") as outfile:
			ct = 0
			for line in ids:
				outfile.write(str(int(line)) + "\n")
				ct+=1
		print ("{} lines written".format(ct))		
		return	
	
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
	if os.path.isfile("super_res_data_wrangler_gui.defaults"):
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
		['&Info', ['&About', '&Help',['&txt Files', '&trc Files', '&trxyt Files', '&ascii Files', '&id Files', '&csv(TrackMate) Files', '&other(manual input) Files', '&Pixel size', '&Acquisition frequency'],'&Licence','&Updates' ]],
		]
		
	# TAB 1 (SELECT FILETYPE TAB)	

	# Tab 1 (contents of input txt tab)
	txt_input_layout = [
		[sg.T("Convert a PalmTracer .txt trajectory file")], 
		[sg.T("TXT files contain 3 headers")],
		[sg.T("Each TXT data line (row) contains 8 tab separated values:\n   [1]Track; (Trajectory#, starting at 1)\n   [2]Plane; (Frame#, starting at 1)\n   [3]CentroidX(px);\n   [4]CentroidY(px);\n  *[5]CentroidZ(um);\n  *[6]Integrated_Intensity;\n  *[7]id;\n  *[8]Pair_Distance(px);\n* = Columns not needed for conversion, can be any number")],
	]
	
	# Tab 1 (contents of input trc tab)
	trc_input_layout = [
		[sg.T("Convert a PalmTracer .trc trajectory file")],
		[sg.T("TRC files contain 0 headers")],
		[sg.T("Each TRC line (row) contains 6 tab separated values:\n   [1]Trajectory#; (starting at 1)\n   [2]Frame#; (starting at 1)\n   [3]X-position(px);\n   [4]Y-position(px);\n  *[5]-1;\n  *[6]Integrated_Intensity;\n* = Columns not needed for conversion, can be any number")],
	]
	
	# Tab 1 (contents of input trxyt tab)
	trxyt_input_layout = [
		[sg.T("Convert a NASTIC .trxyt trajectory file")],
		[sg.T("TRXYT files contain 0 headers")],
		[sg.T("Each TRXYT line (row) contains 4 space separated values:\n   [1]Trajectory#;\n   [2]X-position(um);\n   [3]Y-position(um);\n   [4]Time(s);")],
	]
	
	# Tab 1 (contents of input ascii tab)
	ascii_input_layout = [
		[sg.T("Convert an .ascii and .id trajectory file (before drift correction)")],
		[sg.T("ASCII and ID files contain 0 headers")],
		[sg.T("Each ASCII line (row) contains 9 comma separated values\n  *[1]1;\n   [2]Frame#; (starting at 0)\n   [2]n; (Trajectory# per Frame, starting at 1 each frame)\n   [4]X-position(px);   NOTE: uncorrected=px, corrected=nm\n   [5]Y-position(px);   NOTE: uncorrected=px, corrected=nm\n  *[6]Integrated_intensity;\n  *[7]0;\n  *[8]0;\n  *[9]0;\n* = Columns not needed for conversion, can be any number")],
		[sg.T("Each IDS line (row) contains 1 value\n   [1]Trajectory#;")],
	]
	
	# Tab 1 (contents of input ascii(drift corrected) tab)
	dcascii_input_layout = [
		[sg.T("Convert an .ascii and .id trajectory file (after drift correction)")],
		[sg.T("ASCII and ID files contain 0 headers")],
		[sg.T("Each ASCII line (row) contains 9 comma separated values\n  *[1]1;\n   [2]Frame#; (starting at 0)\n   [2]n; (Trajectory# per Frame, starting at 1 each frame)\n   [4]X-position(nm);   NOTE: uncorrected=px, corrected=nm\n   [5]Y-position(nm);   NOTE: uncorrected=px, corrected=nm\n  *[6]0;\n  *[7]Integrated_intensity;\n  *[8]0;\n  *[9]0;\n* = Columns not needed for conversion, can be any number")],
		[sg.T("Each IDS line contains 1 value\n   [1]Trajectory#;")],
	]
	
	# Tab 1 (contents of input csv(TrackMate) tab)
	csv_input_layout = [
		[sg.T("Convert a TrackMate export_tracks.csv trajectory file")],
		[sg.T("The number of headers can can vary")],
		[sg.T("Each CSV line contains several comma separated columns,\n4 of which must contain the following information:\n  *[  ]Trajectory#; (e.g., TRACK_ID)\n  *[  ]X-position; (e.g., POSITION_X; **px,um or nm)\n  *[  ]Y-position; (e.g., POSITION_Y; **px,um or nm)\n  *[  ]Time; (e.g., FRAME or POSITION_T; **Frames or s)\n* = Columns can be in any order in the input file\n** = Units depend on how FIJI interprets the microscope data")],
	]
	
	# Tab 1 (contents of input other(manual input) tab)
	other_input_layout = [
		[sg.T("Convert a trajectory file by manually selecting parameters")],
		[sg.T("Parameters requiring selection:\n    - # rows which contain headers (before start of data)\n    - Delimiter which separates data into columns\n    - Column # [  ] which contains Trajectory# data\n    - Column # [  ] which contains X-position data\n    - Column # [  ] which contains Y-position data\n    - Column # [  ] which contains Time data\n    - X and Y-position units (um,nm or px)\n   *- Pixel size (um/px) (XY unit conversion)\n    - Time units (Frames or s)\n   *- Acquisition frequency (Hz) (Time unit conversion)\n   *- Starting frame#\n* = Parameters will appear if required for conversion")],
	]
	
	# Tab 1 (contents of input file frame)
	input_filetype_frame_layout = [
		[sg.T("")],
		[sg.T("Input filetype:"), sg.Combo(["txt","trc","trxyt","ascii","ascii(drift corrected)", "csv(TrackMate)", "other(manual input)"], default_value = "txt", key = '-SELECT_INPUT-', enable_events = True)],
		[sg.TabGroup([
			[sg.Tab("txt",txt_input_layout, disabled = False, key = '-TXT_INPUT_TAB-')],
			[sg.Tab("trc", trc_input_layout, disabled = True, key = '-TRC_INPUT_TAB-')],
			[sg.Tab("trxyt", trxyt_input_layout, disabled = True, key = '-TRXYT_INPUT_TAB-')],
			[sg.Tab("ascii", ascii_input_layout, disabled = True, key = '-ASCII_INPUT_TAB-')],
			[sg.Tab("ascii(drift corrected)", dcascii_input_layout, disabled = True, key = '-DCASCII_INPUT_TAB-')],
			[sg.Tab("csv", csv_input_layout, disabled = True, key = '-CSV_INPUT_TAB-')],
			[sg.Tab("other(manual)", other_input_layout, disabled = True, key = '-OTHER_INPUT_TAB-')],
			], key = "-SELECT_INPUT_TAB-")
		],
	]
	
	# Tab 1 (setup input file frame)
	input_filetype_layout = [
		[sg.Frame("Input file", input_filetype_frame_layout)],
	]
	
	# Tab 1 (contents of output txt tab)
	txt_output_layout = [
		[sg.T("Generate a PalmTracer .txt trajectory file")], 
		[sg.T("TXT files contain 3 headers")],
		[sg.T("Each TXT data line (row) contains 8 tab separated values:\n   [1]Track; (Trajectory#, starting at 1)\n   [2]Plane; (Frame#, starting at 1)\n   [3]CentroidX(px);\n   [4]CentroidY(px);\n   [5]0;\n  *[6]Integrated_Intensity;\n   [7]0;\n   [8]0;\n* = Replaced with -1 if not present in input file")],
	]
	
	# Tab 1 (contents of output trc tab)
	trc_output_layout = [
		[sg.T("Generate a PalmTracer .trc trajectory file")],
		[sg.T("TRC files contain 0 headers")],
		[sg.T("Each TRC data line (row) contains 6 tab separated values:\n   [1]Trajectory#; (starting at 1)\n   [2]Frame#; (starting at 1)\n   [3]X-position(px);\n   [4]Y-position(px);\n   [5]-1;\n  *[6]Integrated_Intensity;\n* = Replaced with -1 if not present in input file")],
	]
	
	# Tab 1 (contents of output trxyt tab)
	trxyt_output_layout = [
		[sg.T("Generate a NASTIC .trxyt trajectory file")],
		[sg.T("TRXYT files contain 0 headers")],
		[sg.T("Each TRXYT line (row) contains 4 space separated values:\n   [1]Trajectory#;\n   [2]X-position(um);\n   [3]Y-position(um);\n   [4]Time(s);")],
	]
	
	# Tab 1 (contents of output ascii tab)
	ascii_output_layout = [
		[sg.T("Generate an .ascii and .id trajectory file (before drift correction)")],
		[sg.T("ASCII and ID files contain 0 headers")],
		[sg.T("Each ASCII line (row) contains 9 comma separated values\n   [1]1;\n   [2]Frame#; (starting at 0)\n   [2]n; (Trajectory# per Frame, starting at 1 each frame)\n   [4]X-position(px);   NOTE: uncorrected=px, corrected=nm\n   [5]Y-position(px);   NOTE: uncorrected=px, corrected=nm\n  *[6]Integrated_intensity;\n   [7]0;\n   [8]0;\n   [9]0;\n* = Replaced with -1 if not present in input file")],
		[sg.T("Each IDS line (row) contains 1 value\n   [1]Trajectory#;")],
	]
	
	# Tab 1 (contents of output ascii(drift corrected) tab)
	dcascii_output_layout = [
		[sg.T("Generate an .ascii and .id trajectory file (after drift correction)")],
		[sg.T("ASCII and ID files contain 0 headers")],
		[sg.T("Each ASCII line (row) contains 9 comma separated values\n   [1]1;\n   [2]Frame#; (starting at 0)\n   [2]n; (Trajectory# per Frame, starting at 1 each frame)\n   [4]X-position(nm);   NOTE: uncorrected=px, corrected=nm\n   [5]Y-position(nm);   NOTE: uncorrected=px, corrected=nm\n   [6]0;\n  *[7]Integrated_intensity;\n   [8]0;\n   [9]0;\n* = Replaced with -1 if not present in input file")],
		[sg.T("Each IDS line (row) contains 1 value\n   [1]Trajectory#;")],
	]

	# Tab 1 (contents of output file frame)
	output_filetype_frame_layout = [
		[sg.T("")],
		[sg.T("Output filetype:"), sg.Combo(["txt","trc","trxyt","ascii","ascii(drift corrected)"], default_value = "trxyt", key = '-SELECT_OUTPUT-', enable_events = True)],
		[sg.TabGroup([
			[sg.Tab("txt", txt_output_layout, disabled = True, key = '-TXT_OUTPUT_TAB-')],
			[sg.Tab("trc", trc_output_layout, disabled = True, key = '-TRC_OUTPUT_TAB-')],
			[sg.Tab("trxyt", trxyt_output_layout, disabled = False, key = '-TRXYT_OUTPUT_TAB-')],
			[sg.Tab("ascii", ascii_output_layout, disabled = True, key = '-ASCII_OUTPUT_TAB-')],
			[sg.Tab("ascii(drift corrected)", dcascii_output_layout, disabled = True, key = '-DCASCII_OUTPUT_TAB-')],
			], key = "-SELECT_OUTPUT_TAB-")
		],
	]
	
	# Tab 1 (setup output file frame)
	output_filetype_layout = [
		[sg.Frame("Output file", output_filetype_frame_layout)],
	]
	
	# Tab 1 (setup columns for input file frame and output file frame)
	filetype_layout = [
		[sg.T("")],
		[sg.Column(input_filetype_layout),sg.Column(output_filetype_layout)],
		[sg.Push(),sg.T("Please select input and output filetypes using the dropdowns"),sg.Push()],
		[sg.Push(),sg.B("CONFIRM FILETYPES", key = '-CONFIRM-',enable_events = True),sg.Push()],
	]
	
	# TAB 2 (LOAD FILES TAB)
	
	# Tab 2 (setup Tree) 
	check = [icon(0), icon(1), icon(2)]
	input_headings = ['Input Files']
	treedata = sg.TreeData()
	
	# Tab 2 (contents) 
	load_layout = [
		[sg.T("")],
		[sg.Radio("Browse file",group_id = 1,default = True, key = '-BROWSE_FILE_RADIO-', enable_events = True),sg.Radio("Browse folder (directory)",group_id = 1,default = False, key = '-BROWSE_FOLDER_RADIO-', enable_events = True)],
		[sg.FileBrowse("Browse txt file", key = '-BROWSE_FILE_TXT-', target = '-INPUT_FILE-', size = (13,1), file_types = (("Trajectory Files", "*.txt"),),visible = True, enable_events = True), sg.FileBrowse("Browse trc file", key = '-BROWSE_FILE_TRC-', target = '-INPUT_FILE-', size = (13,1), file_types = (("Trajectory Files", "*.trc"),),visible = False, enable_events = True),sg.FileBrowse("Browse trxyt file", key = '-BROWSE_FILE_TRXYT-', target = '-INPUT_FILE-', size = (13,1), file_types = (("Trajectory Files", "*.trxyt"),),visible = False, enable_events = True), sg.FileBrowse("Browse ascii file", key = '-BROWSE_FILE_ASCII-', target = '-INPUT_FILE-', size = (13,1), file_types = (("Trajectory Files", "*.ascii"),),visible = False, enable_events = True), sg.FileBrowse("Browse csv file", key = '-BROWSE_FILE_CSV-', target = '-INPUT_FILE-', size = (13,1), file_types = (("Trajectory Files", "*.csv"),),visible = False, enable_events = True), sg.FileBrowse('Browse file', key = '-BROWSE_FILE_OTHER-', target = '-INPUT_FILE-', size = (13,1), file_types = (("Trajectory Files", "*"),), visible = False, enable_events = True),sg.In("Select input file", key = '-INPUT_FILE-', expand_x = True, disabled = False, disabled_readonly_background_color = "#313641", enable_events = True)],
		[sg.FolderBrowse("Browse folder",key="-BROWSE_FOLDER-",target="-INPUT_FOLDER-", size = (13,1),initial_folder=initialdir, disabled = True, enable_events = True),sg.In("Select directory containing input files", key = "-INPUT_FOLDER-",expand_x = True, disabled = True, disabled_readonly_background_color = "#313641", disabled_readonly_text_color = "grey", enable_events = True)],
		[sg.B("Find files", key = '-FIND_FILES-', disabled = True),sg.T("Files to include:", key = '-FIND_FILES_TEXT-', text_color = "grey"), sg.T("(Optional) Filenames contain:", key = '-FILES_CONTAIN_TEXT-', tooltip = "Only find files with filenames that contain a selected phrase\nNo input = contains any phrase", visible = True), sg.In("", key = '-FILES_CONTAIN-', size = (25,1), disabled_readonly_background_color = "#313641", text_color = "grey", visible = True), sg.T("(Optional) Filenames end with:", key = '-FILES_EXT_TEXT-', tooltip = "Only find files with filenames that end with a selected phrase\nMust include extension\nNo input = any extension", visible = True),sg.In("", key = '-FILES_EXT-', size = (7,1),disabled_readonly_background_color = "#313641", text_color = "grey", visible = True)],
		[sg.Tree(data=treedata, headings = input_headings, row_height=25, num_rows = 10,select_mode = sg.TABLE_SELECT_MODE_BROWSE,key = '-TREE-', tooltip = "Untick files to exclude them from analysis\nUse ~1s delay between clicks", metadata = [],vertical_scroll_only=True, justification='left', enable_events=True, auto_size_columns = True, expand_x = True, col0_heading = "", col0_width = 1)],
		[sg.Push(),sg.T("Please select trajectory file(s) to load", key = '-LOAD_TRAJ_FILE_TEXT-'),sg.Push()],
		[sg.Push(),sg.B("LOAD DATA",key = "-LOAD-", tooltip = "Load selected files",button_color=("#f5f5f6", "#313641"),disabled=True),sg.Push()],
	]
	
	# TAB 3 (SET PARAMETERS TAB)	
	
	# Tab 3 (contents of input file frame) 
	input_parameter_frame_layout = [
		[sg.T("")],
		[sg.T("Define spatial and temporal information:", key = "-DEFINE_SPACE_TIME_INPUT-", text_color = "white")],
		[sg.T("          X,Y units:", tooltip = "Units for columns containing X- and Y-position data\npx = pixels, um = microns, nm = nanometers", key = '-XY_UNITS_TEXT_INPUT-', text_color = "white"), sg.Combo(["px","um","nm"], key = '-XY_UNITS_INPUT-', default_value= "um", enable_events = True, disabled = False, background_color = "#313641", text_color = "white")],
		[sg.T("          Pixel size (um/px):", key = "-PIXEL_SIZE_TEXT_INPUT-", tooltip = "Number of microns per pixel\nUsed for converting between different X,Y Units\nSee Info>Help>Pixel size for more information", text_color = "white", visible = True, enable_events = True), sg.In("0.106", size = (10,1), key = '-PIXEL_SIZE_INPUT-', text_color = "white", disabled = False, disabled_readonly_background_color = "#313641", visible = True)],
		[sg.Text("          Time units:", tooltip = "Units for the column containing Time data\nFrames = Frame#, s = seconds", key = '-TIME_UNITS_TEXT_INPUT-', text_color = "white"),sg.Combo(["Frames","s"], default_value = "Frames", key = '-TIME_UNITS_INPUT-', enable_events = True, disabled = False, background_color = "#313641", text_color = "white")],
		[sg.T("          Acquisition frequency (Hz):", key = "-ACQ_FREQ_TEXT_INPUT-", tooltip = "Hz = Hertz\nUsed for converting between different Time units\nSee Info>Help>Acquisition frequency for more information", text_color = "white", visible = True), sg.In(50, key = '-ACQ_FREQ_INPUT-', size = (10,1), disabled = False, disabled_readonly_background_color = "#313641", text_color = "white", visible = True)],
		[sg.T("Define file structure:", key = '-DEFINE_FILE_STRUCTURE_INPUT-', text_color = "white")],
		[sg.T("          Number of headers:", tooltip = "Number of rows containing headers (before start of data rows)", key = '-HEADER_NUM_TEXT_INPUT-', text_color = "white"),sg.Combo([0,1,2,3,4,5,6,7,8,9,10], key = '-HEADER_NUM_INPUT-', default_value=3, enable_events = True, background_color = "#313641", text_color = "white",disabled = False)], 
		[sg.T("          Delimiter:", tooltip = "Separates data into columns", key = '-DELIM_TEXT_INPUT-', text_color = "white"),sg.Combo(["comma","tab","1 space", "2 spaces", "3 spaces", "4 spaces", "semicolon"], key = '-DELIM_INPUT-', default_value= "tab",enable_events = True, disabled = False, background_color = "#313641", text_color = "white")],
		[sg.T("Select which columns contain trajectory data:", key = "-TRAJ_DATA_COLS_INPUT-", text_color = "white")],
		[sg.T("          Trajectory col:", tooltip = "Column number that contains Trajectory# data", key = '-TR_COL_TEXT_INPUT-', text_color = "white"),sg.Combo([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],key = '-TR_COL_INPUT-', default_value=1, enable_events = True, disabled = False, background_color = "#313641", text_color = "white")],
		[sg.T("          X-position col:", tooltip = "Column number that contains X-position data", key = '-X_COL_TEXT_INPUT-', text_color = "white", visible = True), sg.Combo([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], default_value=3,key = '-X_COL_INPUT-', enable_events = True, disabled = False, background_color = "#313641", text_color = "white")],
		[sg.T("          Y-position col:", tooltip = "Column number that contains Y-position data", key = '-Y_COL_TEXT_INPUT-', text_color = "white", visible = True),sg.Combo([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],key = '-Y_COL_INPUT-', default_value = 4, enable_events = True, disabled = False, background_color = "#313641", text_color = "white")],
		[sg.T("          Time/Frame col:", tooltip = "Column number that contains Time data", key = '-T_COL_TEXT_INPUT-', text_color = "white"),sg.Combo([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], default_value= 2,key = '-T_COL_INPUT-', enable_events = True, disabled = False, background_color = "#313641", text_color = "white", visible = True), sg.T("Starting frame:", key = '-STARTING_FRAME_TEXT_INPUT-', tooltip = "Frame# of first frame", text_color = "white", visible = True), sg.Combo([0,1],key = '-STARTING_FRAME_INPUT-', default_value = 1, enable_events = True, disabled = False, background_color = "#313641", text_color = "white", visible = True)],
	]
	
	#Tab 3 (contents of output tab frame)
	output_parameter_frame_layout = [
		[sg.T("")],
		[sg.T("Define spatial and temporal information:", key = "-DEFINE_SPACE_TIME_OUTPUT-", text_color = "grey")],
		[sg.T("          X,Y units:", tooltip = "Units for columns containing X- and Y-position data\npx = pixels, um = microns, nm = nanometers", key = '-XY_UNITS_TEXT_OUTPUT-', text_color = "grey"), sg.Combo(["px","um","nm"], key = '-XY_UNITS_OUTPUT-', default_value= "um",text_color = "grey", disabled = True)],
		[sg.T(" ")],
		[sg.Text("          Time units:", tooltip = "Units for the column containing Time data\nFrames = Frame#, s = seconds", key = '-TIME_UNITS_TEXT_OUTPUT-', text_color = "grey"),sg.Combo(["Frames","s"], default_value = "Frames", key = '-TIME_UNITS_OUTPUT-', disabled = True, text_color = "grey")],
		[sg.T(" ")],
		[sg.T("Define file structure:", key = '-DEFINE_FILE_STRUCTURE_OUTPUT-',text_color = "grey")],
		[sg.T("          Number of headers:", tooltip = "Number of rows containing headers (before start of data rows)", key = '-HEADER_NUM_TEXT_OUTPUT-', text_color = "grey"),sg.Combo([0,1,2,3,4,5,6,7,8,9,10], key = '-HEADER_NUM_OUTPUT-', default_value=3, disabled = True, background_color = "#313641", text_color = "grey")], 
		[sg.T("          Delimiter:", tooltip = "Separates data into columns", key = '-DELIM_TEXT_OUTPUT-', text_color = "grey"),sg.Combo(["comma","tab","1 space", "2 spaces", "3 spaces", "4 spaces"], key = '-DELIM_OUTPUT-', default_value= "tab", disabled = True, text_color = "grey")],
		[sg.T("Select which columns contain trajectory data:", text_color = "grey", key = "-TRAJ_DATA_COLS_OUTPUT-")],
		[sg.T("          Trajectory col:", tooltip = "Column number that contains Trajectory# data", key = '-TR_COL_TEXT_OUTPUT-',text_color = "grey"),sg.Combo([1,2,3,4,5,6,7,8,9,10],key = '-TR_COL_OUTPUT-', default_value=1, disabled = True, text_color = "grey")],
		[sg.T("          X-position col:", tooltip = "Column number that contains X-position data", key = '-X_COL_TEXT_OUTPUT-', text_color = "grey"), sg.Combo([1,2,3,4,5,6,7,8,9,10], default_value=3,key = '-X_COL_OUTPUT-', disabled = True, text_color = "grey")],
		[sg.T("          Y-position col:", tooltip = "Column number that contains Y-position data", key = '-Y_COL_TEXT_OUTPUT-', text_color = "grey"),sg.Combo([1,2,3,4,5,6,7,8,9,10],key = '-Y_COL_OUTPUT-', default_value = 4,disabled = True, text_color = "grey")],
		[sg.T("          Time/Frame col:", tooltip = "Column number that contains Time data", key = '-T_COL_TEXT_OUTPUT-', text_color = "grey"),sg.Combo([1,2,3,4,5,6,7,8,9,10], default_value= 2,key = '-T_COL_OUTPUT-', disabled = True, text_color = "grey"),sg.T("Starting frame:", key = '-STARTING_FRAME_TEXT_OUTPUT-', tooltip = "Frame# of first frame", text_color = "grey"),sg.Combo([0,1],key = '-STARTING_FRAME_OUTPUT-', default_value = 1, disabled = True, background_color = "#313641", text_color = "grey")],	
	]
	
	# Tab 3 (setup input file frame)
	input_parameter_layout = [
		[sg.Frame("Input file parameters", input_parameter_frame_layout, key = '-INPUT_FILE_FRAME-', expand_x = True, expand_y = True)],
	]
	
	# Tab 3 (setup output file frame)
	output_parameter_layout = [
		[sg.Frame("Output file parameters", output_parameter_frame_layout, key = '-OUTPUT_FILE_FRAME-', expand_x = True, expand_y = True)],
	]
	
	# Tab 3 (setup columns for input file frame and output file frame)
	parameter_layout = [
		[sg.T("")],
		[sg.Column(input_parameter_layout, expand_x = True), sg.Column(output_parameter_layout, expand_x = True)],
		[sg.Push(),sg.T("Please select parameters for input file", key = '-PARAMETER_MESSAGE-'), sg.Push()],
		[sg.Push(),sg.B("CONVERT", key = '-CONVERT-', enable_events = True),sg.Push()],
	]
	
	# LAYOUT
	layout = [
		[sg.Menu(menu_def)],
		[sg.Push(),sg.T('SUPER RES DATA WRANGLER',font="Any 20"), sg.Push()],
		[sg.Push(), sg.T("Convert between trajectory filetypes", font = "Any 12 italic"), sg.Push()],
		[sg.TabGroup([
			[sg.Tab("Select filetypes", filetype_layout, key = '-SELECT_FILETYPES_TAB-')],
			[sg.Tab("Load files",load_layout,key = '-LOAD_FILES_TAB-', disabled = True)],
			[sg.Tab("Set parameters", parameter_layout, key = '-SET_PARAMETERS_TAB-', disabled = True)],
			], key = "-TRAJ_TABGROUP1-")
		],
		]
		
	window = sg.Window('SUPER RES DATA WRANGLER v{}'.format(last_changed), layout)
	tree = window["-TREE-"]
	popup.close()
	
	# MAIN LOOP
	while True:
		# Read events and values
		event, values = window.read(timeout = 5000)
		
		# Exit	
		if event == sg.WIN_CLOSED or event == 'Exit':  
			break
			
		# Values
		convertfrom = values['-SELECT_INPUT-']
		convertto = values['-SELECT_OUTPUT-']
		browse_file = values['-BROWSE_FILE_RADIO-']
		input_file = values['-INPUT_FILE-']
		input_folder = values['-INPUT_FOLDER-']
		files_contain = values['-FILES_CONTAIN-']
		files_ext = values['-FILES_EXT-']
		acqfreq_input = values['-ACQ_FREQ_INPUT-']
		pix2um_input = values['-PIXEL_SIZE_INPUT-']
		xy_units_input = values['-XY_UNITS_INPUT-']
		xy_units_output = values['-XY_UNITS_OUTPUT-']
		time_units_input = values['-TIME_UNITS_INPUT-']
		time_units_output = values['-TIME_UNITS_OUTPUT-']
		delim_input = values['-DELIM_INPUT-']
		delim_output = values['-DELIM_OUTPUT-']
		tr_col_input = values['-TR_COL_INPUT-']
		tr_col_output = values['-TR_COL_OUTPUT-']
		x_col_input = values['-X_COL_INPUT-']
		x_col_output = values['-X_COL_OUTPUT-']
		y_col_input = values['-Y_COL_INPUT-']
		y_col_output = values['-Y_COL_OUTPUT-']
		t_col_input = values['-T_COL_INPUT-']
		t_col_output = values['-T_COL_OUTPUT-']
		header_num_input = values['-HEADER_NUM_INPUT-']
		header_num_output = values['-HEADER_NUM_OUTPUT-']
		starting_frame_input = values['-STARTING_FRAME_INPUT-']
		starting_frame_output = values['-STARTING_FRAME_OUTPUT-']
		
		# Timestamp
		stamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now()) # datestamp
		
		# Check variables
		check_variables()
		
		# Toggle 'Find files' button, 'LOAD DATA' button and Tree
		if browse_file == True:
			if input_file != "" and input_file != "Select input file":
				load_data_button = True
				find_files = False
				use_tree = False
				update_buttons()
			else:
				load_data_button = False
				find_files = False
				use_tree = False
				update_buttons()
		if browse_file == False:
			if input_folder != "" and input_folder != "Select directory containing input files":
				find_files == True
				use_tree = True
				update_buttons()
			else:
				find_files = False
				load_data_button = False
				use_tree = True
				update_buttons()
				
		# Exit	
		if event in (sg.WIN_CLOSED, 'Exit'):  
			break
		
		# Reset to hard coded default values
		if event == 'Default settings':
			convertfrom_original = convertfrom
			reset_defaults()
			if convertfrom_original == convertfrom:
				update_buttons()
			else:
				window['-TRAJ_TABGROUP1-'].Widget.select(0)
				window.Element('-LOAD_FILES_TAB-').update(disabled = True)
				window.Element('-SET_PARAMETERS_TAB-').update(disabled = True)
				window.Element('-INPUT_FILE-').update("Select input file") 
				window.Element('-INPUT_FOLDER-').update("Select directory containing input files") 
				treedata = sg.TreeData()
				tree.update(values=treedata)
				update_buttons()
		
		# Save settings
		if event == 'Save settings':
			save_defaults()
			update_buttons()

		# Load settings
		if event == 'Load settings':
			convertfrom_original = convertfrom
			load_defaults()
			if convertfrom_original == convertfrom:
				update_buttons()
			else:
				window['-TRAJ_TABGROUP1-'].Widget.select(0)
				window.Element('-LOAD_FILES_TAB-').update(disabled = True)
				window.Element('-SET_PARAMETERS_TAB-').update(disabled = True)
				window.Element('-INPUT_FILE-').update("Select input file") 
				window.Element('-INPUT_FOLDER-').update("Select directory containing input files") 
				treedata = sg.TreeData()
				tree.update(values=treedata)
				update_buttons()
		
		# About
		if event == 'About':
			splash = create_splash()
			keep_on_top = True,			
			splash.close()
		
		# Help
		
		# txt filetype information
		if event == 'txt Files':
			sg.Popup(
				"HELP - txt Files",
				" ",
				"PalmTracer txt files contain 3 headers",
				" ",
				"Each data line(row) contains 8 tab separated columns:",
				"     [1]Track; (Trajectory#, starting at 1)",
				"     [2]Plane; (Frame#, starting at 1)",
				"     [3]CentroidX(px);",
				"     [4]CentroidY(px);",
				"    *[5]CentroidZ(um);",
				"    *[6]Integrated_Intensity;",
				"    *[7]id;",
				"    *[8]Pair_Distance(px);",
				" ",
				"*Columns not needed for conversion, can be any number",
				" ", 
				keep_on_top = True,
			)
		
		# trc filetype information
		if event == 'trc Files':
			sg.Popup(
				"HELP - trc Files",
				" ",
				"PalmTracer trc files contain no headers",
				" ",
				"Each data line(row) contains 6 tab separated columns:",
				"      [1]Trajectory#; (starting at 1)",
				"      [2]Frame#; (starting at 1)",
				"      [3]X-position(px);",
				"      [4]Y-position(px);",
				"     *[5]-1;",
				"     *[6]Integrated_Intensity;",
				" ",
				"*Columns not needed for conversion, can be any number",
				" ", 
				keep_on_top = True,
			)
		
		# trxyt filetype information
		if event == 'trxyt Files':
			sg.Popup(
				"HELP - trxyt Files",
				" ",
				"NASTIC/segNASTIC/BOOSH trxyt files contain no headers",
				" ",
				"Each data line(row) contains 4 space separated columns:",
				"     [1]Trajectory#; (starting at 1)",
				"     [2]X-position (um);",
				"     [3]Y-position (um);",
				"     [4]Frame time (s);",
				" ", 
				keep_on_top = True,
			)
		
		# ascii (uncorrected and drift corrected) filetype information
		if event == 'ascii Files':
			sg.Popup(
				"HELP - ascii Files",
				" ",
				"Converting a file to ascii also generates an id file containing the trajectory# (ID) of each data line in the ascii file. Both files are needed to convert back to another filetype.",
				" ",
				"ascii files contain no headers.",
				" ",
				"Each data line(row) contains 9 comma separated columns:",
				"    *[1]1;",
				"     [2]Frame#, starting at 0;",
				"     [3]n; (Trajectory# per Frame, starting at 1 each frame)",
				"     [4]X-position; (**px=uncorrected; nm=drift corrected)",
				"     [5]Y-position; (**px=uncorrected; nm=drift corrected)",
				"    *[6]Integrated_Intensity",
				"    *[7]0;",
				"    *[8]0;",
				"    *[9]0;",
				" ",
				"*Columns not needed for conversion, can be any number",
				"**IMPORTANT: X,Y values are in pixels before drift correction, and in nanometers following drift correction",
				" ", 
				keep_on_top = True,
			)
		
		# id filetype information
		if event == 'id Files':
			sg.Popup(
				"HELP - id Files",
				" ",
				"id files are generated when converting another filetype into the ascii file format, in the same folder and with the same name as the matching ascii file (e.g., filename_YYMMDD-HHMMSS.ascii and filename_YYMMDD-HHMMSS.id)."
				" ",
				"Both files are needed to convert back to another filetype - the id file should be in the same location and have the same name as the ascii file in order to be found by the GUI."
				" ",
				"id files contain the Trajectory# information of the corresponding ascii file.",
				" ",
				"id files contain no headers",
				" ",
				"Files contain a single column:",
				"     [1]Trajectory#;",
				" ", 
				keep_on_top = True,
			)
		
		# csv(TrackMate) filetype information
		if event == 'csv(TrackMate) Files':
			sg.Popup(
				"HELP - csv(TrackMate) Files",
				" ",
				"The number of headers can vary (requires manual selection)",
				" ",
				"Each data line(row) contains up to 30 comma separated columns, 4 of which must contain the following:",
				"    *[  ]Trajectory#; (e.g., TRACK_ID)",
				"    *[  ]X-position; (e.g., POSITION_X; **px,um or nm)",
				"    *[  ]Y-position; (e.g., POSITION_Y; **px,um or nm)",
				"    *[  ]Time; (e.g., FRAME or POSITION_T, **Frames or s)",
				" ",
				"*Columns can be in any order",
				"** Units depend on how FIJI interprets the microscope data",
				" ", 
				keep_on_top = True,
			)
			
		# other(manual input) filetype information
		if event == 'other(manual input) Files':
			sg.Popup(
				"HELP - other(manual input) Files",
				" ",
				"Parameters can be manually selected in order to convert from other trajectory filetypes",
				"Parameters that require selection:",
				" - X,Y units: spatial units for X and Y position data",
				"     (pixels, microns or nanometers)",
				" - Pixel size (um/px): converts between X,Y units",
				" - Time units: temporal units",
				"     (Frames or seconds)",
				" - Acquisition frequency (Hz): converts between Time units",
				" - Number of headers: number of rows before data starts",
				"     (between 0 and 10)",
				" - Delimiter:separates data into columns",
				"     (tab, comma, 1 space, 2 spaces, 3 spaces, 4 spaces,\n     semicolon)",
				" - Trajectory col: column number containing trajectory data",
				"     (between 1 and 30)",
				" - X-position col: column number containing X-position data",
				"     (between 1 and 30)",
				" - Y-position col: column number containing Y-position data", 
				"     (between 1 and 30)",
				" - Time/Frame col: column number containing Time/Frame# data", 
				"     (between 1 and 30)",
				" - Starting frame: frame number of the first frame",
				"     (0 or 1)"
				" ", 
				keep_on_top = True,
			)
			
		# Pixel size parameter information
		if event == 'Pixel size':
			sg.Popup(
				"HELP - Pixel size(um/px):",
				" ",
				"Spatial units of X and Y position data (pixels, microns or nanometers) depends on the filetype.",
				" ",
				"In order to convert between filetypes in pixels and filetypes in either microns or nanometers, the Pixel size (um/px) parameter is used (number of microns per pixel). The default value for Pixel size is 0.106um/px.",
				" ",
				"The default pixel size value can be changed by typing the desired value in the GUI, then clicking 'File>Save settings' in the menu. The original default values can be restored by clicking 'File>Default settings', which can then be saved by clicking 'File>Save settings'.",
				" ", 
				keep_on_top = True,
			)
		
		# Acquisition frequency parameter information
		if event == 'Acquisition frequency':
			sg.Popup(
				"HELP - Acquisition frequency(Hz):",
				" ",
				"Temporal units (Frame# or seconds) depends on the filetype.",
				" ",
				"The Acquisition frequency (Hz) parameter is used to convert between filetypes with different temporal units. To find the acquisition frequency (Hz), divide 1 by the frame time (s): e.g., for data where a frame is acquired every 0.02s: 1/0.02 = 50Hz. The default value for Acquisition Frequency is 50Hz.",
				" ",
				"The default acquisition frequency value can be changed by typing the desired value in the GUI, then clicking 'File>Save settings' in the menu. The original default values can be restored by clicking 'File>Default settings', which can then be saved by clicking 'File>Save settings'.",
				" ", keep_on_top = True
			)
			
		# Licence	
		if event == 'Licence':
			sg.Popup(
				"Licence",
				"Creative Commons CC BY-NC 4.0",
				"https://creativecommons.org/licenses/by-nc/4.0/legalcode", 
				no_titlebar = True,
				grab_anywhere = True,	
				keep_on_top = True,
			)		
				
		# Check for updates
		if event == 'Updates':
			webbrowser.open("https://github.com/tristanwallis/smlm_clustering/releases",new=2)
		
		# Input filetype selection event
		if event == "-SELECT_INPUT-":
			window.Element('-LOAD_FILES_TAB-').update(disabled = True)
			window.Element('-SET_PARAMETERS_TAB-').update(disabled = True) 
			window.Element('-INPUT_FILE-').update("Select input file") 
			window.Element('-INPUT_FOLDER-').update("Select directory containing input files") 
			treedata = sg.TreeData()
			tree.update(values=treedata)
			update_buttons()
			
		# Output filetype selection event
		if event == "-SELECT_OUTPUT-":
			window.Element('-LOAD_FILES_TAB-').update(disabled = True)
			window.Element('-SET_PARAMETERS_TAB-').update(disabled = True)
			update_buttons()
			
		# Confirm filetypes event
		if event == "-CONFIRM-":			
			if convertfrom == convertto:
				sg.Popup("ALERT", "Selected input and output filetypes are the same.\nPlease select filetypes that are different.", keep_on_top = True)
			else:
				window.Element('-TXT_INPUT_TAB-').update(disabled = False)
				window.Element('-LOAD_FILES_TAB-').update(disabled = False)
				if browse_file == True: 
					window.Element('-LOAD_TRAJ_FILE_TEXT-').update("Please browse for and select 1 {} file to convert".format(convertfrom))
				else:
					window.Element('-LOAD_TRAJ_FILE_TEXT-').update("Please browse for a folder containing {} files, then press 'Find files' and untick files to exclude from conversion".format(convertfrom))
					
				window['-TRAJ_TABGROUP1-'].Widget.select(1)
			if convertfrom == "other(manual input)":
				window.Element('-TR_COL_INPUT-').update(1)
				window.Element('-X_COL_INPUT-').update(2)
				window.Element('-Y_COL_INPUT-').update(3)
				window.Element('-T_COL_INPUT-').update(4)
				window.Element('-STARTING_FRAME_INPUT-').update(1)
			elif convertfrom == "csv(TrackMate)":
				window.Element('-XY_UNITS_INPUT-').update(disabled = False) 
				window.Element('-TIME_UNITS_INPUT-').update(disabled = False) 
				window.Element('-HEADER_NUM_INPUT-').update(disabled = False) 
				window.Element('-TR_COL_INPUT-').update(disabled = False) 
				window.Element('-X_COL_INPUT-').update(disabled = False) 
				window.Element('-Y_COL_INPUT-').update(disabled = False) 
				window.Element('-T_COL_INPUT-').update(disabled = False) 
				window.Element('-STARTING_FRAME_INPUT-').update(disabled = False) 
				xy_units_input = "um" 
				time_units_input = "s" 
				header_num_input = 4 
				tr_col_input = 3 
				x_col_input = 5 
				y_col_input = 6 
				t_col_input = 8
				starting_frame_input = 0
				
			update_buttons()
	
		# TAB 2 EVENTS (LOAD FILES TAB)
		
		# Select browse option (file or folder) event
		if event == '-BROWSE_FILE_RADIO-' or '-BROWSE_FOLDER_RADIO-':
			if browse_file == True:
				find_files = False
				window.Element('-LOAD_TRAJ_FILE_TEXT-').update("Please browse for and select 1 {} file to convert".format(convertfrom))
			elif browse_file == False:
				if input_folder != "" and input_folder != "Select directory containing input files":
					find_files = True
				window.Element('-LOAD_TRAJ_FILE_TEXT-').update("Please browse for a folder containing {} files, then press 'Find files' and untick files to exclude from conversion".format(convertfrom))
			update_buttons()
		
		# Browse Folder event
		if event == "-INPUT_FOLDER-":
			if input_folder != "" and input_folder != "Select directory containing input files":
				find_files = True
				use_tree = True
			else:
				find_files = False
				use_tree = True
			update_buttons()
			
		# Find files event
		if event == "-FIND_FILES-":
			if input_folder != "" and input_folder != "Select directory containing input files":
				split_files_list = []
				files_list = []
				if convertfrom == "txt":
					files = glob.glob(input_folder + '/**/*{}*.txt'.format(files_contain),recursive = True)
				elif convertfrom == "trc":
					files = glob.glob(input_folder + '/**/*{}*.trc'.format(files_contain),recursive = True)
				elif convertfrom == "trxyt":
					files = glob.glob(input_folder + '/**/*{}*.trxyt'.format(files_contain),recursive = True)
				elif convertfrom == "ascii" or convertfrom == "ascii(drift corrected)":
					files = glob.glob(input_folder + '/**/*{}*.ascii'.format(files_contain),recursive = True)
				elif convertfrom == "csv(TrackMate)":
					files = glob.glob(input_folder + '/**/*{}*.csv'.format(files_contain),recursive = True)
				elif convertfrom == "other(manual input)":
					if files_ext == "":
						files = glob.glob(input_folder + '/**/*{}*.*'.format(files_contain),recursive = True)
					else:
						files = glob.glob(input_folder + '/**/*{}*{}'.format(files_contain,files_ext),recursive = True)
				files = [file.replace("\\","/")for file in files] # get all paths into proper forward slash style!_
				for file in files:
					files_list.append(file)
					split_files = file.split("/")
					filename_split_files = split_files[-1]
					split_files_list.append(filename_split_files)
				combolist = [""]+[x for x in range(1,len(files)+1)]
				treedata = sg.TreeData()
				ct = 1
				if len(files_list) == 0:
					tree.update(treedata)
					load_data_button = False
					if convertfrom in ["txt","trc","trxyt","ascii","ascii(drift corrected)","csv(TrackMate)"]: 
						if files_contain == "":
							print("\n\nNo input files found. Make sure selected directory contains .{} files".format(convertfrom))
							sg.Popup("ALERT", "No input files found. Please make sure selected directory contains .{} files".format(convertfrom), keep_on_top = True)
						else:
							print("\n\nNo input files found. \n\nPlease make sure the selected directory has files with filenames that contain the phrase '{}'.".format(files_contain))
							sg.Popup("ALERT", "No input files found.\nPlease make sure the selected directory has files with filenames that contain the phrase '{}'.".format(files_contain), keep_on_top = True)
					else:
						if files_contain == "" and files_ext == "":
							print("\n\nNo input files found. \n\nPlease make sure the selected directory is not empty.")
							sg.Popup("ALERT", "No input files found.\nPlease make sure the selected directory is not empty.", keep_on_top = True)
						
						elif files_contain != "" and files_ext != "":
							print("\n\nNo input files found. \n\nPlease make sure the selected directory has files with filenames that contain the phrase '{}' and end with '{}'.".format(files_contain,files_ext))
							sg.Popup("ALERT", "No input files found.\nPlease make sure the selected directory has files with filenames that contain the phrase '{}' and end with '{}'.".format(files_contain,files_ext), keep_on_top = True)
							
						elif files_contain != "" and files_ext == "":
							print("\n\nNo input files found. \n\nPlease make sure the selected directory has files with filenames that contain the phrase '{}'.".format(files_contain))
							sg.Popup("ALERT", "No input files found.\nPlease make sure the selected directory has files with filenames that contain the phrase '{}'.".format(files_contain), keep_on_top = True)
						
						elif files_contain == "" and files_ext != "":
							print("\n\nNo input files found. \n\nPlease make sure the selected directory has files with filenames that end with '{}'.".format(files_ext))
							sg.Popup("ALERT", "No input files found.\nPlease make sure the selected directory has files with filenames that end with '{}'.".format(files_ext), keep_on_top = True)
						
						treedata = sg.TreeData()
						tree.update(values=treedata)
						load_data_button = False
						window.Element('-SET_PARAMETERS_TAB-').update(disabled = True)
						update_buttons()
				elif len(files_list) >0:
					if len(files_list) == 1:
						print("\n\nInput files found in selected directory: (1 file)\n-------------------------------------------------------")
					elif len(files_list) > 1:
						print("\n\nInput files found in selected directory: ({} files)\n-------------------------------------------------------".format(len(files_list)))
					for selectedfile in split_files_list:
						treedata.insert("",combolist[ct], combolist[ct],values = [selectedfile], icon = check[1])
						print("\nFile {}".format(combolist[ct]),files_list[ct-1])
						tree.update(treedata)
						tree_icon_dict.update({ct:"True"})
						ct +=1
					if "True" in tree_icon_dict.values():
						load_data_button = True
						found_files = True
						use_tree = True
						update_buttons()
					else:
						load_data_button = False
						found_files = False
						use_tree = True
						update_buttons()				
	
		# Tree event
		if event == "-TREE-":
			try:
				filenumber = values['-TREE-'][0]
				ct = filenumber
				if filenumber in tree.metadata:
					tree.metadata.remove(filenumber)
					tree_icon_dict.update({ct:"False"})
					tree.update(key=filenumber,icon=check[0])	
				else:
					tree.metadata.append(filenumber)
					tree_icon_dict.update({ct:"True"})
					tree.update(key=filenumber, icon=check[1])
				if "True" in tree_icon_dict.values():
					load_data_button = True
					update_buttons()
				else:
					load_data_button = False
					update_buttons()
			except:
				pass
		
		# Load data event
		if event == "-LOAD-":
			# input file
			if browse_file == True:
				print("\n\nInput file selected for conversion:\n-------------------------------------------------------")
				print(input_file)
				if convertfrom == "other(manual input)":
					input_file_extension = input_file.split(".")[-1]
					window.Element('-INPUT_FILE_FRAME-').update("Input {} parameters".format(input_file_extension))
			# input folder
			elif browse_file == False:
				selected_files = []
				for ct,infilename in enumerate(files_list,start=1):
					if tree_icon_dict[ct] == "True":	
						selected_files.append(infilename)
						input_files_extension = infilename.split(".")[-1]
						window.Element('-INPUT_FILE_FRAME-').update("Input {} parameters".format(input_files_extension))
				if len(selected_files) == 1:
					print("\n\nInput file selected for conversion: (1 file)\n-------------------------------------------------------")
					print ("\nFile {}".format(ct),infilename)
				elif len(selected_files) >1:
					print("\n\nInput files selected for conversion: ({} files)\n-------------------------------------------------------".format(len(selected_files)))
					for ct,infilename in enumerate(selected_files,start = 1):
						print ("\nFile {}".format(ct),infilename)	
			# change to 'set parameters' tab
			window.Element("-SET_PARAMETERS_TAB-").update(disabled = False) 
			window["-TRAJ_TABGROUP1-"].Widget.select(2)
			update_buttons()
		
		# pixel size event
		if event == "-XY_UNITS_INPUT-":
			update_buttons()
			
		# acquisition frequency event
		if event == "-TIME_UNITS_INPUT-":
			update_buttons()
			
		# Convert files event
		if event == "-CONVERT-":
			infilenames = []
			if browse_file == True:
				infilenames.append(input_file)
			elif browse_file == False:
				for ct,infilename in enumerate(files,start=1):
					if tree_icon_dict[ct] == "True":
						infilenames.append(infilename)	
			# read txt file
			if convertfrom == "txt":
				for ct, infilename in enumerate(infilenames,start=1):
					print("\n\nConverting File {}:\n-------------------------------------------------------".format(ct))
					try:
						rawdata = read_txt(infilename)
						if rawdata != None:
							rawdata = sorted(rawdata, key=lambda x:x[0]) # sort rawdata on trajectory number
							read_file = True
							# Write trc file
							if convertto == "trc":
								try:
									write_trc(rawdata, infilename)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the txt file is in the correct format.\n\nClick 'Info>Help>txt Files' in the menu for more information on the txt file format.\n", keep_on_top = True)
							# Write trxyt file
							elif convertto == "trxyt":
								try:
									write_trxyt(rawdata, infilename)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the txt file is in the correct format.\n\nClick 'Info>Help>txt Files' in the menu for more information on the txt file format.\n", keep_on_top = True)
							# Write ascii file
							elif convertto == "ascii":
								try:
									ids = write_ascii(rawdata, infilename)
									write_ids(ids, infilename)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the txt file is in the correct format.\n\nClick 'Info>Help>txt Files' in the menu for more information on the txt file format.\n", keep_on_top = True)
							# Write ascii(drift corrected) file
							elif convertto == "ascii(drift corrected)":
								try:
									ids = write_dcascii(rawdata, infilename)
									write_ids(ids, infilename)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the txt file is in the correct format.\n\nClick 'Info>Help>txt Files' in the menu for more information on the txt file format.\n", keep_on_top = True)
					except:
						read_file = False
						print("File could not be read")
						sg.Popup("File could not be read.\n\nPlease make sure the txt file is in the correct format.\nClick 'Info>Help>txt Files' in the menu for more information on the txt file format.\n", keep_on_top = True)
				if read_file == True:
					print("\n\nDone!\n=======================================================")
					sg.Popup("Done!", keep_on_top = True)			
			# Read trc file
			elif convertfrom == "trc":
				for ct, infilename in enumerate(infilenames,start=1):
					print("\n\nConverting File {}:\n-------------------------------------------------------".format(ct))
					try:
						rawdata = read_trc(infilename)
						if rawdata != None:
							rawdata = sorted(rawdata, key=lambda x:x[0]) # sort rawdata on trajectory number
							read_file = True
							# Write txt file
							if convertto == "txt":
								try:
									write_txt(rawdata, infilename)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the trc file is in the correct format.\n\nClick 'Info>Help>trc Files' in the menu for more information on the trc file format.\n", keep_on_top = True)
							# Write trxyt file
							elif convertto == "trxyt":
								try:
									write_trxyt(rawdata, infilename)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the trc file is in the correct format.\n\nClick 'Info>Help>trc Files' in the menu for more information on the trc file format.\n", keep_on_top = True)
							# Write ascii file
							elif convertto == "ascii":
								try:
									ids = write_ascii(rawdata, infilename)
									write_ids(ids, infilename)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the trc file is in the correct format.\n\nClick 'Info>Help>trc Files' in the menu for more information on the trc file format.\n", keep_on_top = True)
							# Write ascii(drift corrected) file
							elif convertto == "ascii(drift corrected)":
								try:
									ids = write_dcascii(rawdata, infilename)
									write_ids(ids, infilename)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the trc file is in the correct format.\n\nClick 'Info>Help>trc Files' in the menu for more information on the trc file format.\n", keep_on_top = True)
					except:
						read_file = False
						print("File could not be read")
						sg.Popup("File could not be read.\n\nPlease make sure the trc file is in the correct format.\n\nClick 'Info>Help>trc Files' in the menu for more information on the trc file format.\n", keep_on_top = True)
				if read_file == True:
					print("\n\nDone!\n=======================================================")
					sg.Popup("Done!", keep_on_top = True)	
			# Read trxyt file
			elif convertfrom == "trxyt":
				for ct, infilename in enumerate(infilenames,start=1):
					print("\n\nConverting File {}:\n-------------------------------------------------------".format(ct))
					try:
						rawdata = read_trxyt(infilename)
						if rawdata != None:
							rawdata = sorted(rawdata, key=lambda x:x[0]) # sort rawdata on trajectory number
							read_file = True
							# Write txt file
							if convertto == "txt":
								try:
									write_txt(rawdata, infilename)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the trxyt file is in the correct format.\n\nClick 'Info>Help>trxyt Files' in the menu for more information on the trxyt file format.\n", keep_on_top = True)
							# Write trc file
							elif convertto == "trc":
								try:
									write_trc(rawdata, infilename)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the trxyt file is in the correct format.\n\nClick 'Info>Help>trxyt Files' in the menu for more information on the trxyt file format.\n", keep_on_top = True)
							# Write ascii file
							elif convertto == "ascii":
								try:
									ids = write_ascii(rawdata, infilename)
									write_ids(ids, infilename)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the trxyt file is in the correct format.\n\nClick 'Info>Help>trxyt Files' in the menu for more information on the trxyt file format.\n", keep_on_top = True)
							# Write ascii(drift corrected) file
							elif convertto == "ascii(drift corrected)":
								try:
									ids = write_dcascii(rawdata, infilename)
									write_ids(ids, infilename)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the trxyt file is in the correct format.\n\nClick 'Info>Help>trxyt Files' in the menu for more information on the trxyt file format.\n", keep_on_top = True)
					except:
						read_file = False
						print("File could not be read")
						sg.Popup("File could not be read.\n\nPlease make sure the trxyt file is in the correct format.\n\nClick 'Info>Help>trxyt Files' in the menu for more information on the trxyt file format.\n", keep_on_top = True)
				if read_file == True:
					print("\n\nDone!\n=======================================================")
					sg.Popup("Done!", keep_on_top = True)
			# Read ascii file
			elif convertfrom == "ascii":
				for ct, infilename in enumerate(infilenames,start=1):
					print("\n\nConverting File {}:\n-------------------------------------------------------".format(ct))
					try:
						try:
							ids = read_ids(infilename)
							ids_found = True
						except:
							print("File not found")
							read_file = False
							sg.popup("Matching trajectory ID file not found.\n\nPlease make sure ascii and id files are in the same location (folder) and have the same name.\nClick 'Info>Help>id Files' in the menu for more information on the id file format.\n", keep_on_top = True)
							ids_found = False
						if ids_found == True:
							if ids != None:
								try:
									rawdata = read_ascii(infilename, ids)
									if rawdata != None:
										try:
											rawdata = sorted(rawdata, key=lambda x:x[0]) # sort rawdata on trajectory number
											read_file = True
										except:
											read_file = False
											print("File could not be read")
											sg.Popup("File could not be read.\n\nPlease make sure the ascii and id files are in the correct format.\n\nClick 'Info>Help>ascii Files' and 'Info>Help>id Files' in the menu for more information on the ascii and id file format.\n", keep_on_top = True)
								except:
									read_file = False
									print("File could not be read")
									sg.Popup("File could not be read.\n\nPlease make sure the ascii and id files are in the correct format.\n\nClick 'Info>Help>ascii Files' and 'Info>Help>id Files' in the menu for more information on the ascii and id file format.\n", keep_on_top = True)
								if rawdata == None:
									data_read = False
								else:
									data_read = True
								if data_read == True:
									if len(rawdata)>1:
										# Write txt file
										if convertto == "txt":
											try:
												write_txt(rawdata, infilename)
											except:
												print("File could not be written")
												sg.Popup("File could not be written.\n\nPlease make sure the ascii and id files are in the correct format.\n\nClick 'Info>Help>ascii Files' and 'Info>Help>id Files' in the menu for more information on the ascii and id file format.\n", keep_on_top = True)
										# Write trc file
										elif convertto == "trc":
											try:
												write_trc(rawdata, infilename)
											except:
												print("File could not be written")
												sg.Popup("File could not be written.\n\nPlease make sure the ascii and id files are in the correct format.\n\nClick 'Info>Help>ascii Files' and 'Info>Help>id Files' in the menu for more information on the ascii and id file format.\n", keep_on_top = True)
										# Write trxyt file
										elif convertto == "trxyt":
											try:
												write_trxyt(rawdata, infilename)
											except:
												print("File could not be written")
												sg.Popup("File could not be written.\n\nPlease make sure the ascii and id files are in the correct format.\n\nClick 'Info>Help>ascii Files' and 'Info>Help>id Files' in the menu for more information on the ascii and id file format.\n", keep_on_top = True)
										# Write ascii(drift corrected) file
										elif convertto == "ascii(drift corrected)":
											try:
												ids = write_dcascii(rawdata, infilename)
												write_ids(ids, infilename)
											except:
												print("File could not be written")
												sg.Popup("File could not be written.\n\nPlease make sure the ascii and id files are in the correct format.\n\nClick 'Info>Help>ascii Files' and 'Info>Help>id Files' in the menu for more information on the ascii and id file format.\n", keep_on_top = True)
					except:
						read_file = False
						print("File could not be read")
						sg.Popup("File could not be read.\n\nPlease make sure the ascii and id files are in the correct format.\n\nClick 'Info>Help>ascii Files' and 'Info>Help>id Files' in the menu for more information on the ascii and id file format.\n", keep_on_top = True)
				if read_file == True:
					print("\n\nDone!\n=======================================================")
					sg.Popup("Done!", keep_on_top = True)
			# Read .ascii file (drift corrected)
			elif convertfrom == "ascii(drift corrected)":
				for ct, infilename in enumerate(infilenames,start=1):
					print("\n\nConverting File {}:\n-------------------------------------------------------".format(ct))
					try:
						try:
							ids = read_ids(infilename)
							ids_found = True
						except:
							read_file = False
							print("File not found")
							sg.popup("Matching trajectory ID file not found.\n\nPlease make sure ascii and id files are in the same location (folder) and have the same name.\nClick 'Info>Help>id Files' in the menu for more information on the id file format.\n", keep_on_top = True)
							ids_found = False
						if ids_found == True:
							if ids != None:
								try:
									rawdata = read_dcascii(infilename, ids)
									if rawdata != None:
										try:
											rawdata = sorted(rawdata, key=lambda x:x[0]) # sort rawdata on trajectory number
											read_file = True
										except:
											read_file = False
											print("File could not be read")
											sg.Popup("File could not be read.\n\nPlease make sure the ascii and id files are in the correct format.\n\nClick 'Info>Help>ascii Files' and 'Info>Help>id Files' in the menu for more information on the ascii and id file format.\n", keep_on_top = True)
								except: 
									read_file = False
									print("File could not be read")
									sg.Popup("File could not be read.\n\nPlease make sure the ascii and id files are in the correct format.\n\nClick 'Info>Help>ascii Files' and 'Info>Help>id Files' in the menu for more information on the ascii and id file format.\n", keep_on_top = True)
								if rawdata == None:
									data_read = False
								else:
									data_read = True
								if data_read == True:
									if len(rawdata)>1:
										# Write .txt file
										if convertto == "txt":
											try:
												write_txt(rawdata, infilename)
											except:
												print("File could not be written")
												sg.Popup("File could not be written.\nPlease make sure the drift corrected ascii and id files are in the correct format.\n\nClick 'Info>Help>ascii Files' and 'Info>Help>id Files' in the menu for more information on the drift corrected ascii and id file format.\n", keep_on_top = True)
										# Write .trc file
										elif convertto == "trc":
											try:
												write_trc(rawdata, infilename)
											except:
												print("File could not be written")
												sg.Popup("File could not be written.\n\nPlease make sure the drift corrected ascii and id files are in the correct format.\n\nClick 'Info>Help>ascii Files' and 'Info>Help>id Files' in the menu for more information on the drift corrected ascii and id file format.\n", keep_on_top = True)
										# Write .trxyt file
										elif convertto == "trxyt":
											try:
												write_trxyt(rawdata, infilename)
											except:
												print("File could not be written")
												sg.Popup("File could not be written.\n\nPlease make sure the drift corrected ascii and id files are in the correct format.\n\nClick 'Info>Help>ascii Files' and 'Info>Help>id Files' in the menu for more information on the drift corrected ascii and id file format.\n", keep_on_top = True)
										# Write .ascii file
										elif convertto == "ascii":
											try:
												ids = write_ascii(rawdata, infilename)
												write_ids(ids, infilename)
											except:
												print("File could not be written")
												sg.Popup("File could not be written.\n\nPlease make sure the drift corrected ascii and id files are in the correct format.\n\nClick 'Info>Help>ascii Files' and 'Info>Help>id Files' in the menu for more information on the drift corrected ascii and id file format.\n", keep_on_top = True)
					except:
						read_file = False
						print("File could not be read")
						sg.Popup("File could not be read.\n\nPlease make sure the drift corrected ascii and id files are in the correct format.\n\nClick 'Info>Help>ascii Files' and 'Info>Help>id Files' in the menu for more information on the drift corrected ascii and id file format.\n", keep_on_top = True)
				if read_file == True:
					print("\n\nDone!\n=======================================================")
					sg.Popup("Done!", keep_on_top = True)	
			# Read csv(TrackMate) file
			elif convertfrom == "csv(TrackMate)":
				for ct, infilename in enumerate(infilenames,start=1):
					print("\n\nConverting File {}:\n-------------------------------------------------------".format(ct))
					try:
						rawdata = read_csv(infilename)
						if rawdata != None:
							rawdata = sorted(rawdata, key=lambda x:x[1]) # sort rawdata on frame number first
							rawdata = sorted(rawdata, key=lambda x:x[0]) # sort rawdata on trajectory number second
							read_file = True
							# Write txt file
							if convertto == "txt":
								try:
									write_txt(rawdata, infilename)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the TrackMate csv file is in the correct format.\n\nClick 'Info>Help>csv(TrackMate) Files' in the menu for more information on the TrackMate csv file format.\n", keep_on_top = True)
							# Write trc file
							elif convertto == "trc":
								try:
									write_trc(rawdata, infilename)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the TrackMate csv file is in the correct format.\n\nClick 'Info>Help>csv(TrackMate) Files' in the menu for more information on the TrackMate csv file format.\n", keep_on_top = True)
							# Write trxyt file
							elif convertto == "trxyt":
								try:
									write_trxyt(rawdata, infilename)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the TrackMate csv file is in the correct format.\n\nClick 'Info>Help>csv(TrackMate) Files' in the menu for more information on the TrackMate csv file format.\n", keep_on_top = True)
							# Write ascii file
							elif convertto == "ascii":
								try:
									ids = write_ascii(rawdata, infilename)
									write_ids(ids, infilename)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the TrackMate csv file is in the correct format.\n\nClick 'Info>Help>csv(TrackMate) Files' in the menu for more information on the TrackMate csv file format.\n", keep_on_top = True)
							# Write ascii(drift corrected) file
							elif convertto == "ascii(drift corrected)":
								try:
									ids = write_dcascii(rawdata, infilename)
									write_ids(ids, infilename)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the TrackMate csv file is in the correct format.\n\nClick 'Info>Help>csv(TrackMate) Files' in the menu for more information on the TrackMate csv file format.\n", keep_on_top = True)
					except:
						read_file = False
						print("File could not be read")
						sg.Popup("File could not be read.\n\nPlease make sure the TrackMate csv file is in the correct format.\n\nClick 'Info>Help>csv(TrackMate) Files' in the menu for more information on the TrackMate csv file format.\n\nPlease check that the following input file parameters are correct:\n\n - Number of headers: number of text/blank rows before rows containing trajectory data start\n\n - Trajectory col: column number (starting from 1) that contains trajectory number information\n\n - X-position col: column number (starting from 1) that contains X centroid information in pixels, microns or nanometers\n\n - Y-position col: column number (starting from 1) that contains Y centroid information in pixels, microns or nanometers\n\n - Time/Frame col: column number (starting from 1) that contains time information in seconds or Frames\n\n - Starting frame: the number of the first frame (if Time units are in Frames)\n", keep_on_top = True)
				if read_file == True:
					print("\n\nDone!\n=======================================================")
					sg.Popup("Done!", keep_on_top = True)
			# Read other(manual input) file	
			elif convertfrom == "other(manual input)":
				if tr_col_input in [x_col_input,y_col_input,t_col_input] or x_col_input in [tr_col_input,y_col_input,t_col_input] or y_col_input in [tr_col_input,x_col_input,t_col_input] or t_col_input in [x_col_input,y_col_input,tr_col_input]:
					print("Please make sure column numbers are different")
					sg.Popup("Please make sure column numbers are different\n", keep_on_top = True)
				else:
					for ct, infilename in enumerate(infilenames,start=1):
						print("\n\nConverting File {}:\n-------------------------------------------------------".format(ct))
						try:
							rawdata = read_other(infilename)
							if rawdata != None:
								rawdata = sorted(rawdata, key=lambda x:x[1]) # sort rawdata on frame number first
								rawdata = sorted(rawdata, key=lambda x:x[0]) # sort rawdata on trajectory number
								read_file = True
								# Write .txt file
								if convertto == "txt":
									try:
										write_txt(rawdata, infilename)
									except:
										print("File could not be written")
										sg.Popup("File could not be written.\n\nPlease make sure the file is in the correct format.\n", keep_on_top = True)
								# Write .trc file
								elif convertto == "trc":
									try:
										write_trc(rawdata, infilename)
									except:
										print("File could not be written")
										sg.Popup("File could not be written.\n\nPlease make sure the file is in the correct format.\n", keep_on_top = True)
								# Write .trxyt file
								elif convertto == "trxyt":
									write_trxyt(rawdata, infilename)

								# Write .ascii file
								elif convertto == "ascii":
									try:
										ids = write_ascii(rawdata, infilename)
										write_ids(ids, infilename)
									except:
										print("File could not be written")
										sg.Popup("File could not be written.\n\nPlease make sure the file is in the correct format.\n", keep_on_top = True)
								# Write ascii(drift corrected) file
								elif convertto == "ascii(drift corrected)":
									try:
										ids = write_dcascii(rawdata, infilename)
										write_ids(ids, infilename)
									except:
										sg.Popup("File could not be written.\n\nPlease make sure the file is in the correct format.\n", keep_on_top = True)
						except:
							read_file = False
							print("File could not be read")
							sg.Popup("File could not be read.\n\nPlease make sure the selected input file parameters are correct.\n\nClick 'Info>Help>other(manual input) Files' in the menu for more information.\n", keep_on_top = True)
					if read_file == True:
						print("\n\nDone!\n=======================================================")
						sg.Popup("Done!", keep_on_top = True)
		
		# Update buttons
		if event: 
			update_buttons()
	
	print ("Exiting...")
	window.close()
	quit()