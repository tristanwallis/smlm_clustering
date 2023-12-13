# -*- coding: utf-8 -*-
'''
SUPER_RES_DATA_WRANGLER_GUI
PYSIMPLEGUI BASED GUI FOR THE CONVERSION OF TRAJECTORY FILES
CONVERT BETWEEN MUTLIPLE FILE FORMATS FROM THE PALMTRACER (METAMORPH PLUGIN)/TRACKMATE (IMAGEJ PLUGIN)--> SHARP VISU --> NASTIC/segNASTIC/BOOSH --> NASTIC WRANGLER SUPER RESOLUTION DATA PROCESSING PIPELINE

Design and coding: Tristan Wallis and Alex McCann
Queensland Brain Institute
The University of Queensland
Fred Meunier: f.meunier@uq.edu.au

REQUIRED:
Python 3.8 or greater
python -m pip install pysimplegui colorama scipy numpy scikit-learn

INPUT: 
Any of the below formats can be interchanged with any other:
.txt
.trc
.ascii
.ascii (drift corrected)
.trxyt

Additionally, the below format can be converted into any of the above filetypes:
.csv (TrackMate)

NOTES:
When writing .ascii files, a trajectory .id file containing the Trajectory ID (Trajectory#) of each .ascii data line (row) is generated, which is then used to convert back from .ascii to other formats.

Depending on the file conversion, intensity information is lost. This intensity information is not relevant to subsequent analyses, so no big deal.

Internally the system works in microns, so .trc and .txt formats need to be converted from pixels to microns (usually by using a Pixel size of 0.106um/px as the conversion factor). TrackMate .csv files are in microns. Drift corrected .ascii files are in nanometers whereas uncorrected .ascii files are in microns - this difference is accounted for in the file converter by selecting ".ascii (drift corrected)" for files in nm, and ".ascii" for files in um. 

The time information for each acquired frame in .trxyt files is in seconds, and needs to be converted to Frame# when converting to other filetypes. This is done using the Acquisition frequency (Hz) parameter (usually 50Hz). To find the Acquisition frequency of a file, divide 1 by the frame time (seconds): e.g., for a file where a frame is acquired every 0.02 seconds, 1/0.02 = 50Hz.    

USAGE: 
1 - Run the script (either by double clicking or navigating to the location of the script in the terminal and using the command 'python super_res_data_wrangler_GUI.py')
2 - Specify the file type you want to convert FROM using the dropdown box
3 - Specify the file type you want to convert TO using the dropdown box
4 - Browse for the file that you wish to convert using the 'Browse' button
5 - txt and trc files: specify the Pixel size in um/px
6 - trxyt files: specify the Acquisition frequency in Hz
7 - Press the "CONVERT FILE" button - files will be converted and saved to the same place as the original file, with the appropriate suffix and a date stamp

CHECK FOR UPDATES:
https://github.com/tristanwallis/smlm_clustering/releases
'''

last_changed = "20231212"

# LOAD MODULES (Functions)
import random
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from functools import reduce
import datetime
import webbrowser
import warnings

warnings.filterwarnings("ignore")

# MAIN PROG AND FUNCTIONS
if __name__ == "__main__": # has to be called this way for multiprocessing to work
	
	# LOAD MODULES (GUI and console)
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
	print(f'{Fore.GREEN}SUPER RES DATA WRANGLER {last_changed} initialising...{Style.RESET_ALL}')
	print(f'{Fore.GREEN}=================================================={Style.RESET_ALL}')
	popup = sg.Window("Initialising...",[[sg.T("SUPER RES DATA WRANGLER initialising...",font=("Arial bold",18))]],finalize=True,no_titlebar = True,alpha_channel=0.9)
	
	# VARS
	acqfreq = 50.0 #Hz
	pix2um = 0.106 # microns per pixel

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
		graph.DrawText("PySimpleGUI: https://pypi.org/project/PySimpleGUI/",(0,-150),color="white",font=("Any",10),text_location="center")	

		while True:
			# READ AND UPDATE VALUES
			event, values = splash.read(timeout=timeout) 
			ct += timeout
			# Exit	
			if event in (sg.WIN_CLOSED, 'Exit'): 
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
		global convertfrom, convertto, pix2um, acqfreq
		convertfrom = ".txt"
		convertto = ".trxyt"
		pix2um = 0.106
		acqfreq = 50.0	
		return 

	# SAVE SETTINGS
	def save_defaults():
		print ("\nSaving GUI settings to super_res_data_wrangler_gui.defaults...")
		with open("super_res_data_wrangler_gui.defaults","w") as outfile:
			outfile.write("{}\t{}\n".format("Convert from filetype",convertfrom))
			outfile.write("{}\t{}\n".format("Convert to filetype",convertto))
			outfile.write("{}\t{}\n".format("Pixel size (um/px)",pix2um))
			outfile.write("{}\t{}\n".format("Acquisition frequency (Hz)",acqfreq))	
		return

	# LOAD DEFAULTS
	def load_defaults():
		global convertfrom, convertto, pix2um, acqfreq
		try:
			with open ("super_res_data_wrangler_gui.defaults","r") as infile:
				print ("\nLoading GUI settings from super_res_data_wrangler_gui.defaults...")
				defaultdict = {}
				for line in infile:
					spl = line.split("\t")
					defaultdict[spl[0]] = spl[1].strip()
			convertfrom = (defaultdict["Convert from filetype"])
			convertto = (defaultdict["Convert to filetype"])
			pix2um = float(defaultdict["Pixel size (um/px)"])
			acqfreq = float(defaultdict["Acquisition frequency (Hz)"])
		except:
			print ("\nSettings could not be loaded")
		return
		
	# CHECK VARIABLES
	def check_variables():
		global pix2um, acqfreq
		try:
			pix2um = float(pix2um)
			if pix2um <= 0:
				pix2um = 0.106
		except:
			pix2um = 0.106	
		try:
			acqfreq = float(acqfreq)
			if acqfreq < 1:
				acqfreq = 1.0
		except:
			acqfreq = 50.0	
		return
	
	def update_buttons():
		window.Element("-INPUT_COMBO-").update(convertfrom)
		window.Element("-OUTPUT_COMBO-").update(convertto)
		window.Element("-PIXEL_SIZE-").update(pix2um)
		window.Element("-ACQUISITION_FREQUENCY-").update(acqfreq)
		if convertfrom == ".txt" or convertfrom == ".trc" or convertto == ".txt" or convertto == ".trc":
			window.Element('-PIXEL_SIZE_TEXT-').update(visible = True)
			window.Element('-PIXEL_SIZE-').update(visible = True)
		else:
			window.Element('-PIXEL_SIZE_TEXT-').update(visible = False)
			window.Element('-PIXEL_SIZE-').update(visible = False)
			
		if convertfrom == ".trxyt" or convertto == ".trxyt":
			window.Element('-ACQUISITION_FREQUENCY_TEXT-').update(visible = True)
			window.Element('-ACQUISITION_FREQUENCY-').update(visible = True)
		else:
			window.Element('-ACQUISITION_FREQUENCY_TEXT-').update(visible = False)
			window.Element('-ACQUISITION_FREQUENCY-').update(visible = False)
		return
		
	# Get filename prefix (everything before the suffix)
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
					x = spl[2]*pix2um # convert pixels to microns
					y = spl[3]*pix2um
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
					x = spl[2]*pix2um # convert pixels to microns
					y = spl[3]*pix2um
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
					fr = spl[3]*acqfreq
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
			
	# Read SharpViSu ascii (not drift corrected)
	def read_ascii(infilename,ids):
		'''
		Measurements in microns
		
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
					x = spl[3]
					y = spl[4]
					i = spl[5]
					rawdata.append([tr,fr,x,y,i])
					ct +=1
		if len(rawdata) != 0:
			print ("{} lines read".format(ct))			
			return rawdata
		elif len(rawdata) == 0:
			print("0 lines read")
			sg.popup("0 lines read.\n\nPlease make sure the file is not empty.\n", keep_on_top=True)
			
	# Read SharpViSu ascii (drift corrected)	
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
					x = spl[3]/1000 #convert nanometres to microns
					y = spl[4]/1000
					i = spl[5]
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
			
	# Read Trackmate .csv
	def read_csv(infilename):
		'''
		Measurements in microns
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
		TR_COL = []
		FR_COL = []
		X_COL = []
		Y_COL = []
		ct = 0
		with open(infilename,"r") as infile:
			for line in infile:
				ct+=1
				if ct == 1:
					title = [j for j in line.split(",")]
					col_ct = 0 
					for col in title:
						if col == "TRACK_ID":
							TR_COL = col_ct
						elif col == "POSITION_X":
							X_COL = col_ct
						elif col == "POSITION_Y":
							Y_COL = col_ct
						elif col == "FRAME":
							FR_COL = col_ct
						col_ct+=1
				try:	
					spl = [j for j in line.split(",")]
					tr = float(spl[TR_COL])
					fr = float(spl[FR_COL])+1
					x = float(spl[X_COL])
					y = float(spl[Y_COL])
					i = -1
					rawdata.append([tr,fr,x,y,i])
					ct+=1
				except:
					ct+=1
		if len(rawdata) != 0:
			print ("{} lines read".format(ct-3))				
			return rawdata		
		elif len(rawdata) == 0:
			print("0 lines read")
			sg.popup("0 lines read.\n\nPlease make sure the file is not empty.\n", keep_on_top=True)
			
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
				x = det[2]/pix2um # convert microns to pixels
				y = det[3]/pix2um
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
				x = det[2]/pix2um # convert microns to pixels
				y = det[3]/pix2um
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
				t = det[1]/acqfreq
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
					x = det[2]
					y = det[3]
					i = det[4]
					outstring = "1,{},{},{},{},{},0,0,0\n".format(frame-1,n,x,y,i)
					outfile.write(outstring)
					ids.append(tr)
					ct+=1
		print ("{} lines written".format(ct))					
		return ids	

	# Write SharpViSu ascii (drift corrected)
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
					outstring = "1,{},{},{},{},{},0,0,0\n".format(frame-1,n,x,y,i)
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

	menu_def = [
		['&File', ['&Load settings', '&Save settings','&Default settings','&Exit']],
		['&Info', ['&About', '&Help',['&.txt Files', '&.trc Files', '&.trxyt Files', '&.ascii Files', '&.id Files', '.csv (TrackMate) Files', '&Pixel size', '&Acquisition frequency'],'&Licence','&Updates' ]],
	]
	
	layout = [
		[sg.Menu(menu_def)],
		[sg.Push(),sg.T('SUPER RES DATA WRANGLER',font="Any 20"), sg.Push()],
		[sg.Push(), sg.T("Convert between trajectory filetypes", font = "Any 12 italic"), sg.Push()],
		[sg.T(" ")],
		[sg.T("Convert FROM:"),sg.Combo([".txt",".trc",".trxyt",".ascii", ".ascii (drift corrected)", ".csv (TrackMate)"], key = '-INPUT_COMBO-', default_value = ".txt", size = (18, 1), enable_events = True)],
		[sg.T("Convert TO:     "), sg.Combo([".txt",".trc",".trxyt",".ascii",".ascii (drift corrected)"], key = '-OUTPUT_COMBO-', 
		default_value = ".trxyt", size = (18,1), enable_events = True)],
		[sg.In(key = '-SELECTED_FILE-', size = (40,1), enable_events = True), sg.FileBrowse("Browse .txt", key = '-BROWSE_TXT-', target = '-SELECTED_FILE-', size = (10,1), file_types = (("PalmTracer.txt", "*.txt"),), visible = True), sg.FileBrowse("Browse .trc", key = '-BROWSE_TRC-', target = '-SELECTED_FILE-', size = (10,1), file_types = ((".trc File", "*.trc"),), visible = False),sg.FileBrowse("Browse .trxyt", key = '-BROWSE_TRXYT-', target = '-SELECTED_FILE-', size = (10,1), file_types = ((".trxyt File", "*.trxyt"),), visible = False),sg.FileBrowse("Browse .ascii", key = '-BROWSE_ASCII-', target = '-SELECTED_FILE-', size = (10,1), file_types = ((".ascii File", "*.ascii"),), visible = False), sg.FileBrowse("Browse .csv", key = '-BROWSE_CSV-', target = '-SELECTED_FILE-', size = (10,1), file_types = ((".csv File", "*.csv"),), visible = False)],
		[sg.T("Pixel size (um/px):     ", key = '-PIXEL_SIZE_TEXT-'), sg.In("0.106", key = '-PIXEL_SIZE-', size = (10,1))],
		[sg.T("Acquisition Freq (Hz):", key = '-ACQUISITION_FREQUENCY_TEXT-'), sg.In("50.0", key = '-ACQUISITION_FREQUENCY-', size = (10,1))],
		[sg.T(" ")],
		[sg.Push(),sg.B("CONVERT FILE", key = '-CONVERT-', enable_events = True), sg.Push()],
	]
	
	window = sg.Window('SUPER RES DATA WRANGLER v{}'.format(last_changed), layout)
	popup.close()
	
	# MAIN LOOP
	while True:
		#Read events and values
		event, values = window.read(timeout = 5000)
		
		# Timestamp
		stamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now()) # datestamp
		
		# Variables
		convertfrom = values["-INPUT_COMBO-"]
		convertto = values["-OUTPUT_COMBO-"]
		infilename = values["-SELECTED_FILE-"]
		pix2um = values["-PIXEL_SIZE-"]
		acqfreq = values["-ACQUISITION_FREQUENCY-"]
		
		# Check variables
		check_variables()
		
		# Exit	
		if event in (sg.WIN_CLOSED, 'Exit'):  
			break
		
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
		
		# .txt filetype information
		if event == '.txt Files':
			sg.Popup(
				"HELP - .txt Files",
				" ",
				"PalmTracer .txt files contain 3 headers.",
				" ",
				"Each data line(row) contains 8 tab separated columns:",
				"     [1]Track; (Trajectory#, starting at 1)",
				"     [2]Plane; (Frame#, starting at 1)",
				"     [3]CentroidX (px);",
				"     [4]CentroidY (px);",
				"    *[5]CentroidZ (um);",
				"     [6]Integrated_Intensity;",
				"    *[7]id;",
				"    *[8]Pair_Distance(px);",
				" ",
				"*Column value can be 0.",
				" ", 
				keep_on_top = True,
			)
		
		# .trc filetype information
		if event == '.trc Files':
			sg.Popup(
				"HELP - .trc Files",
				" ",
				"PalmTracer .trc files contain no headers.",
				" ",
				"Each data line(row) contains 6 tab separated columns:",
				"      [1]Track; (Trajectory#, starting at 1)",
				"      [2]Plane; (Frame#, starting at 1)",
				"      [3]CentroidX (px);",
				"      [4]CentroidY (px);",
				"      [5]-1;",
				"     *[6]Integrated_Intensity;",
				" ",
				"*Column value can be -1.",
				" ", 
				keep_on_top = True,
			)
		
		# .trxyt filetype information
		if event == '.trxyt Files':
			sg.Popup(
				"HELP - .trxyt Files",
				" ",
				"NASTIC and segNASTIC .trxyt files contain no headers.",
				" ",
				"Each data line(row) contains 4 space separated columns:",
				"     [1]Track; (Trajectory#, starting at 1)",
				"     [2]CentroidX (um);",
				"     [3]CentroidY (um);",
				"     [4]Frame time (s);",
				" ", 
				keep_on_top = True,
			)
		
		# .ascii (uncorrected and drift corrected) filetype information
		if event == '.ascii Files':
			sg.Popup(
				"HELP - .ascii Files",
				" ",
				"Converting a file to .ascii also generates a .id file containing the trajectory# (ID) of each data line in the .ascii file. Both files are needed to convert back to another filetype.",
				" ",
				".ascii files contain no headers.",
				" ",
				"Each data line(row) contains 9 comma separated columns:",
				"     [1]1;",
				"     [2]Frame#, starting at 0;",
				"     [3]n; (Trajectory# per Frame, starting at 1 each frame)",
				"    *[4]CentroidX (um - uncorrected; nm - drift corrected);",
				"    *[5]CentroidY; (um - uncorrected; nm - drift corrected);",
				"     [6]Integrated_Intensity",
				"     [7]0;",
				"     [8]0;",
				"     [9]0;",
				" ",
				"*IMPORTANT: Centroid values are in microns before drift correction, and in nanometers following drift correction.",
				" ", 
				keep_on_top = True,
			)
		
		# .id filetype information
		if event == '.id Files':
			sg.Popup(
				"HELP - .id Files",
				" ",
				".id files are generated when converting another filetype into the .ascii file format. They contain the Trajectory# information of the corresponding .ascii file. Both files are needed to convert back to another filetype.",
				" ",
				".id files contain no headers.",
				" ",
				"Files contain a single column:",
				"     [1]Trajectory#;",
				" ", 
				keep_on_top = True,
			)
		
		# .csv (TrackMate) filetype information
		if event == '.csv (TrackMate) Files':
			sg.Popup(
				"HELP - .csv (TrackMate) Files",
				" ",
				"TrackMate .csv files contain 4 headers.",
				" ",
				"Each data line(row) contains up to 20 comma separated columns, 4 of which must be the following:",
				"     TRACK_ID;   (Trajectory#)",
				"     X_POSITION;   (um)",
				"     Y_POSITION;   (um)",
				"     FRAME;   (Frame#, starting at 0)",
				" ",
				"Columns can be in any order.",
				" ", 
				keep_on_top = True,
			)
		# Pixel size parameter information
		if event == 'Pixel size':
			sg.Popup(
				"HELP - Pixel size (um/px):",
				" ",
				"Spatial units of X and Y position data depends on filetype:",
				" ",
				"Filetypes in Pixels (px):",
				"     .txt",
				"     .trc",
				"Filetypes in Microns (um):",
				"     .trxyt",
				"     .ascii",
				"     .csv (TrackMate)",
				"Filetypes in Nanometers (nm):",
				"     .ascii (drift corrected)",
				" ",
				"In order to convert between filetypes, the Pixel size (um/px) parameter is used, which is the number of microns per pixel. The default value for Pixel size is 0.106um/px.",
				" ",
				"The default pixel size value can be changed by typing the desired value in the GUI, then clicking 'File>Save settings' in the menu. The original default values can be restored by clicking 'File>Default settings'.",
				" ", 
				keep_on_top = True,
			)
		
		# Acquisition frequency parameter information
		if event == 'Acquisition frequency':
			sg.Popup(
				"HELP - Acquisition frequency (Hz):",
				" ",
				"Temporal units (Frame# or seconds) depends on filetype:",
				" ",
				"Filetypes in Frame#:",
				"     .txt   (from Frame#1)",
				"     .trc   (from Frame#1)",
				"     .ascii   (from Frame#0)",
				"     .ascii (drift corrected)   (from Frame#0)",
				"     .csv (TrackMate)   (from Frame#0)",
				"Filetypes in seconds (s):",
				"     .trxyt",
				" ",
				"The Acquisition frequency (Hz) parameter is used to convert between these filetypes. To find the acquisition frequency (Hz), divide 1 by the frame time (s): e.g., for data where a frame is acquired every 0.02s: 1/0.02 = 50Hz. The default value for Acquisition Frequency is 50Hz.",
				" ",
				"The default acquisition frequency value can be changed by typing the desired value in the GUI, then clicking 'File>Save settings' in the menu. The original default values can be restored by clicking 'File>Default settings'.",
				" ", keep_on_top = True
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
				
		# Check for updates
		if event == 'Updates':
			webbrowser.open("https://github.com/tristanwallis/smlm_clustering/releases",new=2)
		
		# Show/hide browse button corresponding to input file type
		if event == '-INPUT_COMBO-':
			# Show .txt browse button and hide other filetypes
			if convertfrom == ".txt":
				window.Element("-BROWSE_TXT-").update(visible = True)
				window.Element("-BROWSE_TRC-").update(visible = False)
				window.Element("-BROWSE_TRXYT-").update(visible = False)
				window.Element("-BROWSE_ASCII-").update(visible = False)
				window.Element("-BROWSE_CSV-").update(visible = False)
			# Show .trc browse button and hide other filetypes	
			elif convertfrom == ".trc":
				window.Element("-BROWSE_TXT-").update(visible = False)
				window.Element("-BROWSE_TRC-").update(visible = True)
				window.Element("-BROWSE_TRXYT-").update(visible = False)
				window.Element("-BROWSE_ASCII-").update(visible = False)
				window.Element("-BROWSE_CSV-").update(visible = False)
			# Show .trxyt browse button and hide other filetypes
			elif convertfrom == ".trxyt":
				window.Element("-BROWSE_TXT-").update(visible = False)
				window.Element("-BROWSE_TRC-").update(visible = False)
				window.Element("-BROWSE_TRXYT-").update(visible = True)
				window.Element("-BROWSE_ASCII-").update(visible = False)
				window.Element("-BROWSE_CSV-").update(visible = False)
			# Show .ascii browse button and hide other filetypes	
			elif convertfrom == ".ascii" or convertfrom == ".ascii (drift corrected)":
				window.Element("-BROWSE_TXT-").update(visible = False)
				window.Element("-BROWSE_TRC-").update(visible = False)
				window.Element("-BROWSE_TRXYT-").update(visible = False)
				window.Element("-BROWSE_ASCII-").update(visible = True)
				window.Element("-BROWSE_CSV-").update(visible = False)
			# Show .csv browse button and hide other filetypes	
			elif convertfrom == ".csv (TrackMate)":
				window.Element("-BROWSE_TXT-").update(visible = False)
				window.Element("-BROWSE_TRC-").update(visible = False)
				window.Element("-BROWSE_TRXYT-").update(visible = False)
				window.Element("-BROWSE_ASCII-").update(visible = False)
				window.Element("-BROWSE_CSV-").update(visible = True)
				
		# Convert files
		if event == '-CONVERT-':
			# Read .txt file
			if convertfrom == ".txt":
				if infilename.endswith(".txt"):			
					if convertto == ".txt":
						sg.Popup("Input file is already .txt, please select \na different filetype to convert to.", keep_on_top = True)
					else:
						try:
							rawdata = read_txt(infilename)
							if rawdata != None:
								rawdata = sorted(rawdata, key=lambda x:x[0]) # sort rawdata on trajectory number
								# Write .trc file
								if convertto == ".trc":
									try:
										write_trc(rawdata, infilename)
										print("\nDone!\n\n--------------------------------------------------")
										sg.Popup("Done!", keep_on_top = True)
									except:
										print("File could not be written")
										sg.Popup("File could not be written.\n\nPlease make sure the .txt file is in the correct format.\n\nClick 'Info>Help>.txt Files' in the menu for more information on the .txt file format.\n", keep_on_top = True)
								# Write .trxyt file
								elif convertto == ".trxyt":
									try:
										write_trxyt(rawdata, infilename)
										print("\nDone!\n\n--------------------------------------------------")
										sg.Popup("Done!", keep_on_top = True)
									except:
										print("File could not be written")
										sg.Popup("File could not be written.\n\nPlease make sure the .txt file is in the correct format.\n\nClick 'Info>Help>.txt Files' in the menu for more information on the .txt file format.\n", keep_on_top = True)
								# Write .ascii file
								elif convertto == ".ascii":
									try:
										ids = write_ascii(rawdata, infilename)
										write_ids(ids, infilename)
										print("\nDone!\n\n--------------------------------------------------")
										sg.Popup("Done!", keep_on_top = True)
									except:
										print("File could not be written")
										sg.Popup("File could not be written.\n\nPlease make sure the .txt file is in the correct format.\n\nClick 'Info>Help>.txt Files' in the menu for more information on the .txt file format.\n", keep_on_top = True)
								# Write .ascii (drift corrected) file
								elif convertto == ".ascii (drift corrected)":
									try:
										ids = write_dcascii(rawdata, infilename)
										write_ids(ids, infilename)
										print("\nDone!\n\n--------------------------------------------------")
										sg.Popup("Done!", keep_on_top = True)
									except:
										print("File could not be written")
										sg.Popup("File could not be written.\n\nPlease make sure the .txt file is in the correct format.\n\nClick 'Info>Help>.txt Files' in the menu for more information on the .txt file format.\n", keep_on_top = True)
						except:
							print("File could not be read")
							sg.Popup("File could not be read.\n\nPlease make sure the .txt file is in the correct format.\n\nClick 'Info>Help>.txt Files' in the menu for more information on the .txt file format.\n", keep_on_top = True)
				else:
					sg.Popup("Please select a .txt file to convert.", keep_on_top = True)
			# Read .trc file	
			elif convertfrom == ".trc":
				if infilename.endswith(".trc"):
					if convertto == ".trc":
						sg.Popup("Input file is already .trc, please select \na different filetype to convert to.", keep_on_top = True)
					else:
						try:
							rawdata = read_trc(infilename)
							if rawdata != None:
								rawdata = sorted(rawdata, key=lambda x:x[0]) # sort rawdata on trajectory number
								# Write .txt file
								if convertto == ".txt":
									try:
										write_txt(rawdata, infilename)
										print("\nDone!\n\n--------------------------------------------------")
										sg.Popup("Done!", keep_on_top = True)
									except:
										print("File could not be written")
										sg.Popup("File could not be written.\n\nPlease make sure the .trc file is in the correct format.\n\nClick 'Info>Help>.trc Files' in the menu for more information on the .trc file format.\n", keep_on_top = True)
								# Write .trxyt file
								elif convertto == ".trxyt":
									try:
										write_trxyt(rawdata, infilename)
										print("\nDone!\n\n--------------------------------------------------")
										sg.Popup("Done!", keep_on_top = True)
									except:
										print("File could not be written")
										sg.Popup("File could not be written.\n\nPlease make sure the .trc file is in the correct format.\n\nClick 'Info>Help>.trc Files' in the menu for more information on the .trc file format.\n", keep_on_top = True)
								# Write .ascii file
								elif convertto == ".ascii":
									try:
										ids = write_ascii(rawdata, infilename)
										write_ids(ids, infilename)
										print("\nDone!\n\n--------------------------------------------------")
										sg.Popup("Done!", keep_on_top = True)
									except:
										print("File could not be written")
										sg.Popup("File could not be written.\n\nPlease make sure the .trc file is in the correct format.\n\nClick 'Info>Help>.trc Files' in the menu for more information on the .trc file format.\n", keep_on_top = True)
								# Write .ascii (drift corrected) file
								elif convertto == ".ascii (drift corrected)":
									try:
										ids = write_dcascii(rawdata, infilename)
										write_ids(ids, infilename)
										print("\nDone!\n\n--------------------------------------------------")
										sg.Popup("Done!", keep_on_top = True)
									except:
										print("File could not be written")
										sg.Popup("File could not be written.\n\nPlease make sure the .trc file is in the correct format.\n\nClick 'Info>Help>.trc Files' in the menu for more information on the .trc file format.\n", keep_on_top = True)
						except:
							print("File could not be read")
							sg.Popup("File could not be read.\n\nPlease make sure the .trc file is in the correct format.\n\nClick 'Info>Help>.trc Files' in the menu for more information on the .trc file format.\n", keep_on_top = True)
				else:
					sg.Popup("Please select a .trc file to convert.", keep_on_top = True)
			# Read .trxyt file
			elif convertfrom == ".trxyt":
				if infilename.endswith(".trxyt"):
					if convertto == ".trxyt":
						sg.Popup("Input file is already .trxyt, please select \na different filetype to convert to.", keep_on_top = True)
					else:
						try:
							rawdata = read_trxyt(infilename)
							if rawdata != None:
								rawdata = sorted(rawdata, key=lambda x:x[0]) # sort rawdata on trajectory number
								# Write .txt file
								if convertto == ".txt":
									try:
										write_txt(rawdata, infilename)
										print("\nDone!\n\n--------------------------------------------------")
										sg.Popup("Done!", keep_on_top = True)
									except:
										print("File could not be written")
										sg.Popup("File could not be written.\n\nPlease make sure the .trxyt file is in the correct format.\n\nClick 'Info>Help>.trxyt Files' in the menu for more information on the .trxyt file format.\n", keep_on_top = True)
								# Write .trc file
								elif convertto == ".trc":
									try:
										write_trc(rawdata, infilename)
										print("\nDone!\n\n--------------------------------------------------")
										sg.Popup("Done!", keep_on_top = True)
									except:
										print("File could not be written")
										sg.Popup("File could not be written.\n\nPlease make sure the .trxyt file is in the correct format.\n\nClick 'Info>Help>.trxyt Files' in the menu for more information on the .trxyt file format.\n", keep_on_top = True)
								# Write .ascii file
								elif convertto == ".ascii":
									try:
										ids = write_ascii(rawdata, infilename)
										write_ids(ids, infilename)
										print("\nDone!\n\n--------------------------------------------------")
										sg.Popup("Done!", keep_on_top = True)
									except:
										print("File could not be written")
										sg.Popup("File could not be written.\n\nPlease make sure the .trxyt file is in the correct format.\n\nClick 'Info>Help>.trxyt Files' in the menu for more information on the .trxyt file format.\n", keep_on_top = True)
								# Write .ascii (drift corrected) file
								elif convertto == ".ascii (drift corrected)":
									try:
										ids = write_dcascii(rawdata, infilename)
										write_ids(ids, infilename)
										print("\nDone!\n\n--------------------------------------------------")
										sg.Popup("Done!", keep_on_top = True)
									except:
										print("File could not be written")
										sg.Popup("File could not be written.\n\nPlease make sure the .trxyt file is in the correct format.\n\nClick 'Info>Help>.trxyt Files' in the menu for more information on the .trxyt file format.\n", keep_on_top = True)
						except:
							print("File could not be read")
							sg.Popup("File could not be read.\n\nPlease make sure the .trxyt file is in the correct format.\n\nClick 'Info>Help>.trxyt Files' in the menu for more information on the .trxyt file format.\n", keep_on_top = True)
				else:
					sg.Popup("Please select a .trxyt file to convert.", keep_on_top = True)
			# Read .ascii file
			elif convertfrom == ".ascii":
				if infilename.endswith(".ascii"):	
					if convertto == ".ascii":
						sg.Popup("Input file is already .ascii, please select \na different filetype to convert to.", keep_on_top = True)
					else:
						try:
							try:
								ids = read_ids(infilename)
								ids_found = True
							except:
								print("File not found")
								sg.popup("Matching trajectory ID file not found.\n\nPlease make sure ASCII and ID files are in the same location.\n", keep_on_top = True)
								ids_found = False
							if ids_found == True:
								if ids != None:
									try:
										rawdata = read_ascii(infilename, ids)
										if rawdata != None:
											try:
												rawdata = sorted(rawdata, key=lambda x:x[0]) # sort rawdata on trajectory number
											except:
												print("File could not be read")
												sg.Popup("File could not be read.\n\nPlease make sure the .ascii and .id files are in the correct format.\n\nClick 'Info>Help>.ascii Files' and 'Info>Help>.id Files' in the menu for more information on the .ascii and .id file format.\n", keep_on_top = True)
									except:
										print("File could not be read")
										sg.Popup("File could not be read.\n\nPlease make sure the .ascii and .id files are in the correct format.\n\nClick 'Info>Help>.ascii Files' and 'Info>Help>.id Files' in the menu for more information on the .ascii and .id file format.\n", keep_on_top = True)
									if rawdata == None:
										data_read = False
									else:
										data_read = True
									if data_read == True:
										if len(rawdata)>1:
											# Write .txt file
											if convertto == ".txt":
												try:
													write_txt(rawdata, infilename)
													print("\nDone!\n\n--------------------------------------------------")
													sg.Popup("Done!", keep_on_top = True)
												except:
													print("File could not be written")
													sg.Popup("File could not be written.\n\nPlease make sure the .ascii and .id files are in the correct format.\n\nClick 'Info>Help>.ascii Files' and 'Info>Help>.id Files' in the menu for more information on the .ascii and .id file format.\n", keep_on_top = True)
											# Write .trc file
											elif convertto == ".trc":
												try:
													write_trc(rawdata, infilename)
													print("\nDone!\n\n--------------------------------------------------")
													sg.Popup("Done!", keep_on_top = True)
												except:
													print("File could not be written")
													sg.Popup("File could not be written.\n\nPlease make sure the .ascii and .id files are in the correct format.\n\nClick 'Info>Help>.ascii Files' and 'Info>Help>.id Files' in the menu for more information on the .ascii and .id file format.\n", keep_on_top = True)
											# Write .trxyt file
											elif convertto == ".trxyt":
												try:
													write_trxyt(rawdata, infilename)
													print("\nDone!\n\n--------------------------------------------------")
													sg.Popup("Done!", keep_on_top = True)
												except:
													print("File could not be written")
													sg.Popup("File could not be written.\n\nPlease make sure the .ascii and .id files are in the correct format.\n\nClick 'Info>Help>.ascii Files' and 'Info>Help>.id Files' in the menu for more information on the .ascii and .id file format.\n", keep_on_top = True)
											# Write .ascii (drift corrected) file
											elif convertto == ".ascii (drift corrected)":
												try:
													ids = write_dcascii(rawdata, infilename)
													write_ids(ids, infilename)
													print("\nDone!\n\n--------------------------------------------------")
													sg.Popup("Done!", keep_on_top = True)
												except:
													print("File could not be written")
													sg.Popup("File could not be written.\n\nPlease make sure the .ascii and .id files are in the correct format.\n\nClick 'Info>Help>.ascii Files' and 'Info>Help>.id Files' in the menu for more information on the .ascii and .id file format.\n", keep_on_top = True)
						except:
							print("File could not be read")
							sg.Popup("File could not be read.\n\nPlease make sure the .ascii and .id files are in the correct format.\n\nClick 'Info>Help>.ascii Files' and 'Info>Help>.id Files' in the menu for more information on the .ascii and .id file format.\n", keep_on_top = True)
				else:
					sg.Popup("Please select .ascii and .id files to convert.", keep_on_top = True)
			# Read .ascii file (drift corrected)
			elif convertfrom == ".ascii (drift corrected)":
				if infilename.endswith(".ascii"):	
					if convertto == ".ascii (drift corrected)":
						sg.Popup("Input file is already a drift corrected .ascii, please select \na different filetype to convert to.", keep_on_top = True)
					else:
						try:
							try:
								ids = read_ids(infilename)
								ids_found = True
							except:
								print("File not found")
								sg.popup("Matching trajectory ID file not found.\n\nPlease make sure ASCII and ID files are in the same location.\n", keep_on_top = True)
								ids_found = False
							if ids_found == True:
								if ids != None:
									try:
										rawdata = read_dcascii(infilename, ids)
										if rawdata != None:
											try:
												rawdata = sorted(rawdata, key=lambda x:x[0]) # sort rawdata on trajectory number
											except:
												print("File could not be read")
												sg.Popup("File could not be read.\n\nPlease make sure the .ascii and .id files are in the correct format.\n\nClick 'Info>Help>.ascii Files' and 'Info>Help>.id Files' in the menu for more information on the .ascii and .id file format.\n", keep_on_top = True)
									except: 
										print("File could not be read")
										sg.Popup("File could not be read.\n\nPlease make sure the .ascii and .id files are in the correct format.\n\nClick 'Info>Help>.ascii Files' and 'Info>Help>.id Files' in the menu for more information on the .ascii and .id file format.\n", keep_on_top = True)
									if rawdata == None:
										data_read = False
									else:
										data_read = True
									if data_read == True:
										if len(rawdata)>1:
											# Write .txt file
											if convertto == ".txt":
												try:
													write_txt(rawdata, infilename)
													print("\nDone!\n\n--------------------------------------------------")
													sg.Popup("Done!", keep_on_top = True)
												except:
													print("File could not be written")
													sg.Popup("File could not be written.\nPlease make sure the drift corrected .ascii and .id files are in the correct format.\n\nClick 'Info>Help>.ascii Files' and 'Info>Help>.id Files' in the menu for more information on the drift corrected .ascii and .id file format.\n", keep_on_top = True)
											# Write .trc file
											elif convertto == ".trc":
												try:
													write_trc(rawdata, infilename)
													print("\nDone!\n\n--------------------------------------------------")
													sg.Popup("Done!", keep_on_top = True)
												except:
													print("File could not be written")
													sg.Popup("File could not be written.\n\nPlease make sure the drift corrected .ascii and .id files are in the correct format.\n\nClick 'Info>Help>.ascii Files' and 'Info>Help>.id Files' in the menu for more information on the drift corrected .ascii and .id file format.\n", keep_on_top = True)
											# Write .trxyt file
											elif convertto == ".trxyt":
												try:
													write_trxyt(rawdata, infilename)
													print("\nDone!\n\n--------------------------------------------------")
													sg.Popup("Done!", keep_on_top = True)
												except:
													print("File could not be written")
													sg.Popup("File could not be written.\n\nPlease make sure the drift corrected .ascii and .id files are in the correct format.\n\nClick 'Info>Help>.ascii Files' and 'Info>Help>.id Files' in the menu for more information on the drift corrected .ascii and .id file format.\n", keep_on_top = True)
											# Write .ascii file
											elif convertto == ".ascii":
												try:
													ids = write_ascii(rawdata, infilename)
													write_ids(ids, infilename)
													print("\nDone!\n\n--------------------------------------------------")
													sg.Popup("Done!", keep_on_top = True)
												except:
													print("File could not be written")
													sg.Popup("File could not be written.\n\nPlease make sure the drift corrected .ascii and .id files are in the correct format.\n\nClick 'Info>Help>.ascii Files' and 'Info>Help>.id Files' in the menu for more information on the drift corrected .ascii and .id file format.\n", keep_on_top = True)
						except:
							print("File could not be read")
							sg.Popup("File could not be read.\n\nPlease make sure the drift corrected .ascii and .id files are in the correct format.\n\nClick 'Info>Help>.ascii Files' and 'Info>Help>.id Files' in the menu for more information on the drift corrected .ascii and .id file format.\n", keep_on_top = True)
				else:
					sg.Popup("Please select drift corrected .ascii and .id files to convert.", keep_on_top = True)
			# Read .csv (TrackMate) file
			elif convertfrom == ".csv (TrackMate)":
				if infilename.endswith(".csv"):
					try:
						rawdata = read_csv(infilename)
						if rawdata != None:
							rawdata = sorted(rawdata, key=lambda x:x[1]) # sort rawdata on frame number first
							rawdata = sorted(rawdata, key=lambda x:x[0]) # sort rawdata on trajectory number second
							# Write .txt file
							if convertto == ".txt":
								try:
									write_txt(rawdata, infilename)
									print("\nDone!\n\n--------------------------------------------------")
									sg.Popup("Done!", keep_on_top = True)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the TrackMate .csv file is in the correct format.\n\nClick 'Info>Help>.csv (TrackMate) Files' in the menu for more information on the TrackMate .csv file format.\n", keep_on_top = True)
							# Write .trc file
							elif convertto == ".trc":
								try:
									write_trc(rawdata, infilename)
									print("\nDone!\n\n--------------------------------------------------")
									sg.Popup("Done!", keep_on_top = True)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the TrackMate .csv file is in the correct format.\n\nClick 'Info>Help>.csv (TrackMate) Files' in the menu for more information on the TrackMate .csv file format.\n", keep_on_top = True)
							# Write .trxyt file
							elif convertto == ".trxyt":
								try:
									write_trxyt(rawdata, infilename)
									print("\nDone!\n\n--------------------------------------------------")
									sg.Popup("Done!", keep_on_top = True)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the TrackMate .csv file is in the correct format.\n\nClick 'Info>Help>.csv (TrackMate) Files' in the menu for more information on the TrackMate .csv file format.\n", keep_on_top = True)
							# Write .ascii file
							elif convertto == ".ascii":
								try:
									ids = write_ascii(rawdata, infilename)
									write_ids(ids, infilename)
									print("\nDone!\n\n--------------------------------------------------")
									sg.Popup("Done!", keep_on_top = True)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the TrackMate .csv file is in the correct format.\n\nClick 'Info>Help>.csv (TrackMate) Files' in the menu for more information on the TrackMate .csv file format.\n", keep_on_top = True)
							# Write .ascii (drift corrected) file
							elif convertto == ".ascii (drift corrected)":
								try:
									ids = write_dcascii(rawdata, infilename)
									write_ids(ids, infilename)
									print("\nDone!\n\n--------------------------------------------------")
									sg.Popup("Done!", keep_on_top = True)
								except:
									print("File could not be written")
									sg.Popup("File could not be written.\n\nPlease make sure the TrackMate .csv file is in the correct format.\n\nClick 'Info>Help>.csv (TrackMate) Files' in the menu for more information on the TrackMate .csv file format.\n", keep_on_top = True)
					except:
						print("File could not be read")
						sg.Popup("File could not be read.\n\nPlease make sure the TrackMate .csv file is in the correct format.\n\nClick 'Info>Help>.csv (TrackMate) Files' in the menu for more information on the TrackMate .csv file format.\n", keep_on_top = True)
			else:
				pass
				
		# Update buttons
		if event: 
			update_buttons()
			
	print("\nExiting...")						
	window.close()
	quit()
		