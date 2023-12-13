# -*- coding: utf-8 -*-
'''
SUPER_RES_DATA_WRANGLER_CLI
COMMAND LINE (CLI) VERSION FOR THE CONVERSION OF TRAJECTORY FILES
CONVERT BETWEEN MUTLIPLE FILE FORMATS FROM THE PALMTRACER (METAMORPH PLUGIN)/TRACKMATE (IMAGEJ PLUGIN)--> SHARP VISU --> NASTIC/segNASTIC/BOOSH --> NASTIC WRANGLER SUPER RESOLUTION DATA PROCESSING PIPELINE

Design and coding: Tristan Wallis and Alex McCann
Queensland Brain Institute
The University of Queensland
Fred Meunier: f.meunier@uq.edu.au

REQUIRED:
Python 3.8 or greater

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
1 - Copy this script to the top level directory containing the files you are interested in, and run it (either by double clicking or navigating to the location of the script in the terminal and using the command 'python super_res_data_wrangler_cli.py')
2 - Specify the file type you want to convert FROM
3 - Specify the file type you want to convert TO
4 - txt and trc files: specify the Pixel size in um/px
5 - trxyt files: specify the Acquisition frequency in Hz
6 - A list of files found in the current directory and all subdirectories will be generated
7 - Select the files you want to convert (all files = 'a', specific files = desired file numbers separated by a comma), and hit return
8 - Files will be converted and saved to the same place as the original file, with the appropriate suffix and a date stamp

CHECK FOR UPDATES:
https://github.com/tristanwallis/smlm_clustering/releases
'''

lastchanged = "20231212"

# LOAD MODULES (Functions)
from functools import reduce
import os
import glob
import datetime

# VARS
acqfreq_default = 50.0 #Hz
pix2um_default = 0.106 #microns per pixel

# FUNCTIONS

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
	print ("\n\nReading {}...".format(infilename))
	rawdata = []
	ct = 0
	with open(infilename,"r", encoding="utf8") as infile:
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
		print("ALERT: 0 lines read.\nPlease make sure the file is not empty.\n")

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
	print ("\n\nReading {}...".format(infilename))
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
		print("ALERT: 0 lines read.\nPlease make sure the file is not empty.\n")
		
# Read NASTIC/segNASTIC/BOOSH trxyt	
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
	print ("\n\nReading {}...".format(infilename))	
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
		print("ALERT: 0 lines read.\nPlease make sure the file is not empty.\n")
		
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
		print("ALERT: 0 lines read.\nPlease make sure the file is not empty.\n")
		
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
					print ("1.00000,0,1, 209.045592443744, 142.834293217347, 50870.0478515625,0.00000,0.00000,0.00000")
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
		print("ALERT: 0 lines read.\nPlease make sure the file is not empty.\n")
		
# Read idfile (trajectory IDs for ascii)
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
	print ("\n\nReading {}...".format(infilename))
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
		print("ALERT: 0 lines read.\nPlease make sure the file is not empty.\n")
		
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
	print ("\n\nReading {}...".format(infilename))
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
				pass
	if len(rawdata) != 0:
		print ("{} lines read".format(ct-3))
		return rawdata		
	elif len(rawdata) == 0:
		print("ALERT: 0 lines read.\nPlease make sure the file is not empty.\n")

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
	print ("{} lines written\n".format(ct))	
	
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
	print ("{} lines written\n".format(ct))	
	
# Write NASTIC/segNASTIC/BOOSH trxyt
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
	print ("{} lines written\n".format(ct))			

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
	print ("{} lines written\n".format(ct))	
	
######################################################

# RUN IT

# Initial directory
cwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cwd)
initialdir = cwd

# User input
os.system('cls' if os.name == 'nt' else 'clear')
try:
	while {True}:
		print ("SUPER RES DATA WRANGLER CLI - Tristan Wallis {}\n-----------------------------------------------------".format(lastchanged))
		print ("Ctrl-c to quit at any time\n")
		files = []
		while files ==[]:
			# Format to convert from
			convertfrom = input ("Select a file type to convert from:\t[a]scii [d]rift corrected ascii [t]xt t[r]c tr[x]yt [c]sv (TrackMate)\n")
			if convertfrom in ["a","t","x","r","d","c"]:
				fromsuffix = {"a":"ascii","t":"txt","r":"trc","x":"trxyt","d":"ascii","c":"csv"}[convertfrom]
				print("\n")
				# Format to convert to
				convertto = None
				while convertto == None:
					convertto = input ("Select a file type to convert to:\t[a]scii [d]rift corrected ascii [t]xt t[r]c tr[x]yt\n")
					if convertto in ["a","t","x","r","d"]:
						tosuffix = {"a":"ascii","t":"txt","r":"trc","x":"trxyt","d":"ascii"}[convertto]
						# Check if input and output file types are the same 
						if convertfrom == convertto:
							if convertfrom == "t":	
								print("ALERT: Selected input and output file types are both .txt, please select different input and output file types.\n\n\n")
							elif convertfrom == "r":
								print("ALERT: Selected input and output file types are both .trc, please select different input and output file types.\n\n\n")
							elif convertfrom == "x":
								print("ALERT: Selected input and output file types are both .trxyt, please select different input and output file types.\n\n\n")
							elif convertfrom == "a":
								print("ALERT: Selected input and output file types are both .ascii, please select different input and output file types.\n\n\n")
							elif convertfrom == "d":
								print("ALERT: Input and output files are both .ascii (drift corrected), please select different input and output file types.\n\n\n")
						else:
							print("\n")
							# Pixel size of txt or trc file for conversion to/from px<->um
							if convertfrom == "t" or convertfrom == "r":
								pixel_size = None
								while pixel_size == None:
									pixel_size_input = input ("Enter pixel size of {} file in microns/pixel (default = {}um/px):".format(fromsuffix,pix2um_default))
									if pixel_size_input == "0":
										print("ALERT: Please enter a number greater than 0\n")
									elif pixel_size_input != "0":
										try:
											pixel_size = float(pixel_size_input)
											if pixel_size <= 0:
												print("ALERT: Please enter a number greater than 0\n")
												pixel_size = None
										except ValueError:
											print("ALERT: '{input}' is not a number, please enter a number greater than 0\n".format(input = pixel_size_input))
								pix2um = pixel_size
								print("\n")
							elif convertto == "t" or convertto == "r":
								pixel_size = None
								while pixel_size == None:
									pixel_size_input = input ("Enter pixel size of {} file in microns/pixel (default = {}um/px):".format(tosuffix,pix2um_default))
									if pixel_size_input == "0":
										print("ALERT: Please enter a number greater than 0\n")
									elif pixel_size_input != "0":
										try:
											pixel_size = float(pixel_size_input)
											if pixel_size <=0:
												print("ALERT: Please enter a number greater than 0\n")
												pixel_size = None
										except ValueError:
											print("ALERT: '{input}' is not a number, please enter a number greater than 0\n".format(input = pixel_size_input))
									pix2um = pixel_size
									print("\n")
							# Acquisition frequency of trxyt file for conversion to/from Frame#<->seconds
							if convertfrom == "x" or convertto == "x":
								acquisition_frequency = None
								while acquisition_frequency == None:
									acquisition_frequency_input = input("Enter acquistion frequency of trxyt file in Hz (default = {}Hz):".format(acqfreq_default))
									if acquisition_frequency_input == "0":
										print("ALERT: Please enter a number greater than 0\n")
									elif acquisition_frequency_input != "0":
										try:
											acquisition_frequency = float(acquisition_frequency_input)
											if acquisition_frequency <=0:
												print("ALERT: Please enter a number greater than 0\n")
												acquisition_frequency = None
										except ValueError:
											print("ALERT: '{input}' is not a number, please enter a number greater than 0\n".format(input=acquisition_frequency_input))
								acqfreq = acquisition_frequency
								print("\n")
							# Recursively find all files
							files = glob.glob(cwd + '/**/*.{}'.format(fromsuffix), recursive=True)
							if len(files) == 0:
								print("ALERT\nNo {} files were found in this directory.\nPlease check that the files you wish to convert from have the .{} extension.\nPlease check that the Super Res Data Wrangler script is placed in the top level directory containing the {} files that you wish to convert.\n".format(fromsuffix,fromsuffix,fromsuffix))
							elif len(files) == 1:
								print ("1 {} file found in this directory:".format(fromsuffix))
								for n, file in enumerate(files, start=1):
									print ("\t[{}] {}".format(n,file))
							else:
								print("{} {} files found in this directory:".format(len(files),fromsuffix))
								for n, file in enumerate(files, start=1):
									print("\t[{}] {}".format(n,file))
					else:
						print("ALERT: '{input}' is not an accepted input, please type a letter ('a' 'd' 't' 'r' or 'x') to select a valid output file type\n\n".format(input = convertto)) 
						convertto = None
			else:
				print("ALERT: '{input}' is not an accepted input, please type a letter ('a' 'd' 't' 'r' 'x' or 'c') to select a valid input file type\n\n".format(input = convertfrom)) 
		# Select files
		select_files = None
		while select_files == None:
			select_files_input = input("\nSelect files (comma separated, a = select all):")
			if select_files_input == "a":
				select_files = select_files_input.replace(" ","")
				infilenames = files				
			elif select_files_input != "a":
				if select_files_input == "":
					print("ALERT: '{input}' is not an accepted input, please enter comma-separated numbers or 'a'\n".format(input = select_files_input))
				else:
					try:
						select_files = select_files_input.replace(" ","")
						filenums = select_files.split(",")
						if "0" in filenums:
							filenums.remove("0")
						if len(filenums) == 0:
							print("ALERT: File number is not in list, please enter a file number from the list or 'a'\n")
							select_files = None
						elif len(filenums) != 0:
							filenums = [int(x) -1 for x in filenums]
							infilenames = [files[x] for x in filenums]
					except ValueError:
						print("ALERT: '{input}' is not an accepted input, please enter comma-separated numbers or 'a'\n".format(input = select_files_input))
						select_files = None
					except:
						print("ALERT: File number is not in list, please enter a file number from the list or 'a'\n")
						select_files = None
		for infilename in infilenames:
			rawdata = []	
			# Read data
			if convertfrom == "t":
				try:
					rawdata = read_txt(infilename)
				except:
					print("ALERT: File could not be read.\nPlease make sure the file is in the correct format.\n\n")
					pass
			if convertfrom == "r":
				try:
					rawdata = read_trc(infilename)	
				except:
					print("ALERT: File could not be read.\nPlease make sure the file is in the correct format.\n\n")
			if convertfrom == "x":
				try:
					rawdata = read_trxyt(infilename)		
				except:
					print("ALERT: File could not be read.\nPlease make sure the file is in the correct format.\n\n")
			if convertfrom == "a":
				try:
					ids = read_ids(infilename)
					ids_found = True
				except:
					print("ALERT: Matching trajectory ID file not found.\n\n")
					ids_found = False
				if ids_found == True:
					if ids != None:
						try:
							rawdata = read_ascii(infilename,ids)
						except:
							print("ALERT: File could not be read.\nPlease make sure the file is in the correct format.\n\n")
			if convertfrom == "d":
				try:
					ids = read_ids(infilename)
					ids_found = True
				except:
					print("ALERT: Matching trajectory ID file not found.\n\n")
					ids_found = False
				if ids_found == True:
					if ids != None:
						try:
							rawdata = read_dcascii(infilename,ids)
						except:
							print("ALERT: File could not be read.\nPlease make sure the file is in the correct format.\n\n")
			if convertfrom == "c":
				try:
					rawdata = read_csv(infilename)
					rawdata = sorted(rawdata, key=lambda x:x[1]) # sort csv rawdata initially on frame number
				except:
					print("ALERT: File could not be read.\nPlease make sure the file is in the correct format.\n\n")
			# Write data
			data_written = False
			if rawdata == None:
				data_read = False
			else:
				data_read = True
			if data_read == True:
				if len(rawdata) > 1:	
					rawdata = sorted(rawdata, key=lambda x:x[0]) # sort rawdata on trajectory number
					stamp = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now()) # datestamp
					if convertto == "t":
						write_txt(rawdata,infilename)
						data_written = True
					if convertto == "r":
						write_trc(rawdata,infilename)
						data_written = True
					if convertto == "x":
						write_trxyt(rawdata,infilename)
						data_written = True
					if convertto == "a":
						ids = write_ascii(rawdata,infilename)
						write_ids(ids,infilename)
						data_written = True
					if convertto == "d":
						ids = write_dcascii(rawdata,infilename)
						write_ids(ids,infilename)	
						data_written = True				
		if data_written == True:
			print("\n\nDone!\n\n\n")
		if data_written == False:
			print("\n\n")
except KeyboardInterrupt:
	exit()