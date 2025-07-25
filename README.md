# SMLM Clustering
Contains the NASTIC family of scripts - Python based Graphical User Interface (GUI) and command line (CLI) scripts for molecular trajectory clustering analysis.

```
BRIEF OVERVIEW OF STEPS INVOLVING SCRIPTS:

Step 1 - select clustering program to use
    a. use trajectory bounding boxes (NASTIC)
    b. use trajectory segment bounding boxes (SEGNASTIC)
    c. switch between NASTIC and SEGNASTIC (2in1 NASTIC/SEGNASTIC)
    d. use 3D DBSCAN on trajectory detections (BOOSH)
    e. use DBSCAN on trajectory centroids (DBSCAN) (intended for comparison purposes only)
    f. use VORONOI tesselation on trajectory centroids (VORONOI) (intended for comparison purposes only)

Step 2 - select which version of the program to use
    a. GUI.py - multithreaded Graphical User Interface version for use on physical hardware
    b. ST_GUI.py - single-threaded (compilable) Graphical User Interface version
    c. CLI.py - command line single-threaded version
    d. 2C_GUI.py - (where applicable) 2 colored version of the GUI.py version
    e. 2C_ST_GUI.py - (where applicable) 2 colored version of the ST_GUI.py version 

Step 3 - generate trajectory files in the .trxyt file format used as input by NASTIC, SEGNASTIC, BOOSH, DBSCAN and VORONOI scripts
    a. Convert tracked trajectory files to the .trxyt file format using the Super Res Data Wrangler GUI or CLI
    b. Generate synthetic trajectory files using the Synthetic Data Generator GUI or CLI (intended for comparison purposes only)
    c. Use the provided example synthetic trajectory files previously generated using the Synthetic Data Generator (intended for comparison purposes only)  

Step 4 - run the clustering program of choice (from Step 1 and 2)
    a. Option 1: Double click on the script
    b. Option 2: Open a command line terminal, navigate to the location of the script using the terminal, type python followed by the script name (e.g., python nastic_gui.py) followed by the return key

Step 5 - perform meta analysis of clustering data (output from Step 4) using the data wrangler GUIs
    a. NASTIC_WRANGLER_GUI.py - use for data produced by NASTIC, SEGNASTIC and BOOSH scripts
    b. DBSCAN_WRANGLER_GUI.py - use for data produced by DBSCAN and VORONOI scripts (intended for comparison purposes only)

```

# NASTIC (Nanoscale spatiotemporal indexing clustering)

## Overview:
Uses R-tree based spatio-termporal indexing of TRAJECTORY BOUNDING BOXES to cluster trajectories. 

Wallis TP, Jiang A, Young K, Hou H, Kudo K, McCann AJ, Durisic N, Joensuu M, Oelz D, Nguyen H, Gormal RS, Meunier FA. Super-resolved trajectory-derived nanoclustering analysis using spatiotemporal indexing. Nat Commun. 2023 Jun 8;14(1):3353. doi: 10.1038/s41467-023-38866-y.

See nastic_user_manual.pdf for detailed instructions for nastic_gui.py and derivatives.

## Input: 
* .trxyt (file(s) containing trajectory information)

* (Optional) PalmTracer .rgn file containing ROI coordinates

* (Optional) NASTIC roi_coordinates.tsv file containing ROI coordinates

* (Optional) FIJI XY_Coordinates.csv file containing ROI coordinates

## Output:
* metrics.tsv (file containing cluster metrics)

* raw_acquisition.png (plot showing raw detections with selected ROI overlayed) (may also be exported as .eps, .pdf, .ps or .svg)

* main_plot.png (plot showing clustered data inside selected ROI) (may also be exported as .eps, .pdf, .ps or .svg)

* (Optional) .png (any additional plots that were generated and saved) (may also be exported as .eps, .pdf, .ps or .svg)

* roi_coordinates.tsv (file containing coordinates of the ROI that was selected)

## Scripts:
* nastic_gui.py: multithreaded GUI for physical hardware

* 2in1_nastic_segnastic_gui.py: 2-in-1 version (contains nastic and segnastic) of nastic_gui.py

* nastic_st_gui.py: single-threaded (compilable) version of nastic_gui.py 

* 2in1_nastic_segnastic_st_gui.py: single-threaded (compilable) version of 2in1_nastic_segnastic_gui.py

* nastic_cli.py: command line single-threaded version of nastic

* nastic2c_gui.py: multithreaded GUI for physical hardware - two colour version of nastic

* nastic2c_st_gui.py: single-threaded (compilable) version of nastic2c_gui.py

## Notes: 
* Trajectory files in formats other than the .trxyt file format can be converted to the .trxyt file format using the Super Res Data Wrangler (GUI or CLI) for use in NASTIC

* Alternatively, synthetic trajectory .trxyt files can be generated using the Synthetic Data Generator (GUI or CLI) and used as input in NASTIC (intended for comparison purposes only)

* Metrics.tsv files produced by NASTIC can be subsequently analysed using the NASTIC Wrangler GUI


# SEGNASTIC (segment NASTIC)

## Overview:
Uses R-tree based spatio-temporal indexing of TRAJECTORY SEGMENT BOUNDING BOXES to cluster trajectories. 

Wallis TP, Jiang A, Young K, Hou H, Kudo K, McCann AJ, Durisic N, Joensuu M, Oelz D, Nguyen H, Gormal RS, Meunier FA. Super-resolved trajectory-derived nanoclustering analysis using spatiotemporal indexing. Nat Commun. 2023 Jun 8;14(1):3353. doi: 10.1038/s41467-023-38866-y.

See nastic_user_manual.pdf for detailed instructions for nastic_gui.py and derivatives.

## Input:
* .trxyt (file(s) containing trajectory information)

* (Optional) PalmTracer .rgn file containing ROI coordinates

* (Optional) NASTIC roi_coordinates.tsv file containing ROI coordinates

* (Optional) FIJI XY_Coordinates.csv file containing ROI coordinates

## Output:
* metrics.tsv (file containing cluster metrics)

* raw_acquisition.png (plot showing raw detections with selected ROI overlayed) (may also be exported as .eps, .pdf, .ps or .svg)

* main_plot.png (plot showing clustered data inside selected ROI) (may also be exported as .eps, .pdf, .ps or .svg)

* (Optional) .png (any additional plots that were generated and saved) (may also be exported as .eps, .pdf, .ps or .svg)

* roi_coordinates.tsv (file containing coordinates of the ROI that was selected)

## Scripts:
* segnastic_gui.py: multithreaded GUI for physical hardware
 
* 2in1_nastic_segnastic_gui.py: 2-in-1 version (contains nastic and segnastic) of segnastic_gui.py

* segnastic_st_gui.py: single-threaded (compilable) version of segnastic_gui.py  

* 2in1_nastic_segnastic_st_gui.py: single-threaded (compilable) version of 2in1_nastic_segnastic_gui.py

* segnastic_cli.py: command line single-threaded version of segnastic

* segnastic2c_gui.py: multithreaded GUI for physical hardware - two colour version of segnastic

* segnastic2c_st_gui.py: single-threaded (compilable) version of segnastic2c_gui.py  

## Notes: 
* Trajectory files in formats other than the .trxyt file format can be converted to the .trxyt file format using the Super Res Data Wrangler (GUI or CLI) for use in SEGNASTIC

* Alternatively, synthetic trajectory .trxyt files can be generated using the Synthetic Data Generator (GUI or CLI) and used as input in SEGNASTIC (intended for comparison purposes only)

* Metrics.tsv files produced by SEGNASTIC can be subsequently analysed using the NASTIC Wrangler GUI

# BOOSH

## Overview:
Uses 3D DBSCAN to cluster TRAJECTORY DETECTIONS (experimental).

See nastic_user_manual.pdf for detailed instructions for nastic_gui.py and derivatives.

## Input:
* .trxyt (file(s) containing trajectory information)

* (Optional) PalmTracer .rgn file containing ROI coordinates

* (Optional) NASTIC roi_coordinates.tsv file containing ROI coordinates

* (Optional) FIJI XY_Coordinates.csv file containing ROI coordinates

## Output:
* metrics.tsv (file containing cluster metrics)

* raw_acquisition.png (plot showing raw detections with selected ROI overlayed) (may also be exported as .eps, .pdf, .ps or .svg)

* main_plot.png (plot showing clustered data inside selected ROI) (may also be exported as .eps, .pdf, .ps or .svg)

* (Optional) .png (any additional plots that were generated and saved) (may also be exported as .eps, .pdf, .ps or .svg)

* roi_coordinates.tsv (file containing coordinates of the ROI that was selected)

## Scripts:
* boosh_gui.py: multithreaded GUI for physical hardware (experimental)

* boosh_st_gui.py: single-threaded (compilable) version of boosh_gui.py (experimental)

* boosh_cli.py: command line single-threaded version of boosh (experimental)

* boosh2c_gui.py: multithreaded GUI for physical hardware - two colour version of boosh (experimental)

* boosh2c_st_gui.py: single-threaded (compilable) version of boosh2c_gui.py (experimental)

## Notes: 
* Trajectory files in formats other than the .trxyt file format can be converted to the .trxyt file format using the Super Res Data Wrangler (GUI or CLI) for use in BOOSH

* Alternatively, synthetic trajectory .trxyt files can be generated using the Synthetic Data Generator GUI and used as input in BOOSH (intended for comparison purposes only)

* Metrics.tsv files produced by BOOSH can be subsequently analysed using the NASTIC Wrangler GUI

# DBSCAN

## Overview:
Uses DBSCAN to cluster TRAJECTORY CENTROIDS (intended for comparison purposes only).

See nastic_user_manual.pdf for detailed instructions for nastic_gui.py and derivatives.

## Input:
* .trxyt (file(s) containing trajectory information)

* (Optional) NASTIC roi_coordinates.tsv file containing ROI coordinates

## Output: 
* metrics.tsv (file containing cluster metrics)

* raw_acquisition.png (plot showing raw detections with selected ROI overlayed)

* main_plot.png (plot showing clustered data inside selected ROI)

* (Optional) .png (any additional plots that were generated and saved)

* roi_coordinates.tsv (file containing coordinates of the ROI that was selected)

## Scripts:
* dbscan_gui.py: Multithreaded GUI for physical hardware (for comparison purposes)

## Notes:
* Metrics.tsv files produced by DBSCAN can be subsequently analysed using the DBSCAN Wrangler GUI

# VORONOI

## Overview:
Uses VORONOI TESSELATION to cluster TRAJECTORY CENTROIDS (intended for comparison purposes only).

See nastic_user_manual.pdf for detailed instructions for nastic_gui.py and derivatives.

## Input:
* .trxyt (file(s) containing trajectory information)

* (Optional) NASTIC roi_coordinates.tsv file containing ROI coordinates

## Output:
* metrics.tsv (file containing cluster metrics)

* raw_acquisition.png (plot showing raw detections with selected ROI overlayed)

* main_plot.png (plot showing clustered data inside selected ROI)

* (Optional) .png (any additional plots that were generated and saved)

* roi_coordinates.tsv (file containing coordinates of the ROI that was selected)

## Scripts:
* voronoi_gui.py: cluster using Voronoi tesselation of trajectory centroids (for comparison purposes)

## Notes:
* Metrics.tsv files produced by VORONOI can be subsequently analysed using the DBSCAN Wrangler GUI

# NASTIC WRANGLER

## Overview:
Meta analysis of metrics.tsv file outputs from NASTIC, SEGNASTIC and BOOSH scripts.

See nastic_wrangler_user_manual.pdf for detailed instructions for nastic_wrangler_gui.py.

## Input:
* metrics.tsv (files containing cluster metrics produced by NASTIC, SEGNASTIC or BOOSH)

## Output:
* processed_metrics.tsv

* (Optional) pca.png

* (Optional) pca_labels.png

* (Optional) aggregate_plots/cluster_metric_name.png (1 plot for each cluster metric)

* (Optional) average_plots/cluster_metric_name.png (1 plot for each cluster metric)

## Scripts:
* nastic_wrangler_gui.py

# DBSCAN WRANGLER

## Overview:
Meta analysis of output from dbscan_gui.py and voronoi_gui.py.

## Input:
* metrics.tsv (file containing cluster metrics produced by DBSCAN or VORONOI)

## Output:
* processed_metrics.tsv

* aggregate_plots.png

* average_plots.png

* PCA_plot.png

## Scripts:
* dbscan_wrangler_gui.py

# SUPER RES DATA WRANGLER

## Overview:
Convert between various trajectory filetypes for further analysis.

See super_res_data_wrangler_gui_user_manual.pdf and super_res_data_wrangler_cli_user_manual.pdf for detailed instructions.

## Input:
* file/folder containing files with trajectory information
    
    a. (Optional) PalmTracer .txt

    b. (Optional) PalmTracer .trc

    c. (Optional) NASTIC/SEGNASTIC/BOOSH .trxyt

    d. (Optional) SharpViSu .ascii and .id (before drift correction)

    e. (Optional) SharpViSu .ascii and .id (drift corrected)

    f. (Optional) TrackMate .csv 

    g. (Optional) other file type (manually input trajectory file parameters) (GUI only)

## Output:

* file(s) containing trajectory information 

    a. (Optional) PalmTracer .txt

    b. (Optional) PalmTracer .trc

    c. (Optional) NASTIC/SEGNASTIC/BOOSH .trxyt

    d. (Optional) SharpViSu .ascii and .id (before drift correction)

    e. (Optional) SharpViSu .ascii and .id (drift corrected)

## Scripts:
* super_res_data_wrangler_gui.py: multithreaded GUI for physical hardware

* super_res_data_wrangler_cli.py: command line version of super res data wrangler

# SYNTHETIC DATA GENERATOR

## Overview:
Generates synthetic .trxyt files (for use in NASTIC, SEGNASTIC and BOOSH) using parameters input by the user. 

Intended for comparison purposes only.

See synthetic_data_generator_gui_manual.pdf and synthetic_data_generator_cli_manual.pdf for detailed instructions.

## Input:
* (Optional) roi_coordinates.tsv (pre-saved NASTIC ROI file that can be used as the selection area to generate trajectories within)

## Output:
* .trxyt (file(s) containing synthetic trajectory information)
* metrics.tsv (file containing selected parameters and metrics that were generated from them)
* (Optional) .png (plots showing trajectories that were generated within the ROI, with the ROI shown in green)

## Scripts:
* synthetic_data_generator_gui.py: multithreaded GUI for physical hardware - generate synthetic data using multiple parameters

* synthetic_data_generator_cli.py: command line version of synthetic data generator - generate synthetic data using hardcoded default values

# File list
## Multithreaded GUI for physical hardware
* nastic_gui.py: cluster using R-tree based spatio-temporal indexing of trajectory bounding boxes

* nastic2c_gui.py: two colour version of nastic

* segnastic_gui.py: cluster using R-tree based spatio-temporal indexing of trajectory segment bounding boxes

* segnastic2c_gui.py: two colour version of segnastic

* 2in1_nastic_segnastic_gui.py: 2-in-1 version of nastic and segnastic

* boosh_gui.py: cluster using 3D DBSCAN of trajectory detections (experimental)

* boosh2c_gui.py: two colour version of boosh (experimental)

* dbscan_gui.py: cluster using DBSCAN of trajectory centroids (intended for comparison purposes only)

* voronoi_gui.py: cluster using Voronoi tesselation of trajectory centroids (intended for comparison purposes only)

## Single-threaded GUI for virtual computers
* nastic_st_gui.py: single-threaded (compilable) version of nastic_gui.py  

* nastic2c_st_gui.py: single-threaded (compilable) version of nastic2c_gui.py  

* segnastic_st_gui.py: single-threaded (compilable) version of segnastic_gui.py  

* segnastic2c_st_gui.py: single-threaded (compilable) version of segnastic2c_gui.py  

* 2in1_nastic_segnastic_st_gui.py: single-threaded (compilable) version of 2in1_nastic_segnastic_gui.py

* boosh_st_gui.py: single-threaded (compilable) version of boosh_gui.py (experimental)

* boosh2c_st_gui.py: single-threaded (compilable) version of boosh2c_gui.py (experimental)

## Command line
* nastic_cli.py: command line single-threaded version of nastic

* segnastic_cli.py: command line single-threaded version of segnastic

* boosh_cli.py: command line single-threaded version of boosh (experimental)

## Miscellaneous utilities
* nastic_wrangler_gui.py: meta analysis of output from nastic_gui.py, segnastic_gui.py and boosh_gui.py

* super_res_data_wrangler_gui.py: convert between super resolution trajectory filetypes

* super_res_data_wrangler_cli.py: command line version of super_res_data_wrangler_gui.py

* synthetic_data_generator_gui.py: generate synthetic trajectory data using multiple parameters

* synthetic_data_generator_cli.py: command line version of synthetic_data_generator_gui.py

* dbscan_wrangler_gui.py: meta analysis of output from dbscan_gui.py and voronoi_gui.py

## User manuals
* nastic_user_manual.pdf: detailed instructions for nastic_gui.py and derivatives

* nastic_wrangler_user_manual.pdf: detailed instructions for nastic_wrangler_gui.py

* super_res_data_wrangler_gui_user_manual.pdf: detailed instructions for the graphical user interface (GUI) version of the super res data wrangler 

* super_res_data_wrangler_cli_user_manual.pdf: detailed instructions for the command line (cli) version of the super res data wrangler

* synthetic_data_generator_gui_user_manual.pdf: detailed instructions for the graphical user interface (GUI) version of the synthetic data generator

* synthetic_data_generator_cli_user_manual.pdf: detailed instructions for the command line (cli) version of the synthetic data generator

## Data
* synthetic_col1.trxyt: synthetic trajectory data containing "hotspots" of overlapping spatiotemporal trajectory clusters (intended for comparison purposes only)

* synthetic_col2.trxyt: synthetic trajectory data sharing some spatial and temporal clustering with synthetic_col1 (intended for comparison purposes only)

# About

Design and code: Tristan Wallis

Additional code and debugging: Kyle Young, Sophie Huiyi Hou, Kye Kudo, Alex McCann

Documentation: Alex McCann and Tristan Wallis 

Queensland Brain Institute, The University of Queensland

Fred Meunier: f.meunier@uq.edu.au

# Computer requirements

Operating systems: Windows, Linux and Mac

Requires: Python 3.8+

Modules: colorama, matplotlib, matplotlib-venn, numpy, pandas, pillow, freesimplegui, rtree, scikit-learn, scipy, seaborn, statsmodels 

Specific module versions guaranteed working under Python 3.12 at the time of release:
colorama (v0.4.6), matplotlib (v3.8.4), matplotlib-venn (v0.11.7), numpy (v1.23.2), pandas (v1.4.4), pillow (v9.2.0), freesimplegui (v5.2.0.post1), rtree (v1.0.0), scikit-learn (v1.1.2), scipy (v1.13.1), seaborn (v0.12.0), statsmodels (v0.13.2) 

# Installation procedure
1. Open a new instance of the command line
2. Copy and paste the following:
```
python -m pip install freesimplegui colorama matplotlib matplotlib-venn numpy pandas pillow rtree scikit-learn scipy seaborn statsmodels
```
If problems are encountered, please use the following to install the specific module versions:
```
python -m pip install freesimplegui==5.2.0.post1 colorama==0.4.6 matplotlib==3.8.4 matplotlib-venn==0.11.7 numpy==1.23.2 pandas==1.4.4 pillow==9.2.0 rtree==1.0.0 scikit-learn==1.1.2 scipy==1.13.1 seaborn==0.12.0 statsmodels==0.13.2
```
3. Press the return key

# License

[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution NonCommercial 4.0 International License][cc-by-nc].

[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/

[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY%20NC%204.0-lightgrey.svg

[cc-by-nc-image]: https://i.creativecommons.org/l/by-nc/4.0/88x31.png
