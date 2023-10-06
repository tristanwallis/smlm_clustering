# SMLM Clustering
Python GUI/CLI scripts for molecular trajectory clustering analysis.

### Files
Multithreaded GUI for physical hardware
* nastic_gui.py: cluster using R-tree based spatio-temporal indexing of trajectory bounding boxes

* nastic2c_gui.py: two colour version of nastic

* segnastic_gui.py: cluster using R-tree based spatio-temporal indexing of trajectory segment bounding boxes

* segnastic2c_gui.py: two colour version of segnastic

* boosh_gui.py: cluster using 3D DBSCAN of trajectory detections (experimental)

* boosh2c_gui.py: two colour version of boosh (experimental)

* dbscan_gui.py: cluster using DBSCAN of trajectory centroids (for comparison purposes)

* voronoi_gui.py: cluster using Voronoi tesselation of trajectory centroids (for comparison purposes)

Single-threaded GUI for virtual computers
* nastic_st_gui.py: single-threaded (compilable) version of nastic_gui.py  

* nastic2c_st_gui.py: single-threaded (compilable) version of nastic2c_gui.py  

* segnastic_st_gui.py: single-threaded (compilable)version of segnastic_gui.py  

* segnastic2c_st_gui.py: single-threaded (compilable) version of segnastic2c_gui.py  

* boosh_st_gui.py: single-threaded (compilable) version of boosh_gui.py (experimental)

* boosh2c_st_gui.py: single-threaded (compilable) version of boosh2c_gui.py (experimental)

Command line
* nastic_cli.py: command line single-threaded version of nastic

* segnastic_cli.py: command line single-threaded version of segnastic

* boosh_cli.py: command line single-threaded version of boosh (experimental)

Miscellaneous utilities
* nastic_wrangler_gui.py: meta analysis of output from nastic_gui.py, segnastic_gui.py and boosh_gui.py

* super_res_data_wrangler_gui.py: convert between super resolution trajectory filetypes

* synthetic_data_generate_gui.py: generate synthetic data using multiple parameters

* dbscan_wrangler_gui.py: meta analysis of output from dbscan_gui.py and voronoi_gui.py

User manuals
* nastic_user_manual.pdf: detailed instructions for nastic_gui.py and derivatives

* nastic_wrangler_manual.pdf: detailed instructions for nastic_wrangler_gui.py

Data
* synthetic_col1.trxyt: synthetic trajectory data containing "hotspots" of overlapping spatiotemporal trajectory clusters

* synthetic_col2.trxyt: synthetic trajectory data sharing some spatial and temporal clustering with synthetic_col1 

### About

Requires: Python 3.8+

Modules: colorama, matplotlib, matplotlib-venn, numpy, pandas, Pillow, pysimplegui, rtree, scikit-learn, scipy, seaborn, statsmodels 

Design and code: Tristan Wallis

Additional code and debugging: Alex McCann, Kyle Young, Sophie Huiyi Hou, Kye Kudo

Queensland Brain Institute, University of Queensland

Fred Meunier: f.meunier@uq.edu.au


### License

[![CC BY 4.0][cc-by-shield]][cc-by]

[![CC BY 4.0][cc-by-image]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[cc-by]: http://creativecommons.org/licenses/by/4.0/

[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png