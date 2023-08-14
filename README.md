# clustering
Python GUI scripts for molecular trajectory clustering analysis.

### Files
* nastic_gui.py: cluster using R-tree based spatio-temporal indexing of trajectory bounding boxes

* nastic_st_gui.py: compilable single threaded version of nastic_gui.py for virtual machines

* nastic2c_gui.py: two color version of nastic

* nastic2c_st_gui.py: compilable single threaded version of nastic2c_gui.py for virtual machines

* segnastic_gui.py: cluster using R-tree based spatio-temporal indexing of all trajectory segment bounding boxes

* segnastic_st_gui.py: compilable single threaded version of segnastic_gui.py for virtual machines

* segnastic2c_gui.py: two color version of segnastic

* segnastic2c_st_gui.py: compilable single threaded version of segnastic2c_gui.py for virtual machines

* nastic_user_manual.pdf: detailed instructions for nastic_gui.py and derivatives

* nastic_wrangler_gui.py: detailed instructions for nastic_wrangler_gui.py

* nastic_wrangler_manual.pdf: meta analysis of output from nastic_gui.py and segnastic_gui.py

* dbscan_gui.py: cluster using DBSCAN of trajectory centroids

* voronoi_gui.py: cluster using Voronoi tesselation of trajectory centroids

* dbscan_wrangler.py: meta analysis of output from dbscan_gui.py and voronoi_gui.py

* boosh_gui.py: cluster using 3D DBSCAN of trajectory detections (experimental)

* boosh_st_gui.py: compilable single threaded version of boosh_gui.py for virtual machines

* boosh2c_gui.py: two color version of boosh (experimental)

* boosh2c_st_gui.py: compilable single threaded version of boosh2c_gui.py for virtual machines

* synthetic_col1.trxyt: synthetic trajectory data containing "hotspots" of overlapping spatiotemporal trajectory clusters

* synthetic_col2.trxyt: synthetic trajectory data sharing some spatial and temporal clustering with synthetic_col1  

* synthetic_data_generate_gui.py: generate synthetic data using multiple parameters

### About

Requires: Python 3.8+

Modules: scipy, numpy, matplotlib, matplotlib-venn, scikit-learn, rtree, pysimplegui, seaborn, statsmodels, colorama

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
