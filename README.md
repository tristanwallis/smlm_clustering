# clustering
Python GUI scripts for molecular trajectory clustering analysis.

### Files
stic_gui.py: cluster using R-tree based spatio-temporal indexing of trajectory bounding boxes

seg_stic_gui.py: cluster using R-tree based spatio-temporal indexing of all trajectory segment bounding boxes

stic_wrangler_gui.py: meta analysis of output from stic_gui and seg_stic_gui

dbscan_gui.py: cluster using DBSCAN of trajectory centroids

voronoi_gui.py: cluster using Voronoi tesselation of trajectory centroids

dbscan_wrangler.py: meta analysis of output from dbscan_gui and voronoi_gui.py

synthetic_data.trxyt: synthetic trajectory data containing "hotspots" of overlapping spatiotemporal trajectory clusters


### About

All scripts require Python 3.8+
Design and code: Tristan Wallis
Debugging: Sophie Huiyi Hou
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
