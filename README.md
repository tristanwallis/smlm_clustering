# clustering
Python GUI scripts for molecular trajectory clustering analysis

stic_gui.py: cluster using R-tree based spatio-temporal indexing of trajectory bounding boxes

seg_stic_gui.py: cluster using R-tree based spatio-temporal indexing of all trajectory segment bounding boxes

stic_wrangler_gui.py: meta analysis of output from stic_gui and seg_stic_gui

dbscan_gui.py: cluster using DBSCAN of trajectory centroids

voronoi_gui.py: cluster using Voronoi tesselation of trajectory centroids

dbscan_wrangler.py: meta analysis of output from dbscan_gui and voronoi_gui.py

All scripts require Python 3.8+
