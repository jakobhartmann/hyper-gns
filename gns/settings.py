### Edge and Hyper-edge calculation ###
# Connectivity radius is specified in data/metadat.json for normal edges. For HYPEREDGES it is specified HERE.
# construct initial hypergraph using: "Dynamic Hypergraph Neural Networks" https://www.ijcai.org/proceedings/2019/0366.pdf #Algo 1 + smol extension.

### Do we want to use concatenated edge features, hyperedge features, or both ###
USE_BOTH                    = True # when true, uses both hyper_edge_set and hyper-edges regardless of what they are set to.
return_hyperedges           = False # if true, returns hyper-edges as opposed to normal edges. (k-means clustering + knearest neighbours). Takes precedence over hyper_edge_set
hyper_edge_set              = False # if return_hyperedges false, and hyper_edge_set true uses 2 uniform hyperedges.
#if everything is false, uses normal edges. if everything is false, need to chage graph_network.py EncodeProcessDecode forward function!

### cluster based on connectivity radius? k-nearest neighbours? or also do k-means clustering? ###
# If KNN and Radius are both true, then KNN is used. If both are false then radius_clustering is used.
knn_clustering     = True  # K nearest neighbours  
radius_clustering  = False # Radius 
kmeans_clustering  = True  # Split nodes into k clusters

#SETTINGS BELOW ONLY APPLY TO HYPEREDGES#
### K-menas clustering settings
k_m_cl  = 6     #number of clusters
top_s   = 10    #number of nodes to keep 
### KNN settings
k_nn_nr = 4     #number of k nearest neighbours to use
### Connectivity radius settings
con_rad = 0.015 #connectivity radius (0.015) is used by original