### Edge and Hyper-edge calculation ###
# Connectivity radius is specified in data/metadat.json for normal edges. For HYPEREDGES it is specified HERE.
# construct initial hypergraph using: "Dynamic Hypergraph Neural Networks" https://www.ijcai.org/proceedings/2019/0366.pdf #Algo 1 + smol extension.
set_dict = {}

### Do we want to use concatenated edge features, hyperedge features, or both ###
set_dict["USE_BOTH"]                    = True # when true, uses both hyper_edge_set and hyper-edges regardless of what they are set to. Note concatenation will be used for 2 uniform hyperedges (to match the original implementation)
set_dict["return_hyperedges"]           = False # if true, returns hyper-edges as opposed to normal edges. (k-means clustering + knearest neighbours). Takes precedence over hyper_edge_set
set_dict["hyper_edge_set"]              = False # if return_hyperedges false, and hyper_edge_set true uses 2 uniform hyperedges. Note sum will be used.

#if everything is false, uses normal edges. if everything is false, need to chage graph_network.py EncodeProcessDecode forward function!

### cluster based on connectivity radius? k-nearest neighbours? or also do k-means clustering? ###
# If KNN and Radius are both true, then KNN is used. If both are false then radius_clustering is used.
set_dict["knn_clustering"]     = True  # K nearest neighbours  
set_dict["radius_clustering"]  = False # Radius 
set_dict["kmeans_clustering"]  = True  # Split nodes into k clusters

#SETTINGS BELOW ONLY APPLY TO HYPEREDGES#
### K-menas clustering settings
set_dict["k_m_cl"]  = 6     #number of clusters
set_dict["top_s"]   = 10    #number of nodes to keep 
### KNN settings
set_dict["k_nn_nr"] = 4     #number of k nearest neighbours to use
### Connectivity radius settings
set_dict["con_rad"] = 0.015 #connectivity radius (0.015) is used by original



def set_values(USE_BOTH_, return_hyperedges_, hyper_edge_set_, knn_clustering_, radius_clustering_, kmeans_clustering_, k_m_cl_, top_s_, k_nn_nr_, con_rad_):
    global set_dict
    set_dict["USE_BOTH"] = USE_BOTH_ 
    set_dict["return_hyperedges"] = return_hyperedges_
    set_dict["hyper_edge_set"] = hyper_edge_set_
    set_dict["knn_clustering"] = knn_clustering_ 
    set_dict["radius_clustering"] = radius_clustering_ 
    set_dict["kmeans_clustering"] = kmeans_clustering_ 
    set_dict["k_m_cl"] = k_m_cl_
    set_dict["top_s"] = top_s_
    set_dict["k_nn_nr"] = k_nn_nr_ 
    set_dict["con_rad"] = con_rad_
