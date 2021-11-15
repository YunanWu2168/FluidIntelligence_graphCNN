import os
import numpy as np
import scipy.io as sio

def load_inputs(graph_type, filenames):
    ''' Function for generating the graph data inputs given a dictionary of graph types
        
        Args: graph_type (dict): values are dictionaries themselves containing: 
                                    - input data directory
                                    - parts to load for the specific graph type (i.e.: lh_white, lh_pial, ...)
                                    - boolean indicating whether to use halves
        
        Returns: X (dict):  keys - graph type
                            values - tensor (Samples x Nodes x Features_per_node)
    '''
    X = {} # input initialization

    # iterate through the graph types
    for graph in graph_type:
        graph_dir = graph_type[graph]['path']              # locate the dir. all the scans for "graph" surfaces
        input_filename = f"{graph_dir}/{graph}_input.npy"  # filename for NumPy array saving

        # if the file does not exist, create it
        if not(os.path.isfile(input_filename)):
            X[graph] = []
            for file in filenames:
                if graph == 'subcortical':
                    # curr_scan = sio.loadmat(f"{graph_dir}/SubCortical_{file[7:]}")     #abcd
                    curr_scan = sio.loadmat(f"{graph_dir}/{file[0:7]}_SubCortical.mat")    #hcp
                else:
                    curr_scan = sio.loadmat(f"{graph_dir}/{file}")
                if graph_type[graph]['halves']:

                    # extract left and right hemispheres
                    left_hem = [curr_scan[f"lh_{p}"] for p in graph_type[graph]['parts']]
                    right_hem = [curr_scan[f"rh_{p}"] for p in graph_type[graph]['parts']]

                    left_hem = np.concatenate(left_hem.copy(),axis=1)   # concatenate white and pial for left hem.
                    right_hem = np.concatenate(right_hem.copy(),axis=1) # ... ... ... ... ... ... ... ...right ...

                    curr_scan = np.concatenate([left_hem.copy(), right_hem.copy()],axis=0) # concatenate left on top of right
                else:
                    curr_scan = curr_scan[graph_type[graph]['parts']] # extract subcortical surface
                
                curr_scan = np.expand_dims(curr_scan.copy(),axis=0)
                X[graph].append(curr_scan.copy()) # append to input tensor

            X[graph] = np.concatenate(X[graph].copy(),axis=0).astype('float32') # create input tensor
            np.save(input_filename,X[graph])                             # save input tensor file
        else:
            X[graph] = np.load(input_filename)
        
        print(f"{graph.capitalize()}\t(Samples, Nodes, Feat.):\t{X[graph].shape}")
    return X