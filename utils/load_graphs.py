import scipy.io as sio
from scipy import sparse
import os

from glob import glob

def load_graph(graphs_folder, prefix = 'M_w'):
    graph = []
    
    N = glob(f"{graphs_folder}/{prefix}_*.mat") # all the filenames in the graphs folder with matching prefix
    N = len(N)
    
    for idx in range(N):
        file = f"{graphs_folder}/{prefix}_{idx}.mat"        
        sub_graph = sio.loadmat(file)
        sub_graph = sub_graph['TT_w']
        sub_graph = sub_graph.astype('float32')
        sub_graph = sparse.csc_matrix(sub_graph)
        
        graph.append(sub_graph)
    return graph