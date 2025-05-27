import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

def load_pdata(pdata):
    
    """Function using NMRGlue to load processed data from TopSpin. 
    
     Parameters

        ----------

        pdata: str
            path to processed data directory

    """

    # This function (read_pdata) reads the data that have been processed by TopSpin (1i, 1r, etc)
    dic_p,data_p = ng.bruker.read_pdata(dir=pdata, all_components=True) #all_components=True loads both real and imaginary components (needed for DISPA plots)

    return dic_p, data_p
    
    
def parse_dataset(datapath):
    """Function to extract TopSpin processed data from a directory of NMR experiments.
    
    Parameters

    ----------
    
    datapath: str
        relative or absolute path to directory containing datasets

    """
    
    # set up dictionary to hold the data paths
    experiments = {}

    # Find the list of experiment sub-directories
    dirlist = os.listdir(datapath)  
    directories = [entry for entry in dirlist if os.path.isdir(datapath+entry)]
    
    # load all the datasets and store them in the experiments dictionary
    for d in directories:
        exp = os.path.basename(d)
        pdir = datapath+d+"/pdata/1/"
        dic, data = load_pdata(pdir)
        experiments[exp] = [dic,data]

    return experiments
    
    

