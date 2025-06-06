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
    
    
def parse_proc_dataset(datapath):
    """Function to extract TopSpin processed data from a directory of NMR experiments.
    
    Parameters

    ----------
    
    datapath: str
        relative or absolute path to directory containing datasets

    Returns
    
    -------
    
    experiments : dict
        dict containing metadata dict and processed data (2D numpy.array)

    """
    
    # Set up dictionary to hold the data paths
    experiments = {}

    # Find the list of experiment sub-directories
    dirlist = os.listdir(datapath)  
    directories = [entry for entry in dirlist if os.path.isdir(datapath+entry)]

    # Check for a nested 'pdata' dir and for the presence of 1i and 1r processed files
    dir_tmp = []
    for d in directories:
        if not os.path.isdir(datapath+d+"/pdata/1/"):
            if os.path.isfile(datapath+d+"/1i") and os.path.isfile(datapath+d+"/1r"):
                dir_tmp.append(d)
            else:
                pass
        elif os.path.isdir(datapath+d+"/pdata/1/"):
            if os.path.isfile(datapath+d+"/pdata/1/1i") and os.path.isfile(datapath+d+"/pdata/1/1r"):
                dir_tmp.append(d)
            else:
                pass

        else:
            print("Directory structure not understood.")

    directories  = dir_tmp
    
    # Load all the datasets and store them in the experiments dictionary
    # Check for and handle nested 'pdata' dir 
    for d in directories:
        exp = os.path.basename(d)
        if os.path.isdir(datapath+d+"/pdata/1/"):
            pdir = datapath+d+"/pdata/1/"
            dic, data = load_pdata(pdir)
            experiments[exp] = [dic,data]
        elif not os.path.isdir(datapath+d+"/pdata/1/"):
            pdir = datapath+d
            dic, data = load_pdata(pdir)
            experiments[exp] = [dic,data]
        else:
            print("Directory structure not understood.")

    return experiments
    
    

