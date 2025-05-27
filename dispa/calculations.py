from dispa import load_pdata
from scipy.interpolate import interp1d
from scipy.signal import argrelmin
import numpy as np
import matplotlib.pyplot as plt


def get_ppm_scale(dic):    
    """Function to pull parameters from loaded processed data and calculate ppm scale.
    
    Parameters

    ----------
    
    dic: dict
        dictionary of data parameters from nmrglue.bruker.read_pdata
        
    Returns
    -------
    ppm : numpy.array
        ppm scale matched to the real 1D input spectrum
        
    """

    offset_ppm = dic["procs"]["OFFSET"]   # ppm
    sw_Hz      = dic["procs"]["SW_p"]     # Hz
    sf_MHz     = dic["procs"]["SF"]       # MHz
    si         = dic["procs"]["SI"]      # points

    dppm = sw_Hz / sf_MHz / (si - 1)         # ppm per point
    ppm = offset_ppm - dppm * np.array(range(0,si))       # decreasing ppm axis

    return ppm 
    
   
    
def get_ppm_scale_manual(offset_ppm, sw_Hz, sf_MHz, si):    
    """Function to calculate ppm scale from user-provided parameters.
    
    Parameters

    ----------
    
    offset_ppm: int
       	offset ppm value
    sw_Hz: float
        1/dw
    sf_MHz: float
        magnet frequency in MHz   
    si: int
        number of points
                
    Returns
    -------
    ppm : numpy.array
        ppm scale matched to the real 1D input spectrum
        
    """

    dppm = sw_Hz / sf_MHz / (si - 1)         # ppm per point
    ppm = offset_ppm - dppm * np.array(range(0, si))       # decreasing ppm axis

    return ppm 
    
 
 
def rotate(data, origin, angle):
    """Function to rotate a set of points by a specified angle.
    
     Parameters

    ----------
    
    data: numpy.array like
        numpy array from NMRGlue read_pdata() containing real and imaginary components
    origin: tuple
        origin in cartesian coordinates
    angle: float 
        Angle in degrees
        
    Returns
    -------
    rpoints : numpy.array
        Rotated data points
    rpr : numpy.array
        Real component of rotated data points
    rpi : numpy.array
        Imaginary component of rotated data points
    """

    points = np.array([complex(data[0][i],data[1][i]) for i in range(len(data[0]))])
    theta = np.deg2rad(angle)
    
    rpoints = (points - origin) * np.exp(complex(0, theta)) + origin
    
    rpr = rpoints.real
    rpi = rpoints.imag
    return rpoints, rpr, rpi
    

def magnitude_transformation(data):
    """Convert the real and imaginary components of NMR data in Bruker format to Magnitude Mode.
    
    Parameters

    ----------
    
    data: numpy.array like
        numpy array from NMRGlue read_pdata() containing real and imaginary components
        
    Returns
    -------
    magnitude : numpy.array
        Spectrum converted to magnitude mode
    """

    magnitude = np.sqrt(data[0]**2 + data[1]**2)
    
    return magnitude


def add_noise(data, snr):
    """Function to add Gaussian noise to processed 1D NMR data.
    
    Parameters

    ----------
    
    data: object
        NMRGlue numpy array data obejct from nmrglue.bruker.read_pdata or dispatools.utils.load_pdata
    snr: float
        desired SNR value for adding noise

    Returns
    
    -------
    
    dsp0_rn : numpy.array
        Real dimension data with added Gaussian noise
    dsp0_in : numpy.array
        Imaginary dimension data with added Gaussian noise
    rnoise : numpy.array
        Noise for the real dimension
    inoise : numpy.array
        Noise for the imaginary dimension
    """
    
    # transform to magnitude mode
    mag = magnitude_transformation(data)
    maxi = np.max(np.abs(mag)) 
    
    # set length of data/noise
    n = len(mag)
    
    # find noise scale 
    s = (maxi / snr) / 2 

    # draw noise arrays from Gaussian distribution
    rnoise = np.random.normal(0, s, n)
    inoise = np.random.normal(0, s, n)
    
    # add noise to the real and imag. components
    dsp0_rn = data[0] + rnoise
    dsp0_in = data[1] + inoise
    
    return dsp0_rn, dsp0_in, rnoise, inoise
