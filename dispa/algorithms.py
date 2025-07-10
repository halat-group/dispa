from dispa import load_pdata, get_ppm_scale, get_ppm_scale_manual, magnitude_transformation, rotate
from dispa import fidgen, phaseshift, specgen, addnoise, fidcomb, write_fid_TS
from scipy.interpolate import interp1d
from scipy.signal import argrelmin
import numpy as np
import matplotlib.pyplot as plt
import os
import nmrglue as ng
import pandas as pd

def find_saddle(params, data, ppm_region, plot=False, plotname="saddle-plot"):
    """Function to identify local saddle points between peaks in 1D NMR data.
    
    Parameters

    ----------

    params: dict
        dictionary of data parameters from nmrglue.bruker.read_pdata or dispatools.utils.load_pdata
    data: object
        NMRGlue numpy array data obejct from nmrglue.bruker.read_pdata or dispatools.utils.load_pdata
    ppm_region: tuple
        Upper and lower limits for the ppm region of interest
    plot: bool
        whether or not to plot the saddle point location on magnitude mode data
    plotname: str
        name of optional plot

    Returns
    
    -------
    
    saddle_ppm : float
        The location along the ppm axis of the saddle point
        
    """

    # Calculate the ppm scale for the data
    ppm  = get_ppm_scale(params)

    # Transform the data to magnitude mode
    magnitude = magnitude_transformation(data)

    # Extract region
    lo = ppm_region[0];
    hi = ppm_region[1];
    idx = np.where((ppm >= lo) & (ppm <= hi))
    mag_roi = magnitude[idx]
    ppm_roi = ppm[idx]

    # Find the local minimum in the region
    saddle_idx = argrelmin(mag_roi, order=20)[0]
    saddle_ppm = ppm_roi[saddle_idx]
    
    # Get the saddle point intensity in R and I dimensions
    pidx = np.where(ppm == saddle_ppm)
    R_sp = data[0][pidx]
    I_sp = data[1][pidx]
    
    # Plot the location of the saddle point on magnitude spectrum
    if plot==True:
    
        # set up fig and gridspec for paired plots
        fig = plt.figure(constrained_layout=True, figsize=(8,4))
        gs = fig.add_gridspec(nrows=1, ncols=2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        # plot the 1D magnitude and absorptive spectra
        ax1.plot(ppm_roi, mag_roi, color="midnightblue")
        ax1.plot(ppm_roi, data[0][idx], color="orange")
        # add the saddle point markers
        ax1.scatter(saddle_ppm, mag_roi[saddle_idx], color="red", s=10)
        ax1.axvline(x=saddle_ppm, ls="--", color="red", lw=1)
        ax1.set_xlabel("ppm", fontsize=11)
        ax1.set_ylabel("a.u", fontsize=11)
        ax1.set_title("1D spectrum", fontsize=11)
        ytick_range = list(ax1.get_yticks())
        xtext=np.round(saddle_ppm[0], decimals=3)
        ax1.text(x=xtext-np.abs(0.005*xtext), y=np.median(ytick_range), s=str(xtext), color="red") 
        ax1.legend({"magnitude":"midnightblue","absorptive":"orange"})
        ax1.invert_xaxis()


        # plot the polar representation
        
        # find the polar coordinates for saddle point
        ax2.plot(data[0], data[1], color="midnightblue"); 
        ax2.scatter(R_sp, I_sp, color="red", s=20); 
        # label axes
        ax2.set_xlabel("Real (a.u.)", fontsize=11)
        ax2.set_ylabel("Imaginary (a.u)", fontsize=11); 
        ax2.set_title("Polar Plot", fontsize=11)
        # set aspect to equal for square plot
        plt.gca().set_aspect("equal")


        fig.suptitle(plotname, fontsize=12)
        plt.savefig(plotname+".pdf")
        plt.savefig(plotname+".png", dpi=300)
        
    return saddle_ppm, pidx, (R_sp, I_sp)
    
    
def optimize_rotation_rms_file(data_path1, data_path2, ppm_region, step_deg=0.001, plot=True, plotname="RMS-Angles", origin=(0,0), angle_range=(-180,180), figsize=(8,4)):
    """Optimize phase rotation of dataset 2 to match dataset 1 using RMS error, taking data directly from Bruker processed data directories. 
    Returns the angle with lowest RMS, and optionally plots RMS vs angle. 
    
    Parameters

    ----------
    
    data_path1: str
        Path to the first Bruker processed data directory
    data_path2: str
        Path to the second Bruker processed data directory
    ppm_region: tuple
        Upper and lower limits for the ppm region of interest
    step_deg: float
        Step size for sweeping angles in second pass (can be non-integer)
    plot: bool
        Whether to generate the optional plots of optimized rotation
    plotname: str
    	Name for the plot file
    origin: tuple
        cartesian coordinates of origin for rotation plot
    angle_range: tuple
    	range of angles to sweep during RMSE minimization
    figsize: tuple
    	figure size for optional plot (inches)
    	
    Returns
    
    -------
    
    best_angle : float
        The value of the angle in degrees that minimizes RMSE
    min_rms : float
        The minimum value of RMSE identified by algorithm
    angles_deg : np.array
        The 1D array of angles (in degrees) for optimization
    rms_values : np.array
        The 1D array of calculated RMSE values 
    data_rotated : np.array
        The 2D array of rotated data   
        
    """

    # Load the datasets
    params1, spec1 = load_pdata(data_path1)
    ppm1  = get_ppm_scale(params1)
    
    params2, spec2 = load_pdata(data_path2)
    ppm2  = get_ppm_scale(params2)

    # scale the spec and project to 1D
    R1 = spec1[0]
    I1 = spec1[1]
    nc_proc1 = params1['procs']["NC_proc"]
    scale1 = 2**nc_proc1;
    SPEC1 = (R1 + 1*np.sqrt(-1+0j) * I1) * scale1

    R2 = spec2[0]
    I2 = spec2[1]
    nc_proc2 = params2['procs']["NC_proc"]
    scale2 = 2**nc_proc2;
    SPEC2 = (R2 + 1*np.sqrt(-1+0j) * I2) * scale2

    # Interpolate the spectra using a quadratic spline
    f_splineS1 = interp1d(ppm1, SPEC1, kind='quadratic')
    f_splineS2 = interp1d(ppm2, SPEC2, kind='quadratic')

    SPEC1 = f_splineS1(ppm1)
    SPEC2 = f_splineS2(ppm2)

    # Extract region
    lo = ppm_region[0];
    hi = ppm_region[1];
    idx1 = np.where((ppm1 >= lo) & (ppm1 <= hi))
    idx2 = np.where((ppm2 >= lo) & (ppm2 <= hi))

    spec1r = SPEC1[idx1]
    spec2r = SPEC2[idx2]

    # Run sanity check 
    N = min(len(spec1r), len(spec2r))
    spec1r = spec1r[0:N]
    spec2r = spec2r[0:N]

    # Sweep angles
    angles_deg_init = np.array(np.arange(angle_range[0], angle_range[1]+1, 1))
    rms_values_init = np.zeros(len(angles_deg_init))

    # Do an initial pass with 1 deg step size
    for k in range(0, len(rms_values_init)):
        theta_init = np.deg2rad(angles_deg_init[k])
        rotated_init = spec2r * np.exp(1*np.sqrt(-1+0j) * theta_init)
        rms_values_init[k] = np.sqrt(np.mean(np.abs(spec1r - rotated_init)**2))

    # Find best intial estimate of angle
    min_rms_init = np.min(rms_values_init)
    min_idx_init = np.where(rms_values_init == min_rms_init)
    best_angle_init = angles_deg_init[min_idx_init][0]

    # get new bounds from initial angle estimate
    low_bound = best_angle_init - 0.5
    high_bound = best_angle_init + 0.5 + 0.01
    angles_deg = np.array(np.arange(low_bound, high_bound, step_deg))
    
    # Sweep angles
    rms_values = np.zeros(len(angles_deg))
        
    for k in range(0, len(rms_values)):
        theta = np.deg2rad(angles_deg[k])
        rotated = spec2r * np.exp(1*np.sqrt(-1+0j) * theta)
        rms_values[k] = np.sqrt(np.mean(np.abs(spec1r - rotated)**2))


    # Find best angle
    min_rms = np.min(rms_values)
    min_idx = np.where(rms_values == min_rms)
    best_angle = angles_deg[min_idx][0]
    num_decimals = len(str(step_deg).split(".")[1])
    best_angle = np.round(best_angle, decimals=num_decimals)

    # rotate data for vizualation - this should plot over relevant ppm range
    data_rotated, dr_rotated, di_rotated = rotate(spec2, complex(origin[0], origin[1]), best_angle)

    # Plot the result
    if plot==True:

        # set up plot and gridspec
        fig = plt.figure(constrained_layout=True, figsize=(figsize[0],figsize[1]))
        gs = fig.add_gridspec(nrows=1, ncols=2)

        # generate the RMSE subplot
        f_ax1 = fig.add_subplot(gs[0,0])
        f_ax1.set_title('Optimal Angle (RMSE)')
        f_ax1.plot(angles_deg_init, rms_values_init, color="midnightblue");
        #f_ax1.scatter(angles_deg, rms_values, color="darkorange", s=2);
        f_ax1.set_xlabel("Rotation Angle (degrees)")
        f_ax1.set_ylabel("RMS Error")
        plt.axvline(best_angle, lw=0.5, linestyle="--", color="blue")
        ytick_range = list(f_ax1.get_yticks())
        plt.text(x=best_angle+np.abs(0.05*best_angle), y=np.median(ytick_range), s=r"{}$^\circ$".format(best_angle), color="blue")
        

        # generate the rotated polar subplot
        f_ax2 = fig.add_subplot(gs[0,1])
        f_ax2.plot(dr_rotated[idx2], di_rotated[idx2], color="darkviolet")
        f_ax2.plot(R1[idx1], I1[idx1], color="orange")
        f_ax2.plot(R2[idx2], I2[idx2], color="teal")       
        f_ax2.set_xlabel("Real (a.u.)")
        f_ax2.set_ylabel("Imaginary (a.u.)")
        f_ax2.set_title('Optimal Rotation')
        plt.axhline(0, lw=1, linestyle="--", color="gray")
        plt.axvline(0, lw=1, linestyle="--", color="gray")
        plt.gca().set_aspect("equal")
        plt.legend({"rotated":"darkviolet", "reference":"orange", "unrotated":"teal"})
        
        plt.savefig(plotname+".png", dpi=300)
        plt.savefig(plotname+".pdf")

        
    return best_angle, min_rms, angles_deg, rms_values, data_rotated
    
    
def optimize_rotation_rms_mem(spec1, spec2, params1, params2, ppm_region, plot=False, plotname="RMS-angles", origin=(0,0), 
                               step_deg=0.1, angle_range=(-180,180), figsize=(8,4)):
    """Optimize phase rotation of dataset 2 to match dataset 1 using RMS error for files loaded in memory as NMRGlue datasets. 
    
    Parameters

    ----------
    
    spec1: numpy.array
        2D array of real and imaginary components from nmgrglue.read_pdata/distpatools.read_pdata
    spec2: numpy.array
        2D array of real and imaginary components from nmgrglue.read_pdata/distpatools.read_pdata
    params1: dict
        metadata dictionary from nmgrglue.read_pdata/distpatools.read_pdata
    params2: dict
        metadata dictionary from nmgrglue.read_pdata/distpatools.read_pdata
    ppm_region: tuple
        Upper and lower limits for the ppm region of interest
    plot: bool
        Whether to generate the optional plots of optimized rotation
    plotname: str
    	Name for the plot file
    origin: tuple
        cartesian coordinates of origin for rotation plot
    step_deg: float
        Step size for sweeping angles after initial pass (can be non-integer)
    angle_range: tuple
    	range of angles to sweep during RMSE minimization
    figsize: tuple
    	figure size for optional plot (inches)
    		
    Returns
    
    -------
    
    best_angle : float
        The value of the angle in degrees that minimizes RMSE
    min_rms : float
        The minimum value of RMSE identified by algorithm
    angles_deg : np.array
        The 1D array of angles (in degrees) for final optimization
    rms_values : np.array
        The 1D array of calculated RMSE values for final optimization
    angles_deg_init : np.array
        The 1D array of angles (in degrees) for initial optimization
    rms_values_init : np.array
        The 1D array of calculated RMSE values for initial optimization        
    """
    
    # Load the datasets
    #params1, spec1 = load_pdata(data_path1)
    ppm1  = get_ppm_scale(params1)
    
    #params2, spec2 = load_pdata(data_path2)
    ppm2  = get_ppm_scale(params2)

    # scale the spec and project to 1D
    R1 = spec1[0]
    I1 = spec1[1]
    nc_proc1 = params1['procs']["NC_proc"]
    scale1 = 2**nc_proc1;
    SPEC1 = (R1 + 1*np.sqrt(-1+0j) * I1) * scale1

    R2 = spec2[0]
    I2 = spec2[1]
    nc_proc2 = params2['procs']["NC_proc"]
    scale2 = 2**nc_proc2;
    SPEC2 = (R2 + 1*np.sqrt(-1+0j) * I2) * scale2

    # Interpolate the spectra using a quadratic spline
    #f_splineS1 = interp1d(ppm1, SPEC1, kind='quadratic')
    #f_splineS2 = interp1d(ppm2, SPEC2, kind='quadratic')

    #SPEC1 = f_splineS1(ppm1)
    #SPEC2 = f_splineS2(ppm2)

    # Extract region
    lo = ppm_region[0];
    hi = ppm_region[1];
    idx1 = np.where((ppm1 >= lo) & (ppm1 <= hi))
    idx2 = np.where((ppm2 >= lo) & (ppm2 <= hi))

    spec1r = SPEC1[idx1]
    spec2r = SPEC2[idx2]

    # Run sanity check 
    N = min(len(spec1r), len(spec2r))
    spec1r = spec1r[0:N]
    spec2r = spec2r[0:N]

    # Sweep angles
    angles_deg_init = np.array(np.arange(angle_range[0], angle_range[1]+1, 1))
    rms_values_init = np.zeros(len(angles_deg_init))

    # Do an initial pass with 1 deg step size
    for k in range(0, len(rms_values_init)):
        theta_init = np.deg2rad(angles_deg_init[k])
        rotated_init = spec2r * np.exp(1*np.sqrt(-1+0j) * theta_init)
        rms_values_init[k] = np.sqrt(np.mean(np.abs(spec1r - rotated_init)**2))

    # Find best intial estimate of angle
    min_rms_init = np.min(rms_values_init)
    min_idx_init = np.where(rms_values_init == min_rms_init)
    best_angle_init = angles_deg_init[min_idx_init][0]

    # get new bounds from initial angle estimate
    low_bound = best_angle_init - 0.5
    high_bound = best_angle_init + 0.5 + 0.01
    angles_deg = np.array(np.arange(low_bound, high_bound, step_deg))
    
    # Sweep angles
    rms_values = np.zeros(len(angles_deg))
        
    for k in range(0, len(rms_values)):
        theta = np.deg2rad(angles_deg[k])
        rotated = spec2r * np.exp(1*np.sqrt(-1+0j) * theta)
        rms_values[k] = np.sqrt(np.mean(np.abs(spec1r - rotated)**2))


    # Find best angle
    min_rms = np.min(rms_values)
    min_idx = np.where(rms_values == min_rms)
    best_angle = angles_deg[min_idx][0]
    

    # rotate data for vizualation - this should plot over relevant ppm range
    data_rotated, dr_rotated, di_rotated = rotate(spec2, complex(origin[0], origin[1]), best_angle)
  
    # Plot the result
    if plot==True:

        #if show_frequency == True: #whether or not to add the 1D spectra panel

        # set up plot and gridspec
        fig = plt.figure(constrained_layout=True, figsize=(figsize[0],figsize[1]))
        gs = fig.add_gridspec(nrows=1, ncols=3)

        # generate the RMSE subplot
        f_ax1 = fig.add_subplot(gs[0,0])
        f_ax1.set_title('Optimal Angle (RMSE)')
        f_ax1.plot(angles_deg_init, rms_values_init, color="midnightblue");
        #f_ax1.scatter(angles_deg, rms_values, color="darkorange", s=2);
        f_ax1.set_xlabel("Rotation Angle (degrees)")
        f_ax1.set_ylabel("RMS Error")
        plt.axvline(best_angle, lw=0.5, linestyle="--", color="blue")
        ytick_range = list(f_ax1.get_yticks())
        plt.text(x=best_angle+np.abs(0.05*best_angle), y=np.median(ytick_range), s=r"{}$^\circ$".format(np.round(best_angle, decimals=3)), color="blue")
        

        # generate the rotated polar subplot
        f_ax2 = fig.add_subplot(gs[0,1])
        f_ax2.scatter(dr_rotated[idx2], di_rotated[idx2], color="darkviolet", s=2, alpha=0.5)
        f_ax2.scatter(R1[idx1], I1[idx1], color="orange", s=2, alpha=0.5)
        f_ax2.scatter(R2[idx2], I2[idx2], color="teal", s=2, alpha=0.5)
        
        f_ax2.set_xlabel("Real (a.u.)")
        f_ax2.set_ylabel("Imaginary (a.u.)")
        f_ax2.set_title('Optimal Rotation')
        plt.axhline(0, lw=1, linestyle="--", color="gray")
        plt.axvline(0, lw=1, linestyle="--", color="gray")
        plt.gca().set_aspect("equal")
        plt.legend({"rotated":"darkviolet", "reference":"orange", "unrotated":"teal"})
        
        # generate a 1D frequency-domain spectra plot
        f_ax3 = fig.add_subplot(gs[0,2])
        f_ax3.scatter(ppm1[idx1], R1[idx1], color="orange", s=2, alpha=0.5)
        f_ax3.scatter(ppm2[idx2], R2[idx2], color="teal", s=2, alpha=0.5)
        f_ax3.set_title('Frequency Domain')
        f_ax3.set_xlabel("ppm")
        f_ax3.set_ylabel("Intensity (a.u.)")
        
        plt.savefig(plotname+".png", dpi=300)
        plt.savefig(plotname+".pdf")

    return best_angle, min_rms, angles_deg, rms_values, data_rotated, rms_values_init
    
    
    
def optimize_rotation_rms_NoFiles(spec1, spec2, ppm_region, step_deg=1, plot=True, plotname="RMS-Angles", origin=(0,0), 
                          angle_range=(-180,180), figsize=(8,4), nc_proc=11, offset_ppm=5, dw=0.001):
    """Optimize phase rotation of spectrum 2 to match spectrum 1 using RMS error. 
    
    Parameters

    ----------
    
    spec1: numpy.array
        2D array of real and imaginary components from nmgrglue.read_pdata/distpatools.read_pdata
    spec2: numpy.array
        2D array of real and imaginary components from nmgrglue.read_pdata/distpatools.read_pdata
    params1: dict
        metadata dictionary from nmgrglue.read_pdata/distpatools.read_pdata
    params2: dict
        metadata dictionary from nmgrglue.read_pdata/distpatools.read_pdata
    ppm_region: tuple
        Upper and lower limits for the ppm region of interest
    step_deg: float
        Step size for sweeping angles after initial pass (can be non-integer)
    plot: bool
        Whether to generate the optional plots of optimized rotation
    plotname: str
    	Name for the plot file
    origin: tuple
        cartesian coordinates of origin for rotation plot
    angle_range: tuple
    	range of angles to sweep during RMSE minimization
    figsize: tuple
    	figure size for optional plot (inches)
    nc_proc: int
    	Bruker NC_proc parameter (input manually in this version)
    offset_ppm: int
    	Buker OFFSET parameter (input manually in this version)
    dw: float
    	Dwell Time (DW) for data collection/simulation
    		
    Returns
    
    -------
    
    best_angle : float
        The value of the angle in degrees that minimizes RMSE
    min_rms : float
        The minimum value of RMSE identified by algorithm
    angles_deg : np.array
        The 1D array of angles (in degrees) for final optimization
    rms_values : np.array
        The 1D array of calculated RMSE values for final optimization
    angles_deg_init : np.array
        The 1D array of angles (in degrees) for initial optimization
    rms_values_init : np.array
        The 1D array of calculated RMSE values for initial optimization        
    """
    
    
    # Calculate ppm with manually-input parameters
    ppm1  = get_ppm_scale_manual(offset_ppm=offset_ppm, sw_Hz=1/dw, sf_MHz=100, si=len(spec1[0])) # why is ppm flipped?
    ppm2  = get_ppm_scale_manual(offset_ppm=offset_ppm, sw_Hz=1/dw, sf_MHz=100, si=len(spec2[0]))

    
    #print(ppm1)
    # scale the spec and project to 1D
    R1 = spec1[0]
    I1 = spec1[1]
    nc_proc1 = nc_proc#params1['procs']["NC_proc"]
    scale1 = 2**nc_proc1;
    SPEC1 = (R1 + 1j*I1) * scale1
    #print(SPEC1)
    R2 = spec2[0]
    I2 = spec2[1]
    nc_proc2 = nc_proc#params2['procs'][""]
    scale2 = 2**nc_proc2;
    SPEC2 = (R2 + 1j*I2) * scale2

    # Interpolate the spectra using a quadratic spline
    #f_splineS1 = interp1d(ppm1, SPEC1, kind='quadratic')
    #f_splineS2 = interp1d(ppm2, SPEC2, kind='quadratic')

    #SPEC1 = f_splineS1(ppm1)
    #SPEC2 = f_splineS2(ppm2)

    # Extract region
    lo = ppm_region[0];
    hi = ppm_region[1];
    idx1 = np.where((ppm1 >= lo) & (ppm1 <= hi))
    idx2 = np.where((ppm2 >= lo) & (ppm2 <= hi))
    #print(idx1)
    spec1r = SPEC1[idx1]
    spec2r = SPEC2[idx2]
    #print(spec1r)
    # Run sanity check 
    N = min(len(spec1r), len(spec2r))
    spec1r = spec1r[0:N]
    spec2r = spec2r[0:N]
    #print(spec1r)
    # Sweep angles
    angles_deg_init = np.array(np.arange(angle_range[0], angle_range[1]+1, 1))
    rms_values_init = np.zeros(len(angles_deg_init))

    # Do an initial pass with 1 deg step size
    for k in range(0, len(rms_values_init)):
        theta_init = np.deg2rad(angles_deg_init[k])
        rotated_init = spec2r * np.exp(1*np.sqrt(-1+0j) * theta_init)
        rms_values_init[k] = np.sqrt(np.mean(np.abs(spec1r - rotated_init)**2))

    #print(rms_values_init)
    # Find best intial estimate of angle
    min_rms_init = np.min(rms_values_init)
    min_idx_init = np.where(rms_values_init == min_rms_init)
    #print(min_idx_init)
    best_angle_init = angles_deg_init[min_idx_init][0]
    
    # get new bounds from initial angle estimate
    low_bound = best_angle_init - 0.5
    high_bound = best_angle_init + 0.5 + 0.01
    angles_deg = list(np.arange(low_bound, high_bound, step_deg))
    # make sure the exact value of the initial best angle estimate is included
    angles_deg.append(best_angle_init)
    angles_deg = np.array(np.unique(angles_deg))
    
    # Sweep angles
    rms_values = np.zeros(len(angles_deg))
        
    for k in range(0, len(rms_values)):
        theta = np.deg2rad(angles_deg[k])
        rotated = spec2r * np.exp(1*np.sqrt(-1+0j) * theta)
        rms_values[k] = np.sqrt(np.mean(np.abs(spec1r - rotated)**2))


    # Find best angle
    min_rms = np.min(rms_values)
    min_idx = np.where(rms_values == min_rms)
    best_angle = angles_deg[min_idx][0]
    num_decimals = len(str(step_deg).split(".")[1])
    best_angle = np.round(best_angle, decimals=num_decimals)

    # rotate data for vizualation - this should plot over relevant ppm range
    data_rotated, dr_rotated, di_rotated = rotate(spec2, complex(origin[0], origin[1]), best_angle)
    
    # Plot the result
    if plot==True:

        #if show_frequency == True: #whether or not to add the 1D spectra panel

        # set up plot and gridspec
        fig = plt.figure(constrained_layout=True, figsize=(figsize[0],figsize[1]))
        gs = fig.add_gridspec(nrows=1, ncols=3)

        # generate the RMSE subplot
        f_ax1 = fig.add_subplot(gs[0,0])
        f_ax1.set_title('Optimal Angle (RMSE)')
        f_ax1.plot(angles_deg_init, rms_values_init, color="midnightblue");
        #f_ax1.scatter(angles_deg, rms_values, color="darkorange", s=2);
        f_ax1.set_xlabel("Rotation Angle (degrees)")
        f_ax1.set_ylabel("RMS Error")
        plt.axvline(best_angle, lw=0.5, linestyle="--", color="blue")
        ytick_range = list(f_ax1.get_yticks())
        plt.text(x=best_angle+np.abs(0.05*best_angle), y=np.median(ytick_range), s=r"{}$^\circ$".format(best_angle), color="blue")
        

        # generate the rotated polar subplot
        f_ax2 = fig.add_subplot(gs[0,1])
        f_ax2.scatter(dr_rotated[idx2], di_rotated[idx2], color="darkviolet", s=2, alpha=0.5)
        f_ax2.scatter(R1[idx1], I1[idx1], color="orange", s=2, alpha=0.5)
        f_ax2.scatter(R2[idx2], I2[idx2], color="teal", s=2, alpha=0.5)
        
        f_ax2.set_xlabel("Real (a.u.)")
        f_ax2.set_ylabel("Imaginary (a.u.)")
        f_ax2.set_title('Optimal Rotation')
        plt.axhline(0, lw=1, linestyle="--", color="gray")
        plt.axvline(0, lw=1, linestyle="--", color="gray")
        plt.gca().set_aspect("equal")
        plt.legend({"rotated":"darkviolet", "reference":"orange", "unrotated":"teal"})
        
        # generate a 1D frequency-domain spectra plot
        f_ax3 = fig.add_subplot(gs[0,2])
        f_ax3.scatter(ppm1[idx1], R1[idx1], color="orange", s=2, alpha=0.5)
        f_ax3.scatter(ppm2[idx2], R2[idx2], color="teal", s=2, alpha=0.5)
        f_ax3.set_title('Frequency Domain')
        f_ax3.set_xlabel("ppm")
        f_ax3.set_ylabel("Intensity (a.u.)")
        
        plt.savefig(plotname+".png", dpi=300)
        plt.savefig(plotname+".pdf")

        
    return best_angle, min_rms, angles_deg, rms_values, data_rotated, rms_values_init
  
    
def estimate_separation_error(focal_width, width2, focal_intensity, intensity2, focal_center, center2, parameters, output_folder, theta_ps, plot=True):
    """Function to estimate the error in phase-shift estimates caused by peak overlap via simulation.
    
    Parameters

    ----------

    focal_width: float
        Width in Hz of focal peak
    width2: float
        Width in Hz of non-focal peak
    focal_intensity: float
        Intensity of focal peak
    intensity2: float
        Intensity of non-focal peak
    focal_center: float
        Center (in Hz) of focal peak
    center2: float
        Center (in Hz) of non-focal peak
    parameters: dict
        Dictionary containing simulation parameters (e.g. {"n":4096, "dw":0.001, "bf":100, "SW_p":1000, "OFFSET":5, "nc_proc":11})
    output_folder: str
        name of folder to store outputs
    theta_ps: float
        Phase shift angle for focal peak (degrees)
    plot: bool
        Whether or not to generate and save optional plots of spectra and ROI error curves
        
    Returns
    
    -------
    
    expansions_p0 : pandas.DataFrame
        Dataframe containing estimates and errors for non-focal peak
    expansions_p10 : pandas.DataFrame
        Dataframe containing estimates and errors for focal phase-shifted peak
    min_err_focal : float
        Minimum ROI error calculated from the ROI scan for focal peak
    min_err_nops : float
        Minimum ROI error calculated from the ROI scan for non-focal peak     
    """
    
    ### Simulate comparable no-noise spectra based on user-input parameters
    
    # Make sure the user specifies focal_center as the peak of interest
    
    theta_noshift = 0         # phase shift to apply to first peak
    theta_shift = theta_ps         # phase shift to apply to second peak

    # need to focus on the peak of interest that might have phase shift (width1)
    
    # Get ratio of largest to smallest peak intensities for scaling simulations
    intensities = [focal_intensity, intensity2]
    intensity_ratio = max(intensities)/min(intensities)
    
    # Automatically identify a relevant range of peak separations
    centers = [focal_center, center2]
    
    # Create descriptive expnos for write function to save files
    expno = "FocalPeakShift" + str(focal_center) + "_" + "FocalPeakWidth" + str(focal_width) + "_" + "PhaseShift" + "0deg"
    expno_ps = "FocalPeakShift" + str(focal_center) + "_" + "FocalPeakWidth" + str(focal_width) + "_" + "PhaseShift" + str(theta_shift)+ "deg"
    
    ### Simulate data for peaks with phase shift ### 

    # Generate both peaks
    fid1 = fidgen(focal_center, focal_width, parameters["n"], parameters["dw"]) #this is the focal peak, which will have phase shift
    fid2 = fidgen(center2, width2, parameters["n"], parameters["dw"])
        
    # Remove DC
    fid1[0] = fid1[0]/2
    fid2[0] = fid2[0]/2

    # alter the intensity of the second peak
    if focal_intensity == max(intensities):
        fid1 = fid1*intensity_ratio
    elif intensity2 == max(intensities):
        fid2 = fid2*intensity_ratio
        
    # Apply phase shift
    fid1 = phaseshift(fid1, parameters["dw"], theta_shift)
    fid2 = phaseshift(fid2, parameters["dw"], theta_noshift)

    # Combine the fids - do not add noise (assume max possible SNR)
    fid = fid1 + fid2

    # Assign folder path for this experiment
    foldername = output_folder +"/" + str(expno_ps)
    
    # Write to TopSpin format
    write_fid_TS(fid, parameters["dw"], parameters["bf"], foldername)
    
    
    ### Simulate data for peaks without phase shift ###
    
    # Generate both peaks
    fid3 = fidgen(focal_center, focal_width, parameters["n"], parameters["dw"]) #this is the focal peak, but without phase shift
    fid4 = fidgen(center2, width2, parameters["n"], parameters["dw"])
        
    # Remove DC
    fid3[0] = fid3[0]/2
    fid4[0] = fid4[0]/2

    # alter the intensity of the second peak
    if focal_intensity == max(intensities):
        fid3 = fid3*intensity_ratio
    elif intensity2 == max(intensities):
        fid4 = fid4*intensity_ratio
        
    # Apply phase shift
    fid3 = phaseshift(fid3, parameters["dw"], theta_noshift)
    fid4 = phaseshift(fid4, parameters["dw"], theta_noshift)

    # Combine the fids - do not add noise (assume max possible SNR)
    fid_ns = fid3 + fid4

    # Assign folder path for this experiment
    foldername = output_folder +"/" + str(expno)
    
    # Write to TopSpin format
    write_fid_TS(fid_ns, parameters["dw"], parameters["bf"], foldername)


    # Load the simualate spectra back from files into memory - store in experiments dictionary
    experiments = {}

    for d in os.listdir(output_folder):
        #if os.path.isdir(os.path.join(d, output_folder)):
        meta, data = ng.bruker.read(output_folder+"/"+d, "fid")
        fid_fft, f_fft = specgen(data, dw=parameters["dw"])
        test_data = np.array([fid_fft.real, fid_fft.imag])
    
        meta['procs']['OFFSET'] = parameters["OFFSET"] #this is a kludge to deal with missing parameter in procs for now
        meta['procs']['SW_p'] = parameters["SW_p"] #this is a kludge to deal with missing parameter in procs for now
    
        experiments[d] = [meta, test_data]
    
    # Generate an array of seed ROIs for each spectrum based on input peak centers
    seed_roi_p10 = [focal_center/parameters["bf"]-0.005, focal_center/parameters["bf"]+0.005]
    seed_roi_p0 = [center2/parameters["bf"]-0.005, center2/parameters["bf"]+0.005]


    ### Walk outward from seed ROIs and estimate phase shift ###

    # Assign test datasets to new variables with simple labels
    test_data_0 = experiments["FocalPeakShift" + str(focal_center) + "_" + "FocalPeakWidth" + str(focal_width) + "_" + "PhaseShift" + "0deg"][1]
    test_data_10 = experiments["FocalPeakShift" + str(focal_center) + "_" + "FocalPeakWidth" + str(focal_width) + "_" + "PhaseShift" + str(theta_shift)+ "deg"][1]

    # Set up number of iterations and increment for ROI walkout
    num_iter = 100
    increment = 0.005
    
    # Set up dictionaries to hold angle estimates
    expansions_p0 = {}
    expansions_p10 = {}

    
    # Initial estimate point with seed roi
    best_angle_p0, min_rms_p0, angles_deg_p0, rms_values_p0, data_rotated_p0, rms_values_init_p0 = optimize_rotation_rms_NoFiles(spec1=test_data_0, spec2=test_data_10, 
                                                                                                                                 ppm_region=seed_roi_p0,step_deg=0.001,
                                                                                                                                 plot=False, 
                                                                                                                                 plotname="NULL0", figsize=(12,4), 
                                                                                                                                 nc_proc=parameters["nc_proc"], 
                                                                                                                                 offset_ppm=parameters["OFFSET"], 
                                                                                                                                 dw=parameters["dw"])   

    
    best_angle_p10, min_rms_p10, angles_deg_p10, rms_values_p10, data_rotated_p10, rms_values_init_p10 = optimize_rotation_rms_NoFiles(spec1=test_data_0, spec2=test_data_10, 
                                                                                                                                       ppm_region=seed_roi_p10,step_deg=0.001, 
                                                                                                                                       plot=False, 
                                                                                                                                       plotname="NULL10", figsize=(12,4),
                                                                                                                                       nc_proc=parameters["nc_proc"], 
                                                                                                                                       offset_ppm=parameters["OFFSET"], 
                                                                                                                                       dw=parameters["dw"])   

    
    expansions_p0[tuple(seed_roi_p0)] = best_angle_p0
    expansions_p10[tuple(seed_roi_p10)] = best_angle_p10
    
    
    # Iterate over the seed ROI walkout to estimate errors
    for j in range(0, num_iter):
    
        seed_roi_p0[0] -= increment
        seed_roi_p0[1] += increment

        seed_roi_p10[0] -= increment
        seed_roi_p10[1] += increment
        
        best_angle_p0, min_rms_p0, angles_deg_p0, rms_values_p0, data_rotated_p0, rms_values_init_p0 = optimize_rotation_rms_NoFiles(spec1=test_data_0, spec2=test_data_10, 
                                                                                                                                     ppm_region=seed_roi_p0, step_deg=0.001,
                                                                                                                                     plot=False, 
                                                                                                                                     plotname="NULL0", figsize=(12,4),
                                                                                                                                     nc_proc=parameters["nc_proc"], 
                                                                                                                                     offset_ppm=parameters["OFFSET"], 
                                                                                                                                     dw=parameters["dw"])    
        
        best_angle_p10, min_rms_p10, angles_deg_p10, rms_values_p10, data_rotated_p10, rms_values_init_p10 = optimize_rotation_rms_NoFiles(spec1=test_data_0, spec2=test_data_10, 
                                                                                                                                           ppm_region=seed_roi_p10, step_deg=0.001, 
                                                                                                                                           plot=False, 
                                                                                                                                           plotname="NULL10", figsize=(12,4),
                                                                                                                                           nc_proc=parameters["nc_proc"], 
                                                                                                                                           offset_ppm=parameters["OFFSET"], 
                                                                                                                                           dw=parameters["dw"])     
    
        expansions_p0[tuple(seed_roi_p0)] = best_angle_p0
        expansions_p10[tuple(seed_roi_p10)] = best_angle_p10

        
    # convert dictionaries to pd.DataFrame
    expansions_p0 = pd.DataFrame.from_dict(expansions_p0, orient="index").sort_index()
    expansions_p10 = pd.DataFrame.from_dict(expansions_p10, orient="index").sort_index()

    expansions_p0["width"] = [i[1] - i[0] for i in expansions_p0.index]
    expansions_p10["width"] = [i[1] - i[0] for i in expansions_p10.index]

    expansions_p0.columns = ["estimate", "width"]
    expansions_p10.columns = ["estimate", "width"]


    expansions_p0["AbsErr"] = theta_noshift - expansions_p0["estimate"]
    expansions_p10["AbsErr"] = theta_shift - expansions_p10["estimate"]

    # Get min errors from the tables
    min_err_focal = np.min(expansions_p10["AbsErr"])
    min_err_nops = np.min(expansions_p0["AbsErr"])


    if plot == True:
        ## Plot the simulated data ##
        ppm = get_ppm_scale_manual(offset_ppm=parameters["OFFSET"], sw_Hz=1/parameters["dw"], sf_MHz=parameters["bf"], si=len(test_data_0[0]))
        
        plt.plot(ppm, test_data_0[0], color="blue", alpha=0.8)
        plt.plot(ppm, test_data_10[0], color="red", alpha=0.8)
        plt.xlim(max([focal_center, center2])/parameters["bf"]+0.25, min([focal_center, center2])/parameters["bf"]-0.25)
        
        plt.legend({'No Phase Shift':"blue", '{} Phase Shift'.format(theta_shift):"red"})
        plt.xlabel("PPM")
        plt.ylabel("a.u."); 
        
        plt.axvline(focal_center/parameters["bf"], ls="--", color="black", lw=1)
        plt.axvline(center2/parameters["bf"], ls="--", color="black", lw=1)
        
        plt.savefig(output_folder + "/simulated-twopeak-spectra-{}.png".format(theta_shift), dpi=300)
        plt.show()
    
        
        ## Plot ROI error curves ##
        plt.scatter(np.log10(expansions_p10["width"]), expansions_p10["estimate"], s=5, color="red")
        plt.scatter(np.log10(expansions_p0["width"]), expansions_p0["estimate"], s=5, color="blue")
        
        plt.axhline(theta_shift, ls="--", lw=1, color="red")
        #plt.text(0.8*min(np.log10(expansions_p10["width"])), theta_shift, "True Phase Shift", color="red")
        
        plt.axhline(0, ls="--", lw=1, color="blue")
        #plt.text(0.8*min(np.log10(expansions_p0["width"])), theta_noshift, "True Phase Shift", color="blue")
        
        plt.legend({"{} PS".format(theta_shift):"red", "No PS":"blue"})
        plt.title("ROI vs Angle Estimate Curves (0 and {}deg PS)".format(theta_shift), fontsize=11)
        
        plt.xlabel("log$_{10}$(ROI width (ppm))")
        plt.ylabel("Estimated Rotation (degrees)");
        
        plt.savefig(output_folder + "/ROI-vs-BestAngle-2Peak-curves-log-{}.png".format(theta_shift), dpi=300)
        plt.show()

    return expansions_p0, expansions_p10, min_err_focal, min_err_nops
