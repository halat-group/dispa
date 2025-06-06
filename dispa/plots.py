import numpy as np
import nmrglue as ng
import matplotlib.pyplot as plt
import pandas as pd
from dispa import get_ppm_scale

def polar_plot(data, filename, color, frame=True, units="a.u."):
    """ Function to plot the DISPA polar plot using real and imaginary components of NMR signal. 

     Parameters

        ----------
        
        data: numpy.array
            numpy array object from NMRGlue read_pdata 
        filename: str
            name of file to save plot
        color: str
            color of DISPA polar plot line
        frame: bool
            include right and top frame lines or not
        units: str
            units of measurement for axes

    """


    # Plot the real vs imaginary components of the data to generate the DISPA circular plot
    if frame==True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        
        # add dashed lines through the origin
        plt.axhline(0, lw=1, linestyle="--", color="gray")
        plt.axvline(0, lw=1, linestyle="--", color="gray")

        # plot real and imaginary dimensions
        ax.plot(data[0], data[1], color=color); 

        # label axes
        plt.xlabel("Real"+ " " + "(" + units + ")", fontsize=12)
        plt.ylabel("Imaginary"+ " " + "(" + units + ")", fontsize=12);
        
        # set aspect to equal for square plot
        plt.gca().set_aspect("equal")

        # save figure in png and pdf format
        plt.savefig(filename+".png", dpi=300)
        plt.savefig(filename+".pdf")

    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        # add dashed lines through the origin
        plt.axhline(0, lw=1, linestyle="--", color="gray")
        plt.axvline(0, lw=1, linestyle="--", color="gray")

        # plot real and imaginary dimensions
        ax.plot(data[0], data[1], color=color);
        
        # turn off the top and right lines of border frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # label axes
        plt.xlabel("Real"+ " " + "(" + units + ")", fontsize=12)
        plt.ylabel("Imaginary"+ " " + "(" + units + ")", fontsize=12);

        # set aspect to equal for square plot
        plt.gca().set_aspect("equal")

        # save figure in png and pdf format
        plt.savefig(filename+".png", dpi=300)
        plt.savefig(filename+".pdf")
     
   
def panel_plot(experiments, plotname,  labels = None, threshold = 0.05, units_polar="a.u.", units_1d="Hz", 
               figsize=(8, 18), color="midnightblue"):
    """Function to generate side-by-side visual comparisons of 1D and polar plots for a set of NMR datasets.
    
     Parameters

    ----------
    
    experiments: dict
        dictionary of processed data from NMR experiments generated by dispatools.utils.parse_dataset()
    plotname: str
        name of file (without extension) to save figure into
    labels: dict
        optional dictionary mapping experiment directory names to label names for 1D spectra
    threshold: float
        fractional threshold for calling peaks/relevant data region
    units_polar: str
        units of the intensity on polar plots
    units_1d: str
        units of frequency (ppm or Hz) for spectra axis
    figsize: tuple
        dimensions for the figure (inches)
    color: str
        color for the 1D and polar plot traces
            
    """

    # Create a list of the keys from experiments to use as labels
    experiments_ord = list(sorted(experiments.keys()))

    # load the ascii spectrum to use for x axis scale
    #axis_df = pd.read_csv(axis_scale, sep=",", header=None)
    #axis_df.columns = ["point_number", "intensity", "Hz", "ppm"]

    # Set up the figure and gridspec
    nrows = len(experiments_ord)
    ncols = 2
    
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols)

    # Detect shared axes limits by finding widest region containing peaks
    # Detect x limits for 1D plots (simple peak finding by a threshold of max)
    maxes = []
    mins = []
    for j in range(0,len(experiments_ord)):
        r = np.abs(experiments[experiments_ord[j]][1][0])
        
        peaks = np.where(r >= threshold*np.max(r))
        mins.append(np.min(peaks))
        maxes.append(np.max(peaks))
        
    # Find the global min and max indexes
    MIN = np.min(mins)
    MAX = np.max(maxes)

    
    # Iterate through the gridspec and add the individual plots
    for i in range(0, gs.nrows):
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1])
        
        # get the real and imaginary components from the individual experiments
        dr = experiments[experiments_ord[i]][1][0]
        di = experiments[experiments_ord[i]][1][1]
        sf_MHz = experiments[experiments_ord[i]][0]["procs"]["SF"]
        ppm = get_ppm_scale(experiments[experiments_ord[i]][0])
        Hz = ppm*sf_MHz
        
        # extract the 1D spectra labels from dictionary
        if labels != None:
            try:
                label = labels[experiments_ord[i]]
            except:
                label = experiments_ord[i]
                print("Labels dict is not formatted correctly")
        else:
            label = experiments_ord[i]

        
        ## plot the 1D spectra ##
        # plot the spectrum against calculated reference x scale
        if units_1d == "Hz":
            ax1.plot(Hz[MIN:MAX], dr[MIN:MAX], lw=1, color=color)
        elif  units_1d == "ppm":
            ax1.plot(ppm[MIN:MAX], dr[MIN:MAX], lw=1, color=color)
        else:
            print("1D spectra units not properly specified. Using Hz.")
            units_1d = "Hz"
            ax1.plot(Hz[MIN:MAX], dr[MIN:MAX], lw=1, color=color)

            
        #ax1.plot(axis_df.iloc[MIN:MAX][units_1d], dr[MIN:MAX], lw=1, color=color)
        # shut off the top, right, and left spines for 1D spectra
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        #ax1.set_xlim(MIN, MAX)
        
        ax1.set_title(label, fontsize=11, color="midnightblue")
        ax1.yaxis.set_visible(False)
        ax1.invert_xaxis()
        # Add the x-axis label
        ax1.set_xlabel(units_1d, fontsize=10); 
        
        ## plot the polar i vs r plots ##
        ax2.plot(dr,di, lw=1, color=color)
        plt.axhline(0, lw=0.8, linestyle="--", color="gray")
        plt.axvline(0, lw=0.8, linestyle="--", color="gray")
        # turn off the top and right lines of border frame
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        # label and style the axes
        ax2.axes.get_xaxis().set_ticks([])
        ax2.axes.get_yaxis().set_ticks([])
        ax2.set_xlabel("Real"+ " " + "(" + units_polar + ")", fontsize=10)
        ax2.set_ylabel("Imaginary"+ " " + "(" + units_polar + ")", fontsize=10);
        
        ax2.set_aspect("equal"); 
    
    plt.savefig(plotname+".pdf")
    plt.savefig(plotname+".png", dpi=300)
    
    
def overlay_plot(experiments, plotname, colors, labels = None, threshold = 0.05, units_polar="a.u.", units_1d="Hz", figsize=(4, 9)):  
    """Function to generate visual comparisons of 1D and polar plots with shared axes.
    
     Parameters

    ----------
    
    experiments: dict
        dictionary of processed data from NMR experiments generated by dispatools.utils.parse_dataset()
    plotname: str
        name of file (without extension) to save figure into
    colors: list
        list of colors with length equal to number of invididual spectra
    labels: dict
        optional dictionary mapping experiment directory names to label names for 1D spectra
    threshold: float
        fractional threshold for calling peaks/relevant data region
    units_polar: str
        units of the intensity on polar plots
    units_1d: str
        units of frequency (ppm or Hz) for spectra axis
    figsize: tuple
        dimensions for the figure (inches)
    """


    # Create a list of the keys from experiments to use as labels
    experiments_ord = list(sorted(experiments.keys()))

    # Set up the figure and gridspec
    nrows = len(experiments_ord)
    ncols = 2
    
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols)
    
    
    ## Plot the polar plot ##
    # This plot takes up the entire righthand side of the gridspec (all rows)
    f_ax1 = fig.add_subplot(gs[0:,1:])
    f_ax1.set_title('Polar Plot')
    
    # plot real and imaginary dimensions
    for i in range(0, gs.nrows):
        # get the real and imaginary components from the individual experiments
        dr = experiments[experiments_ord[i]][1][0]
        di = experiments[experiments_ord[i]][1][1]
        f_ax1.plot(dr, di, color=colors[i]);

    # add dashed lines through the origin
    plt.axhline(0, lw=1, linestyle="--", color="gray")
    plt.axvline(0, lw=1, linestyle="--", color="gray")
    
    # turn off the top and right lines of border frame
    f_ax1.spines['top'].set_visible(False)
    f_ax1.spines['right'].set_visible(False)
    
    # label axes
    plt.xlabel("Real"+ " " + "(" + units_polar + ")", fontsize=9)
    plt.ylabel("Imaginary"+ " " + "(" + units_polar + ")", fontsize=9);
    
    # set aspect to equal for square plot
    plt.gca().set_aspect("equal"); 
    
        
    ## Plot the individual stacked 1D plots ##

    # Detect shared x limits for 1D plots (simple peak finding by a threshold of max)
    maxes = []
    mins = []
    for j in range(0,len(experiments_ord)):
        r = np.abs(experiments[experiments_ord[j]][1][0])
        
        peaks = np.where(r >= threshold*np.max(r))
        mins.append(np.min(peaks))
        maxes.append(np.max(peaks))
        
    # Find the global min and max indexes
    MIN = np.min(mins)
    MAX = np.max(maxes) #this needs to account for negative peaks!


    # Iterate through the gridspec and add the individual plots
    for i in range(0, gs.nrows):
        # get the real and imaginary components from the individual experiments
        dr = experiments[experiments_ord[i]][1][0]
        di = experiments[experiments_ord[i]][1][1]
        sf_MHz = experiments[experiments_ord[i]][0]["procs"]["SF"]
        ppm = get_ppm_scale(experiments[experiments_ord[i]][0])
        Hz = ppm*sf_MHz
        
        # extract the 1D spectra labels from dictionary
        if labels != None:
            try:
                label = labels[experiments_ord[i]]
            except:
                label = experiments_ord[i]
                print("Labels dict is not formatted correctly")
        else:
            label = experiments_ord[i]

        # add suplot for individual spectrum
        ax1 = fig.add_subplot(gs[i, 0])

        # plot the spectrum against user-input reference x scale
        if units_1d == "Hz":
            ax1.plot(Hz[MIN:MAX], dr[MIN:MAX], lw=1, color=colors[i])
        elif  units_1d == "ppm":
            ax1.plot(ppm[MIN:MAX], dr[MIN:MAX], lw=1, color=colors[i])
        else:
            print("1D spectra units not properly specified. Using Hz.")
            units_1d = "Hz"
            ax1.plot(Hz[MIN:MAX], dr[MIN:MAX], lw=1, color=colors[i])
        
        # turn off the top and right lines of border frame
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)

        # Set the title/label based on labels dictionary or experiment name
        ax1.set_title(label, fontsize=10, color=colors[i])
        
        # Style axes and axis labels
        ax1.yaxis.set_visible(False)
        ax1.invert_xaxis()
        ax1.set_xlabel(units_1d, fontsize=9); 
        
    # Save figure in png and pdf formats
    plt.savefig(plotname+".pdf")
    plt.savefig(plotname+".png", dpi=300)
