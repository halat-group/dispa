from dispa import load_pdata, get_ppm_scale, magnitude_transformation, rotate
from scipy.interpolate import interp1d
from scipy.signal import argrelmin
import numpy as np
import matplotlib.pyplot as plt
import os

def fidgen(shift, width, n, dw=0.001): 
    """Function to generate FID from specified shift, width using quadrature detection - i.e. real and imag
    are equal to amplitude multiplied by cos/sin(2pi*shift*t).
    
    Parameters

    ----------

    shift: float
        phase shift of FID
    width: float
        FWHM for FID peak
    n: int
        number of points in FID
    dw: float
        detection width (0.1 gives 10 Hz spectral width)

    Returns
    
    -------
    
    fid : numpy.array like
        The simulated fid
        
    """
    
    t = np.arange(0, n*dw, dw)
    
    fidamp = np.exp(-t*np.pi*width) # FID amplitude
    #exp decay of 1 Hz gives 1/pi Hz FWHM, so scale appropriately
    
    #now get real, imag components
    fidr = fidamp / 2 * np.cos(2 * np.pi * shift * t)  # carrier at 0 Hz, so shift - 0 = shift
    fidi = fidamp / 2 * np.sin(2 * np.pi * shift * t)
    fid = fidr + 1j*fidi

    return fid


def phaseshift(fid, dw, theta):
    """Function to apply corrected phase shift to FID.
    
    Parameters

    ----------

    fid: numpy.array like
        phase shift of FID
    dw: float
        dwell time (0.1 gives 10 Hz spectral width)
    theta: float
        angle (in degrees) to phase shift FID

    Returns
    
    -------
    
    fidps : numpy.array like
        The phase-shifted FID
        
    """
    
    # dw not used but prefer to 
    # specify as 2nd arg whenever fid used

    # phase shifts the FID by theta degrees
    fidps = fid*np.exp(-1j*theta*np.pi/180)

    return fidps


def specgen(fid, dw):
    """Function to generate complex spectrum from FID, assuming quadrature detection.
    
    Parameters

    ----------

    fid: numpy.array like
        phase shift of FID
    dw: float
        dwell time (0.1 gives 10 Hz spectral width)
    SNR: float
        desired signal-to-noise ratio

    Returns
    
    -------
    
    spec: numpy.array like
        The complex spectrum
    f: numpy.array like
        spectral frequency
        
    """
    
    # carrier implicitly assumed 0 Hz
    n = len(fid); # number of points in FID
    
    sp = (1/dw)/(n-1) # freq spacing in spectral points
    #f = np.arange(-(1/dw)/2-sp/2, (1/dw)/2-sp/2, sp) # spectral frequency
    # this includes a point at 0 freq, and one extra point at neg freq

    f = [-(1/dw)/2 - sp/2 + i * sp for i in range(n)]  # spectral frequency
        
    specpre = np.fft.fft(fid)
    
    spec = np.zeros_like(specpre)
    spec[:len(spec)//2] = specpre[len(specpre)//2:]
    spec[len(spec)//2:] = specpre[:len(specpre)//2]
    
    # no spectrum flipping - do this in plotting
    spec = np.flip(spec)
    f = np.flip(f)

    return spec, f
    
    
def addnoise(fid, dw, SNR):
    """Function to add Gaussian noise to simulated FID.
    
    Parameters

    ----------

    fid: numpy.array like
        input noiseless FID
    dw: float
        dwell time (0.1 gives 10 Hz spectral width)
    SNR: float
        desired signal-to-noise ratio

    Returns
    
    -------
    
    fid : numpy.array like
        The noise-added FID
        
    """
    
    n = len(fid)
    spec, f = specgen(fid, dw)
    
    # Use magnitude spectrum to be phase-independent
    # Multiply by sqrt(2) to be consistent with previous SNR definition
    maxi = np.max(np.abs(spec)) #* np.sqrt(2)

    # RMS of Gaussian noise (randn) is 1
    # SNR = max intensity / rms(noise) / 2
    # Scale further by sqrt(n) as we are adding to complex FID
    s = maxi / SNR / 2 / np.sqrt(n)

    rnoise = np.random.randn(1, n)[0] * s
    inoise = np.random.randn(1, n)[0] * s
    
    # Check deviation of rnoise, inoise from ideal
    ract = np.sqrt(np.mean(rnoise**2)) * np.sqrt(n)
    iact = np.sqrt(np.mean(inoise**2)) * np.sqrt(n)

    # Scale noise accordingly
    sr = (maxi / SNR / 2) / ract
    si = (maxi / SNR / 2) / iact

    rnoise = rnoise * sr
    inoise = inoise * si

    fid = fid + rnoise + 1j*inoise

    return fid
    
    
def fidcomb(fid):
    """Function to convert single complex FID to combined.
    
    Parameters

    ----------

    fid: numpy.array like
        input noiseless FID

    Returns
    
    -------
    
    fidc : numpy.array like
        The combined FID
    """
    
    # Takes single complex FID and converts to combined,
    # alternating real/imag format needed by TopSpin
    # Using same method as previous CPMG script: fid_lbCPMGecho_2018_01_working

    lenc = len(fid)*2
    fidr = np.real(fid) 
    fidi = np.imag(fid) 
    fidc = np.zeros(lenc, dtype=np.float64)
    
    for i in range(lenc // 2):
        fidc[2 * i] = fidr[i]
        fidc[2 * i + 1] = fidi[i]

    return fidc
    
    
# upgraded version of this function that ensures TopSpin compatibility
def write_fid_TS(fid, dw, bf, foldername):
    """Function to write simulated FID to file in TopSpin format (int32).
    
    Parameters
    ----------
    fid: numpy.array like
        Input noiseless FID (complex)
    dw: float
        Dwell time (s)
    bf: float
        Base frequency (MHz)
    foldername: str
        Output folder name
    """

    import numpy as np
    import os

    def fidcomb(fid):
        fidr = np.real(fid)
        fidi = np.imag(fid)
        fidc = np.zeros(len(fid) * 2, dtype=np.float64)
        for i in range(len(fid)):
            fidc[2*i] = fidr[i]
            fidc[2*i+1] = fidi[i]
        return fidc

    fidc = fidcomb(fid)
    td = len(fidc)

    os.makedirs(foldername, exist_ok=True)

    fidc_max = np.max(np.abs(fidc))
    if fidc_max == 0:
        raise ValueError("FID has zero amplitude.")
    fidc = fidc * (2**31 / fidc_max)
    fidc = np.clip(fidc, -2**31, 2**31 - 1)
    fidc.astype('<i4').tofile(os.path.join(foldername, 'fid'))  # little-endian int32

    # Write acqu/acqus with DTYPA = 0
    dw_TS = dw / 2 * 1e6
    sfo1 = bf
    sw = int(1e6 / (2 * dw_TS * sfo1))
    
    acqu = (
        "##TITLE=\n"
        "##$AQ_mod= 1\n"
        f"##$BF1= {bf}\n"
        "##$BYTORDA= 0\n"
        "##$DTYPA= 0\n"
        "##$NUC1= <1H>\n"
        "##$PARMODE= 0\n"
        f"##$SFO1= {sfo1}\n"
        f"##$SW= {sw}\n"
        f"##$TD= {td}\n"
        "##$NUCLEUS= <off>\n"
        "##$SOLVENT= <>\n"
        "##END="
    )

    with open(os.path.join(foldername, 'acqu'), 'w') as f_acqu:
        f_acqu.write(acqu)
    with open(os.path.join(foldername, 'acqus'), 'w') as f_acqus:
        f_acqus.write(acqu)

    # Write proc/procs
    si = int(2 ** np.ceil(np.log2(td))) * 8 # zero-filling x8
    sf = bf

    proc = (
        "##TITLE=\n"
        "##DATMOD = 1\n"
        "##$LB= 0\n"
        "##$PKNL= no\n"
        f"##$SF= {sf}\n"
        f"##$SI= {si}\n"
        "##END="
    )

    pdata_path = os.path.join(foldername, "pdata", "1")
    os.makedirs(pdata_path, exist_ok=True)

    with open(os.path.join(pdata_path, 'proc'), 'w') as f_proc:
        f_proc.write(proc)
    with open(os.path.join(pdata_path, 'procs'), 'w') as f_procs:
        f_procs.write(proc)
        

