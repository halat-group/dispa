# dispa
Python package for analyzing electrophoretic NMR data using the DISPA framework.

Example: 

Create a side-by-side plot showing the 1D spectra and matched 2D DISPA polar plots.

```python
from dispatools import overlay_plot

# Use the parse_dataset() function to load the example dataset from specified directory
op = "../example_data/OnePeak/"
tol = parse_proc_dataset(op)

# Create a list of colors to pass to the plotting function
colors = ["darkcyan","darkorange","indigo","violet"]

# Create dictionary of phase shifts for labeling the subplots
labels = {'1000':r"0$^\circ$", '1090':r"90$^\circ$", '1180':r"180$^\circ$", '1270':r"270$^\circ$"}

# Use the overlay_plot() function to generate and save the figure
overlay_plot(tol, "test_overlay_plot", colors = colors, labels = labels, 
                              threshold = 0.05, units_polar="a.u.", units_1d="Hz", figsize=(8, 5))

```
![](docs/test_overlay_plot.png)


Jupyter notebooks containing additional examples of plotting and data analysis can be found [here.](https://github.com/lcwheeler/dispatools/tree/main/notebooks)



## Install

Clone this repository and install a development version using `pip`:
```
pip install -e .
```
