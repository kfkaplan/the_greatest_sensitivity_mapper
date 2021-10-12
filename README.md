# The GREAT Exposure Mapper Program

This program is for mapping exposure time, sensitivity (noise), simulated data, and signal-to-noise for the GREAT instrument on SOFIA based on an uploaded .aor file for your project.  It can be run either from your own machine or online in a web-browser.  The code consists of the main GUI jupyter notebook GUI.ipynb and the library sens_map_lib.py.

Questions, comments, found a bug?  Email me at: kkaplan@usra.edu

## Using the GUI

The GUI notebook will open in a web-browser.  Activate the cell with the code by clicking it and pressing SHIFT+RETURN.  When it has finished running after a few seconds, scroll down and you should see the GUI.  Follow the instructions written on it.

## To run online using mybinder.org

The program can be conveniently run online using the service http://mybinder.org.    Note that if you find mybinder.org too slow, we recommend you install and run the GREAT Exposure Mapper Program on your own machine.

To run online, click the following link (when first run, it can take several minutes for the environment to build so please be patient):
https://mybinder.org/v2/gh/kfkaplan/the_greatest_sensitivity_mapper/ceb43748a0e37b8d985202fddbadc93b9661f057?urlpath=lab%2Ftree%2FGUI.ipynb

## To install on your own machine

Clone or download the master branch of the git repo to your machine.

Install Anaconda (https://www.anaconda.com/products/individual) if you do not already have it installed.

You can either install all the required python packages listed in environment.yml or you can create an anaconda environment.  To create an anaconda environment, go to this repo's directory on your machine and run the following in your terminal:
```
conda env create -f environment.yml
```

## To run on your own machine

Go to the git repo directory

If you created a conda environment from environment.yml (see above), run the following in your terminal to activate the environment:
```
conda activate the_great_sensitivty_mapper

```
Run the jupyter notebook GUI:
```
jupyter notebook GUI.ipynb
```



