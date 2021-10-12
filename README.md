# The GREAT Exposure Mapper Progrm

This program is for mapping exposure time, sensitivity (noise), simualted data, and signal-to-noise for the GREAT instrument on SOFIA.  It can be run either from your own machine or online in a webbrowser.  The code consists of the main GUI upyter notebook GUI.ipynb and the library sens_map_lib.py.

## To install on your own machine

Clone or download the master branch of the git repo to your machine.

Install Anaconda (https://www.anaconda.com/products/individual) if you do not already have it installed.

You can either install all the required python packages listed in environment.yml or you can create an anaconda environment.  To create an anaconda environment, go to this repo's directory on your machine and run the following in your terminal:
```
conda env create -f environment.yml
```

## To run on your own machine

Go to the git repo directory

If you created a conda enviroment from environment.yml (see above), run the following in your termianl to activate the environment:
```
conda activate the_great_sensitivty_mapper
```
