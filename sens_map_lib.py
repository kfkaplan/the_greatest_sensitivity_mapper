#Library for the GREATest sensitivity mapper
#
#Created by Kyle Kaplan and Simon Coudé June 4, 2020
#

#Import python libraries 
import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting

#Class that stores the 2D array representing the sky and it's associated coordinate system
class sky:
	def __init__(self, max_x, max_y, plate_scale):
		nx = int(max_x/plate_scale)
		ny = int(max_y/plate_scale)
		data = np.zeros([nx, ny])
		y, x = np.mgrid[0:nx,0:ny]
		self.data = data #This is the actual 2D array that stores the model array profiles painted onto the sky
		self.x = x * plate_scale #2D x coords 
		self.y = y * plate_scale #2D y coords
	def paint(self, array_obj, integration_time=1.0): #Paint a single instance of an array profile onto the sky (e.g. a single pointing)
		self.data += array_obj.array_profile(self.x, self.y) * integration_time




#Parent class for storing the array profile for the LFA, HFA, and 4GREAT
class GREAT_array:
	def __init__(self):
		self.array_profile = None #Holder for array profile
	def rotate(self, angle): #Rotates entire array model, based off of https://docs.astropy.org/en/stable/modeling/compound-models.html#operators
		self.array_profile = models.Rotation2D(angle) | self.array_profile
	def position(self, x, y): #Move center of array to (x,y)
		for pixel in self.array_profile:
			try: #Just ignore pieces of the compound astropy model that can't undergo translation
				pixel.x_mean = pixel.x_mean + x
				pixel.y_mean = pixel.y_mean + y
			except:
				pass



#Child class to store array profile for the LFA
class LFA_array(GREAT_array):
	def __init__(self, fwhm=2.0, amplitude=1.0, r=31.7):
		stddev = fwhm / (2.0 * np.sqrt(np.log(2.0)))#Convert FWHM to stddev
		#Defines the profile for each pixel in a hexagon pattern, relative pixel positions in a hexagon are from https://www.quora.com/How-can-you-find-the-coordinates-in-a-hexagon
		pix0 =  models.Gaussian2D(amplitude=amplitude, x_mean=0.0, y_mean=0.0, x_stddev=stddev, y_stddev=stddev)
		pix1 =  models.Gaussian2D(amplitude=amplitude, x_mean=r/2.0, y_mean=-np.sqrt(3)*r/2.0, x_stddev=stddev, y_stddev=stddev)
		pix2 =  models.Gaussian2D(amplitude=amplitude, x_mean=r, y_mean=0.0, x_stddev=stddev, y_stddev=stddev)
		pix3 =  models.Gaussian2D(amplitude=amplitude, x_mean=r/2.0, y_mean=np.sqrt(3)*r/2.0, x_stddev=stddev, y_stddev=stddev)
		pix4 =  models.Gaussian2D(amplitude=amplitude, x_mean=-r/2.0, y_mean=np.sqrt(3)*r/2.0, x_stddev=stddev, y_stddev=stddev)
		pix5 =  models.Gaussian2D(amplitude=amplitude, x_mean=-r, y_mean=0.0, x_stddev=stddev, y_stddev=stddev)
		pix6 =  models.Gaussian2D(amplitude=amplitude, x_mean=-r/2.0, y_mean=-np.sqrt(3)*r/2.0, x_stddev=stddev, y_stddev=stddev)
		self.array_profile = pix0 + pix1 + pix2 + pix3 + pix4 + pix5 + pix6 #Generate an astropy compound model of the array profile
