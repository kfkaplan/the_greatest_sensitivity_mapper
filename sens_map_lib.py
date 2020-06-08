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
		y, x = np.mgrid[0:nx,0:ny] * plate_scale
		self.data = data #This is the actual 2D array that stores the model array profiles painted onto the sky
		self.x = x[:,::-1] #2D x coords (note the x coordinates inncrese to the left since they are RA)
		self.y = y #2D y coords
		self.extent = [np.max(x), np.min(x), np.min(y), np.max(y)]  #Gives the x and y coordinate extents for proper plotting using imshow
	def paint(self, array_obj, time=1.0): #Paint a single instance of an array profile onto the sky (e.g. a single pointing)
		self.data += array_obj.array_profile(self.x, self.y) * time
	def clear(self): #Erase everything on the grid
		self.data[:] = 0.





#Parent class for storing the array profile for the LFA, HFA, and 4GREAT
class GREAT_array:
	def __init__(self):
		self.array_profile = None #Holder for array profile
		self.angle = 0.
	def position(self, x, y): #Move center of array to (x,y)
		self.reset_position() #First zero position
		for pixel in self.array_profile:
			pixel.x_mean = pixel.x_mean + x
			pixel.y_mean = pixel.y_mean + y
	def reset_position(self): #Rezeros position around the 0th pixel
		zeroth_pixel = self.array_profile[0]
		dx = zeroth_pixel.x_mean.value
		dy = zeroth_pixel.y_mean.value
		for pixel in self.array_profile: #Move back position zero point
			pixel.x_mean = pixel.x_mean - dx
			pixel.y_mean = pixel.y_mean - dy
		self.angle = 0.
	def rotate(self, angle): #Rotate array by the given angle
		self.reset_array_rotation() #Zero rotation angle by reversing the previous rotation
		self.set_array_rotation(angle) #Set new rotation angle
	def set_array_rotation(self, angle): #Rotate the positions of all the pixels around the 0th pixel (angle is inputted in degrees)
		zeroth_pixel = self.array_profile[0] #Grab central pixel
		zero_point = np.array([zeroth_pixel.x_mean, zeroth_pixel.y_mean]).T #Store zero point of central pixel as a 
		c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle)) #Construct rotation matrix
		rot_matrix = np.array([[c, s], [-s, c]]).T
		for pixel in self.array_profile:
			position_vector = np.array([pixel.x_mean.value, pixel.y_mean.value]).T - zero_point #Cast pixel centroid position as a vector relative to the center
			position_vector = position_vector.dot(rot_matrix) #Apply rotation matrix
			new_position = (position_vector + zero_point).T  #Move centroid of pixel relative to the center
			pixel.x_mean = new_position[0]
			pixel.y_mean = new_position[1]
		self.angle += angle #Store overall angle of rotation
	def reset_array_rotation(self): #Rotate back to an angle of zero
		self.set_array_rotation(-self.angle)
	def paint(self, skyobj, time=1.0, cycles=1): #Paint a single instance of the array profile onto a sky object, this is the base for all observation types including single pointing and maps
		skyobj.data += self.array_profile(skyobj.x, skyobj.y) * time * cycles
	def single_point(self, skyobj, x=0., y=0., time=1.0, array_angle=0., cycles=1): #Paint a single point observation onto the sky object	
		self.rotate(array_angle) #Set rotation angle
		self.position(x, y) #Set central position for the single pointing
		self.paint(skyobj, time=time, cycles=cycles) #Paint the single pointing to the sky object
	def raster(self, skyobj, x=0., y=0., nx=1, ny=1, dx=1.0, dy=1.0, time=1.0, cycles=1, array_angle=0., map_angle=0.):
		self.rotate(array_angle) #Set rotation angle
		map_y, map_x = np.mgrid[0:ny,0:nx] #Generate map coordinates
		map_x = map_x * dx
		map_y = map_y * dy
		cos_map_angle = np.cos(np.radians(map_angle)) #Rotate map coordiunates by map angle (using a rotation matrix) and add starting position to map coordinates
		sin_map_angle = np.sin(np.radians(map_angle))
		rotated_map_x = x + (cos_map_angle*map_x + sin_map_angle*map_y)
		rotated_map_y = y + (-sin_map_angle*map_x + cos_map_angle*map_y)
		for ix in range(nx): #Loop through each step in the map
			for iy in range(ny):
			# 	print('x=', map_x[iy,ix])
			# 	print('y=', map_y[iy,ix])
				self.position(rotated_map_x[iy,ix], rotated_map_y[iy,ix]) #Set array position at this step
				self.paint(skyobj, time=time, cycles=cycles) #Paint the current step to the sky object
				

#Child class to store array profile for the LFA
class LFA_array(GREAT_array):
	def __init__(self, fwhm=15.85, amplitude=1.0, r=31.7, angle=0.):
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
		self.angle = 0.
		self.rotate(angle)		
		
