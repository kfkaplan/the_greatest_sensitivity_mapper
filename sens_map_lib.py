#Library for the GREATest sensitivity mapper
#
#Created by Kyle Kaplan and Simon Coudé June 4, 2020
#

#Import python libraries 
import numpy as np
from matplotlib import pyplot
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.coordinates import SkyCoord
import xmltodict #For reading in AORs

#import timeit #used for profiling code


#Honeycomb pattern from Randolf Klein as a vector with [x,y], The unit is one fifth of the distance between the pixels in each array. 
#That distance is 13.8” for the HFA and 31.7” for the LFA. Thus, to get to arc seconds multiply the offsets with 2.76” or 6.34” for the
#HFA or LFA, respectively. In the y-direction, you see multiples of 0.86603. That is actually sqrt(3.)/2.
honeycomb_pattern = np.array([ 
	[0.00000, 0.0000], #1
	[-1.00000, 0.00000], #2
	[-0.50000, 0.86603], #3
	[0.50000, 0.86603], #4
	[1.00000, 0.00000], #5
	[0.50000, -0.86603], #6
	[-0.50000, -0.86603], #7
	[-1.50000, -0.86603], #8
	[-2.00000, 0.00000], #9
	[-1.50000, 0.86603], #10
	[-2.00000, 1.73205], #11
	[-1.00000, 1.73205], #12
	[-0.50000, 2.59808], #13
	[-0.00000, 1.73205], #14
	[1.00000, 1.73205], #15
	[1.50000, 0.86603], #16
	[2.50000, 0.86603], #17
	[2.00000, 0.00000], #18
	[2.50000, -0.86603], #19
	[1.50000, -0.86603], #20
	[1.00000, -1.73205], #21
	[0.00000, -1.73205], #22
	[-0.50000, -2.59808], #23
	[-1.00000, -1.73205], #24
	[-2.00000, -1.73205], #25
	])



#Find nearest value in an array
def find_nearest(arr, value):
    return (np.abs(arr - value)).argmin()


#Reads in a .aor file and returns aor objects (see aor class below) for each aor in the file
def open_aors(filename):
	#The following code from Ed Chambers' aor_inpar_translator.py was plundered on the high seas by the nortorious software pirate knwon as Kyle Kaplan
	#who then proceeded to modify it without approval of the Gov'ner or the Crown™.
	aor_file = open(filename, 'rb') #Open .aor file
	xml_dict = xmltodict.parse(aor_file) #Parse the xml from the .aor file into python dictionaries so we can grab information out of them
	aor_file.close() #close .aor file
	aor_dict_list = []
	if type(xml_dict['AORs']['list']['vector']['Request']) == list: #multiple aors in .aor file, 
		for aor_dict in xml_dict['AORs']['list']['vector']['Request']:
			aor_dict_list.append(aor_dict)
	else:
		# one aor in .aor file
		aor_dict = xml_dict['AORs']['list']['vector']['Request']
		aor_dict_list.append(aor_dict)
	aor_list = []
	for aor_dict in aor_dict_list: #Loop through each aor python dictionary and create an aor object that stores the relavent information
		aor_list.append(aor(aor_dict))
	return aor_list




#Class that the relavent array into and information necessory to paint an array onto the sky
class aor:
	def __init__(self, aor_dict): #Contstruct aor object
		instr_data = aor_dict['instrument']['data']
		# self.array1 = HFA_array()
		# if instr_data['InstrumentSpectralElement2'] == 'GRE_LFA':
		# 	self.array2 = LFA_array()
		# else: #4GREAT
		# 	self.array2 = FOURGREAT_array()
		self.map_type = aor_dict['instrument']['@class'].split('.')[-1] #'GREAT_SP', 'GREAT_Raster', 'GREAT_OTF', OR 'GREAT_ON_THE_FLY_HONEYCOMB_MAP'
		self.cycles = float(instr_data['Repeat'])
		self.array_angle = float(instr_data['ArrayRotationAngle'])
		if self.map_type == 'GREAT_SP':
			self.time = 0.5 * float(instr_data['TotalTime'])
			self.x = float(instr_data['TargetOffsetRA'])
			self.y = float(instr_data['TargetOffsetDec'])
			#STUFF
		elif self.map_type == 'GREAT_Raster' or self.map_type == 'GREAT_OTF':
			self.dx = float(instr_data['MapStepSizeRA'])
			self.dy = float(instr_data['MapStepSizeDec'])
			self.nx = int(instr_data['NumStepRA'])
			self.ny = int(instr_data['NumStepDec'])
			self.time = float(instr_data['TimePerPoint'])
			self.map_angle = float(instr_data['MapRotationAngle'])
			self.x = float(instr_data['MapCenterOffsetRA'])
			self.y = float(instr_data['MapCenterOffsetDec'])
			#STUFF
		elif self.map_type == 'GREAT_ON_THE_FLY_HONEYCOMB_MAP':
			self.time = float(instr_data['TimePerPoint'])
			self.map_angle = float(instr_data['ArrayRotationAngle'])
			self.x = float(instr_data['TargetOffsetRA'])
			self.y = float(instr_data['TargetOffsetDec'])
		else:
			print('ERROR: '+self.map_type+' is not a valid map type to paint in the sky.')
	def paint(self, skyobj, which_array): #Paint AOR onto sky object with specified array ("HFA", "LFA", or "4GREAT")
		# #Determine which array to use and generate the appropriate object
		if which_array.upper() == 'HFA':
			array_obj = HFA_array()
		elif which_array.upper() == 'LFA':
			array_obj = LFA_array()
		elif which_array.upper() == '4G1':
			array_obj = FG1_array()
		elif which_array.upper() == '4G2':
			array_obj = FG2_array()
		elif which_array.upper() == '4G3':
			array_obj = FG3_array()
		elif which_array.upper() == '4G4':
			array_obj = FG4_array()
		else:
			print('ERROR: '+which_array+' is not a valid array. Please set to be either HFA, LFA, 4G1, 4G2, 4G3, or 4G4')
			return
		#Determine the map type then paint the array
		if self.map_type == 'GREAT_SP':
			array_obj.single_point(skyobj, x=self.x, y=self.y, time=self.time, array_angle=self.array_angle, cycles=self.cycles)
		elif self.map_type == 'GREAT_Raster' or self.map_type == 'GREAT_OTF':
			array_obj.map(skyobj, x=self.x, y=self.y, nx=self.nx, ny=self.ny, dx=self.dx, dy=self.dy, array_angle=self.array_angle, cycles=self.cycles, time=self.time)
		elif self.map_type == 'GREAT_ON_THE_FLY_HONEYCOMB_MAP':
			array_obj.honeycomb(skyobj, x=self.x, y=self.y, array_angle=self.array_angle, map_angle=self.map_angle, cycles=self.cycles, time=self.time)



#Class that stores the 2D array representing the sky and it's associated coordinate system
class sky:
	def __init__(self, x_range, y_range, plate_scale):
		if np.size(x_range) == 1:
			x_range = [0.0, x_range]
		if np.size(y_range) == 1:
			y_range = [0.0, y_range]
		nx = int((x_range[1]-x_range[0])/plate_scale)
		ny = int((y_range[1]-y_range[0])/plate_scale)
		data = np.zeros([nx, ny])
		y, x = np.mgrid[0:nx,0:ny] * plate_scale
		x += x_range[0]
		y += y_range[0]
		self.data = data #This is the actual 2D array that stores the model array profiles painted onto the sky
		self.x = x[:,::-1] #2D x coords (note the x coordinates inncrese to the left since they are RA)
		self.y = y #2D y coords
		self.x_1d = self.x[0,:] #Grab 1D arrays for x and y coordinates
		self.y_1d = self.y[:,0]
		self.extent = [np.max(x), np.min(x), np.min(y), np.max(y)]  #Gives the x and y coordinate extents for proper plotting using imshow
		self.total_exptime = 0. #Store total cumulative exposure time which is incremented every time a map is added
		self.pixel_area = plate_scale**2 #Area per pixel, used in calculating (exptime/arcsec^2)
		#self.set_sky_coords(0.0, 0.0)
	def clear(self): #Erase everything on the grid
		self.data[:] = 0.
	def get_range_indicies(self, xmin, xmax, ymin, ymax): #Returns the index numbers for a range of (xmin, xmax, ymin, ymax)
		ixmin = find_nearest(xmin, self.x_1d)
		ixmax = find_nearest(xmax, self.x_1d)
		iymin = find_nearest(ymin, self.y_1d)
		iymax = find_nearest(ymax, self.y_1d)
		return ixmin, ixmax, iymin, iymax
	def plot(self, **kwargs): #Generate an expsoure map plot
		self.normalize()
		pyplot.imshow(self.data, origin='bottom', extent=self.extent, **kwargs)
		pyplot.xlabel('Relative RA (arcsec)')
		pyplot.ylabel('Relative Dec. (arcsec)')
		pyplot.colorbar(label='s arcsec$^{-2}$')
	def normalize(self): #Normalize entire sky map by the total exposure time and the pixel size so the resulting values are in units of (exptime/arcsec^2)
		self.data *= (self.total_exptime/self.pixel_area) / np.nansum(self.data)
	#def set_sky_coords(self, ra, dec): #Set the sky coordinates for the 0,0 point using astropy.coordinates SkyCoord
	#	self.coords = SkyCoord(ra, dec, frame='icrs', unit ='arcsec')





#Parent class for storing the array profile for the LFA, HFA, and 4GREAT
class GREAT_array:
	def __init__(self):
		self.array_profile = None #Holder for array profile
		self.angle = 0.
		self.type = ''
		self.range = 100.0 #Maximum range in arcsec +/- x and y to paint, this is for optimization
		self.x = 0.
		self.y = 0.
	def position(self, x, y): #Move center of array to (x,y)
		self.reset_position() #First zero position
		for pixel in self.array_profile:
			pixel.x_mean = pixel.x_mean + x
			pixel.y_mean = pixel.y_mean + y
		self.x = x
		self.y = y
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
		self.angle = 0.
	def paint(self, skyobj, time=1.0, cycles=1): #Paint a single instance of the array profile onto a sky object, this is the base for all observation types including single pointing and maps
		skyobj.total_exptime += time*cycles #Add exposure time from this to the total
		sky_xmax, sky_xmin, sky_ymin, sky_ymax = skyobj.extent #Grab limits of the sky coordinates
		paint_xmin, paint_xmax = self.x - self.range, self.x + self.range #Calculate coordinate range to paint (this is for optimization)
		paint_ymin, paint_ymax = self.y - self.range, self.y + self.range
		if paint_xmin < sky_xmin: paint_xmin = sky_xmin #Bring coordinate ranges within bounds if they fall outside of sky object's bounds
		if paint_ymin < sky_ymin: paint_ymin = sky_ymin
		if paint_xmax > sky_xmax: paint_xmax = sky_xmax
		if paint_ymax > sky_ymax: paint_ymax = sky_ymax
		ix2, ix1, iy1, iy2 = skyobj.get_range_indicies(paint_xmin, paint_xmax, paint_ymin, paint_ymax)  #Grab the indicies for the pixels on the sky over witch to paint onto (NOTE: x axis is inverted because RA increases to the left)
		if np.size(self.array_profile) > 1: #If LFA or HFA
			skyobj.data[iy1:iy2, ix1:ix2] += self.array_profile(skyobj.x[iy1:iy2, ix1:ix2], skyobj.y[iy1:iy2, ix1:ix2]) * time * cycles #Paint pattern onto the sky
		else: #Else if 4GREAT
			skyobj.data[iy1:iy2, ix1:ix2] += self.array_profile[0](skyobj.x[iy1:iy2, ix1:ix2], skyobj.y[iy1:iy2, ix1:ix2]) * time * cycles #Paint pattern onto the sky
	def single_point(self, skyobj, x=0., y=0., time=1.0, array_angle=0., cycles=1): #Paint a single point observation onto the sky object	
		self.rotate(array_angle) #Set rotation angle
		self.position(x, y) #Set central position for the single pointing
		self.paint(skyobj, time=time, cycles=cycles) #Paint the single pointing to the sky object
	def map(self, skyobj, x=0., y=0., nx=1, ny=1, dx=1.0, dy=1.0, time=1.0, cycles=1, array_angle=0., map_angle=0.): #Paint a raster or OFT map observation onto the sky object
		self.rotate(array_angle) #Set rotation angle
		map_y, map_x = np.mgrid[0:ny,0:nx] #Generate map coordinates
		map_x = (map_x - 0.5*(nx-1.0))*dx #Center map and scale map coordinates to the proper step size
		map_y = (map_y - 0.5*(ny-1.0))*dy
		cos_map_angle = np.cos(np.radians(map_angle)) #Rotate map coordiunates by map angle (using a rotation matrix) and add starting position to map coordinates
		sin_map_angle = np.sin(np.radians(map_angle))
		rotated_map_x = x + (cos_map_angle*map_x + sin_map_angle*map_y)
		rotated_map_y = y + (-sin_map_angle*map_x + cos_map_angle*map_y)
		for ix in range(nx): #Loop through each step in the map
			for iy in range(ny):
				self.position(rotated_map_x[iy,ix], rotated_map_y[iy,ix]) #Set array position at this step
				self.paint(skyobj, time=time, cycles=cycles) #Paint the current step to the sky object
	def honeycomb(self, skyobj, x=0., y=0., time=1.0, cycles=1, array_angle=0., map_angle=0.):
		self.rotate(array_angle) #Set rotation angle
		if self.type == 'LFA': #Set multiplier for honeycomb offsets based on which array you are using
			honeycomb_multiplier = 6.34
		elif self.type == 'HFA':
			honeycomb_multiplier = 2.76
		c, s = np.cos(np.radians(map_angle)), np.sin(np.radians(map_angle)) #Construct rotation matrix
		rot_matrix = np.array([[c, s], [-s, c]]).T
		honeycomb_map_positions = np.array([x,y]).T + (honeycomb_pattern*honeycomb_multiplier).dot(rot_matrix) #Construct a vector of honeycomb positions and dot it with the rotation matrix
		for honeycomb_map_position in honeycomb_map_positions:
			self.position(honeycomb_map_position[0], honeycomb_map_position[1]) #Set array position at this step
			self.paint(skyobj, time=time, cycles=cycles) #Paint the current step to the sky object			



#Child class to store array profile for the LFA
class LFA_array(GREAT_array):
	def __init__(self, fwhm=14.1, amplitude=1.0, r=31.8, angle=0.):
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
		self.type = 'LFA'
		self.x = 0.
		self.y = 0.
		self.range = 2.0 * r #Maximum range in arcsec +/- x and y to paint, this is for optimization
		
#Child class to store array profile for the HFA
class HFA_array(GREAT_array):
	def __init__(self, fwhm=6.3, amplitude=1.0, r=13.6, angle=0.):
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
		self.type = 'HFA'
		self.x = 0.
		self.y = 0.
		self.range = 2.0 * r #Maximum range in arcsec +/- x and y to paint, this is for optimization



#Child class to store array profile for 4GREAT 1, FWHM is from the ICD
class FG1_array(GREAT_array):
	def __init__(self, fwhm=50.0, type='4G1', amplitude=1.0):
		stddev = fwhm / (2.0 * np.sqrt(np.log(2.0)))#Convert FWHM to stddev
		pix0 =  models.Gaussian2D(amplitude=amplitude, x_mean=0.0, y_mean=0.0, x_stddev=stddev, y_stddev=stddev) #Define pixel profile
		self.array_profile = [pix0]
		self.angle = 0.
		self.rotate(self.angle)
		self.type = type
		self.x = 0.
		self.y = 0.
		self.range = 3.0*fwhm #Maximum range in arcsec +/- x and y to paint, this is for optimization

#Child class to store array profile for 4GREAT 1, FWHM is from the ICD
class FG2_array(GREAT_array):
	def __init__(self, fwhm=30.0, type='4G2', amplitude=1.0):
		stddev = fwhm / (2.0 * np.sqrt(np.log(2.0)))#Convert FWHM to stddev
		pix0 =  models.Gaussian2D(amplitude=amplitude, x_mean=0.0, y_mean=0.0, x_stddev=stddev, y_stddev=stddev) #Define pixel profile
		self.array_profile = [pix0]
		self.angle = 0.
		self.rotate(self.angle)
		self.type = type
		self.x = 0.
		self.y = 0.
		self.range = 3.0*fwhm #Maximum range in arcsec +/- x and y to paint, this is for optimization

#Child class to store array profile for 4GREAT 1, FWHM is from the ICD
class FG3_array(GREAT_array):
	def __init__(self, fwhm=19.0, type='4G3', amplitude=1.0):
		stddev = fwhm / (2.0 * np.sqrt(np.log(2.0)))#Convert FWHM to stddev
		pix0 =  models.Gaussian2D(amplitude=amplitude, x_mean=0.0, y_mean=0.0, x_stddev=stddev, y_stddev=stddev) #Define pixel profile
		self.array_profile = [pix0]
		self.angle = 0.
		self.rotate(self.angle)
		self.type = type
		self.x = 0.
		self.y = 0.
		self.range = 3.0*fwhm #Maximum range in arcsec +/- x and y to paint, this is for optimization

#Child class to store array profile for 4GREAT 1, FWHM is from the ICD
class FG4_array(GREAT_array):
	def __init__(self, fwhm=11.0, type='4G4', amplitude=1.0):
		stddev = fwhm / (2.0 * np.sqrt(np.log(2.0)))#Convert FWHM to stddev
		pix0 =  models.Gaussian2D(amplitude=amplitude, x_mean=0.0, y_mean=0.0, x_stddev=stddev, y_stddev=stddev) #Define pixel profile
		self.array_profile = [pix0]
		self.angle = 0.
		self.rotate(self.angle)
		self.type = type
		self.x = 0.
		self.y = 0.
		self.range = 3.0*fwhm #Maximum range in arcsec +/- x and y to paint, this is for optimization



