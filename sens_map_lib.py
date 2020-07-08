#Library for the GREATest sensitivity mapper
#
#Created by Kyle Kaplan and Simon Coudé June 4, 2020
#

#Import python libraries 
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.coordinates import SkyCoord
from astropy.nddata.utils import block_reduce
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


def fwhm2std(fwhm): #Convert FWHM to stddev (for a gaussian)
	return   fwhm / (2.0 * np.sqrt(np.log(2.0)))


def std2fwhm(stddev): #Convert stddev to FWHM  (for a gaussian)
	return   stddev * (2.0 * np.sqrt(np.log(2.0)))


# def get_deltaTa(Tsys=0., deltafreq=1.0, Non=1.0, time=1.0, TPOTF=False): #Get the RMS antenna temperature (delta-Ta) for a single pointing (when time on = time off)
# 	if not TPOTF: #If not a Total Power OTF map (most observations)...
# 		deltaTa = 2.0 * Tsys / (time * deltafreq)**0.5 #Calulate RMS temperature using Equation 6-5 in the observer's handbook
# 	else: #If a Total Power OTF map....
# 		deltaTa = Tsys * (1.0 + Non**-0.5)**0.5 / (time * deltafreq)*0.5 #Calculate RMS temp. for TP OTF maps
# 	return deltaTa
	


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
		frequencies = [] #Grab  frequencies 1,2,3,4,5
		frequencies.append(float(instr_data['Frequency'])) #HFA
		frequencies.append(float(instr_data['Frequency2'])) #4G1
		frequencies.append(float(instr_data['Frequency3'])) #4G2
		frequencies.append(float(instr_data['Frequency4'])) #4G3 or LFAH
		frequencies.append(float(instr_data['Frequency5'])) #4G4 or LFAV
		self.frequencies = frequencies #
		if self.map_type == 'GREAT_SP': #Grab mapping parameters
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
			array_obj.freq = self.frequencies[0]
		elif which_array.upper() == 'LFAH':
			array_obj = LFAH_array()
			array_obj.freq = self.frequencies[3]
		elif which_array.upper() == 'LFAV':
			array_obj = LFAV_array()
			array_obj.freq = self.frequencies[4]
		elif which_array.upper() == '4G1':
			array_obj = FG1_array()
			array_obj.freq = self.frequencies[1]
		elif which_array.upper() == '4G2':
			array_obj = FG2_array()
			array_obj.freq = self.frequencies[2]
		elif which_array.upper() == '4G3':
			array_obj = FG3_array()
			array_obj.freq = self.frequencies[3]
		elif which_array.upper() == '4G4':
			array_obj = FG4_array()
			array_obj.freq = self.frequencies[4]
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
	def __init__(self, x_range, y_range, plate_scale): #, Tsky=178.0, Ttel=188.0, deltaTa=0.):
		if np.size(x_range) == 1:
			x_range = [0.0, x_range]
		if np.size(y_range) == 1:
			y_range = [0.0, y_range]
		nx = int((x_range[1]-x_range[0])/plate_scale)
		ny = int((y_range[1]-y_range[0])/plate_scale)
		signal = np.zeros([nx, ny])
		data = np.zeros([nx, ny])
		noise = np.zeros([nx, ny])
		exptime = np.zeros([nx, ny])
		y, x = np.mgrid[0:nx,0:ny] * plate_scale
		x += x_range[0]
		y += y_range[0]
		self.x_range = x_range #Save x,y ranges and plate scale
		self.y_range = y_range
		self.plate_scale = plate_scale
		self.data = data #This is the actual 2D array that stores the simulated data (in units of T_a)
		self.noise = noise #This is the 2D noise array for the data (in units of delta-T_a)
		self.signal = signal #This is the the model "true" signal exepected signal from the sky stored in a (in units of T_a)
		self.exptime = exptime #Save total exposure time per pixel
		self.x = x[:,::-1] #2D x coords (note the x coordinates inncrese to the left since they are RA)
		self.y = y #2D y coords
		self.x_1d = self.x[0,:] #Grab 1D arrays for x and y coordinates
		self.y_1d = self.y[:,0]
		self.extent = [np.max(x), np.min(x), np.min(y), np.max(y)]  #Gives the x and y coordinate extents for proper plotting using imshow
		#self.total_exptime = 0. #Store total cumulative exposure time which is incremented every time a map is added
		#self.pixel_area = plate_scale**2 #Area per pixel, used in calculating (exptime/arcsec^2)
		self.fwhm = 0. #Store latest beam profile FWHM used on this sky object
		self.freq = 0. #Store latest frequency painted onto this sky object]
		#self.Tsys = Tsys
		#self.deltaTa = deltaTa
		#self.Tsky = Tsky #Ambient temperature for the atmosphere
		#self.Ttel = Ttel #Physical temperature of the telescope
		#self.set_sky_coords(0.0, 0.0)
	def clear(self): #Erase everything on the grid except for the signal
		self.data[:] = 0.
		self.exptime[:] = 0.
		self.noise[:] = 0.
	def get_range_indicies(self, xmin, xmax, ymin, ymax): #Returns the index numbers for a range of (xmin, xmax, ymin, ymax)
		ixmin = find_nearest(xmin, self.x_1d)
		ixmax = find_nearest(xmax, self.x_1d)
		iymin = find_nearest(ymin, self.y_1d)
		iymax = find_nearest(ymax, self.y_1d)
		return ixmin, ixmax, iymin, iymax
	def plot(self, map_type='data', **kwargs): #Generate an expsoure map plot
		# if np.any(self.signal != 0.): #Error catch to ensure we are not dividing by zero
		# 	self.normalize()
		if map_type == 'exposure':
			pyplot.imshow(self.exptime, origin='bottom', extent=self.extent, **kwargs)
			label = r'Time (s)'
			title = r'Exposure Time (s)'
		elif map_type == 'signal': # if signal is set to true, plot the modeled signal instead
			pyplot.imshow(self.signal, origin='bottom', extent=self.extent, **kwargs)
			label = r'T_a'
			title = r'Signal (T_a)'
		elif map_type == 'noise':
			min_noise = np.nanmin(self.noise)
			print(min_noise)
			pyplot.imshow(self.noise, origin='bottom', extent=self.extent, vmax=min_noise*1.5, vmin=min_noise, **kwargs)
			label = r'\Delta T_a'
			title = r'Noise (\Delta Ta)'
		elif map_type == 's2n': #If s2n is true, plot the signal-to-noise
			pyplot.imshow(self.data/self.noise, origin='bottom', extent=self.extent, **kwargs)
			label = r'S/N'
			title = r'S/N'
		else: #Normally plot the simulated data
			pyplot.imshow(self.data, origin='bottom', extent=self.extent, **kwargs)
			label = r'T_a'
			title = r'T_a'
		pyplot.suptitle(title)
		pyplot.xlabel('Relative RA (arcsec)')
		pyplot.ylabel('Relative Dec. (arcsec)')
		pyplot.colorbar(label=label)
	# def normalize(self): #Normalize entire sky map by the total exposure time and the pixel size so the resulting values are in units of (exptime/arcsec^2)
	# 	scale_by = (self.total_exptime/self.pixel_area) / np.nansum(self.data)
	# 	self.data *= scale_by
	# 	self.noise *= scale_by
	def simulate_observation(self, Tsys=0., TPOTF=False, Non=1.0, deltafreq=1.0): #Calculate noise and smooth the data and noisea by convolving with a 2D gausasian kernel with a FHWM that is 1/3 the beam profile, this is the final step for simulating data
		if not TPOTF: #If not a Total Power OTF map (most observations)...
			self.noise = 2.0 * Tsys / (self.exptime * deltafreq)**0.5 #Calulate RMS temperature using Equation 6-5 in the observer's handbook
		else: #If a Total Power OTF map....
			self.noise = Tsys * (1.0 + Non**-0.5)**0.5 / (self.exptime * deltafreq)*0.5 #Calculate RMS temp. for TP OTF maps
		stddev = fwhm2std(self.fwhm) / 3.0
		kernel = Gaussian2DKernel(x_stddev=stddev, y_stddev=stddev) #Define the gaussian kernel to be 1/3 the FWHM of the beam profile
		self.data = convolve(self.data, kernel) #Apply convolution to the data
		self.noise = convolve(self.noise**2, kernel)**0.5 #Apply convolution to the variance and convert back to noise
	def input(self, model_shape): #Draw an astropy model shape onto  sigal (e.g. create a model of the "true" signal)
		self.signal += model_shape(self.x, self.y)
	def downsample(self, factor): #Downsample sky grid by an integer factor (1/2, 1/3, 1/4, ect.) using the astropy function 
		plate_scale = self.plate_scale * factor #calculate new plate scale
		nx = int((self.x_range[1]-self.x_range[0])/plate_scale)
		ny = int((self.y_range[1]-self.y_range[0])/plate_scale)
		y, x = np.mgrid[0:nx,0:ny] * plate_scale
		x += self.x_range[0]
		y += self.y_range[0]
		self.plate_scale = plate_scale #Save the new plate scale and x,y coordinate 2D and 1D grids
		self.x = x[:,::-1] #2D x coords (note the x coordinates inncrese to the left since they are RA)
		self.y = y #2D y coords
		self.x_1d = self.x[0,:] #Grab 1D arrays for x and y coordinates
		self.y_1d = self.y[:,0]
		self.data = block_reduce(self.data, factor, func=np.mean) #Resample all grids using astropy block_reduce
		self.noise = block_reduce(self.noise**2, factor, func=np.mean)**0.5 / factor #Note the 1/factor is actually 1/sqrt(factor*^2) since factor^2 is the number of pixels being averaged together
		self.exptime = block_reduce(self.exptime, factor, func=np.mean)
		self.signal = block_reduce(self.signal, factor, func=np.mean)





#Parent class for storing the array profile for the LFA, HFA, and 4GREAT
class GREAT_array:
	def __init__(self):
		self.array_profile = None #Holder for array profile
		self.angle = 0.
		self.type = ''
		self.range = 100.0 #Maximum range in arcsec +/- x and y to paint, this is for optimization
		self.x = 0. #In arcsec
		self.y = 0. #In arcsec
		self.fwhm = 0. #In arcsec
		self.freq = 0. #In GHz
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
	def paint(self, skyobj, time=1.0, cycles=1, Tsys=0., TPOTF=False): #Paint a single instance of the array profile onto a sky object, this is the base for all observation types including single pointing and maps
		#skyobj.total_exptime += time*cycles #Add exposure time from this to the total
		# deltaTa = get_deltaTa(Tsys=Tsys, TPOTF=TPOTF) #Get RMS antenna temperature 
		sky_xmax, sky_xmin, sky_ymin, sky_ymax = skyobj.extent #Grab limits of the sky coordinates
		paint_xmin, paint_xmax = self.x - self.range, self.x + self.range #Calculate coordinate range to paint (this is for optimization)
		paint_ymin, paint_ymax = self.y - self.range, self.y + self.range
		if paint_xmin < sky_xmin: paint_xmin = sky_xmin #Bring coordinate ranges within bounds if they fall outside of sky object's bounds
		if paint_ymin < sky_ymin: paint_ymin = sky_ymin
		if paint_xmax > sky_xmax: paint_xmax = sky_xmax
		if paint_ymax > sky_ymax: paint_ymax = sky_ymax
		ix2, ix1, iy1, iy2 = skyobj.get_range_indicies(paint_xmin, paint_xmax, paint_ymin, paint_ymax)  #Grab the indicies for the pixels on the sky over witch to paint onto (NOTE: x axis is inverted because RA increases to the left)
		chunk_of_signal = skyobj.signal[iy1:iy2, ix1:ix2] #Isolate the chunk of the signal array for painting at this particular position
		chunk_of_noise = skyobj.noise[iy1:iy2, ix1:ix2]
		# if np.size(self.array_profile) > 1: #If LFA or HFA
		for this_array_profile in self.array_profile: #Loop through each individual pixel
			chunk_of_array_profile = this_array_profile(skyobj.x[iy1:iy2, ix1:ix2], skyobj.y[iy1:iy2, ix1:ix2]) #Isolate the piece of the array profile to use 
			sum_chunk_of_array_profile = np.nansum(chunk_of_array_profile)
			convolved_signal = chunk_of_array_profile *  np.nansum(chunk_of_signal * chunk_of_array_profile) / sum_chunk_of_array_profile
			skyobj.data[iy1:iy2, ix1:ix2] += convolved_signal * time * cycles #Convolve pattern with expected signal and paint result onto the sky
			skyobj.exptime[iy1:iy2, ix1:ix2] += time * cycles * chunk_of_array_profile #Convolve the exposure time with the profile for this particular pixel
		skyobj.fwhm = std2fwhm(self.array_profile[0].x_stddev) #Copy beam profile FWHM to sky object for later using to determine the convolution kernel to smooth with
		# else: #Else if 4GREAT
		# 	chunk_of_array_profile = self.array_profile(skyobj.x[iy1:iy2, ix1:ix2], skyobj.y[iy1:iy2, ix1:ix2]) #Isolate the piece of the array profile to use 
		# 	convolved_signal = chunk_of_array_profile * np.nansum(chunk_of_signal * chunk_of_array_profile) / np.nansum(chunk_of_array_profile) #Convolve the signal with the profile for this particular pixel
		# 	skyobj.data[iy1:iy2, ix1:ix2] += convolved_signal * time * cycles #Convolve pattern with expected signal and paint result onto the sky
		# 	skyobj.exptime[iy1:iy2, ix1:ix2] += time * cycles * chunk_of_array_profile #Convolve the exposure time with the profile for this particular pixel
		# 	skyobj.fwhm = std2fwhm(self.array_profile.x_stddev) #Copy beam profile FWHM to sky object for later using to determine the convolution kernel to smooth with
	def single_point(self, skyobj, x=0., y=0., time=1.0, array_angle=0., cycles=1, Tsys=0.): #Paint a single point observation onto the sky object	
		self.rotate(array_angle) #Set rotation angle
		self.position(x, y) #Set central position for the single pointing
		self.paint(skyobj, time=time, cycles=cycles, Tsys=Tsys) #Paint the single pointing to the sky object
	def map(self, skyobj, x=0., y=0., nx=1, ny=1, dx=1.0, dy=1.0, time=1.0, cycles=1, array_angle=0., map_angle=0., Tsys=0.): #Paint a raster or OFT map observation onto the sky object
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
				self.paint(skyobj, time=time, cycles=cycles, Tsys=Tsys) #Paint the current step to the sky object
	def honeycomb(self, skyobj, x=0., y=0., time=1.0, cycles=1, array_angle=0., map_angle=0., Tsys=0.):
		self.rotate(array_angle) #Set rotation angle
		if self.type == 'LFAV' or self.type == 'LFAH': #Set multiplier for honeycomb offsets based on which array you are using
			honeycomb_multiplier = 6.34
		elif self.type == 'HFA':
			honeycomb_multiplier = 2.76
		c, s = np.cos(np.radians(map_angle)), np.sin(np.radians(map_angle)) #Construct rotation matrix
		rot_matrix = np.array([[c, s], [-s, c]]).T
		honeycomb_map_positions = np.array([x,y]).T + (honeycomb_pattern*honeycomb_multiplier).dot(rot_matrix) #Construct a vector of honeycomb positions and dot it with the rotation matrix
		for honeycomb_map_position in honeycomb_map_positions:
			self.position(honeycomb_map_position[0], honeycomb_map_position[1]) #Set array position at this step
			self.paint(skyobj, time=time, cycles=cycles, Tsys=Tsys) #Paint the current step to the sky object
	# def get_deltaTa_singlepoint(self, deltafreq=1.0, time=1.0): #Get the RMS antenna temperature (delta-Ta) for a single pointing (when time on = time off)
	# 	deltaTa = 2.0 * self.get_Tsys() / (time * deltafreq)**0.5 #Equation 6-5 in the observer's handbook
	# 	return deltaTa
	# def get_detlaTa_TPOTF(self, deltafreq=1.0, time=1.0, Non=1.0: #Get RMS antenna temperature (delta-Ta) for OTF TP maps
	# 	deltaTa = self.get_Tsys() * (1.0 + Non**-0.5)**0.5 / (time * deltafreq)*0.5
	# def get_Tsys(self, ntel=0.92, nsky=1.0, Tsky=220.0, Ttel=230.0): #Get the single sideband system temperature (Tsys) for use in calculating the antenna temperature (delta Ta)
	# 	Trx = self.get_Trx() #Get singe sideband system temperature
	# 	Tsys = 2.9 * (Trx + ntel*Tsky + Ttel) / (ntel*nsky) #Equation 6-1 in the observer's handbook (cycle 9)
	# 	return Tsys #Return the single sideband system temperature
	# def get_Trx(self): #Get reciever temperature (Trx) from classTools tables
	# 	freq_arr, Trx_arr = np.loadtxt('reciever_temps/trx'+self.type+'.dat', unpack=True) #Load Trx table used in ClassTools
	# 	interp_obj = interp1d(freq_arr, Trx_arr, kind='linear') #Generate an interp1d object to linearly interpolate table
	# 	Trx = interp_obj(self.freq) #Grab Trx by interpolating the table using the frequency set for this array object
	# 	return Trx #Return reciever temperature based on this array object's frequency



#Child class to store array profile for the LFA
class LFAH_array(GREAT_array):
	def __init__(self, fwhm=14.1, type='LFAH', amplitude=1.0, r=31.8, angle=0., freq=1897.420):
		stddev = fwhm2std(fwhm) #Convert FWHM to stddev
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
		self.type = type
		self.x = 0.
		self.y = 0.
		self.range = 2.0 * r #Maximum range in arcsec +/- x and y to paint, this is for optimization
		self.fwhm = fwhm
		self.freq = freq


#Child class to store array profile for the LFA
class LFAV_array(GREAT_array):
	def __init__(self, fwhm=14.1, type='LFAV', amplitude=1.0, r=31.8, angle=0., freq=1897.420):
		stddev = fwhm2std(fwhm) #Convert FWHM to stddev
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
		self.type = type
		self.x = 0.
		self.y = 0.
		self.range = 2.0 * r #Maximum range in arcsec +/- x and y to paint, this is for optimization
		self.fwhm = fwhm
		self.freq = freq


		
#Child class to store array profile for the HFA
class HFA_array(GREAT_array):
	def __init__(self, fwhm=6.3, type='HFA', amplitude=1.0, r=13.6, angle=0., freq=4744.77749):
		stddev = fwhm2std(fwhm) #Convert FWHM to stddev
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
		self.type = type
		self.x = 0.
		self.y = 0.
		self.range = 2.0 * r #Maximum range in arcsec +/- x and y to paint, this is for optimization
		self.fwhm = fwhm
		self.freq = freq

#Child class to store array profile for 4GREAT 1, FWHM is from the ICD
class FG1_array(GREAT_array):
	def __init__(self, fwhm=50.0, type='4G1', amplitude=1.0, freq=550.0):
		stddev = fwhm2std(fwhm) #Convert FWHM to stddev
		pix0 =  models.Gaussian2D(amplitude=amplitude, x_mean=0.0, y_mean=0.0, x_stddev=stddev, y_stddev=stddev) #Define pixel profile
		self.array_profile = [pix0]
		self.angle = 0.
		self.rotate(self.angle)
		self.type = type
		self.x = 0.
		self.y = 0.
		self.range = 3.0*fwhm #Maximum range in arcsec +/- x and y to paint, this is for optimization
		self.fwhm = fwhm
		self.freq = freq


#Child class to store array profile for 4GREAT 1, FWHM is from the ICD
class FG2_array(GREAT_array):
	def __init__(self, fwhm=30.0, type='4G2', amplitude=1.0, freq=980.0):
		stddev = fwhm2std(fwhm) #Convert FWHM to stddev
		pix0 =  models.Gaussian2D(amplitude=amplitude, x_mean=0.0, y_mean=0.0, x_stddev=stddev, y_stddev=stddev) #Define pixel profile
		self.array_profile = [pix0]
		self.angle = 0.
		self.rotate(self.angle)
		self.type = type
		self.x = 0.
		self.y = 0.
		self.range = 3.0*fwhm #Maximum range in arcsec +/- x and y to paint, this is for optimization
		self.fwhm = fwhm
		self.freq = freq

#Child class to store array profile for 4GREAT 1, FWHM is from the ICD
class FG3_array(GREAT_array):
	def __init__(self, fwhm=19.0, type='4G3', amplitude=1.0, freq=1390.0):
		stddev = fwhm2std(fwhm) #Convert FWHM to stddev
		pix0 =  models.Gaussian2D(amplitude=amplitude, x_mean=0.0, y_mean=0.0, x_stddev=stddev, y_stddev=stddev) #Define pixel profile
		self.array_profile = [pix0]
		self.angle = 0.
		self.rotate(self.angle)
		self.type = type
		self.x = 0.
		self.y = 0.
		self.range = 3.0*fwhm #Maximum range in arcsec +/- x and y to paint, this is for optimization
		self.fwhm = fwhm
		self.freq = freq

#Child class to store array profile for 4GREAT 1, FWHM is from the ICD
class FG4_array(GREAT_array):
	def __init__(self, fwhm=11.0, type='4G4', amplitude=1.0, freq=2540.0):
		stddev = fwhm2std(fwhm) #Convert FWHM to stddev
		pix0 =  models.Gaussian2D(amplitude=amplitude, x_mean=0.0, y_mean=0.0, x_stddev=stddev, y_stddev=stddev) #Define pixel profile
		self.array_profile = [pix0]
		self.angle = 0.
		self.rotate(self.angle)
		self.type = type
		self.x = 0.
		self.y = 0.
		self.range = 3.0*fwhm #Maximum range in arcsec +/- x and y to paint, this is for optimization
		self.fwhm = fwhm
		self.freq = freq



