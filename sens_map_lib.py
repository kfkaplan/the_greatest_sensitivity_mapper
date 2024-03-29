#Library for the GREATest sensitivity mapper
#
#Created by Kyle Kaplan and Simon Coudé June 4, 2020
#

#Import python libraries 
import numpy as np
import copy
#from scipy.interpolate import interp1d
from matplotlib import pyplot
#from astropy.io import fits
from astropy.modeling import models
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.nddata.blocks import block_reduce
from astropy.io import fits #Read in FITS files and headers
from astropy import wcs #For reprojecting fits files=
from reproject import reproject_exact#, reproject_interp #To reproeject WISE band 3 data
import xmltodict #For reading in AORs
#import timeit #used for profiling code
from numba import jit
import ipywidgets

np.seterr(divide='ignore', invalid='ignore') #Hide those pesky divide by zero warnings


try:  #Try to import bottleneck library, this greatly speeds up things such as nanmedian, nanmax, and nanmin
	import bottleneck as bn #Library to speed up some numpy routines
except ImportError:
	print("Bottleneck library not installed.  Code will still run but might be slower.  You can try to bottleneck with 'conda install bottleneck' or 'pip install bottleneck' for a speed up.")
	import numpy as bn
import matplotlib.gridspec as grd



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
	return   fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def std2fwhm(stddev): #Convert stddev to FWHM  (for a gaussian)
	return   stddev * (2.0 * np.sqrt(2.0 * np.log(2.0)))

# @jit(nopython=True, parallel=True, fastmath=True)
# def gauss2d_simulate_obs(xpos=0.0, ypos=0.0, x=0.0, y=0.0, stddev=1.0, amplitude=1.0): #2D gaussian kernel (not normalized)
# 	gauss2d_result = amplitude * np.exp(-((xpos-x)**2 + (ypos-y)**2) / (2.0 * stddev**2))
# 	return gauss2d_result / np.nansum(gauss2d_result) #normalize results


@jit(nopython=True, parallel=True)
def run_simulate_observation(x1d, y1d, x_array, y_array, signal_array, exptime_array, one_third_stddev, data, exptime):
	#nx = len(x1d)
	#for ix, x in enumerate(x1d): #Loop through each pixel in the sky object and use a kernel with 1/3 the FWHM of the beam size to 
	#	for iy, y in enumerate(y1d):
	nx = len(x1d)
	ny = len(y1d)
	for ix in range(nx):
		for iy in range(ny):
			#weights = gauss2d_simulate_obs(amplitude=1.0, xpos=x_array, ypos=y_array, x=x, y=y, stddev=one_third_stddev) #Generate weights for this position using the kernel
			#weights = 1.0 * np.exp(-((x_array-x)**2 + (y_array-y)**2) / (2.0 * one_third_stddev**2))#Generate weights for this position using the kernel
			weights = np.exp(-((x_array-x1d[ix])**2 + (y_array-y1d[iy])**2) / (2.0 * one_third_stddev**2))#Generate weights for this position using the kernel
			#weights[~np.isfinite(weights)] = np.nan
			#weights /= np.nansum(weights) #normalize weights
			exptime_pixel = np.nansum(exptime_array * weights) #Convolve exposure time with kernel to calulate the exposure time for this specific pixel
			if exptime_pixel != 0.: #Error catch
				exptime[iy, ix] = exptime_pixel
				data[iy, ix] = np.nansum(signal_array * weights) / exptime_pixel #Convolve simualted signal on sky with kernel to claculate signal at this pixel
			else:
				exptime[iy, ix] = 0.
				data[iy, ix] = 0.
		#print('Progress:', ix/nx)
	#data /= exptime  #normalize simulated data by exposure time
	return data, exptime


# def get_deltaTa(Tsys=0., deltafreq=1.0, Non=1.0, time=1.0, TPOTF=False): #Get the RMS antenna temperature (delta-Ta) for a single pointing (when time on = time off)
# 	if not TPOTF: #If not a Total Power OTF map (most observations)...
# 		deltaTa = 2.0 * Tsys / (time * deltafreq)**0.5 #Calulate RMS temperature using Equation 6-5 in the observer's handbook
# 	else: #If a Total Power OTF map....
# 		deltaTa = Tsys * (1.0 + Non**-0.5)**0.5 / (time * deltafreq)*0.5 #Calculate RMS temp. for TP OTF maps
# 	return deltaTa
	


#Find nearest value in an array
@jit(nopython=True, parallel=True, fastmath=True)
def find_nearest(arr, value):
    return (np.abs(arr - value)).argmin()


#Reads in a .aor file and returns aor objects (see aor class below) for each aor in the file
def open_aors(user_input):
	#The following code from Ed Chambers' aor_inpar_translator.py was plundered on the high seas by the nortorious software pirate knwon as Kyle Kaplan
	#who then proceeded to modify it without approval of the Gov'ner or the Crown™.
	#print('the input type is', type(user_input))
	if type(user_input) == str: #If user input is a file path
		aor_file = open(user_input, 'rb') #Open .aor file
		xml_dict = xmltodict.parse(aor_file) #Parse the xml from the .aor file into python dictionaries so we can grab information out of them
		aor_file.close() #close .aor file
	elif type(user_input) == ipywidgets.widgets.widget_upload.FileUpload: #Else if user input is a file upload widget
		xml_dict = xmltodict.parse(user_input.data[0])  #Parse the xml from the .aor file into python dictionaries so we can grab information out of them
	else:
		print("WARNING: User input type "+str(type(user_input)) + "is not a valid parameter for open_aors.  User input must be a file path string or file upload widget.")
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

#Reads in a .aor file that was previously opened as a binary (for use with ipython widgets file upload)
def open_aors_binary(data):
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
		self.aor_dict = aor_dict #Carry the aor dictionary, mostly for debugging purposes
		instr_data = aor_dict['instrument']['data']
		targets = aor_dict['target']
		# self.array1 = HFA_array()
		# if instr_data['InstrumentSpectralElement2'] == 'GRE_LFA':
		# 	self.array2 = LFA_array()
		# else: #4GREAT
		# 	self.array2 = FOURGREAT_array()
		self.map_type = aor_dict['instrument']['@class'].split('.')[-1] #'GREAT_SP', 'GREAT_Raster', 'GREAT_OTF', 'GREAT_ON_THE_FLY_HONEYCOMB_MAP', or 'GREAT_ON_THE_FLY_ARRAY_MAPPING'
		self.cycles = float(instr_data['Repeat'])
		self.array_angle = float(instr_data['ArrayRotationAngle'])
		self.primary_frequency = instr_data['PrimaryFrequency']
		if self.primary_frequency == 'Frequency1':
			self.primary_frequency = 'HFA'
		elif self.primary_frequency == 'Frequency2':
			self.primary_frequency = 'LFAV'
		frequencies = [] #Grab  frequencies 1,2,3,4,5
		frequencies.append(float(instr_data['Frequency']) * 1e9) #HFA
		frequencies.append(float(instr_data['Frequency2']) * 1e9) #4G1 or LFAV
		frequencies.append(float(instr_data['Frequency3']) * 1e9) #4G2 or LFAH
		if "Frequency4" in instr_data: #Check if using 4GREAT
			frequencies.append(float(instr_data['Frequency4']) * 1e9) #4G3
		else: #If not, just put zero as a placeholder
			frequencies.append(0.0)
		if "Frequency5" in instr_data: #Check if using 4GREAT
			frequencies.append(float(instr_data['Frequency5']) * 1e9) #4G4 
		else: #If not, just put zero as a placeholder
			frequencies.append(0.0)
		self.frequencies = frequencies #
		self.aor_id = instr_data['aorID'] #Carry the aor ID through so it is easier to identify what is what
		self.target = targets['name']
		self.skycoord = SkyCoord(ra=float(targets['position']['lon'])*units.degree, dec=float(targets['position']['lat'])*units.degree, frame='icrs')
		self.Non = 1
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
			self.nod_type = instr_data['NodType'] #'Total_Power', 'Dual_Beam_Switch', ect.
			if self.nod_type == 'Total_Power':
				if self.map_type == 'GREAT_Raster':
					self.Non = float(instr_data['OnsPerOff'])
				elif self.map_type == 'GREAT_OTF':
					#self.Non = float(instr_data['LinesPerOff'])
					if instr_data["ScanDirection"] == "x_direction": #Non if scan direction is x
						self.Non = self.nx * float(instr_data['LinesPerOff'])
					elif instr_data["ScanDirection"] == "y_direction": #Non if scan direction is y
						self.Non = self.ny * float(instr_data['LinesPerOff'])
		elif self.map_type == 'GREAT_ON_THE_FLY_HONEYCOMB_MAP':
			self.time = float(instr_data['TimePerPoint'])
			self.map_angle = float(instr_data['ArrayRotationAngle'])
			self.x = float(instr_data['TargetOffsetRA'])
			self.y = float(instr_data['TargetOffsetDec'])
			self.nod_type = instr_data['NodType'] #'Total_Power', 'Dual_Beam_Switch', ect.
			self.Non = 25 / float(instr_data['OffPerPattern'])
		elif self.map_type == 'GREAT_ON_THE_FLY_ARRAY_MAPPING':
			self.nod_type = instr_data['NodType'] #'Total_Power', 'Dual_Beam_Switch', ect.
			#self.Non = float(instr_data['LinesPerOff'])
			self.map_angle = float(instr_data['MapRotationAngle'])
			self.nscans = int(instr_data['NumFillOTF'])
			self.x = float(instr_data['MapCenterOffsetRA'])
			self.y = float(instr_data['MapCenterOffsetDec'])
			self.stepsize = float(instr_data['MapStepSize'])
			self.scan_direction = instr_data['ScanDirection'] #'x_direction', 'y_direction', or 'x_and_y_directions'
			if self.scan_direction == 'x_direction' or self.scan_direction == 'x_and_y_directions':
				self.x_length = float(instr_data['XMapScanLength'])
				self.x_time = float(instr_data['XMapTimePerPoint'])
				self.x_blocks_scan = int(instr_data['XMapNumBlocksX'])
				self.x_blocks_perp = int(instr_data['XMapNumBlocksY'])
				if self.nod_type == "Total_Power":
					self.x_Non = self.x_length * float(instr_data['ArrayWidth']) / self.stepsize
				else:
					self.x_Non = 1
			if self.scan_direction == 'y_direction' or self.scan_direction == 'x_and_y_directions':
				self.y_length = float(instr_data['YMapScanLength'])
				self.y_time = float(instr_data['YMapTimePerPoint'])
				self.y_blocks_scan = int(instr_data['YMapNumBlocksY'])
				self.y_blocks_perp = int(instr_data['YMapNumBlocksX'])
				if self.nod_type == "Total_Power":
					self.y_Non = self.y_length * float(instr_data['ArrayWidth']) / self.stepsize
				else:
					self.y_Non = 1
		else:
			print('ERROR: '+self.map_type+' is not a valid map type to paint in the sky.')
	def target_offsets(self, input_aor): #Get offsets between two different AOR targets in arcsec in RA and Dec. (see https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html#astropy.coordinates.SkyCoord.spherical_offsets_to)
		dra, ddec = self.skycoord.spherical_offsets_to(input_aor.skycoord)
		return dra.to(units.arcsec).value, ddec.to(units.arcsec).value
	def paint(self, skyobj, which_array, type=''): #Paint AOR onto sky object with specified array ("HFA", "LFA", or "4GREAT")
		# #Determine which array to use and generate the appropriate object
		# print('frequencies = ', self.frequencies)
		if which_array.upper() == 'HFA':
			array_obj = HFA_array()
			array_obj.freq = self.frequencies[0]
		elif which_array.upper() == 'LFAV':
			array_obj = LFAV_array()
			array_obj.freq = self.frequencies[1]
		elif which_array.upper() == 'LFAH':
			array_obj = LFAH_array()
			array_obj.freq = self.frequencies[2]
		elif which_array.upper() == '4G_1':
			array_obj = FG1_array()
			array_obj.freq = self.frequencies[1] 
		elif which_array.upper() == '4G_2':
			array_obj = FG2_array()
			array_obj.freq = self.frequencies[2]
		elif which_array.upper() == '4G_3':
			array_obj = FG3_array()
			array_obj.freq = self.frequencies[3]
		elif which_array.upper() == '4G_4':
			array_obj = FG4_array()
			array_obj.freq = self.frequencies[4]
		else:
			print('ERROR: '+which_array+' is not a valid array. Please set to be either HFA, LFAH, LFAV, 4G_1, 4G_2, 4G_3, or 4G_4')
			return
		if type != '': #Manually set type (important for array OTF and honeycomb maps) if not the primary frequency
			array_obj.type = type
		#Determine the map type then paint the array
		# print('map_type = ', self.map_type)
		# print('nod_type = ', self.nod_type)
		if self.map_type == 'GREAT_SP':
			array_obj.single_point(skyobj, x=self.x, y=self.y, time=self.time, array_angle=self.array_angle, cycles=self.cycles, Non=self.Non)
		elif self.map_type == 'GREAT_Raster' or self.map_type == 'GREAT_OTF':
			#if self.map_type == 'GREAT_OTF' and self.nod_type == 'Total_Power':  #Set a few parameters to ensure proper calculation of Total Power OTF maps
			# if self.nod_type == 'Total_Power':  #Set a few parameters to ensure proper calculation of Total Power OTF (or raster) maps
			# 	skyobj.TPOTF = True
			# 	skyobj.Non = self.Non
			array_obj.map(skyobj, x=self.x, y=self.y, nx=self.nx, ny=self.ny, dx=self.dx, dy=self.dy, array_angle=self.array_angle, map_angle=self.map_angle, cycles=self.cycles, time=self.time, Non=self.Non)
		elif self.map_type == 'GREAT_ON_THE_FLY_HONEYCOMB_MAP':
			array_obj.primary_frequency = self.primary_frequency
			array_obj.honeycomb(skyobj, x=self.x, y=self.y, array_angle=self.array_angle, map_angle=self.map_angle, cycles=self.cycles, time=self.time, Non=self.Non)
			# if self.nod_type == 'Total_Power': #Set a few parameters to ensure proper calculation of Total Power Array OTF maps
			# 	skyobj.TPOTF = True
			# 	skyobj.Non = self.Non
		elif self.map_type == 'GREAT_ON_THE_FLY_ARRAY_MAPPING':
			# if self.nod_type == 'Total_Power': #Set a few parameters to ensure proper calculation of Total Power Array OTF maps
			# 	skyobj.TPOTF = True
			# 	skyobj.Non = self.Non
			if self.scan_direction == 'x_direction' or self.scan_direction == 'x_and_y_directions': #Scans in x direction
					array_obj.array_otf(skyobj, x=self.x, y=self.y, nblock_scan=self.x_blocks_scan, nblock_perp=self.x_blocks_perp, step=self.stepsize, length=self.x_length, 
							time=self.x_time, cycles=self.cycles, map_angle=self.map_angle, direction='x', nscans=self.nscans, Non=self.x_Non)
			if self.scan_direction == 'y_direction' or self.scan_direction == 'x_and_y_directions': #Scans in y direction
					array_obj.array_otf(skyobj, x=self.x, y=self.y, nblock_scan=self.y_blocks_scan, nblock_perp=self.y_blocks_perp, step=self.stepsize, length=self.y_length, 
							time=self.y_time, cycles=self.cycles, map_angle=self.map_angle, direction='y', nscans=self.nscans, Non=self.y_Non)


#Class that stores the 2D array representing the sky and it's associated coordinate system
class sky:
	def __init__(self, x_range, y_range, plate_scale): #, Tsky=178.0, Ttel=188.0, deltaTa=0.):
		self.update(x_range, y_range, plate_scale)
	def update(self, x_range, y_range, plate_scale): #Initialize or update sky object
		if np.size(x_range) == 1:
			x_range = [0.0, x_range]
		if np.size(y_range) == 1:
			y_range = [0.0, y_range]
		nx = int((x_range[1]-x_range[0])/plate_scale) + 1
		ny = int((y_range[1]-y_range[0])/plate_scale) + 1
		signal = np.zeros([ny, nx])
		noise = np.zeros([ny, nx])
		noise[:] = np.inf #Make noise infinite
		data = np.zeros([ny, nx])
		sigma = np.zeros([ny, nx])
		exptime = np.zeros([ny, nx])
		y, x = np.mgrid[0:ny,0:nx]*plate_scale
		x += x_range[0]
		y += y_range[0]
		self.x_range = x_range #Save x,y ranges and plate scale
		self.y_range = y_range
		self.nx = nx
		self.ny = ny
		self.plate_scale = plate_scale
		self.data = data #This is the actual 2D array that stores the simulated data (in units of T_a)
		self.noise = noise #This is the 2D array that stores point sources (narroiw gaussians) for the noise (in units of delta-T_a) which are later convolutionally regridded, we are treating noise like we treat signal
		self.signal = signal #This is the the model "true" signal exepected signal from the sky stored in a (in units of T_a)
		self.sigma = sigma #This is the 2D array that stores the regridded noise (1 sigma uncertainity)
		self.exptime = exptime #Save total exposure time per pixel
		self.x = x[:,::-1] #2D x coords (note the x coordinates inncrese to the left since they are RA)
		self.y = y #2D y coords
		self.x_1d = self.x[0,:] #Grab 1D arrays for x and y coordinates
		self.y_1d = self.y[:,0]
		self.extent = [np.max(x), np.min(x), np.min(y), np.max(y)]  #Gives the x and y coordinate extents for proper plotting using imshow
		self.area = (np.max(x)-np.min(x)) * (np.max(y)-np.min(y)) #Area of sky in square arcsec
		#self.total_exptime = 0. #Store total cumulative exposure time which is incremented every time a map is added
		#self.pixel_area = plate_scale**2 #Area per pixel, used in calculating (exptime/arcsec^2)
		self.fwhm = 0. #Store latest beam profile FWHM used on this sky object
		self.freq = 0. #Store latest frequency painted onto this sky object]
		self.TPOTF = False #Store if this is a Total Power OTF map or not (used for noise calculations)
		self.Non = 1.0 #Store N_on, used to caculate noise if this is a Total Power OTF map
		self.x_map_center_points = [] #self.x_map_center_points and self.y_map_cete_points are designed to hold x and y coordinates for the centers of individual maps or blocks for later plotting/checking
		self.y_map_center_points = []
		self.total_time = 0. #Track the total time integrated by summing up all the pointings
		self.x_beam = [] #Lists that store the x position, y position, exposure time, and convolved signal for each "beam" to later paint onto the sky object when simulating an observation
		self.y_beam = []
		self.exptime_beam = []
		self.signal_beam = []
		self.beam_profiles = []
		self.Non = []
		self.WCS = wcs.WCS(naxis=2) #Define an astropy WCS object to allow fits files to be projected into this sky object
		self.WCS.wcs.cdelt = [-plate_scale/3600.0, plate_scale/3600.0] #Define the X and Y scales
		self.WCS.wcs.ctype = ["RA---TAN", "DEC--TAN"] #Define coordinates projection
		self.per_beam = False #Stores if units for exp. time, noise, and S/N should be in per pixel (=False) or per beam (=True)
		# ref_pix_x = 0
		# ref_pix_y = 0
		ref_pix_x = (self.x_1d[0] - self.x_1d[-1])/(2.0*plate_scale)+1
		ref_pix_y = (self.y_1d[-1] - self.y_1d[0])/(2.0*plate_scale)+1
		ref_pix_ra =  (self.x_1d[0] + self.x_1d[-1]) / (2.0 * 3600.0)
		ref_pix_dec = (self.y_1d[0] + self.y_1d[-1]) / (2.0 * 3600.0)
		# ref_pix_x = np.where(self.x_1d == 0.0)[0][0]
		# ref_pix_y = np.where(self.y_1d == 0.0)[0][0]
		# ref_pix_x = self.x_1d[0] / plate_scale
		# ref_pix_y = - self.y_1d[0] / plate_scale
		#self.WCS.wcs.crval = [-ref_pix_x*plate_scale/3600.0, ref_pix_y*plate_scale/3600.0] #RA and Dec of reference pixel
		self.WCS.wcs.crval = [ref_pix_ra, ref_pix_dec]
		self.WCS.wcs.crpix = [ref_pix_x, ref_pix_y] #Define reference pixel
		#self.Tsys = Tsys
		#self.deltaTa = deltaTa
		#self.Tsky = Tsky #Ambient temperature for the atmosphere
		#self.Ttel = Ttel #Physical temperature of the telescope
		#self.set_sky_coords(0.0, 0.0)
	def clear(self): #Erase everything on the grid except for the signal
		self.data[:] = 0.
		self.exptime[:] = 0.
		self.noise[:] = np.inf
		self.x_beam = []
		self.y_beam = []
		self.exptime_beam = []
		self.signal_beam = []
	def get_range_indicies(self, xmin, xmax, ymin, ymax): #Returns the index numbers for a range of (xmin, xmax, ymin, ymax)
		ixmin = find_nearest(xmin, self.x_1d)
		ixmax = find_nearest(xmax, self.x_1d)
		iymin = find_nearest(ymin, self.y_1d)
		iymax = find_nearest(ymax, self.y_1d)
		return ixmin, ixmax, iymin, iymax
	def plot(self, map_type='data', show_points=True, fraction_completion=1.0, **kwargs): #Generate an expsoure map plot
		# if np.any(self.signal != 0.): #Error catch to ensure we are not dividing by zero
		# 	self.normalize()
		if map_type == 'exposure':
			min_exptime = np.nanpercentile(self.exptime, 2) / fraction_completion
			max_exptime = np.nanpercentile(self.exptime, 98) / fraction_completion
			pyplot.imshow(self.exptime, origin='lower', extent=self.extent, vmax=max_exptime, vmin=min_exptime, **kwargs)
			if self.per_beam:
				label = r'Time (s beam$^{-1}$)'
				title = r'Exposure Time (s beam$^{-1}$)'
			else: #Per pixel
				label = r'Time (s pixel$^{-1}$)'
				title = r'Exposure Time (s pixel$^{-1}$)'
		elif map_type == 'signal': # if signal is set to true, plot the modeled signal instead
			min_Ta = np.nanpercentile(self.signal, 2)
			max_Ta = np.nanpercentile(self.signal, 99.5)
			pyplot.imshow(self.signal, origin='lower', extent=self.extent, vmin=min_Ta, vmax=max_Ta, **kwargs)
			label = r'$T_a$'
			title = r'Signal ($T_a$)'
		elif map_type == 'noise':
			#min_noise = np.nanmin(self.noise)
			min_noise = np.nanpercentile(self.noise, 2) * fraction_completion**0.5
			pyplot.imshow(self.noise, origin='lower', extent=self.extent, vmax=2.5*min_noise, vmin=0.8*min_noise, **kwargs)
			#pyplot.imshow(self.noise, origin='lower', extent=self.extent, **kwargs)
			if self.per_beam:
				label = r'$\Delta T_a$ beam$^{-1}$'
				title = r'Noise ($\Delta T_a^*$ beam$^{-1}$)'
			else: #Per pixel
				label = r'$\Delta T_a$ pixel$^{-1}$'
				title = r'Noise ($\Delta T_a^*$ pixel$^{-1}$)'
		elif map_type == 's2n': #If s2n is true, plot the signal-to-noise
			s2n = self.data/self.noise
			# min_s2n = np.nanpercentile(s2n, 5) / fraction_completion*0.5
			# max_s2n = np.nanpercentile(s2n, 95) / fraction_completion*0.5
			max_s2n = np.nanmax(s2n) / fraction_completion**0.5
			pyplot.imshow(s2n, origin='lower', extent=self.extent, vmax=max_s2n, vmin=0.0, **kwargs)
			# pyplot.imshow(s2n, origin='lower', extent=self.extent, **kwargs)
			if self.per_beam:
				label = r'S/N beam$^{-1}$'
				title = r'S/N beam$^{-1}$'
			else: #Per pixel
				label = r'S/N pixel$^{-1}$'
				title = r'S/N pixel$^{-1}$'
		else: #Normally plot the simulated data
			#min_Ta = 0.9 * bn.nanmin(self.data[np.isfinite(self.data)]) #Fix colorbar scale, especially for a uniform background
			min_Ta = np.nanpercentile(self.data, 2)
			max_Ta = np.nanpercentile(self.data, 98)
			pyplot.imshow(self.data, origin='lower', extent=self.extent, vmin=min_Ta, vmax=max_Ta, **kwargs)
			label = r'$T_a$'
			title = r'$T_a$'
		if show_points: #If user specifies to show points (usually map or block centers), plot them
			pyplot.plot(self.x_map_center_points, self.y_map_center_points, 'o', color='red')
		pyplot.suptitle(title)
		pyplot.xlabel('Relative RA (arcsec)')
		pyplot.ylabel('Relative Dec. (arcsec)')
		pyplot.colorbar(label=label)
	# def normalize(self): #Normalize entire sky map by the total exposure time and the pixel size so the resulting values are in units of (exptime/arcsec^2)
	# 	scale_by = (self.total_exptime/self.pixel_area) / np.nansum(self.data)
	# 	self.data *= scale_by
	# 	self.noise *= scale_by
	def simulate_observation(self, Tsys=0., deltafreq=1e6, deltav=0., TPOTF=False, Non=1, freq=0., simulate_noise=False, per_beam=False, use_both_LFA_polarizations=False, fraction_completion=1.0): #Calculate noise and smooth the data and noisea by convolving with a 2D gausasian kernel with a FHWM that is 1/3 the beam profile, this is the final step for simulating data
		if freq !=0.: #Allow user to manually set frequency
			self.freq = freq
		if deltav > 0: #If user specifies the size of the spectral element in km/s, use that to calculate deltafreq instead of deltafreq being provided
			self.deltafreq = (deltav / 299792.458) * self.freq

		one_third_stddev = fwhm2std(self.fwhm) / 3.0 #Set up convolving kerneal for cygrid to be a 2D guassian with 1/3 the FWHM of the beam profiles
		x_array = np.array(self.x_beam) #- self.plate_scale
		y_array = np.array(self.y_beam) #- self.plate_scale
		signal_array = np.array(self.signal_beam)
		exptime_array = np.array(self.exptime_beam)
		Non_array = np.array(self.Non)


		find_unique_Non = np.unique(Non_array)
		data_shape = np.shape(self.data)
		# self.total_exptime = bn.nansum(exptime_array)
		for unique_Non in find_unique_Non: #Loop through each unique value of Non
			i = Non_array == unique_Non
			noise_for_one_second = Tsys * (1.0 + unique_Non**-0.5)**0.5 / (self.deltafreq)**0.5  #Calculate RMS temp. (noise) for all maps (note: for non TP maps, Non should = 1 thus reduce the equation to noise_for_one_second = (np.sqrt(2.0) * Tsys) / ((self.deltafreq)**0.5) )
			total_exp_time = bn.nansum(exptime_array[i])
			data_out, exptime_out = run_simulate_observation(self.x_1d, self.y_1d, x_array[i], y_array[i], signal_array[i], exptime_array[i], one_third_stddev, np.zeros(data_shape), np.zeros(data_shape)) #Simulate this chunk of observations with all the same Non
			exptime_out *=  fraction_completion * (total_exp_time/bn.nansum(exptime_out)) #Scale exptime to fraction of observations complete
			noise_out = noise_for_one_second  / ((exptime_out)**0.5) #Calculate noise based on exp time

			 #Combine data exptime and noise maps for all unique instances of Non
			self.exptime += exptime_out
			inverse_variance = self.noise**-2
			inverse_variance_out =  noise_out**-2
			sum_inverse_variance = inverse_variance + inverse_variance_out
			data_times_inverse_variance = self.data * inverse_variance
			data_out_times_inverse_variance_out = data_out * inverse_variance_out
			goodpix = (sum_inverse_variance > 0.) & np.isfinite(data_times_inverse_variance) & np.isfinite(data_out_times_inverse_variance_out) #Error catch


			#Do a weighted sum for simulated data and noise
			self.data[goodpix] = ((data_times_inverse_variance + data_out_times_inverse_variance_out) / sum_inverse_variance)[goodpix]
			self.noise[goodpix] = (sum_inverse_variance[goodpix])**-0.5

		# #convolved_variance = np.zeros(np.shape(self.data))



		# noise_array = np.zeros(len(x_array))




		# #Calculate variance
		# if TPOTF: #If user specifies Total Power OTF, set the proper variables
		# 	self.TPOTF = True
		# 	self.Non = Non

		# else:
		# 	self.deltafreq = deltafreq
		# 		# print('self.freq = ', self.freq)
		# 		# print('deltafreq = ', deltafreq)
		# if not self.TPOTF: #If not a Total Power OTF map (most observations)...
		# 	noise_for_one_second = (np.sqrt(2.0) * Tsys) / ((self.deltafreq)**0.5) #Calulate RMS temperature (noise) using Equation 6-5 in the observer's handbook ***NOTE***: MODIFIED 2 TO SQRT(2) BECAUSE WE ARE ONLY USING THE ON TIME, NOT THE ON+OFF TIME AS SHOWN IN THE EQUATION
		# else: #If a Total Power Array OTF map....
		# 	noise_for_one_second = Tsys * (1.0 + self.Non**-0.5)**0.5 / (self.deltafreq)**0.5 #Calculate RMS temp. (noise) for TP OTF maps
		# self.noise_for_one_second = noise_for_one_second #tore the noise for one second in case it is needed for later recalculations
		# #self.noise_beam = noise_for_one_second / exptime_array
		# #self.s2n_beam = signal_array / self.noise_beam
		# #noise_array = noise_array * exptime_array
		# # noise_array = noise #/ (exptime_array**0.5)
		# # print('Noise per beam', noise_array)

		# # print('S/N per beam', signal_array/noise_array)
		# # print()
		# #total_beam_signal = bn.nansum(signal_array/exptime_array)
		# #total_beam_noise = bn.nansum((noise_for_one_second*len(signal_array))**2/exptime_array)**0.5
		# # print('Total beam signal:', total_beam_signal)
		# # print('Total beam noise:', total_beam_noise)
		# # print('Total beam S/N:', total_beam_signal/total_beam_noise)

		# #convolved_variance = np.zeros(np.shape(self.data))
		# # #Paint variance onto 2D variance array starting by making the variance point sources as narrow gaussians
		# # fwhm = self.fwhm / 10.0 #Set parameters for 2D gaussian
		# # stddev = fwhm / (2.0 * np.sqrt(np.log(2.0)))#Convert FWHM to stddev
		# # for i in range(len(self.x_beam)): #Loop through each beam and create a point source for the variance
		# # 	gauss_model =  models.Gaussian2D(amplitude=1.0, x_mean=self.x_beam[i], y_mean=self.y_beam[i], x_stddev=stddev, y_stddev=stddev)

		# # 	point_source_gaussian = gauss_model(self.x, self.y)
		# # 	point_source_gaussian = noise_array[i]**2 * point_source_gaussian / bn.nansum(point_source_gaussian) #Normalize gaussian


		# # # 	variance +=  point_source_gaussian
		# # n_beams = len(self.beam_profiles)
		# # variance_array = np.zeros(n_beams)
		# # for i in range(n_beams):
		# # 	chunk_of_array_profile = self.beam_profiles[i](self.x, self.y) #Isolate the beam profile on the sky to use 
		# # 	sum_chunk_of_array_profile = bn.nansum(chunk_of_array_profile)
		# # 	variance_array[i] = bn.nansum(variance * chunk_of_array_profile) / sum_chunk_of_array_profile #Convovle assumed signal on sky with beam profile

		# # nx = len(self.x_1d)
		# # for ix, x in enumerate(self.x_1d): #Loop through each pixel in the sky object and use a kernel with 1/3 the FWHM of the beam size to 
		# # 	for iy, y in enumerate(self.y_1d):
		# # 		weights = gauss2d_simulate_obs(amplitude=1.0, xpos=x_array, ypos=y_array, x=x, y=y, stddev=one_third_stddev) #Generate weights for this position using the kernel
		# # 		#weights /= bn.nansum(weights) #Normalize weights
		# # 		self.data[iy, ix] = bn.nansum(signal_array * weights) #Convolve simualted signal on sky with kernel to claculate signal at this pixel
		# # 		self.exptime[iy, ix] = bn.nansum(exptime_array * weights)#/self.plate_scale**2 #Convolve exposure time with kernel to calulate the exposure time for this specific pixel
		# # 		#exptime_weighted_for_noise[iy, ix] = bn.nansum(exptime_array * weights**0.5)
		# # 		#convolved_variance[iy, ix] = bn.nansum(variance_array * weights)

		# # 		#convolved_variance[iy, ix] = bn.nansum(noise_array**2 * exptime_array * weights)
		# # 	print('Progress: ', ix / nx)

		# #self.data, self.exptime = run_simulate_observation(self.x_1d - 0.5*self.plate_scale, self.y_1d - 0.5*self.plate_scale, x_array, y_array, signal_array, exptime_array, one_third_stddev, self.data, self.exptime)



		# self.data, self.exptime = run_simulate_observation(self.x_1d, self.y_1d, x_array, y_array, signal_array, exptime_array, one_third_stddev, self.data, self.exptime)

		# #self.data /= (self.exptime) #normalize simulated data by exposure time
		# self.exptime *=  fraction_completion * self.total_exptime / bn.nansum(self.exptime)
		# #self.noise = (convolved_variance / self.exptime)**0.5
		# #self.noise = noise  / ((self.exptime * self.plate_scale**2)**0.5)
		# self.noise = self.noise_for_one_second  / ((self.exptime)**0.5)
		mask = self.exptime < 1e-5 #Mask out pixels with near zero exposure
		self.exptime[mask] = 0.0
		self.data[mask] = 0.0
		self.noise[mask] = np.nan

		if use_both_LFA_polarizations: #If user specifies they want to use both the LFAH and LFAV polarizations, double the exposure time
			self.exptime *= 2.0
			self.noise /= np.sqrt(2.0)


		if simulate_noise: #If user wants to add simulated noise to the simulated data
			self.data += np.random.normal(0.0, self.noise, self.data.shape)

		if per_beam: #If user wants variables in per beam, convert them
			#pixel_area = abs( (x_array[1]-x_array[0]) * (y_array[1]-y_array[0]) ) #Take the ratio of the beam area to pixel area
			pixel_area = self.plate_scale**2
			beam_area = 2.0 * np.pi * fwhm2std(self.fwhm)**2
			beam_to_pixel_area_ratio = beam_area / pixel_area
			self.exptime *= beam_to_pixel_area_ratio #Scale exptime and noise to be s/beam and K/beam respectively
			self.noise /= np.sqrt(beam_to_pixel_area_ratio)
			self.per_beam = True #Store the fact these units are now in per beam for later calcualtions such as downsampling or to properly label units when making plots

		# print('Total S/N: ',self.s2n())
	def input(self, model_shape): #Draw an astropy model shape onto  sigal (e.g. create a model of the "true" signal)
		self.signal += model_shape(self.x, self.y)
	def uniform(self, T): #Make sky grid have a uniform signal (mainly used for testing)
		self.signal[:] = T
	def downsample(self, factor): #Downsample sky grid by an integer factor (1/2, 1/3, 1/4, ect.) using the astropy function
		plate_scale = self.plate_scale * factor #calculate new plate scale
		new_WCS = copy.deepcopy(self.WCS) #Need to set up a new WCS
		new_WCS.wcs.cdelt[0] = self.WCS.wcs.cdelt[0] * factor
		new_WCS.wcs.cdelt[1] = self.WCS.wcs.cdelt[1] * factor
		#ref_pix_x = self.x_1d[0] / plate_scale
		#ref_pix_y = - self.y_1d[0] / plate_scale
		ref_pix_x = self.WCS.wcs.crpix[0] / factor
		ref_pix_y = self.WCS.wcs.crpix[1] / factor
		new_WCS.wcs.crpix = [ref_pix_x, ref_pix_y] #Define reference pixel
		new_header = new_WCS.to_header()
		new_header['NAXIS'] =  2 #Looks like i need to manually set the NASXIS keywords in the sky header
		nx = abs(int((self.x_range[1]-self.x_range[0])/plate_scale))
		ny = abs(int((self.y_range[1]-self.y_range[0])/plate_scale))
		# nx = int((self.x_range[1]-self.x_range[0])/plate_scale)
		# ny = int((self.y_range[1]-self.y_range[0])/plate_scale)
		new_header['NAXIS1'] =  nx
		new_header['NAXIS2'] =  ny
		#Reproject artificial signal
		img_hdu = fits.ImageHDU(header=self.WCS.to_header()) #Encapsulate everything an ImageHDU object for running repropject
		img_hdu.data = self.signal 
		array, footprint = reproject_exact(img_hdu, new_header, parallel=True)
		self.signal = array
		#Reproject simulated signal
		img_hdu.data = self.data 
		array, footprint = reproject_exact(img_hdu, new_header, parallel=True)
		self.data = array
		 #Reproject exposure time
		img_hdu.data = self.exptime
		array, footprint = reproject_exact(img_hdu, new_header, parallel=True)
		if not self.per_beam:  #If in units of per pixel and not per beam
			self.exptime = array * factor**2
		else:
			self.exptime = array
		#Reproject noise
		img_hdu.data = self.noise**-2
		array, footprint = reproject_exact(img_hdu, new_header, parallel=True)
		if not self.per_beam:  #If in units of per pixel and not per beam
			self.noise = (array * factor**2)**-0.5
		else:
			self.noise = array**-0.5

		#$ self.noise = self.noise_for_one_second / self.exptime**0.5 #Recalculate noise using the new exposure time grid
		# else: #If calculating beam per pixel, downsample the noise array as well
		# 	img_hdu.data = self.noise**2 #Reproject artificial signal
		# 	array, footprint = reproject_exact(img_hdu, new_header, parallel=True) 
		# 	self.noise = np.sqrt(array)
		self.WCS = new_WCS #Save updated wcs

		# plate_scale = self.plate_scale * factor #calculate new plate scale
		# nx = int((self.x_range[1]-self.x_range[0])/plate_scale)
		# ny = int((self.y_range[1]-self.y_range[0])/plate_scale)
		# y, x = np.mgrid[0:ny,0:nx] * plate_scale
		# x += self.x_range[0]# - 0.5*plate_scale
		# y += self.y_range[0]# + 0.5*plate_scale
		# self.plate_scale = plate_scale #Save the new plate scale and x,y coordinate 2D and 1D grids
		# self.x = x[:,::-1] #2D x coords (note the x coordinates inncrese to the left since they are RA)
		# self.y = y #2D y coords
		# self.x_1d = self.x[0,:] #Grab 1D arrays for x and y coordinates
		# self.y_1d = self.y[:,0]
		# #Resample all grids using astropy block_reduce
		# #total_data_unreduced = bn.nansum(self.data) #Resample data normalized to the total value (ie. T_a won't change)
		# self.data = block_reduce(self.data, factor)  / float(factor**2)
		# #total_data_reduced = bn.nansum(self.data)
		# #print('Block reduce ratio = ', total_data_unreduced / total_data_reduced)
		# #self.data = self.data * (total_data_unreduced / total_data_reduced) 
		# #Resample signal (treat like the data)
		# #total_signal_unreduced = bn.nansum(self.signal) #Resample data normalized to the total value (ie. T_a won't change)
		# self.signal = block_reduce(self.signal, factor)  / float(factor**2)
		# #total_signal_reduced = bn.nansum(self.signal)
		# #self.signal = self.signal * (total_signal_unreduced / total_signal_reduced)
		# self.exptime = block_reduce(self.exptime, factor)  #Resample exposure time (here we just sum, no normalization)
		# self.noise = self.noise_for_one_second / self.exptime**0.5 #Recalculate noise using the new exposure time grid
		# #Update WCS after downsampling
		# #self.WCS.cdelt =  [-plate_scale/3600.0, plate_scale/3600.0] #Define the X and Y scales
		# self.WCS.wcs.cdelt[0] = self.WCS.wcs.cdelt[0] * factor
		# self.WCS.wcs.cdelt[1] = self.WCS.wcs.cdelt[1] * factor
		# ref_pix_x = self.x_1d[0] / plate_scale
		# ref_pix_y = - self.y_1d[0] / plate_scale
		# self.WCS.wcs.crpix = [ref_pix_x, ref_pix_y] #Define reference pixel
		# # print('Total S/N: ',self.s2n())

	def gaussian_smooth(self, fwhm): #Gaussian smooth to a stated FWHM, useful for visualizing binned data without downsizing or smoothing out artifacts, can be done before downsampling
		standard_deviation = fwhm2std(fwhm)
		kernel = Gaussian2DKernel(x_stddev=standard_deviation, y_stddev=standard_deviation)
		self.data = convolve(self.data, kernel, boundary='extend') #Gaussian smooth data
		self.exptime = convolve(self.data, kernel, boundary='extend') #Gaussian smooth exposure time
		self.exptime = np.sqrt(convolve(self.noise**2, kernel, boundary='extend')) #Gaussian smooth noise
	def s2n(self, s2n_cut=1e-5): #Return total signal-to-noise value as a sanity check
		goodpix = np.isfinite(self.noise) & np.isfinite(self.data)# & (self.data/self.noise > s2n_cut)
		total_signal = bn.nansum(self.data[goodpix])
		total_noise = bn.nansum(self.noise[goodpix]**2)**0.5
		# print('total signal =',total_signal)
		# print('total noise = ', total_noise)
		return (total_signal / total_noise /np.size(self.data)**0.5) #/ self.area
	def import_wise_band3(self, file_data, primary_aor, background_percentile=2.0, deltafreq=1e6, deltav=0., frequency=0.): #Import a WISE Band 3 (12 micron) image and convert to [CII] flux units
		hdulist = fits.HDUList.fromstring(file_data) #Load fits file
		data = hdulist[0].data #Set up pointer to data
		#data = data * (1.8326e-6) #Convert WISE Band 3 DN to Jy (1.8326x10^-6 Jy/DN)  Source: https://wise2.ipac.caltech.edu/docs/release/allwise/expsup/sec4_3a.html#tbl1
		data = data* 2.9045e-3 * 1.091 #Convert WISE Band 3 DN to Jy  (2.9045e-06 Jy/DN) following the convention used by Anderson et al (2019), Apj, 882, 11   Source:https://wise2.ipac.caltech.edu/docs/release/prelim/expsup/sec2_3f.html, then apply the same color correction they used
		#data = 14.92 + 0.51*data #Convert WISE Band 3 12 micron data in mJ to [CII] intensity in units of K km s^-1 source: Anderson et al (2019), Apj, 882, 11 https://ui.adsabs.harvard.edu/abs/2019ApJ...882...11A]
		background = np.nanpercentile(data, background_percentile) #Attempt to subtract the background by finding a lowe end percentile (e.g. 2%), WISE image should be large enough to include background
		# data = 4.39138186e+01*data -1.93542638e+00*data**2 +  3.99690075e-02*data**3 #My own calibration comparing the Orion CII and WISE Band 3 data, see jupyter notebook
		data = 10.0*(data - background) #Trying my own calibration
		hdulist[0].data = data
		self.WCS.wcs.crval = [primary_aor.skycoord.ra.deg, primary_aor.skycoord.dec.deg] #set WCS reference pixel coordinates to be the RA and Dec of the primary AOR
		sky_header = self.WCS.to_header()
		sky_header['NAXIS'] =  2 #Looks like i need to manually set the NASXIS keywords in the sky header
		sky_header['NAXIS1'] =  self.nx
		sky_header['NAXIS2'] =  self.ny
		array, footprint = reproject_exact(hdulist[0], sky_header, parallel=True) #Reproject WISE Band 3 fits data onto sky
		# array, footprint = reproject_interp(hdulist[0], sky_header) #Reproject WISE Band 3 fits data onto sky
		#pyplot.imshow(array)
		self.signal = array
		hdulist.close() #Close the HDU List
	def import_wise_band1(self, file_data, primary_aor, background_percentile=2.0, deltafreq=1e6, deltav=0., frequency=0.): #Import a WISE Band 1 (3.4 micron) image and use it to remove the stellar continuum from an already imported WISE Band 3 image, put through the exact same conversion process but subtract at the end (see citation at the end)
		hdulist = fits.HDUList.fromstring(file_data) #Load fits file
		data = hdulist[0].data #Set up pointer to data
		#data = data * (1.8326e-6) #Convert WISE Band 3 DN to Jy (1.8326x10^-6 Jy/DN)  Source: https://wise2.ipac.caltech.edu/docs/release/allwise/expsup/sec4_3a.html#tbl1
		data = data * 2.9045e-3 * 1.091 #Convert WISE Band 3 DN to Jy  (2.9045e-06 Jy/DN) following the convention used by Anderson et al (2019), Apj, 882, 11   Source:https://wise2.ipac.caltech.edu/docs/release/prelim/expsup/sec2_3f.html, then apply the same color correction they used
		#data = 14.92 + 0.51*data #Convert WISE Band 3 12 micron data in mJ to [CII] intensity in units of K km s^-1 source: Anderson et al (2019), Apj, 882, 11 https://ui.adsabs.harvard.edu/abs/2019ApJ...882...11A]
		background = np.nanpercentile(data, background_percentile) #Attempt to subtract the background by finding a lowe end percentile (e.g. 2%), WISE image should be large enough to include background
		# data = 4.39138186e+01*data -1.93542638e+00*data**2 +  3.99690075e-02*data**3 #My own calibration comparing the Orion CII and WISE Band 3 data, see jupyter notebook
		data = 10.0*(data - background) #Trying my own calibration (same calib as WISE Band 3)
		hdulist[0].data = data
		self.WCS.wcs.crval = [primary_aor.skycoord.ra.deg, primary_aor.skycoord.dec.deg] #set WCS reference pixel coordinates to be the RA and Dec of the primary AOR
		sky_header = self.WCS.to_header()
		sky_header['NAXIS'] =  2 #Looks like i need to manually set the NASXIS keywords in the sky header
		sky_header['NAXIS1'] =  self.nx
		sky_header['NAXIS2'] =  self.ny
		array, footprint = reproject_exact(hdulist[0], sky_header, parallel=True) #Reproject WISE Band 1 fits data onto sky
		self.signal -= 0.158 * array #See stellar contribution removal method in Section 6.1.2 of Asabere et al. (2016) https://arxiv.org/pdf/1605.07565.pdf
		hdulist.close() #Close the HDU List
	def savefits(self, filename, kind='exposure'):
		sky_header = self.WCS.to_header() #Get header
		#Get data dpeneding on what user inputs
		if kind == 'exposure': #exp time in seconds per spatial resolution element
			output = self.exptime
		elif kind == 'signal': #artificial signal used to simulate data
			output = self.signal
		elif kind == 'noise': #Noise
			output = self.noise
		elif kind == 's2n': #signal-to-noise of simulated data
			output = self.data/self.noise
		elif kind == 'data': #simulated data
			output = self.data
		else: #catch error
			print("WARNING: Argument kind="+str(type(user_input)) + "is not valid.  Set kind equal to 'exposure', 'signal', 'noise', 's2n', or 'data'.")
		fits.writeto(filename, output, header=sky_header, overwrite=True) #Output fits file



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
		self.primary_frequency = '' #Primary frequency, used to define honeycomb step size from an aor
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
		#self.angle = 0.
	def rotate(self, angle): #Rotate array by the given angle
		self.reset_array_rotation() #Zero rotation angle by reversing the previous rotation
		self.set_array_rotation(angle) #Set new rotation angle
	def set_array_rotation(self, angle): #Rotate the positions of all the pixels around the 0th pixel (angle is inputted in degrees)
		zeroth_pixel = self.array_profile[0] #Grab central pixel
		zero_point = np.array([zeroth_pixel.x_mean.value, zeroth_pixel.y_mean.value]).T #Store zero point of central pixel as a 
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
	def paint(self, skyobj, time=1.0, cycles=1, TPOTF=False, Non=1): #Paint a single instance of the array profile onto a sky object, this is the base for all observation types including single pointing and maps
		#skyobj.total_exptime += time*cycles #Add exposure time from this to the total
		# deltaTa = get_deltaTa(Tsys=Tsys, TPOTF=TPOTF) #Get RMS antenna temperature 
		skyobj.fwhm = std2fwhm(self.array_profile[0].x_stddev) #Copy beam profile FWHM to sky object for later using to determine the convolution kernel to smooth with
		sky_xmax, sky_xmin, sky_ymin, sky_ymax = skyobj.extent #Grab limits of the sky coordinates
		paint_xmin, paint_xmax = self.x - self.range, self.x + self.range #Calculate coordinate range to paint (this is for optimization)
		paint_ymin, paint_ymax = self.y - self.range, self.y + self.range
		if paint_xmin < sky_xmin: paint_xmin = sky_xmin #Bring coordinate ranges within bounds if they fall outside of sky object's bounds
		if paint_ymin < sky_ymin: paint_ymin = sky_ymin
		if paint_xmax > sky_xmax: paint_xmax = sky_xmax
		if paint_ymax > sky_ymax: paint_ymax = sky_ymax
		ix2, ix1, iy1, iy2 = skyobj.get_range_indicies(paint_xmin, paint_xmax, paint_ymin, paint_ymax)  #Grab the indicies for the pixels on the sky over witch to paint onto (NOTE: x axis is inverted because RA increases to the left)
		chunk_of_signal = skyobj.signal[iy1:iy2, ix1:ix2] #Isolate the chunk of the signal array for painting at this particular position
		chunk_of_x = skyobj.x[iy1:iy2, ix1:ix2]
		chunk_of_y = skyobj.y[iy1:iy2, ix1:ix2]
		exptime = time * cycles
		for this_array_profile in self.array_profile: #Loop through each individual beam in array
			chunk_of_array_profile = this_array_profile(chunk_of_x, chunk_of_y) #Isolate the beam profile on the sky to use 
			sum_chunk_of_array_profile = bn.nansum(chunk_of_array_profile)
			if np.size(chunk_of_array_profile) > 0: #Error catch, only record beams that are inside the sky map
				convolved_signal = bn.nansum(chunk_of_signal * chunk_of_array_profile) / sum_chunk_of_array_profile #Convovle assumed signal on sky with beam profile
				skyobj.x_beam.append(this_array_profile.x_mean.value) #Save position, convolved signal, and exposure time for each beam in a list of the sky object for later regridding
				skyobj.y_beam.append(this_array_profile.y_mean.value)
				skyobj.exptime_beam.append(exptime)
				skyobj.signal_beam.append(convolved_signal * exptime)
				skyobj.beam_profiles.append(this_array_profile)
				skyobj.Non.append(Non)
	##### backup of old paint method
	# def paint(self, skyobj, time=1.0, cycles=1, TPOTF=False): #Paint a single instance of the array profile onto a sky object, this is the base for all observation types including single pointing and maps
	# 	#skyobj.total_exptime += time*cycles #Add exposure time from this to the total
	# 	# deltaTa = get_deltaTa(Tsys=Tsys, TPOTF=TPOTF) #Get RMS antenna temperature 
	# 	sky_xmax, sky_xmin, sky_ymin, sky_ymax = skyobj.extent #Grab limits of the sky coordinates
	# 	paint_xmin, paint_xmax = self.x - self.range, self.x + self.range #Calculate coordinate range to paint (this is for optimization)
	# 	paint_ymin, paint_ymax = self.y - self.range, self.y + self.range
	# 	if paint_xmin < sky_xmin: paint_xmin = sky_xmin #Bring coordinate ranges within bounds if they fall outside of sky object's bounds
	# 	if paint_ymin < sky_ymin: paint_ymin = sky_ymin
	# 	if paint_xmax > sky_xmax: paint_xmax = sky_xmax
	# 	if paint_ymax > sky_ymax: paint_ymax = sky_ymax
	# 	ix2, ix1, iy1, iy2 = skyobj.get_range_indicies(paint_xmin, paint_xmax, paint_ymin, paint_ymax)  #Grab the indicies for the pixels on the sky over witch to paint onto (NOTE: x axis is inverted because RA increases to the left)
	# 	chunk_of_signal = skyobj.signal[iy1:iy2, ix1:ix2] #Isolate the chunk of the signal array for painting at this particular position
	# 	#chunk_of_noise = skyobj.noise[iy1:iy2, ix1:ix2]
	# 	# if np.size(self.array_profile) > 1: #If LFA or HFA
	# 	for this_array_profile in self.array_profile: #Loop through each individual pixel
	# 		chunk_of_array_profile = this_array_profile(skyobj.x[iy1:iy2, ix1:ix2], skyobj.y[iy1:iy2, ix1:ix2]) #Isolate the piece of the array profile to use 
	# 		sum_chunk_of_array_profile = np.nansum(chunk_of_array_profile)
	# 		convolved_signal = chunk_of_array_profile *  np.nansum(chunk_of_signal * chunk_of_array_profile) / sum_chunk_of_array_profile
	# 		skyobj.data[iy1:iy2, ix1:ix2] += convolved_signal * time * cycles #Convolve pattern with expected signal and paint result onto the sky
	# 		skyobj.exptime[iy1:iy2, ix1:ix2] += time * cycles * chunk_of_array_profile #Convolve the exposure time with the profile for this particular pixel
	# 	skyobj.fwhm = std2fwhm(self.array_profile[0].x_stddev) #Copy beam profile FWHM to sky object for later using to determine the convolution kernel to smooth with
	# 	# else: #Else if 4GREAT
	# 	# 	chunk_of_array_profile = self.array_profile(skyobj.x[iy1:iy2, ix1:ix2], skyobj.y[iy1:iy2, ix1:ix2]) #Isolate the piece of the array profile to use 
	# 	# 	convolved_signal = chunk_of_array_profile * np.nansum(chunk_of_signal * chunk_of_array_profile) / np.nansum(chunk_of_array_profile) #Convolve the signal with the profile for this particular pixel
	# 	# 	skyobj.data[iy1:iy2, ix1:ix2] += convolved_signal * time * cycles #Convolve pattern with expected signal and paint result onto the sky
	# 	# 	skyobj.exptime[iy1:iy2, ix1:ix2] += time * cycles * chunk_of_array_profile #Convolve the exposure time with the profile for this particular pixel
	# 	# 	skyobj.fwhm = std2fwhm(self.array_profile.x_stddev) #Copy beam profile FWHM to sky object for later using to determine the convolution kernel to smooth with
	def single_point(self, skyobj, x=0., y=0., time=1.0, array_angle=0., cycles=1, Non=1): #Paint a single point observation onto the sky object	
		if self.freq > 0.: #Pass through the frequency to the sky object if it is specified for this array
			skyobj.freq = self.freq
		self.rotate(array_angle) #Set rotation angle
		self.position(x, y) #Set central position for the single pointing
		self.paint(skyobj, time=time, cycles=cycles, Non=Non) #Paint the single pointing to the sky object
	def map(self, skyobj, x=0., y=0., nx=1, ny=1, dx=1.0, dy=1.0, time=1.0, cycles=1, array_angle=0., map_angle=0., Non=1): #Paint a raster or OFT map observation onto the sky object
		if self.freq > 0.: #Pass through the frequency to the sky object if it is specified for this array
			skyobj.freq = self.freq
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
				self.paint(skyobj, time=time, cycles=cycles, Non=Non) #Paint the current step to the sky object
	def honeycomb(self, skyobj, x=0., y=0., time=1.0, cycles=1, array_angle=0., map_angle=0., LFA=False, HFA=False, Non=1):
		if self.freq > 0.: #Pass through the frequency to the sky object if it is specified for this array
			skyobj.freq = self.freq
		self.rotate(array_angle) #Set rotation angle
		if LFA: #Set multiplier for honeycomb offsets based on if the user specifically selects LFA or HFA or based which array you are using (default)
			honeycomb_multiplier = 6.34
		elif HFA:
			honeycomb_multiplier = 2.76
		elif self.primary_frequency == 'LFAV' or self.primary_frequency == 'LFAH':
			honeycomb_multiplier = 6.34
		elif self.primary_frequency == 'HFA':
			honeycomb_multiplier = 2.76
		else: 
			if self.type == 'LFAV' or self.type == 'LFAH': 
				honeycomb_multiplier = 6.34
			if self.type == 'HFA':
				honeycomb_multiplier = 2.76
		c, s = np.cos(np.radians(map_angle)), np.sin(np.radians(map_angle)) #Construct rotation matrix
		rot_matrix = np.array([[c, s], [-s, c]]).T
		honeycomb_map_positions = np.array([x,y]).T + (honeycomb_pattern*honeycomb_multiplier).dot(rot_matrix) #Construct a vector of honeycomb positions and dot it with the rotation matrix
		for honeycomb_map_position in honeycomb_map_positions:
			self.position(honeycomb_map_position[0], honeycomb_map_position[1]) #Set array position at this step
			self.paint(skyobj, time=time, cycles=cycles, Non=Non) #Paint the current step to the sky object
	def array_otf(self, skyobj, nblock_scan=1, nblock_perp=1, x=0, y=0, step=1.0, length=1.0, time=1.0, cycles=1, map_angle=0., direction='x', nscans=2, Non=1): #Paint a set of array otf blocks in a particular direction (x or y)
		if self.freq > 0.: #Pass through the frequency to the sky object if it is specified for this array
			skyobj.freq = self.freq
		block_angle = map_angle #the block angle is the same as the map angle
		if self.type == 'LFAV' or self.type == 'LFAH': #Set the size (in arcsec) of each block in the scan and perpendicular directions depending on what array is being used
			array_size = 72.6
		elif self.type == 'HFA':
			array_size = 31.6
		if direction.lower() == 'x':
			length_scan_x = array_size * length #- 0.5*step
			length_scan_y = array_size
			n_block_x = nblock_scan
			n_block_y = nblock_perp
		elif direction.lower() == 'y':
			length_scan_x = array_size 
			length_scan_y = array_size * length #- 0.5*step
			n_block_x = nblock_perp
			n_block_y = nblock_scan
		block_y, block_x = np.mgrid[0:n_block_y,0:n_block_x] #Generate block coordinates
		block_x = (block_x - 0.5*(n_block_x-1.0))*length_scan_x #Center block and scale coordinates to the proper lengths
		block_y = (block_y - 0.5*(n_block_y-1.0))*length_scan_y
		cos_block_angle = np.cos(np.radians(block_angle)) #Rotate blockcoordiunates by map angle (using a rotation matrix) and add starting position to block coordinates
		sin_block_angle = np.sin(np.radians(block_angle))
		rotated_block_x = x + (cos_block_angle*block_x + sin_block_angle*block_y)
		rotated_block_y = y + (-sin_block_angle*block_x + cos_block_angle*block_y)
		for ix in range(n_block_x): #Paint blocks along x direction
			for iy in range(n_block_y): #Paint blocks along y direction
				skyobj.x_map_center_points.append(rotated_block_x[iy, ix]) #Save center of each block into the sky object for later checking and/or plotting
				skyobj.y_map_center_points.append(rotated_block_y[iy, ix])
				self.array_otf_block(skyobj, x=rotated_block_x[iy, ix], y=rotated_block_y[iy, ix], step=step, length=length, time=time, cycles=cycles, map_angle=map_angle, direction=direction, nscans=nscans, Non=Non)
	def array_otf_block(self, skyobj, x=0., y=0., step=1.0, length=1.0, time=1.0, cycles=1, map_angle=0., direction='x', nscans=2, TPOTF=False, Non=1): #Paint a single block for an Array OTF Map onto
		if self.type == 'LFAV' or self.type == 'LFAH': #Set length of block in arcseconds to be length * array size
			length_arcsec = length * 72.6
			scan_spacing = 72.6 / (7.0 * nscans)
		elif self.type == 'HFA':
			length_arcsec = length * 31.6
			scan_spacing = 31.6 / (7.0 * nscans)
		if direction.lower() == 'x': 
			array_angle = -19.1 + map_angle #Set array angle to always be -19.1 for x-direction array OTF maps (note this angle is relative to the map angle)
			nx = np.ceil(length_arcsec/step).astype('int') #Note that observations round the number of steps per block scan up so we use np.ceil
			ny = nscans
			dx = step
			dy = scan_spacing
		elif direction.lower() == 'y':
			array_angle  = 10.9 + map_angle #Set array angle to always be +10.9 for y-direction array OTF maps (note this angle is relative to the map angle)
			nx = nscans
			ny = np.ceil(length_arcsec/step).astype('int') #Note that observations round the number of steps per block scan up so we use np.ceil
			dx = scan_spacing
			dy = step
		else:
			print('ERROR: Array OTF block direction needs to be specified as x or y.')
		self.map(skyobj, x=x, y=y, nx=nx, ny=ny, dx=dx, dy=dy, time=time, cycles=cycles, array_angle=array_angle, map_angle=map_angle, Non=Non) #Paint a map
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
	def __init__(self, fwhm=14.1, type='LFAH', amplitude=1.0, r=31.8, angle=0., freq=1897.420e9):
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
	def __init__(self, fwhm=14.1, type='LFAV', amplitude=1.0, r=31.8, angle=0., freq=1897.420e9):
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
	def __init__(self, fwhm=6.3, type='HFA', amplitude=1.0, r=13.6, angle=0., freq=4744.77749e9):
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
	def __init__(self, fwhm=50.0, type='4G_1', amplitude=1.0, freq=550.0e9):
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
	def __init__(self, fwhm=30.0, type='4G_2', amplitude=1.0, freq=980.0e9):
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
	def __init__(self, fwhm=19.0, type='4G_3', amplitude=1.0, freq=1390.0e9):
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
	def __init__(self, fwhm=11.0, type='4G_4', amplitude=1.0, freq=2540.0e9):
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



