import numpy as np
from scipy.interpolate import griddata
from netCDF4 import Dataset
import math
from scipy.ndimage import gaussian_filter
import logging

try:
    import gsw
    HAS_GSW = True
except ImportError:
    HAS_GSW = False

log = logging.getLogger("Functions")

# Constants for missing value thresholds
MISSING_VAL_THRESHOLD_STRICT = 1000.0  # Divisor for strict missing value checks
MISSING_VAL_THRESHOLD_LOOSE = 10.0     # Divisor for loose missing value checks

# ---------- 1 REGRID DATA ----------
def regrid(var, var_lons, var_lats, target_lon, target_lat, tmask, missingVal):

	tDim = var.shape[0]

	# duplicate edge pixels 
	for iy in range(0,var_lons.shape[0]):
		var[:,iy,0] = (var[:,iy,1]+var[:,iy,-2])/2
		var[:,iy,-1] =  var[:,iy,0]
	for ix in range(0,var_lons.shape[0]):
		var[:,-1,ix] = var[:,-2,ix] 

	# adjust nav_lon to WOA grid 
	var_lons[ var_lons < 0 ] = var_lons[ var_lons < 0 ] + 360

	list_lons = var_lons.ravel()
	list_lats = var_lats.ravel()
	points = np.column_stack([list_lats,list_lons])

	if( len(target_lon.shape) < 2):
		# create obs_lon and obs_lat
		# -179.5->179.5; -89.5->89.5, 1 deg res
		target_lon = np.arange(0.5,360.5,1)
		target_lat = np.arange(-89.5,90.5,1)

	# re-interpolate onto WOA grid, centred on greenwich meridian
	data_out = np.zeros((var.shape[0],target_lat.shape[0],target_lon.shape[0]),dtype=float) + np.nan

	for t in range(0,tDim):
		# preprocesss input data to remove mask
		vals = []
		maskVals = []
		for iy in range(0,var_lons.shape[0]):
			for ix in range(0,var_lons.shape[1]):
				maskVals.append(tmask[iy,ix])
				vals.append(var[t,iy,ix])

		vals = np.array(vals)

		valsFilt = []
		pointsFilt = []
		for p in range(0,points.shape[0]):
			if maskVals[p] == 1:
				valsFilt.append(vals[p])
				pointsFilt.append( ( points[p,0], points[p,1] ) )

		pointsFilt = np.array(pointsFilt)
		valsFilt = np.array(valsFilt)

		if pointsFilt.shape[0] > 0:
			grid_lon,grid_lat = np.meshgrid(target_lon,target_lat)
			data_out[t,:,:] = griddata(pointsFilt,valsFilt,(grid_lat,grid_lon), method='linear')

			data_out_near = griddata(pointsFilt,valsFilt,(grid_lat[:,0:2],grid_lon[:,0:2]), method='nearest') # take these values as has errors at date line due to itnerpoloation
			data_out[t,:,0:2] = data_out_near
			data_out_near = griddata(pointsFilt,valsFilt,(grid_lat[:,-3:],grid_lon[:,-3:]), method='nearest') # take these values as has errors at date line due to itnerpoloation
			data_out[t,:,-3:] = data_out_near[:,-3:]

			# northern boundary
			data_out_near = griddata(pointsFilt,valsFilt,(grid_lat[-3:,:],grid_lon[-3:,:]), method='nearest') # take these values as has errors at date line due to itnerpoloation
			data_out[t,-3:,:] = data_out_near[-3:,:]

	# tidy up
	data_out[ data_out > missingVal/MISSING_VAL_THRESHOLD_STRICT ] = missingVal

	return data_out

# ---------- 2 SUB DOMAIN DATA ----------
def subDomain(lonLim, latLim, in_data):
	lonStart = int(lonLim[0])+180
	lonEnd = int(lonLim[1])+180
	latStart = int(latLim[0])+90
	latEnd = int(latLim[1])+90
	subDiff = np.zeros(in_data.shape) * np.nan
	subDiff[:, latStart:latEnd, lonStart:lonEnd] = in_data[:, latStart:latEnd, lonStart:lonEnd]
	return subDiff

# ---------- 3 SUB DOMAIN ORCA DATA ----------
def subDomainORCA(lonLim, latLim, var_lons, var_lats, in_data, landMask, volMask, missingVal):
	"""
	Apply spatial domain masking to ORCA grid data.

	Fully vectorized implementation - no Python loops over time/depth.
	"""
	lonStart = int(lonLim[0])
	lonEnd = int(lonLim[1])
	latStart = int(latLim[0])
	latEnd = int(latLim[1])

	# Create 2D mask for lat/lon bounds (vectorized)
	mask_2d = (
		(var_lons >= lonStart) &
		(var_lons <= lonEnd) &
		(var_lats >= latStart) &
		(var_lats <= latEnd)
	)

	# Combine with land mask
	invalid_2d = ~mask_2d | np.isnan(landMask)

	log.debug(f"volMask shape: {volMask.shape}, in_data shape: {in_data.shape}")

	if len(in_data.shape) == 3:
		# 3D data (time, y, x): broadcast 2D mask across time dimension
		# Using advanced indexing with broadcasting
		in_data[:, invalid_2d] = missingVal

	elif len(in_data.shape) == 4:
		# 4D data (time, depth, y, x): broadcast 2D mask across time and depth
		# Apply 2D spatial mask to all time and depth levels
		in_data[:, :, invalid_2d] = missingVal

		# Apply 3D volume mask if dimensions match
		if volMask.shape[0] == in_data.shape[1]:
			ind_vol = np.isnan(volMask)
			# Broadcast vol mask across time dimension
			in_data[:, ind_vol] = missingVal

	return in_data

# ---------- 4 SURFACE DATA (sum at surface) ----------
def surfaceData(var, var_lons, var_lats, units, area, landMask, volMask, missingVal, lonLim, latLim):

	tDim = var.shape[0]

	var = subDomainORCA(lonLim, latLim, var_lons, var_lats, var, landMask, volMask, missingVal)

	# filter data for missingVal (in-place to avoid copy)
	varNan = var
	varNan[ varNan > missingVal/MISSING_VAL_THRESHOLD_LOOSE ] = np.nan
	varNan[ varNan < -missingVal/MISSING_VAL_THRESHOLD_LOOSE ] = np.nan

	# Vectorized calculation: compute total across all time steps at once
	total = np.nansum(varNan * area * units / tDim)

	# Vectorized monthly statistics computation
	varScaled = varNan * units
	monthly_sums = np.nansum(varNan * area * units / tDim, axis=(1, 2))
	percentiles = np.nanquantile(varScaled, [0.05, 0.25, 0.5, 0.75, 0.95], axis=(1, 2))
	monthly_min, monthly_first, monthly_median, monthly_third, monthly_max = percentiles

	# Build monthly list from vectorized arrays
	monthly = [[monthly_sums[t], monthly_min[t], monthly_first[t], monthly_median[t], monthly_third[t], monthly_max[t]]
	           for t in range(tDim)]

	return total, monthly

# ---------- 5 VOLUME DATA (sum over all depths) ----------
def volumeData(var, var_lons, var_lats, units, vol, landMask, volMask, missingVal, lonLim, latLim):

	var = subDomainORCA(lonLim, latLim, var_lons, var_lats, var, landMask, volMask, missingVal)

	tDim = var.shape[0]

	# filter data for missingVal (in-place to avoid copy)
	varNan = var
	varNan[ varNan > missingVal/MISSING_VAL_THRESHOLD_LOOSE ] = np.nan
	varNan[ varNan < -missingVal/MISSING_VAL_THRESHOLD_LOOSE ] = np.nan
	
	# Vectorized calculation: compute total across all time steps at once
	total = np.nansum(varNan * vol * units / tDim)

	# Vectorized monthly statistics computation
	varScaled = varNan * units
	monthly_sums = np.nansum(varNan * vol * units / tDim, axis=(1, 2, 3))
	percentiles = np.nanquantile(varScaled, [0.05, 0.25, 0.5, 0.75, 0.95], axis=(1, 2, 3))
	monthly_min, monthly_first, monthly_median, monthly_third, monthly_max = percentiles

	# Build monthly list from vectorized arrays
	monthly = [[monthly_sums[t], monthly_min[t], monthly_first[t], monthly_median[t], monthly_third[t], monthly_max[t]]
	           for t in range(tDim)]

	return total, monthly

# ---------- 6 LEVEL DATA (sum at certain depth) ----------
def levelData(var, var_lons, var_lats, units, area, landMask, volMask, missingVal, lonLim, latLim, level):

	var = subDomainORCA(lonLim, latLim, var_lons, var_lats, var, landMask, volMask, missingVal)

	tDim = var.shape[0]

	# filter data for missingVal (in-place to avoid copy)
	varNan = var
	varNan[ varNan > missingVal/MISSING_VAL_THRESHOLD_LOOSE ] = np.nan
	varNan[ varNan < -missingVal/MISSING_VAL_THRESHOLD_LOOSE ] = np.nan

	# Vectorized calculation: compute total across all time steps at once
	total = np.nansum(varNan[:, level, :, :] * area * units / tDim)

	# Vectorized monthly statistics computation
	varScaled = varNan[:, level, :, :] * units
	monthly_sums = np.nansum(varNan[:, level, :, :] * area * units / tDim, axis=(1, 2))
	percentiles = np.nanquantile(varScaled, [0.05, 0.25, 0.5, 0.75, 0.95], axis=(1, 2))
	monthly_min, monthly_first, monthly_median, monthly_third, monthly_max = percentiles

	# Build monthly list from vectorized arrays
	monthly = [[monthly_sums[t], monthly_min[t], monthly_first[t], monthly_median[t], monthly_third[t], monthly_max[t]]
	           for t in range(tDim)]

	return total, monthly

# ---------- 7 INTEGRATE DATA (sum over range of depths) ----------
def integrateData(var, var_lons, var_lats, depthFrom, depthTo, units, vol, landMask, volMask, missingVal, lonLim, latLim):

	var = subDomainORCA(lonLim, latLim, var_lons, var_lats, var, landMask, volMask, missingVal)

	tDim = var.shape[0]

	# filter data for missingVal (in-place to avoid copy)
	varNan = var
	varNan[ varNan > missingVal/MISSING_VAL_THRESHOLD_LOOSE ] = np.nan
	varNan[ varNan < -missingVal/MISSING_VAL_THRESHOLD_LOOSE ] = np.nan

	# Vectorized calculation: compute total across all time steps at once
	total = np.nansum(varNan[:, depthFrom:depthTo+1, :, :] * vol[depthFrom:depthTo+1, :, :] * units / tDim)

	# Vectorized monthly statistics computation
	varScaled = varNan[:, depthFrom:depthTo+1, :, :] * units
	monthly_sums = np.nansum(varNan[:, depthFrom:depthTo+1, :, :] * vol[depthFrom:depthTo+1, :, :] * units / tDim, axis=(1, 2, 3))
	percentiles = np.nanquantile(varScaled, [0.05, 0.25, 0.5, 0.75, 0.95], axis=(1, 2, 3))
	monthly_min, monthly_first, monthly_median, monthly_third, monthly_max = percentiles

	# Build monthly list from vectorized arrays
	monthly = [[monthly_sums[t], monthly_min[t], monthly_first[t], monthly_median[t], monthly_third[t], monthly_max[t]]
	           for t in range(tDim)]

	return total, monthly

# ---------- 8 AVERAGE DATA (average over range of depths) ----------
def volumeDataAverage(var, var_lons, var_lats, depthFrom, depthTo, units, vol, landMask, volMask, missingVal, lonLim, latLim):

	var = subDomainORCA(lonLim, latLim, var_lons, var_lats, var, landMask, volMask, missingVal)

	tDim = var.shape[0]

	# filter data for missingVal (in-place to avoid copy)
	varNan = var
	varNan[ varNan > missingVal/MISSING_VAL_THRESHOLD_LOOSE ] = np.nan
	varNan[ varNan < -missingVal/MISSING_VAL_THRESHOLD_LOOSE ] = np.nan

	if vol.shape[0] == varNan.shape[1]:
		vol_masked = vol.copy()  # Need copy here as we modify it
	if varNan.shape[1] == 1:
		vol_masked = np.reshape( vol[0,:,:].copy(), (1,varNan.shape[2], varNan.shape[3]) )  # Need copy for reshape
	ind_masked = np.isnan(varNan[0,:,:,:])

	vol_masked[ ind_masked ] = np.nan
	volNorm = np.nansum(vol_masked[depthFrom:depthTo+1,:,:])

	# Vectorized calculation: compute total across all time steps at once
	total = np.nansum(varNan[:, depthFrom:depthTo+1, :, :] * vol_masked[depthFrom:depthTo+1, :, :] / volNorm * units / tDim)

	# Vectorized monthly statistics computation
	varScaled = varNan[:, depthFrom:depthTo+1, :, :] * units
	monthly_sums = np.nansum(varNan[:, depthFrom:depthTo+1, :, :] * vol_masked[depthFrom:depthTo+1, :, :] / volNorm * units, axis=(1, 2, 3))
	percentiles = np.nanquantile(varScaled, [0.05, 0.25, 0.5, 0.75, 0.95], axis=(1, 2, 3))
	monthly_min, monthly_first, monthly_median, monthly_third, monthly_max = percentiles

	# Build monthly list from vectorized arrays
	monthly = [[monthly_sums[t], monthly_min[t], monthly_first[t], monthly_median[t], monthly_third[t], monthly_max[t]]
	           for t in range(tDim)]

	return total, monthly

# ---------- 9 OBS DATA (RMSE to class fo observations) ----------
def unit_vector(vector):
	return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def observationData(obs, obs_lon, obs_lat, var, var_lons, var_lats, obsCenter, tmask, missingVal, gammaFlag, lonLim, latLim):

	# wrap in long if obs data centered differently (target is greenwich centering of array)
	if obsCenter == 'zero':
		obs = np.roll(obs,int(len(obs_lon)/2),axis=2)
		obs_lon = np.roll(obs_lon,int(len(obs_lon)/2),axis=0)

		for i in range(0,len(obs_lon)):
			if obs_lon[i] > 180:
				obs_lon[i] = obs_lon[i] - 360

	data_out = regrid(var, var_lons, var_lats, obs_lon, obs_lat, tmask, missingVal)

	obs[ obs > missingVal/MISSING_VAL_THRESHOLD_STRICT ] = missingVal

	# spread obs to account for DTA
	spread = 2 # degree std
    
	modified_weights = np.copy(obs)
	modified_obs = np.copy(obs)

	for t in range(0,12):
		slice = np.copy(obs[t,:,:])
		weight = np.copy(slice)
		slice[ slice <= -missingVal ] = 0
		slice[ slice != 0 ] = 1
		slice_spead = gaussian_filter(slice, sigma=spread)
		ind = np.where( slice_spead < 0.1)
		slice_spead[ ind ] = 0
		grid_lon,grid_lat = np.meshgrid(obs_lon,obs_lat)

		if np.max(slice) == 1: # not empty
			# source data
			ind_obs = np.where( slice == 1)
			list_lons = grid_lon[ ind_obs ].ravel()
			list_lats = grid_lat[ ind_obs ].ravel()
			points = np.column_stack([list_lats,list_lons])
			vals = []

			for p in range(0, len(ind_obs[0])):
				vals.append(obs[t, ind_obs[0][p], ind_obs[1][p] ])

			vals = np.array(vals)
			filled_obs = griddata(points,vals,(grid_lat,grid_lon), method='nearest')
			filled_obs[ ind ] = np.nan
			modified_weights[t,:,:] = slice_spead
			modified_obs[t,:,:] = filled_obs
		else:
			modified_weights[t,:,:] = np.zeros( slice.shape )
			modified_obs[t,:,:] = np.zeros( slice.shape )

	# Now, compare to obs data as its now on the same grid
	data_out[ data_out > missingVal/MISSING_VAL_THRESHOLD_STRICT ] = np.nan

	diff = (data_out-obs) * (data_out-obs)
	diff_mod = (data_out-modified_obs) * (data_out-modified_obs)
	diff[ diff < -missingVal/MISSING_VAL_THRESHOLD_STRICT ] = np.nan
	diff_mod[ diff_mod < -missingVal/MISSING_VAL_THRESHOLD_STRICT ] = np.nan

	diff[ diff >  1E6 ] = np.nan
	diff_mod[ diff_mod >  1E6 ] = np.nan
    
	# calc mean and std differences for filtering outliers
	mean = np.nanmean(diff)
	std = np.nanstd(diff)
	z_dat = (diff-mean)/std # Z statistic for outliers 
	diff[ z_dat > 10 ] = np.nan
	diff[ z_dat < -10 ] = np.nan
	mean = np.nanmean(diff_mod)
	std = np.nanstd(diff_mod)
	z_dat = (diff-mean)/std 
	diff_mod[ z_dat > 10 ] = np.nan
	diff_mod[ z_dat < -10 ] = np.nan

	# filter the diff by the limits to long and lat
	diff = subDomain(lonLim, latLim, diff)
	diff_mod = subDomain(lonLim, latLim, diff_mod)

	ind = np.isnan(diff_mod)
	modified_weights[ ind ] = 0

	# annual rmse
	total = np.nansum(diff[:,:,:])
	count = np.count_nonzero(~np.isnan(diff[:,:,:]))
	rmse_annual = np.sqrt( (total/count) )

	total_mod = np.nansum( diff_mod[:,:,:] * modified_weights[:,:,:] )
	count_mod = np.count_nonzero( modified_weights[:,:,:] )
	rmse_annual_mod = np.sqrt( (total_mod/count_mod) )

	# repeat on monthly basis using same z-dat criteria for outliers
	monthly_rms = []
	for t in range(0,12):
		total = np.nansum(diff[t,:,:])
		count = np.count_nonzero(~np.isnan(diff[t,:,:]))
		rmse = np.sqrt( (total/count) )

		if np.max(modified_weights[t,:,:]) != np.min(modified_weights[t,:,:]) :
			total_mod = np.nansum(diff_mod[t,:,:] * modified_weights[t,:,:])
			count_mod = np.count_nonzero(modified_weights[t,:,:])
			rmse_mod = np.sqrt( (total_mod/count_mod) )

			monthly_rms.append(rmse_mod)
		else:
			monthly_rms.append(-1)

	gammaResults = [[],[]]
	if gammaFlag == True:
		# create an array of vectors for each point on the surface
		vectors = np.zeros((3,180,360))
		latlon = np.zeros((2,180,360))
		for ix in range(0,360):
			for iy in range(0,180):
				vectors[0,iy,ix] = math.cos(np.radians(obs_lon[ix]))*math.cos(np.radians(obs_lat[iy]))
				vectors[1,iy,ix] = math.sin(np.radians(obs_lon[ix]))*math.cos(np.radians(obs_lat[iy]))
				vectors[2,iy,ix] = math.sin(np.radians(obs_lat[iy]))
				latlon[0,iy,ix] = obs_lat[iy]
				latlon[1,iy,ix] = obs_lon[ix]

		a_vals = [0.1,1,2,3,5,10] # this sampling of the relationship
		b_vals = [ [],[],[],[],[],[] ] #create empty lists to average
        
		d1 = np.absolute(data_out - obs)

		for ix in range(0,obs.shape[2]):
			for iy in range(0,obs.shape[1]):
				for t in range(0,obs.shape[0]):
					if obs[t,iy,ix] > -999 and obs[t,iy,ix] < missingVal/10 and data_out[t,iy,ix] < missingVal/10:
						# create dot product array
						angs = np.zeros((180,360))
						vect = vectors[:,iy,ix]

						for x in range(0,360):
							for y in range(0,180):
								angs[y,x] = angle_between(vect,vectors[:,y,x])

						for ai in range(0,len(a_vals)):
							a = a_vals[ai]

							prev_r = False

							for r in range(1,10):
								if prev_r == False:
									sub = np.copy(angs)
									rad = np.radians(r)
									sub[ sub > rad ] = missingVal
									sub_dist = np.copy(sub)
									sub[ sub < rad ] = 1

									d = np.copy(data_out[t,:,:])
									d = d * sub

									sub_diff = np.abs(d-obs[t,iy,ix])/obs[t,iy,ix] * 100 # % change
									min_diff = np.amin(sub_diff)

									c1 = sub_diff / a
									c2 = np.degrees(sub_dist) / r 
									gamma = np.sqrt( c1*c1 + c2*c2 )
									min_gamma = np.amin(gamma)
									if min_gamma < 1:
										b_vals[ai].append(r)
										log.debug(f"GAM fit: ix={ix}, iy={iy}, obs={obs[t,iy,ix]}, data_out={data_out[t,iy,ix]}, a={a}, r={r}, min_gamma={min_gamma}")
										prev_r = True

		# output average b-vals for each a-val
		for ai in range(0,len(a_vals)):
			mean_b = np.mean(b_vals[ai])

			gammaResults[0].append(a_vals[ai])
			gammaResults[1].append(mean_b)

	return rmse_annual_mod, monthly_rms #, gammaResults

# ---------- 10 BLOOM FUNCTION ----------
def interpolate(x, x1, x2, y1, y2):
	return y1 + (x-x1)*(y2-y1)/(x2-x1)

def bloom(var, var_lons, var_lats, missingVal, lonLim, latLim):

	data_out = regrid(var, var_lons, var_lats, np.array([0]), np.array([0]), missingVal)

	bloomVal=np.zeros((5,data_out.shape[1],data_out.shape[2]))

	# fit a time series for each point
	for x in range(0,data_out.shape[2]):
		for y in range(0,data_out.shape[1]):
			# get time series 
			if np.sum(data_out[0,y,x]) < missingVal/10 :
				timeSeries = data_out[:,y,x]

				# extend for interpolation
				timeSeries = np.concatenate((timeSeries,timeSeries,timeSeries), axis=0)

				# set threshold and initialise indecis
				thresh = 1.05 * np.median(timeSeries)
				max_t = np.argmax(timeSeries) + 12
				t_low = max_t  
				t_high = max_t
				meanAtPoint = np.mean(timeSeries)
                
				# find start/end points
				while timeSeries[t_low] > thresh:
					t_low = t_low - 1
				t_low = interpolate(thresh, timeSeries[t_low], timeSeries[t_low+1], t_low, t_low+1)
				while timeSeries[t_high] > thresh:
					t_high = t_high + 1
				t_high = interpolate(thresh, timeSeries[t_high-1], timeSeries[t_high], t_high-1, t_high)

				# get the peak bloom value
				maxVal = timeSeries[max_t]

				# set the start/end points to month
				while t_low > 11:
					t_low = t_low - 12
				while t_high > 11:
					t_high = t_high - 12
				while max_t > 11:
					max_t = max_t - 12
				if t_low < 0:
					t_low = t_low + 12

				# get duration
				duration = 0
				if t_high > t_low:
					duration = t_high - t_low
				else:
					duration = (t_high +12) - t_low
                
				bloomVal[0,y,x] = maxVal/meanAtPoint
				bloomVal[1,y,x] = max_t
				bloomVal[2,y,x] = t_low
				bloomVal[3,y,x] = t_high
				bloomVal[4,y,x] = duration

	# filter the diff by the limits to long and lat
	bloomVal = subDomain(lonLim, latLim, bloomVal)

	# scale according to area and give stats
	# max peak, median peak, avg duration, s.d. duration

	maxPeak = np.nanmax(bloomVal[0,:,:])
	peaks = bloomVal[0,:,:]
	peaks[ peaks == 0 ] = np.nan
	medPeak = np.nanmedian(peaks)

	duration = bloomVal[4,:,:]
	duration[ duration == 0 ] = np.nan
	meanDur = np.nanmean(duration)
	sdDur = np.nanstd(duration)

	# add column headings we'll need for output (last one is monthly output holder)
	return maxPeak, medPeak, meanDur, sdDur, ['maxPeak','medPeak', 'meanDur', 'sdDur'], None

def getSlope(xs,ys):
	mx = np.mean(xs)
	my = np.mean(ys)
	mxy = np.mean(xs*ys)
	mxx = np.mean(xs*xs)
	m = ( mx*my - mxy) / ( mx*mx - mxx ) 
	c = my - m * mx

	return m, c

def trophic(var, var_lons, var_lats, missingVal, lonLim, latLim):

	trophicVals=np.zeros((4, 12, 180, 360 ))
 
	for p in range(0,3):
		data_out = regrid(var[p], var_lons, var_lats, np.array([0]), np.array([0]), missingVal)
		trophicVals[p,:,:,:] = data_out

	y_vals = [ 0, 1, 2 ]
	for x in range(0,trophicVals[0,:,:,:].shape[2]):
		for y in range(0,trophicVals[0,:,:,:].shape[1]):
			for t in range(0,trophicVals[0,:,:,:].shape[0]):
				x_vals = trophicVals[0:3,t,y,x]
				# fit
				if np.min(x_vals) != 0:
					if ~np.isnan(x_vals).any():
						m, c = getSlope(x_vals,y_vals)
						trophicVals[3,t,y,x] = m

	trophicVals[ trophicVals > missingVal/MISSING_VAL_THRESHOLD_LOOSE ] = np.nan

	trophicVals = subDomain(lonLim, latLim, trophicVals)

	mean0 = np.nanmean(trophicVals[0,:,:,:])
	mean1 = np.nanmean(trophicVals[1,:,:,:])
	mean2 = np.nanmean(trophicVals[2,:,:,:])
	mean_m = np.nanmean(trophicVals[3,:,:,:])
	mean_m_monthly = np.zeros((12))

	for t in range(0,12):
		mean_m_monthly[t]= np.nanmean(trophicVals[3,t,:,:])

	return mean0, mean1, mean2, mean_m, ['level1','level2', 'level3', 'meanSlope'], mean_m_monthly


# ---------- 11 AOU COMPUTATION ----------
# ORCA2 depth values (meters) for pressure calculation
ORCA2_DEPTHS = np.array([
    5.0, 15.0, 25.0, 35.5, 46.5, 58.5, 71.5, 86.0,
    102.5, 121.5, 143.5, 169.0, 198.5, 233.0, 273.5,
    321.5, 378.5, 446.0, 526.5, 622.5, 737.5, 875.5,
    1041.5, 1241.5, 1482.5, 1772.5, 2121.5, 2541.5,
    3046.5, 3653.5, 4383.5
])


def computeAOU(o2_data, temp_data, sal_data, depth_index=17, missingVal=1e20):
    """
    Compute Apparent Oxygen Utilization (AOU) at a specific depth.

    AOU = O2_saturation - O2_measured

    Args:
        o2_data: Oxygen concentration (time, depth, y, x) in mol/L
        temp_data: Temperature (time, depth, y, x) in degrees C
        sal_data: Salinity (time, depth, y, x) in PSU
        depth_index: Depth level index (default 17 for ~300m)
        missingVal: Missing value indicator

    Returns:
        AOU in µmol/L as (time, y, x) array
    """
    if not HAS_GSW:
        log.warning("GSW library not available, cannot compute AOU")
        return None

    # Get pressure from depth (dbar ≈ meters)
    pressure = ORCA2_DEPTHS[depth_index]

    # Extract data at specified depth
    o2_at_depth = o2_data[:, depth_index, :, :]
    temp_at_depth = temp_data[:, depth_index, :, :]
    sal_at_depth = sal_data[:, depth_index, :, :]

    # Mask missing values
    o2_at_depth = np.where(o2_at_depth > missingVal / MISSING_VAL_THRESHOLD_LOOSE, np.nan, o2_at_depth)
    temp_at_depth = np.where(temp_at_depth > missingVal / MISSING_VAL_THRESHOLD_LOOSE, np.nan, temp_at_depth)
    sal_at_depth = np.where(sal_at_depth > missingVal / MISSING_VAL_THRESHOLD_LOOSE, np.nan, sal_at_depth)

    # Calculate O2 saturation using GSW
    # gsw.O2sol returns saturation in µmol/kg
    o2_sat = gsw.O2sol(sal_at_depth, temp_at_depth, pressure, 0, 0)

    # Convert O2 from mol/L to µmol/L (multiply by 1e6)
    o2_measured = o2_at_depth * 1e6

    # Convert O2_sat from µmol/kg to µmol/L (multiply by density ~1.025)
    o2_sat_umol_L = o2_sat * 1.025

    # Calculate AOU
    aou = o2_sat_umol_L - o2_measured

    return aou


def aouData(o2_data, temp_data, sal_data, var_lons, var_lats, landMask, volMask, missingVal, lonLim, latLim, depth_index=17):
    """
    Compute AOU statistics for analyser output.

    Args:
        o2_data: Oxygen concentration (time, depth, y, x) in mol/L
        temp_data: Temperature (time, depth, y, x) in degrees C
        sal_data: Salinity (time, depth, y, x) in PSU
        var_lons: Longitude coordinates
        var_lats: Latitude coordinates
        landMask: 2D land mask
        volMask: 3D volume mask
        missingVal: Missing value indicator
        lonLim: Longitude limits [min, max]
        latLim: Latitude limits [min, max]
        depth_index: Depth level index (default 17 for ~300m)

    Returns:
        Tuple of (annual_mean, monthly_stats) matching volumeDataAverage format
    """
    # Compute AOU
    aou = computeAOU(o2_data, temp_data, sal_data, depth_index, missingVal)

    if aou is None:
        return -1, np.full((12, 6), -1.0)

    tDim = aou.shape[0]

    # Apply spatial domain masking
    aou = subDomainORCA(lonLim, latLim, var_lons, var_lats, aou, landMask, volMask, missingVal)

    # Filter missing values
    aou[aou > missingVal / MISSING_VAL_THRESHOLD_LOOSE] = np.nan
    aou[aou < -missingVal / MISSING_VAL_THRESHOLD_LOOSE] = np.nan

    # Compute annual mean (global average)
    total = np.nanmean(aou)

    # Compute monthly statistics
    monthly_means = np.nanmean(aou, axis=(1, 2))
    percentiles = np.nanquantile(aou, [0.05, 0.25, 0.5, 0.75, 0.95], axis=(1, 2))
    monthly_min, monthly_first, monthly_median, monthly_third, monthly_max = percentiles

    # Build monthly list
    monthly = [[monthly_means[t], monthly_min[t], monthly_first[t], monthly_median[t], monthly_third[t], monthly_max[t]]
               for t in range(tDim)]

    return total, monthly


# ---------- 12 E-DEPTH (Z_STAR) COMPUTATION ----------
def calculate_e_depth(exp_flux, depth_vals, mld_2d, landMask, missingVal):
    """
    Calculate e-folding depth (z_star) for carbon export flux remineralization.

    z_star is the depth scale over which export flux decreases by a factor of e,
    measured from a reference depth z0 = MLD + 10m.

    Args:
        exp_flux: Export flux (depth, y, x) - time-averaged
        depth_vals: 1D array of depth values in meters
        mld_2d: Mixed layer depth (y, x) in meters - time-averaged
        landMask: 2D land mask (1 = ocean, NaN = land)
        missingVal: Missing value indicator

    Returns:
        z_star: 2D array (y, x) of e-folding depths in meters
    """
    nz, ny, nx = exp_flux.shape
    z_star = np.full((ny, nx), np.nan)

    for i in range(ny):
        for j in range(nx):
            if np.isnan(landMask[i, j]):
                continue

            mld_val = mld_2d[i, j]
            if np.isnan(mld_val) or mld_val <= 0:
                continue

            z0_depth = mld_val + 10.0
            z0_idx = np.argmin(np.abs(depth_vals - z0_depth))

            flux_z0 = exp_flux[z0_idx, i, j]
            if np.isnan(flux_z0) or flux_z0 <= 0 or flux_z0 > missingVal / MISSING_VAL_THRESHOLD_LOOSE:
                continue

            target_flux = flux_z0 / np.e

            for k in range(z0_idx, nz):
                flux_k = exp_flux[k, i, j]
                if np.isnan(flux_k) or flux_k > missingVal / MISSING_VAL_THRESHOLD_LOOSE:
                    continue

                if flux_k <= target_flux:
                    if k == z0_idx:
                        z_star[i, j] = depth_vals[k] - z0_depth
                    else:
                        flux_before = exp_flux[k-1, i, j]
                        if not np.isnan(flux_before) and flux_before > flux_k:
                            frac = (flux_before - target_flux) / (flux_before - flux_k)
                            depth_at_target = depth_vals[k-1] + frac * (depth_vals[k] - depth_vals[k-1])
                            z_star[i, j] = depth_at_target - z0_depth
                    break

    return z_star
