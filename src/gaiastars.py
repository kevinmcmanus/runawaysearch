#AUTHOR: KEVIN MCMANUS
import tempfile, os
import numpy as np
import pandas as pd
import pickle

import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.table import Table
from astropy.units import Quantity
from astroquery.gaia import Gaia
#from pyvo.dal import TAPService

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm







class gaiastars():
	
	#default query columns:
	column_list = ['source_id', 'ra','dec','parallax','pmra','pmdec','radial_velocity',
					'phot_g_mean_mag','phot_bp_mean_mag', 'phot_rp_mean_mag', 'e_bp_min_rp_val', 'a_g_val'] #,'r_est'] # r_est not in gaia archive
	tap_service_url = "http://gaia.ari.uni-heidelberg.de/tap" #need to change to use just gaia archive
	
	source_constraints = ' AND '.join([
		'parallax_over_error > 10',
		'phot_g_mean_flux_over_error>50',
		'phot_rp_mean_flux_over_error>20',
		'phot_bp_mean_flux_over_error>20',
		'phot_bp_rp_excess_factor < 1.3+0.06*power(phot_bp_mean_mag-phot_rp_mean_mag,2)',
		'phot_bp_rp_excess_factor > 1.0+0.015*power(phot_bp_mean_mag-phot_rp_mean_mag,2)',
		'visibility_periods_used>8',
		'astrometric_chi2_al/(astrometric_n_good_obs_al-5)<1.44*greatest(1,exp(-0.4*(phot_g_mean_mag-19.5)))'])

	def __init__(self, **kwargs):
		self.name=kwargs.pop('name',None)
		self.description = kwargs.pop('description', None)

		self.coords = None
		self.tap_query_string = None
		self.objs = None

	def __repr__(self):
		str = 'GaiaStars Object'
		if self.name is not None:
			str += f', Name: {self.name}'
		if self.description is not None:
			str += f', Description: {self.description}'
		if self.objs is not None:
			str += f', {len(self.objs)} objects' #hope its a dataframe
		return str

	def __str__(self):
		return self.__repr__()

	def _get_col_list(self, prefix='gs'):
		collist = f'{prefix}.' +  f'\n\t\t,{prefix}.'.join(self.column_list)
		return collist
	
	@u.quantity_input(ra='angle', dec='angle', rad='angle')
	def conesearch(self, ra, dec, radius, **kwargs):
		## ToDo: rewrite to use Gaia Archive, not tap service.  Need to ditch distance param and just use parallax
		# input parameters in degrees:
		ra_ = ra.to(u.degree); dec_= dec.to(u.degree); rad_=radius.to(u.degree)

		maxrec = kwargs.get('maxrec', 20000)

		columnlist = self._get_col_list() + ', gd.r_est'
		dbsource = '\n\t'.join(['\nFROM external.gaiadr2_geometric_distance gd',
					'INNER JOIN gaiadr2.gaia_source gs using (source_id) '])

		constraints =  '\n\t'.join(['\nWHERE ', 
				'CONTAINS(POINT(\'\', gs.ra, gs.dec), ',
				'\tCIRCLE(\'\', {ra}, {dec}, {rad})) = 1 '.format(ra=ra_.value, dec=dec_.value, rad=rad_.value)])

		if self.source_constraints is not None:
			constraints = constraints + ' AND '+ self.source_constraints

		self.tap_query_string = 'SELECT \n\t\t'+ columnlist + dbsource + constraints
		
		#tap_service = TAPService(self.tap_service_url)
		#tap_results = tap_service.search(self.tap_query_string, maxrec=maxrec)
		#self.objs = tap_results.to_table().to_pandas()
		#fetch the data
		
		job = Gaia.launch_job_async(query=self.tap_query_string)
		self.objs = job.get_results().to_pandas()

		self.objs.set_index('source_id', inplace=True)

	def from_source_idlist(self, source_idlist, source_idcol=None, query_type='async'):
		#xml-ify the source_idlist to a file
		if isinstance(source_idlist, Table):
			#guess which column contains the source ids
			if source_idcol is None:
				if 'source_id' in source_idlist.colnames:
					sidcol = 'source_id'
				elif 'source' in source_idlist.colnames:
					sidcol = 'source'
				else:
					raise ValueError('no column to use as source_id')
			else:
				if source_idcol not in source_idlist.colnames:
					raise ValueError(f'invalid column specified as source id column: {source_idcol}')
				sidcol = source_idcol
				
			tbl = source_idlist
		elif isinstance(source_idlist, np.ndarray) or isinstance(source_idlist, list):
			sidcol = 'source_id'
			tbl = Table({sidcol:source_idlist})
		else:
			raise ValueError(f'invalid source_idlist type: {type(source_idlist)}')

		#need tempfile for source id list
		fh =  tempfile.mkstemp()
		os.close(fh[0]) #fh[0] is the file descriptor; fh[1] is the path

		try:
			tbl.write(fh[1], table_id='source_idlist', format='votable', overwrite=True)
			
			#build the query:
			col_list = self._get_col_list() + ', gd.r_est' #because r_est is in different table
			
			dbsource =  ''.join([' FROM tap_upload.source_idlist sidl',
								f' LEFT JOIN gaiadr2.gaia_source gs ON gs.source_id = sidl.{sidcol}',
								f' LEFT JOIN external.gaiadr2_geometric_distance gd on gs.source_id = gd.source_id' ])
			
			# note: no source filter constraints on a source_id query; just get the objects
			query_str = f'SELECT sidl.{sidcol} as "source", '+ col_list + dbsource

			self.tap_query_string = query_str
			
			#fetch the data into a pandas data frame
			#synchronous job gives better error message than async, use for debugging queries
			#synchronous job limited to 2000 results or thereabouts.
			if query_type == 'async':
				job = Gaia.launch_job_async(query=query_str, upload_resource=fh[1], upload_table_name='source_idlist')
			elif query_type == 'sync':
				job = Gaia.launch_job(query=query_str, upload_resource=fh[1], upload_table_name='source_idlist')
			else:
				raise ValueError(f'invalid query_type parameter: {query_type}; valid values are: \'async\' and \'sync\'')

			self.objs = job.get_results().to_pandas()

			self.objs.set_index('source', inplace=True)

		finally:
			#ditch the temporary file
			os.remove(fh[1])           
	
	def merge(self, right):
		# joins right gaiastars to self; returns result
		consol_df = self.objs.merge(right.objs,left_index=True, right_index=True, how='outer', indicator=True)

		consol_df['which'] = consol_df._merge.apply(lambda s: right.name if s == 'right_only' else self.name if s == 'left_only' else 'both')
		
		#fix up columns; preferential treatment for self's columns
		mycols = self.objs.columns
		for c in mycols:
			consol_df[c]=np.where(np.isnan(consol_df[c+'_x']), consol_df[c+'_y'], consol_df[c+'_x'])

		consol_df.drop(columns = [s+'_x' for s in mycols]+[s+'_y' for s in mycols], inplace=True)
		
		my_fs = gaiastars(name = self.name + ' merged with ' + right.name)
		my_fs.objs = consol_df
		my_fs.tap_query_string = [self.tap_query_string, right.tap_query_string]
		
		return my_fs
	
	def to_pickle(self, picklepath):
		"""
		pickles and dumps the object to file designated by picklepath
		"""
		with open(picklepath,'wb') as pkl:
			pickle.dump(self, pkl)
		

	def plot_hrdiagram(self, **kwargs):
		ax = kwargs.get('ax')
		title = kwargs.get('title', 'HR Diagram')
		color = kwargs.get('color', 'blue')
		alpha = kwargs.get('alpha', 1.0)      
		absmag = kwargs.get('absmag', True)
		
		if ax is None:
			yax = plt.subplot(111)
		else:
			yax = ax
			
		distmod = 5*np.log10(self.objs.r_est)-5
		#distmod = (10.0 - 5.0*np.log10(self.objs.parallax)) if absmag else 0.0
		#distance = coord.Distance(parallax=u.Quantity(np.array(self.objs.Plx)*u.mas),allow_negative=True)

		abs_mag = self.objs.phot_g_mean_mag - (distmod if absmag else 0)
		BP_RP = self.objs.phot_bp_mean_mag - self.objs.phot_rp_mean_mag

		yax.scatter(BP_RP,abs_mag, s=1,  label=self.name, color=color)
		if not yax.yaxis_inverted():
			yax.invert_yaxis()
		yax.set_xlim(-1,5)
		#yax.set_ylim(20, -1)

		yax.set_title(title)
		yax.set_ylabel('$M_G$',fontsize=14)
		yax.set_xlabel('$G_{BP}\ -\ G_{RP}$', fontsize=14)
		if ax is None:
			yax.legend()
			
	def plot_motions(self, **kwargs):
		from scipy.stats import kde
		ax = kwargs.get('ax')
		title = kwargs.get('title', 'Proper Motions')
		cmap = kwargs.get('cmap', 'viridis')
		nbins = kwargs.get('nbins', 300)
   
		if ax is None:
			fig, yax = plt.subplots()
		else:
			yax = ax
			
		x = np.array(self.objs.pmra)
		y = np.array(self.objs.pmdec)
		
		# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
		k = kde.gaussian_kde([x,y])
		xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
		zi = k(np.vstack([xi.flatten(), yi.flatten()]))
		
		pcm = yax.pcolormesh(xi, yi, zi.reshape(xi.shape),cmap=cmap)
		
		yax.set_title(title)
		yax.set_ylabel('PM Dec (mas/yr)')
		yax.set_xlabel('PM RA (mas/yr)')
		
		if ax is None:
			fig.colorbar(pcm)
		
		return pcm
	
	def __get_coords__(self, recalc, default_rv):
		#computes and caches sky coordinates for the objects
		#set recalc=True to force recalculation
		if self.coords is None or recalc:

			if default_rv is not None:
				rv = default_rv
			else:
				rv = np.array(self.objs.radial_velocity)*u.km/u.s

			self.coords = coord.SkyCoord(ra=np.array(self.objs.ra)*u.degree,
				   dec=np.array(self.objs.dec)*u.degree,
				   distance=np.array(self.objs.r_est)*u.pc,
				   pm_ra_cosdec=np.array(self.objs.pmra)*u.mas/u.yr,
				   pm_dec=np.array(self.objs.pmdec)*u.mas/u.yr,
				   radial_velocity=rv)

	def get_coords(self, recalc=False, default_rv = None):
		#returns sky coordinates for the objects
		self.__get_coords__(
			 recalc, default_rv)
		return self.coords

	def maxsep(self):
		#computes maximum separation from mean of objects
		ra_mean = self.objs.ra.mean()*u.degree
		dec_mean = self.objs.dec.mean()*u.degree
		c_mean=coord.SkyCoord(ra=ra_mean, dec=dec_mean)
		seps = c_mean.separation(self.get_coords())
		return seps.max()

	def from_pandas(self, df, colmapper):

		src_cols = [c for c in colmapper.values()]
		dest_cols = [c for c in colmapper.keys()]

		arg_df = df.reset_index()

		# source columns in argument dataframe?
		inv_cols = set(src_cols).difference(arg_df.columns)
		if len(inv_cols) != 0:
			raise ValueError('invalid source column(s): '+str(inv_cols))

		# get the right destination columns?
		# case 1: too few dest columns given
		missing_cols = set(self.column_list).difference(dest_cols)
		if len(missing_cols) != 0:
			raise ValueError('Missing column mapping for: '+str(missing_cols))
		# case 2: too many dest columns given:
		extra_cols = set(dest_cols).difference(self.column_list)
		if len(extra_cols) != 0:
			raise ValueError('Invalid destination column supplied: '+str(extra_cols))

		#swap the keys and values for purposes of renaming:
		col_renamer = {s:d for s,d in zip(src_cols, dest_cols)}
		self.objs = arg_df[src_cols].rename(columns = col_renamer, copy=True)

		self.objs.set_index('source_id', inplace=True)
		self.coords = None
		
		return
	
	def project_center_motion(self, cen_coord, return_df=True):
		"""
		calculates the motion of the center projected onto the lines of sight to the constituent stars
		Arguments:
			self: gaiastars instance
			cen_coord: SkyCoord instance capturing position presumably of a cluster center
			return_df: Boolean: if True return pandas data frame otherwise return numpy array
		Returns:
			if return_df, pandas dataframe, one row for each of self.objs, columns for the projected PMRA and PMDEC.
			if not return_df, numpy array, one row for each of self.objs, column0 projected PMRA, column1 projected PMDEC
		Reference:
			https://www.aanda.org/articles/aa/full_html/2009/13/aa11382-08/aa11382-08.html (see eq3 and eq4)
		"""

		k = 4.74047 #magic number to convert mas/yr to km/sec at 1 kpc (see reference)

		nobjs = len(self.objs) #number of stars

		#get necessary params for the center, need 3x1 velocity vector
		cen_plx = cen_coord.distance.to(u.mas, equivalencies = u.parallax())
		cen_vel = np.array([(cen_coord.radial_velocity).value,
							(k*cen_coord.pm_ra/cen_plx).value,
							(k*cen_coord.pm_dec/cen_plx).value]).reshape(3,1)

		#get SkyCoords for each object
		star_coord = self.get_coords()

		#below are 1 x nobjs vectors of sins & cos of each obj's ra and dec
		sin_ra  = np.sin(star_coord.ra.radian); cos_ra = np.cos(star_coord.ra.radian)
		sin_dec = np.sin(star_coord.dec.radian); cos_dec = np.cos(star_coord.dec.radian)

		#create and transpose projection matrix (see equation 3)
		#array is 9 x nobjs, then transposed to nobjs x 9 (one row for each obj)
		proj = np.array( [cos_ra*cos_dec,      sin_ra*cos_dec,      sin_dec,
						 -sin_ra,              cos_ra,              np.zeros(nobjs),
						 -cos_ra*sin_dec,     -sin_ra*sin_dec,      cos_dec]).T
		# reshape to 3d array; one 3x3 projection matrix for each star 
		proj = proj.reshape((nobjs, 3, 3))

		#projection is the matrix product of each obj's proj matrix (3x3) with the center's velocity column vector
		proj_vel = proj.dot(cen_vel) #returns 2d array of column vectors
		assert proj_vel.shape == (nobjs, 3, 1)

		#ditch the awkward low order dimension
		proj_vel = proj_vel.reshape((nobjs, 3))


		# get the motions in pmra and pmdec in mas/year (see equation 2)
		# pmra is column 1, pmdec is column 2
		star_plx = star_coord.distance.to(u.mas, equivalencies = u.parallax()).value
		pmra = proj_vel[:,1]*star_plx/k
		pmdec = proj_vel[:,2]*star_plx/k

		#package up the goods:
		if return_df:
			retval = pd.DataFrame({'proj_pmra': pmra, 'proj_pmdec':pmdec},
								  index=pd.Index(self.objs.index, name=self.objs.index.name))
		else:
			retval = np.array([pmra, pmdec]).T #transpose to nbojs x 2

		return (retval)

	def points_to(self, center, center_radius, center_dist_tol = 0.25):
		"""
		for each obj in self, computes whether the obj's motion points back to the center.
		
		Arguments:
			self: gaiastars instance with populated objs
			center:  astropy.SkyCoord instance
			center_radius: (float) angular radius of cluster in degrees
			center_dist_tol: (float) +/- allowable fraction of distance to cluster to allow
		
		Returns:
			pandas series of booleans, index matching that of self.objs            
		"""
		#get necessary measures into numpy arrays (objects on low-order dimension)
		cen_radec = np.array([center.ra.value, center.dec.value]).reshape(2,1)
		cen_dist = center.distance.value
		obj_radec_abs = np.array([self.objs.ra, self.objs.dec])
		obj_pm = np.array([self.objs.pmra, self.objs.pmdec])

		#move the reference frame to cluster center
		obj_radec = obj_radec_abs - cen_radec

		#calculate phi, the angle of the star wrt. axis RA=0, this would be theta if we were using polar coords
		phi = np.arctan2(obj_radec[1],obj_radec[0])
		#arctan2 returns negative angles, so put into radians ccw from RA=0
		phi = np.where(phi<0, phi+2.0*np.pi, phi)

		#calculate distance from center for each star (distance in ra/dec space)
		d = np.sqrt((obj_radec**2).sum(axis=0))

		#calculate theta: angle btwn center line to star and outer edge of cluster
		theta = np.arctan(center_radius/d)
		assert np.all(np.logical_and(theta >= 0, theta <= np.pi/2.0))

		#calculate upper and lower bounds of PM direction, considering phi and wrapping
		upper = (phi + theta) % (2.0*np.pi)
		lower = phi - theta
		lower = np.where(lower < 0, lower+2*np.pi, lower)

		#calculate the PM direction and adjust to positive angle
		pm_dir = np.arctan2(obj_pm[1], obj_pm[0])
		pm_dir = np.where(pm_dir < 0, pm_dir+2.0*np.pi, pm_dir)
		assert np.all(np.logical_and(pm_dir >= 0, pm_dir <= 2.0*np.pi))

		#test whether PM direction within bounds
		inbounds = np.logical_and(pm_dir >= lower, pm_dir <= upper)

		#test distance constraint
		within_dist = np.logical_and(self.objs.r_est >= cen_dist*(1-center_dist_tol),
									 self.objs.r_est <= cen_dist*(1+center_dist_tol))

		# put the constrainsts together
		points_to = np.logical_and(inbounds, within_dist)

		#############
		# calculate magnitude of pm and how long it should've taken to move to star's position
		#############

		#distance from sun (in pc)
		dist_from_sun = self.objs.r_est 

		#Get pm's in radians/year
		pm_rad_radec = np.radians(obj_pm/3.6e6) #to convert mas to degrees then to radians
		pm_rad = np.sqrt((pm_rad_radec**2).sum(axis=0)) # radians per year
		# convert to mas per year
		mas_per_radian = 2.0628e8
		pm_v = pm_rad*mas_per_radian #mas per year

		#how many years to get from cluster center to current position?
		d_years = d/(pm_v/3.6e6) #to convert mas to degrees (the units of d)

		#return whole buncha stuff for debugging purposes
		ret_df = pd.DataFrame({
							'CenRA': cen_radec[0,0],
							'CenDec': cen_radec[1,0],
							'CenDist': cen_dist,
							'ObjRA':obj_radec_abs[0],
							'ObjDec': obj_radec_abs[1],
							'ObjRelRA': obj_radec[0],
							'ObjRelDec': obj_radec[1],
							'ObjPMRA': obj_pm[0],
							'ObjPMDec': obj_pm[1],
							'ObjPMDir': pm_dir,
							'ObjDistCen': d,
							'ObjVelMag':pm_v,
							'ObjDistSun': self.objs.r_est,
							'Phi': phi,
							'Theta': theta,
							'Upper': upper,
							'Lower': lower,
							'Inbounds': inbounds,
							'WithinDist': within_dist,
							'PointsTo': points_to,
							'YrsDist': d_years},
					index = pd.Index(self.objs.index, name=self.objs.index.name))
		return ret_df

	def _trace_back_obj(self, o, c, t):
		"""
		computes minimum time and distance of object trajectory with that of c
		Arguments:
		o: object, SkyCoord
		c: object, SkyCoord, presumably cluster center
		t: time vector in years over which to perform calculations
		Returns:
			dict with keys: d_min, t_min and tol
		Referenct:
		https://arxiv.org/pdf/2005.04762.pdf 
		"""

		#function to constrain dec between -90 and 90 (argument in radians)
		fix_lat = np.vectorize(lambda x:  x if x <= np.pi/2.0 else \
			( np.pi-x if x <=3.0*np.pi/2.0 else  x-2.0*np.pi))

		# get the ra's and dec's of the two objects at time t
		# relying on astropy.units to do proper conversion to degrees
		o_ra =  (o.ra  + o.pm_ra_cosdec*t).wrap_at(360*u.degree);
		c_ra  = (c.ra + c.pm_ra*t).wrap_at(360*u.degree)

		o_dec = coord.Angle(fix_lat((o.dec + o.pm_dec*t).radian % (2.0*np.pi))*u.radian)
		c_dec = coord.Angle(fix_lat((c.dec + c.pm_dec*t).radian % (2.0*np.pi))*u.radian)

		#sky coords at time t for both, using constant distance:
		o_t = SkyCoord(ra=o_ra, dec=o_dec, distance=o.distance)
		c_t = SkyCoord(ra=c_ra, dec=c_dec, distance=c.distance)

		#angular separation as function of time
		sep = o_t.separation(c_t)

		#find minimum separation and time
		min_i = sep.argmin()
		d_min = o_t[min_i].separation(c_t[min_i])
		t_min = t[min_i]

		#calculate the tolerance (see equation 1 in reference)
		tol = 10+1.3*c.separation(o)/(1*u.degree)

		return {"t_min":t_min, "d_min":d_min, "tol": tol,
			 'tracesback':(d_min/(1.0*u.arcminute)) <= tol}



	def traces_back(self, center, time_max=-2.0e9, time_step=0.1e9):
		"""
		for each obj in self, computes whether the obj's motion traces back to the center in space and time.
		
		Arguments:
			self: gaiastars instance with populated objs
			center:  astropy.SkyCoord instance
			time_max: float, maximum time to trace back to, in years, negative number
			time_step: float, time increments in years
		
		Returns:
			pandas dataframe, same rows as self.objs, columns: d_min, t_min, tol, tracesback           
		"""
		#timeframe (round out to make closed interval [time_max,0] evenly space by time_step)
		t = np.linspace(time_max, 0, int((abs(time_max)+time_step)/time_step))*u.year
		
		#object coordinates
		obj_coords = self.get_coords()

		# in future, transform to local standard of rest:
		#transform to default lsr frame
		#obj_coords_lsr = obj_coords.transform_to(coords.lsr)
		#center_lsr = center.transform_to(coords.lsr)
		obj_coords_lsr = obj_coords
		center_lsr = center

		#calculate traceback for each star:
		tb_df = pd.DataFrame([self._trace_back_obj(o_lsr, center_lsr, t) for o_lsr in obj_coords_lsr],
			index=pd.Index(self.objs.index, name=self.objs.index.name))

		return tb_df





def from_pickle(picklefile):
	"""
	reads up a pickle file and hopefully it's a pickled gaiastars object
	"""
	with open(picklefile,'rb') as pkl:
		my_fs = pickle.load(pkl)
		
	if not isinstance(my_fs, gaiastars):
		raise ValueError(f'pickle file contained a {type(my_fs)}')
		
	return my_fs

if __name__ == '__main__':

	fs = gaiastars(name='test cone search', description='test of the new capabilities of GaiaStars')
	print(fs)
	fs.conesearch(52.074625695066345*u.degree, 48.932707471347136*u.degree, 1.0*u.degree, plx_error_thresh=5, r_est=(175,185))
	print(fs)

	print(f'There are {len(fs.objs)} field stars')

	id_list = list(fs.objs.index)
	fs2 = gaiastars(name='test id search')
	fs2.from_source_idlist(id_list)
	print(fs2)

	fs3 = gaiastars(name='test missing id handling')
	idlist2 = id_list[0:5]+[0,1234] #invalid source ids
	fs3.from_source_idlist(idlist2, query_type='sync')
	print(fs3)
	print(fs3.objs)


	print(fs.tap_query_string)
