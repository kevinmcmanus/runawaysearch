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
	
	#default tables and columns (they differ from one release to the next):
	#dr2
	gaia_column_dict_gaiadr2 ={'gaiadr2.gaia_source': [ 'ra','dec','parallax','pmra','pmdec','radial_velocity',
								'phot_g_mean_mag','phot_bp_mean_mag', 'phot_rp_mean_mag',
								'e_bp_min_rp_val', 'a_g_val'],
			'external.gaiadr2_geometric_distance': ['r_est']}
	#early release dr3
	gaia_column_dict_gaiaedr3 ={'gaiaedr3.gaia_source': [ 'ra','dec','parallax','pmra','pmdec','dr2_radial_velocity',
								'phot_g_mean_mag','phot_bp_mean_mag', 'phot_rp_mean_mag', 'ruwe']}


	
	def gaia_column_dict(self, schema):
		
		if schema == 'gaiadr2':
			coldict = self.gaia_column_dict_gaiadr2
		elif schema=='gaiaedr3':
			coldict = self.gaia_column_dict_gaiaedr3
		else:
			errorstr = f'Invalid schema specification: {schema}'
			raise ValueError(errorstr)

		return coldict

	gaia_source_constraints = [
		'{schema}.gaia_source.parallax_over_error > 10',
		'{schema}.gaia_source.phot_g_mean_flux_over_error>50',
		'{schema}.gaia_source.phot_rp_mean_flux_over_error>20',
		'{schema}.gaia_source.phot_bp_mean_flux_over_error>20',
		'{schema}.gaia_source.phot_bp_rp_excess_factor < 1.3+0.06*power({schema}.gaia_source.phot_bp_mean_mag-{schema}.gaia_source.phot_rp_mean_mag,2)',
		'{schema}.gaia_source.phot_bp_rp_excess_factor > 1.0+0.015*power({schema}.gaia_source.phot_bp_mean_mag-{schema}.gaia_source.phot_rp_mean_mag,2)',
		'{schema}.gaia_source.visibility_periods_used>8',
		'{schema}.gaia_source.astrometric_chi2_al/({schema}.gaia_source.astrometric_n_good_obs_al-5)<1.44*greatest(1,exp(-0.4*({schema}.gaia_source.phot_g_mean_mag-19.5)))'
		]


	def set_gaia_source_constraints(self, strlist):
		previous_constraints = self.gaia_source_constraints
		self.gaia_source_constraints = strlist
		return previous_constraints

	def get_gaia_source_constraints(self, schema):
		constraints = ' AND '.join([s.format(schema=schema) for s in self.gaia_source_constraints])
		return constraints
		
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

	def _get_col_list(self, coldict):
		"""
		makes an ADQL column list from the supplied dictionary, ensuring first table supplies a source_id column
		Arguments:
			coldict: dictionary of form <table>:[column,...],...
		Returns:
			collist: string, an ADQL-syntax column list
		"""
		strs = []
		first = True
		for t in coldict:

			s = f' {t}.'
			cols = coldict[t]

			#if source_id isn't in the first table's column list, prepend it on
			if first:
				first=False
				if not 'source_id' in cols:
					cols = ['source_id']+cols

			tstr = s + (', '+t+'.').join(cols)

			strs.append(tstr)

		return ','.join(strs)

	def _get_db_source(self, coldict, join_type='LEFT JOIN'):
		"""
		makes an ADQL database source from the supplied dictionary, suitable for FROM clause of ADQL query
		assumes LEFT JOINs among the tables
		Arguments:
			coldict: dictionary of form <table>:[column,...],...
			join_type: 'LEFT JOIN' or 'INNER JOIN'
		Returns:
			dbsource: string, suitable as operand of FROM ADQL query clause
		"""
		tbl_list = list(coldict.keys())
		first = tbl_list[0]
		dbsrc = first
		for t in tbl_list[1:]:
			dbsrc += f' {join_type} {t} ON {first}.source_id = {t}.source_id'

		return dbsrc

	@u.quantity_input(ra='angle', dec='angle', ra_ext='angle', dec_ext='angle')
	def boxsearch(self, ra, dec, ra_ext, dec_ext, **kwargs):
		"""
		performs a box search of the Gaia Archive at the supplied coordinates.
		Box center is at (ra, dec) and box sides are half the extents in both directions.
		Arguments:
			ra, dec: astropy.quantity.Angle objects specifying coords of box center
			ra_ext, dec_ext: astropy.quantity.Angle objects specifying (total) width and height of box
		Returns:
			Nothing, updates self.objs with panda query results
		"""
		# input parameters in degrees:
		ra_ = ra.to(u.degree); dec_= dec.to(u.degree)
		ra_ext_ = ra_ext.to(u.degree); dec_ext_ = dec_ext.to(u.degree)

		#construct minimal box search query filter (column_dict needs to deliver an ra and dec)
		query_filter =  'CONTAINS(POINT(\'\', ra, dec), '\
			' BOX(\'\', {ra}, {dec}, {ra_ext}, {dec_ext})) = 1 '.format(ra=ra_.value, dec=dec_.value,
																		ra_ext=ra_ext_.value, dec_ext=dec_ext_.value)

		self.regionsearch(query_filter, **kwargs)


	@u.quantity_input(ra='angle', dec='angle', rad='angle')
	def conesearch(self, ra, dec, radius, **kwargs):
		"""
		Performs a cone search of the Gaia Archive at the supplied coordinates and radius
		Arguments:
			ra, dec: astropy.quantity.Angle objects specifying coords of center of cone seacrh
			rad: astropy.quantity.Angle object specifying the angular radius of the cone search
			column_dict: optional dictionary of form: <gaiatable>:[column list]; default:gaiastars.gaia_column_dict
		Returns:
			Nothing, updates self.objs with pandas query results
		"""

		# input parameters in degrees:
		ra_ = ra.to(u.degree); dec_= dec.to(u.degree); rad_=radius.to(u.degree)

		#construct minimal cone search query filter (column_dict needs to deliver an ra and dec)
		query_filter =  'CONTAINS(POINT(\'\', ra, dec), '\
			' CIRCLE(\'\', {ra}, {dec}, {rad})) = 1 '.format(ra=ra_.value, dec=dec_.value, rad=rad_.value)

		self.regionsearch(query_filter, **kwargs)

	def regionsearch(self, query_filter, **kwargs):
		"""
		executes a Gaia search in a region defined by query_filter
		Should not be called directly, use gaiastars.conesearch() or gaiastars.boxsearch()
		"""
		
		#use default table columns and constraints?
		schema = kwargs.pop('schema', 'gaiaedr3')
		column_dict = kwargs.pop('column_dict', self.gaia_column_dict(schema=schema))
		parallax = kwargs.pop('parallax', None)

		col_list = self._get_col_list( column_dict)
		db_source = self._get_db_source(column_dict, join_type='INNER JOIN')

		#tack the constraints onto the query filter
		constraints = self.get_gaia_source_constraints(schema=schema)
		query_filter = f'{query_filter} AND {constraints}'

		#deal with parallax constraint:
		if parallax is not None:
			query_filter = f'{query_filter} AND parallax >= {parallax[0]} AND parallax <= {parallax[1]}'

		#build the query string and ship it off
		query_string = f'SELECT {col_list} FROM {db_source} WHERE {query_filter}'
		self.gaia_query(query_string, query_type='async')	

	def gaia_query(self, query_str, query_type, upload_resource=None, upload_tablename=None):
		"""
		Queries Gaia archive with query_str and updates self with results
		Arguments:
			query_str: string: valid ADQL query in context of Gaia Archive
			query_type: string, one of {'sync','async'} (use sync for debugging)
			upload_resource: string, path to xml file containing list of gaia source ids
			upload_tablename: string, name of table in upload_resource
		Returns:
			Nothing, self.objs updated in place with query results
					 self.tap_query_string updated with supplied query string
		"""
		#save the query string for posterity
		self.tap_query_string = query_str
		
		#fetch the data into a pandas data frame
		#synchronous job gives better error message than async, use for debugging queries
		#synchronous job limited to 2000 results or thereabouts.
		if query_type == 'async':
			job = Gaia.launch_job_async(query=query_str,
										upload_resource=upload_resource,
										upload_table_name=upload_tablename)
		elif query_type == 'sync':
			job = Gaia.launch_job(query=query_str,
							upload_resource=upload_resource,
							upload_table_name=upload_tablename)
		else:
			raise ValueError(f'invalid query_type parameter: {query_type}; valid values are: \'async\' and \'sync\'')

		#get the results as pandas dataframe and index it
		self.objs = job.get_results().to_pandas().set_index('source_id')
		
		# hacks for edr3
		if not ('r_est' in self.objs.columns):
			if 'parallax' in self.objs.columns:
				self.objs['r_est'] = 1000.0/self.objs.parallax
		self.objs.rename(columns={'dr2_radial_velocity':'radial_velocity',
								 'dr2_radial_velocity_error':'radial_velocity_error'}, inplace=True)


	def from_source_idlist(self, source_idlist, column_dict=None,
				schema='gaiaedr3',
				query_type='async'):
		"""
		Queries Gaia Archive for specified records; returns columns as spec'd in column_dict
		Arguments:
			source_idlist: list of Gaia Source Ids
			column_dict: optional dictionary of form: <gaiatable>:[column list]; default:gaiastars.gaia_column_dict
			query_type: string, one of {'sync','async'} (use sync for debugging)
		Returns:
			Nothing; updates self in place with pandas DataFrame in property self.objs
		"""
		#use default column list if one wasn't passed in
		coldict = self.gaia_column_dict(schema=schema) if column_dict is None else column_dict

		#need tempfile for source id list
		upload_tablename, upload_resource, sidcol = source_id_to_xmlfile(source_idlist)
		try:
			#build the query:
			col_list = self._get_col_list( coldict)
			db_source = self._get_db_source({'tap_upload.source_idlist':[], **coldict})
			query_str = f'SELECT {col_list} FROM {db_source}'

			#do the deed
			self.gaia_query(query_str, query_type, upload_resource=upload_resource, upload_tablename=upload_tablename)

		finally:
			#ditch the temporary file
			os.remove(upload_resource)           
	
	def merge(self, right):
		"""
		# joins right gaiastars to self; returns result
		"""
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
		
	def get_colors(self, absmag=True, r_est=True):
		"""
		Returns absolute magnitude and color for each star
		Arguments:
			absmag: Boolean, determines whether absolute or apparent magnitude should be return, defalut:True
			r_est: Boolean, determines whether estimated distance (default) or parallax should be used in distance calculation
		Returns:
			BP_RP: star color
			M_G: absolute or apparent magnitude
		"""

		if r_est:
			#use estimated distance
			dist = self.objs.r_est 
		else:
			#use parallax (assumed to be in MAS) to calc distance
			dist = 1000/self.objs.parallax

		distmod = 5*np.log10(dist)-5

		#absolute or apparent magnitude
		M_G = self.objs.phot_g_mean_mag - (distmod if absmag else 0)

		#color
		BP_RP = self.objs.phot_bp_mean_mag - self.objs.phot_rp_mean_mag

		return BP_RP, M_G

	def plot_hrdiagram(self, **kwargs):
		ax = kwargs.pop('ax',None)
		title = kwargs.pop('title', 'HR Diagram')
		label = kwargs.pop('label',self.name)
		s = kwargs.pop('s', 1) #default size = 1
		absmag = kwargs.pop('absmag', True) #Absolute or Apparent magnitude?
		r_est = kwargs.pop('r_est',True) #estimated distance or calc from parallax
   
		if ax is None:
			yax = plt.subplot(111)
		else:
			yax = ax

		BP_RP, M_G = self.get_colors(absmag=absmag, r_est=r_est)

		pcm = yax.scatter(BP_RP, M_G, label=label, s=s, **kwargs)

		if not yax.yaxis_inverted():
			yax.invert_yaxis()
		yax.set_xlim(-1,5)
		#yax.set_ylim(20, -1)

		yax.set_title(title)
		yax.set_ylabel(r'$M_G$')
		yax.set_xlabel(r'$G_{BP}\ -\ G_{RP}$')
		if ax is None:
			yax.legend()
			
		return pcm

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
			if default_rv is None:
				rv = None
			elif isinstance(default_rv, bool):
				rv = np.array(self.objs.radial_velocity)*u.km/u.s if default_rv else None
			else:
				#use the reported rv if available otherwise the default rv
				rv = np.where(np.isfinite(self.objs.radial_velocity),
							  self.objs.radial_velocity,
							  default_rv)*u.km/u.s

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

	def maxsep(self, center=None):
		#computes maximum separation from mean of objects
		if center is None:
			#calculate center from mean of members
			ra_mean = self.objs.ra.mean()*u.degree
			dec_mean = self.objs.dec.mean()*u.degree
			c_mean=coord.SkyCoord(ra=ra_mean, dec=dec_mean)
		else:
			#use the center passed in
			c_mean = center
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

	def query(self, query_str, inplace=False):
		objs = self.objs.query(query_str).copy()
		desc = f'{self.description} with query: {query_str}'
		if inplace:
			self.objs=objs
			self.description = desc
		else:
			mystars = gaiastars(name=self.name, description = desc)
			mystars.tap_query_string = self.tap_query_string
			mystars.objs = objs
			return mystars
	
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
	
	def motion_direction(self, point, inplace=False, dt=10*u.year):
		"""
		Computes whether the individual stars are moving toward or away from the point
		Arguments:
			point: astropy.SkyCoord object
			dt: time delta over which to calculate the direction of movement, positive=> forward in time
			inplace: whehter the objs member will be updated
		Returns:
			pandas series, index same as objs, values: -1: => movement toward point, 1 => movement away
		"""
		
		#get the current separation
		coords_now = self.get_coords(recalc=True)
		seps_now = point.separation(coords_now)
		
		#future separation
		point_then = point.apply_space_motion(dt=dt)
		coords_then = coords_now.apply_space_motion(dt=dt)
		seps_then = point_then.separation(coords_then)
		
		direction = np.sign(seps_then-seps_now)
		
		if inplace:
			self.objs['Direction'] = direction
			retser = None
		else:
			retser = pd.Series(direction, index=pd.Index(self.objs.index, name = self.objs.index.name))

		return retser
	
	def travel_time(self, center, inplace=False, allcalcs=True):
		"""
		Calculates the travel time needed for each star to achieve current separation from Center
		Args:
			self: Gaia stars object populated stars (objs member)
			center: astropy.Skycoord instance of center
		Returns:
			
		"""
		objs = self.objs
		rad_delta = np.radians(objs.dec)
		sin_delta = np.sin(rad_delta)
		cos_delta = np.cos(rad_delta)
		
		######### determine travel time for current separation of stars from center ########
		# get theta, angle btwn LOS to star and LOS to center
		# compute theta as great circle distance (theta in radians)
		theta = np.arccos(np.sin(center.dec.radian)*sin_delta +
					np.cos(center.dec.radian)*cos_delta*np.cos(center.dec.radian - rad_delta))
		
		#distance from center to stars using law of cosines
		cen_dist = np.sqrt(objs.r_est**2 + center.distance.value**2 -
					 2*objs.r_est*center.distance.value*np.cos(theta))
		
		#calculate beta, angle btwn the (LOS to the stars and the LOS from star to cluster) - 90 degrees
		beta = np.arccos(center.distance*np.sin(theta)/cen_dist)

		#center pm_ra may or may not include cos(dec), but needs to for purpposes here
		cen_pm_ra_cosdec = center.pm_ra_cosdec.value if hasattr(center, 'pm_ra_cosdec') else center.pm_ra.value*np.cos(center.dec)

		#proper motions in center's rest frame: (2d array: {pm_ra, pm_dec} x N)
		#note pm_ra includes cos(dec) term per gaia practice
		pm = np.array([objs.pmra, objs.pmdec]) - np.array([cen_pm_ra_cosdec, center.pm_dec.value]).reshape(2,1)

		#project pm onto LOS btwn stars and center
		proj_pm = pm/np.vstack([np.cos(beta), np.cos(beta)])

		# magnitude of projected proper motion: (in mas/year)
		mag_ppm = np.sqrt((proj_pm**2).sum(axis=0))

		#convert to radians per year
		mas_per_degree = 3.6e6
		mag_ppm_r = np.radians(mag_ppm/mas_per_degree)

		#convert the magnitude to pc per year
		mag_ppm_pc = mag_ppm_r*objs.r_est

		#compute travel time in years
		travel_time = cen_dist/mag_ppm_pc

		#build up the return value
		ret_df = pd.DataFrame({'theta':theta,
								'cen_dist':cen_dist,
								'beta': beta,
								'cen_pm_ra_cosdec': cen_pm_ra_cosdec,
								'cen_pm_dec': center.pm_dec,
								'pm_ra_cosdec_censr':pm[0],
								'pm_dec_censr':pm[1],
								'proj_pmra': proj_pm[0],
								'proj_pmdec': proj_pm[1],
								'mag_ppm': mag_ppm,
								'mag_ppm_r':mag_ppm_r,
								'mag_ppm_pc': mag_ppm_pc,
								'travel_time':travel_time},
				index=pd.Index(objs.index, name=objs.index.name))
		if allcalcs:
			ret_cols =  list(ret_df.columns)
		else:
			ret_cols = ['travel_time']

		if inplace:
			for c in ret_cols:
				self.objs[c] = ret_df[c]
			return None
		else:
			return ret_df[ret_cols]		

	def points_to(self, center, center_radius, inplace=False, allcalcs=True):
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
		objs = self.objs

		def normalize_angle(ang):
			return np.remainder(ang+2.0*np.pi, 2.0*np.pi)

		#get some radians and co-latitudes for the stars and center
		s_rad_delta = np.radians(objs.dec)
		s_colat = np.pi/2.0 - s_rad_delta
		s_sin_colat = np.sin(s_colat)
		s_cos_colat = np.cos(s_colat)

		c_rad_delta = center.dec.radian
		c_colat = np.pi/2.0 - c_rad_delta
		c_sin_colat =  np.sin(c_colat)
		c_cos_colat = np.cos(c_colat)

		#cosine of differenece in RA:
		cos_delta_ra = np.cos(np.radians(objs.ra - center.ra.degree))

		# Algorithm:
		#1. Calculate theta, spherical angle star -> cluster wrt. star's meridian
		#2. Calculate gamma: spherical angle btwn star-cluster line of site and search radius around cluster
		#3. Calculate phi_prime: opposite of direction of star's PM wrt cluster standard of rest.
		#4. Test wheter theta-gamma <= phi_prime <= theta+gamma

		####### determine whether stars' motions point back to center #########
		# 1. Calculate theta: spherical angle great circle containing star and cluster

		#1a: Calc. great circle distance in radians star-> cluster
		cen_dist = np.arccos(s_cos_colat*c_cos_colat + s_sin_colat*c_sin_colat*cos_delta_ra)
		cen_dist = normalize_angle(cen_dist)
		cos_cen_dist = np.cos(cen_dist)
		sin_cen_dist = np.sin(cen_dist)

		#1b: Given the cendist, calculate theta using law of cosines
		theta = (c_cos_colat - cos_cen_dist*s_cos_colat)/(sin_cen_dist*s_sin_colat)

		#2: Calculate gamma, spherical angle between star-cluster los and search radius
		#2a. Calculate c: length of arc from outer edge of cluster to center of star
		cos_center_radius = np.cos(np.radians(center_radius))
		cos_c = (cos_cen_dist*cos_center_radius)
		c = np.arccos(cos_c)
		sin_c = np.sin(c)

		#2b. Calculate gamma
		gamma = np.arccos((cos_center_radius-cos_cen_dist*cos_c)/(sin_cen_dist*sin_c))

		#3: calculate phi_prime: opposite direction of star's motion, in cluster reference frame
		delta_pm_ra = objs.pmra - center.pm_ra_cosdec.value
		delta_pm_dec = objs.pmdec - center.pm_dec.value
		pm_dir = np.arctan2(delta_pm_ra, delta_pm_dec)
		#need opposite direction, so subtract off pi
		pm_dir_prime = pm_dir - np.pi
		pm_dir_prime = normalize_angle(pm_dir_prime)

		#4: calculate points_to
		points_to = np.logical_and(theta-gamma <= pm_dir_prime, pm_dir_prime <= theta+gamma)

		#build up return value
		ret_df = pd.DataFrame({'cen_dist':cen_dist,
								'theta':theta,
								'c': np.arccos(cos_c),
								'gamma': gamma,
								'delta_pm_ra':delta_pm_ra,
								'delta_pm_dec':delta_pm_dec,
								'pm_dir': pm_dir,
								'pm_dir_prime': pm_dir_prime,
								'points_to': points_to})

		cols = ret_df.columns if allcalcs else [ 'points_to']
		if inplace:
			for c in cols:
				self.objs[c] = ret_df[c]
			return None
		else:
			return ret_df[cols]


def from_pickle(picklefile):
	"""
	reads up a pickle file and hopefully it's a pickled gaiastars object
	"""
	with open(picklefile,'rb') as pkl:
		my_fs = pickle.load(pkl)
		
	if not isinstance(my_fs, gaiastars):
		raise ValueError(f'pickle file contained a {type(my_fs)}')
		
	return my_fs

def source_id_to_xmlfile(source_idlist, sidcol='source_id', table_id='source_idlist'):
	#need tempfile for source id list
	fh =  tempfile.mkstemp()
	os.close(fh[0]) #fh[0] is the file descriptor; fh[1] is the path

	#xml-ify the source_idlist to a file
	tbl = Table({sidcol:source_idlist})
	tbl.write(fh[1], table_id=table_id, format='votable', overwrite=True)

	return table_id, fh[1], sidcol
			

def gaiadr2xdr3(source_idlist,nearest=True):
	"""
	returns the dr2 to dr3 cross matches for the given source_idlist
	"""

	upload_tablename, upload_resource, sidcol = source_id_to_xmlfile(source_idlist)

	query_str = ' '.join([f'SELECT tu.{sidcol}, dr2.*',
				f'from tap_upload.{upload_tablename} tu left join gaiaedr3.dr2_neighbourhood dr2',
				f'on tu.{sidcol} = dr2.dr2_source_id'])
	try:
		job = Gaia.launch_job_async(query=query_str,
								upload_resource=upload_resource,
								upload_table_name=upload_tablename)
		
		df = job.get_results().to_pandas()
	finally:
		os.remove(upload_resource)
	
	if nearest:
		#just return the nearest dr3 source id based on angular distance
		ret_df = df.sort_values(['dr2_source_id','angular_distance']).groupby('dr2_source_id',
					as_index=False).first().set_index('source_id')
	else:
		ret_df = df.set_index('source_id')

	return ret_df

if __name__ == '__main__':

	import sys
	sys.path.append('./src')

	from data_queries import  getClusterInfo, getGAIAKnownMembers

	fs = gaiastars(name='test cone search', description='test of the new capabilities of GaiaStars')
	print(fs)
	fs.conesearch(52.074625695066345*u.degree, 48.932707471347136*u.degree, 1.0*u.degree)
	print(fs)

	print(f'There are {len(fs.objs)} field stars')

	cluster_info = getClusterInfo()

	fs.points_to(cluster_info.loc['Pleiades']['coords'], 6.0, inplace=True)
	n = fs.objs.points_to.sum()
	print(f'Number of points_to: {n}')

""" 
	id_list = list(fs.objs.index)
	fs2 = gaiastars(name='test id search')
	fs2.from_source_idlist(id_list, schema='gaiadr2')
	print(fs2)

	df = gaiadr2xdr3(id_list)
	print(df.head())

	fs3 = gaiastars(name='test missing id handling')
	idlist2 = id_list[0:5]+[0,1234] #invalid source ids
	fs3.from_source_idlist(idlist2, query_type='sync')
	print(fs3)
	print(fs3.objs)

	fs4 = gaiastars(name='test box search', description='test of the new box search of GaiaStars')
	print(fs)
	fs4.boxsearch(52.074625695066345*u.degree, 48.932707471347136*u.degree, 5.0*u.degree, 5.0*u.degree, parallax=(5.0,10.0))
	print(fs4)

	print(fs4.tap_query_string)
 """