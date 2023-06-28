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
		cos_theta = (c_cos_colat - cos_cen_dist*s_cos_colat)/(sin_cen_dist*s_sin_colat)
		theta = np.arccos(cos_theta)

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

	def _travel_time3d(self, star, center, tt_rng, rv, ret_sample=False):
		#get velocity column vectors for star for each element of rv
		d_xyz_s = pm_to_dxyz(star.ra.value, star.dec.value, star.distance.value,
				star.pm_ra_cosdec.value, star.pm_dec.value, rv.value)*u.km/u.second
		
		#center velocity column vector
		d_xyz_c = center.velocity.d_xyz.reshape(3,1)
		#velocity in center reference frame
		d_xyz_s_csr = d_xyz_s - d_xyz_c
		
		#space velocity and convert to pc/year
		vel_csr = np.sqrt((d_xyz_s_csr**2).sum(axis=0)).to(u.pc/u.year)
		
		#distance btwn center and star
		d=center.separation_3d(star) #comes back in pc
		
		#how long did it take given the sample of velocities
		tt_sample = d/vel_csr # should be in years
		
		valid_tt = np.logical_and(tt_sample >= tt_rng[0], tt_sample <= tt_rng[1])
		tt_cand = np.any(valid_tt)
		if tt_cand:
			tt_min = tt_sample[valid_tt].min().value
			tt_max = tt_sample[valid_tt].max().value
			rv_min = rv[valid_tt].min().value
			rv_max = rv[valid_tt].max().value
		else:
			tt_min = tt_max = rv_min = rv_max = np.nan

		if ret_sample:
			return rv, tt_sample	
		else:		
			return {"tt_3d_candidate": tt_cand,
					"tt_3d_min":tt_min,
					"rv_min": rv_min,
					"tt_3d_max":tt_max,
					"rv_max": rv_max}
	
	def travel_time3d(self, center, tt_rng, rv=None, inplace=False):
		if rv is None:
			my_rv = np.concatenate([np.linspace(-100, -20, 800, endpoint=False),
						np.linspace(-20,20,1000, endpoint=False),
						np.linspace(20, 100, 800)])*u.km/u.second
		else:
			my_rv = rv

		coords = self.get_coords()
		
		tt_df = pd.DataFrame([self._travel_time3d(s, center,tt_rng,my_rv) for s in coords],
					index = pd.Index(self.objs.index, name = self.objs.index.name))
		
		return tt_df


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
