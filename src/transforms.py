import numpy as np

#conversion factors:
#import astropy.units as u
#degree_per_mas = 1/(1000*60*60)
#seconds_per_year = (1.0*u.year).to_value(u.second)
#pc_per_km = (1.0*u.km).to_value(u.pc)
    
degree_per_mas = 2.7777777777777776e-07
seconds_per_year = 31557600.0
pc_per_km = 3.240779289469756e-14

def pm_to_dxyz(ra:float, dec:float, distance:float, pm_ra_cosdec:float, pm_dec:float, radial_velocity):
    """
    computes d_xyz from ra, dec,d distance, pmra_cosdec and pm_dec, given a sample of radial velocity
    Arguemnts:
        ra, dec: ra and dec in degrees;
        distance: distance to object in pc;
        pm_ra_cosdec, pm_dec: proper motions in mas/year
        radial_velocity: in km/s, possibly a vector of these
    Returns np.array(3,) if radial velocity is scalar , np.array(3, len(radial_velocity)) of d_x, d_y and d_z  
    """

    #get some radians
    rarad = np.radians(ra);
    decrad = np.radians(dec)
    pmrarad = np.radians(pm_ra_cosdec*degree_per_mas) # radians per year
    pmdecrad = np.radians(pm_dec*degree_per_mas) # radians per year

    v_ra = (pmrarad*distance/pc_per_km)/seconds_per_year # km/second
    v_dec = (pmdecrad*distance/pc_per_km)/seconds_per_year #km/second
    
    #form the transform matrix
    sin_alpha = np.sin(rarad); cos_alpha = np.cos(rarad)
    sin_delta = np.sin(decrad); cos_delta=np.cos(decrad)

    dm = np.array([
                   [cos_delta*cos_alpha, -sin_alpha,   -sin_delta*cos_alpha],
                   [cos_delta*sin_alpha,  cos_alpha,   -sin_alpha*sin_delta],
                   [sin_delta,            0,            cos_delta]
                ])

    if hasattr(radial_velocity,"__len__"):
        n = len(radial_velocity)
        eq_v = np.array([radial_velocity, np.full(n,v_ra), np.full(n, v_dec)])
    else:
        eq_v = np.array([radial_velocity, v_ra, v_dec])

    d_xyz = dm.dot(eq_v)
    
    return d_xyz

def spherical_to_cartesian(ra, dec, r):
    alpha = np.radians(ra)
    delta = np.radians(dec)
    
    x = r * np.cos(delta)*np.cos(alpha)
    y = r * np.cos(delta)*np.sin(alpha)
    z = r * np.sin(delta)
    
    xyz = np.array([x, y, z])
    return xyz

def cartesian_to_spherical(xyz):
    #xyz: one column for each star; one row for x, y and z
    r = np.sqrt((xyz**2).sum(axis=0))
    delta = np.arctan(xyz[2]/np.sqrt(xyz[0]**2+xyz[1]**2))
    alpha = np.arctan2(xyz[1], xyz[0])
    alpha = np.where(alpha<0, alpha+2.0*np.pi, alpha)
    
    spherical = {'distance':r,
                 'alpha': np.rad2deg(alpha),
                 'delta': np.rad2deg(delta)
    }
    
    return spherical

def dxyz_to_pm(xyz, d_xyz):
    
    N = xyz.shape[1] # number of stars
    
    #get the spherical coords from xyz:
    spherical = cartesian_to_spherical(xyz)
    
    alpha = np.deg2rad(spherical['alpha'])
    delta = np.deg2rad(spherical['delta'])
    distance = spherical['distance']/pc_per_km
    

    #sines and cosines needed for transform matrix
    sin_alpha = np.sin(alpha); cos_alpha = np.cos(alpha)
    sin_delta = np.sin(delta); cos_delta = np.cos(delta)
    
    #form the derivative transformation matrix    
    #dm_x is 9xN in which each column (star) is the star's 3x3 transform matrix in row-major form.
    dm_x = np.array([cos_delta*cos_alpha, -sin_alpha,   -sin_delta*cos_alpha,
                  cos_delta*sin_alpha,   cos_alpha,   -sin_alpha*sin_delta,
                  sin_delta,             np.zeros(N),  cos_delta])

    #rearrange dm_x to be Nx3x3, i.e. stars on the high order axis
    dm_t = dm_x.transpose().reshape(-1, 3, 3)
    assert dm_t.shape == (N, 3, 3)
    
    # need the invers of the transform matrix
    # invert them all in one go; linalg.inv operates on the two lowest order dims
    dm_i = np.linalg.inv(dm_t)
    assert dm_i.shape == (N, 3, 3)

    #get the velocities in km/s:
    #for each star, pre-multiply its velocity vector (as column matrix) by its inverted transform matrix.
    #result for each star is 3x1 col vector; resulting matrix across all stars is Nx3x1
    d_spherical_kms = np.array([dm_i[i].dot(d_xyz[:,i].reshape(3,1)) for i in range(N)])
    assert d_spherical_kms.shape == (N, 3, 1)
    
    d_spherical_kms = d_spherical_kms.squeeze().transpose()
    assert d_spherical_kms.shape == (3, N)
    
    #for reference, how we transform pm_ra_cosdec in mas/year to km/s
    #v_ra = (pm_ra_cosdec.to(u.radian/u.year)*distance*(1/u.radian)).to(u.km/u.s)
    
    #convert to mas/year for proper motions
    d_spherical = {'radial_velocity': d_spherical_kms[0]*u.km/u.second,
                   'pm_ra_cosdec':  ((d_spherical_kms[1]/distance)*u.radian/u.second).to(u.mas/u.year),
                   'pm_dec': ((d_spherical_kms[2]/distance)*u.radian/u.second).to(u.mas/u.year)}
    
    return d_spherical
if __name__ == "__main__":
    print(pm_to_dxyz(56.44, 23.86, 135.79576317,19.997, -45.548, np.array([5.65, 5.65])))

    #rv as a scalar:
    print(pm_to_dxyz(56.44, 23.86, 135.79576317,19.997, -45.548, 5.65))

    #get some stars:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    #matplotlib inline

    from matplotlib.ticker import FormatStrFormatter
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    import sys
    sys.path.append('./src')

    from data_queries import  getClusterInfo, getGAIAKnownMembers
    from gaiastars import gaiastars as gs

    import astropy.units as u

    known_cluster_members, cluster_names = getGAIAKnownMembers()
    print(cluster_names)

    # gaiadr2 to gaiaedr3 mapper
    from  gaiastars import gaiadr2xdr3
    # just deal with Pleiades and alphaPer for now
    cluster_names = ['Pleiades', 'alphaPer']
    xmatches = {}
    cluster_members={}
    #for cl in cluster_names:
    for cl in cluster_names:
        known_members_dr2 = list(known_cluster_members.query('Cluster == @cl').index)
        xmatches[cl] = gaiadr2xdr3(known_members_dr2)
        cluster_members[cl]  = gs(name = cl, description=f'{cl} sources from Table 1a records from Gaia archive')
        cluster_members[cl].from_source_idlist(list(xmatches[cl].dr3_source_id),schema='gaiaedr3', query_type='sync')
    
    #just the stars with RV
    coords = {}
    for cl in cluster_names:
        cluster_members[cl].objs.dropna(inplace=True)
        coords[cl] = cluster_members[cl].get_coords(default_rv=True)


    #calculate d_xyz and astropy's
    calc_d_xyz = {}
    astropy_d_xyz = {}
    for cl in cluster_names:
        astropy_d_xyz[cl] = coords[cl].velocity.d_xyz.value
        calc_d_xyz[cl] = np.array([pm_to_dxyz(s.ra, s.dec, s.r_est, s.pmra, s.pmdec, s.radial_velocity) for s in cluster_members[cl].objs.itertuples()]).T

    print('\nDrumroll please...')
    for cl in cluster_names:
        print(f'Cluster: {cl}, passed: {np.allclose(astropy_d_xyz[cl],calc_d_xyz[cl])}')