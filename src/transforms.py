import numpy as np 

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
    seconds_per_year = 31557600.0
    mas_per_degree = 3600000.0
    km_per_pc = 30856775814671.914

    #get some radians
    rarad = np.radians(ra);
    decrad = np.radians(dec)
    pmrarad = np.radians(pm_ra_cosdec/mas_per_degree) # radians per year
    pmdecrad = np.radians(pm_dec/mas_per_degree) # radians per year

    v_ra = (pmrarad*distance*km_per_pc)/seconds_per_year # km/second
    v_dec = (pmdecrad*distance*km_per_pc)/seconds_per_year #km/second
    
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