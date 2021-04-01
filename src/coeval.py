import numpy as np 
from astropy.coordinates import SkyCoord
import astropy.units as u 
from transforms import pm_to_dxyz


def coeval(star, center, times, rv):
    # get the center's position at each time t
    cen_xyz = center.cartesian.xyz #shape(3,)
    cen_d_xyz = center.velocity.d_xyz #shape(3,)

    #relying on astropy to do unit conversion from km/s to pc/year

    #center displacements in {x,y,z} at each time step; shape(time,{x,y,z})
    cen_delta_t = np.outer(times, cen_d_xyz) #shape(times,{x,y,z})

    #ceneter position at time t: shape(time, {x,y,z})
    cen_pos_t = cen_xyz[np.newaxis,:]+cen_delta_t

    #work on the star
    star_xyz = star.cartesian.xyz

    #need 2d array (rv x {dx,dy,dz}) for each rv (need to transpose result of pm_to_dxyz)
    star_rv_d_xyz = pm_to_dxyz(star.ra.value, star.dec.value, star.distance.value,
                star.pm_ra_cosdec.value, star.pm_dec.value, rv.value).T*u.km/u.second
    
    #3d array (rv x time x {x,y,z}) of star's displacements
    star_delta_rv_t2 = np.outer(times, star_rv_d_xyz)
    star_delta_rv_t1 = star_delta_rv_t2.reshape( len(times), len(rv), 3)
    star_delta_rv_t = star_delta_rv_t1.transpose(1,0,2)

    #3d postion of star for each rv, time pair: shape(rv, times, {x,y,z})
    star_pos_rv_t = star_xyz[np.newaxis, np.newaxis, :] + star_delta_rv_t

    #get star position wrt center at time t
    star_off_rv_t = star_pos_rv_t - cen_pos_t[np.newaxis,:]

    star_cen_dist_rv_t2 = (star_off_rv_t**2).sum(axis=2)

    assert star_cen_dist_rv_t2.shape == (len(rv), len(times))

    return np.sqrt(star_cen_dist_rv_t2)

if __name__ == "__main__":

    center = SkyCoord(ra=56.44*u.degree, dec= 23.86*u.degree, distance=135.79576317*u.pc,
            pm_ra_cosdec=19.997*u.mas/u.year, pm_dec=-45.548*u.mas/u.year,
            radial_velocity= 5.65*u.km/u.second)
    
    print(center)

    star = SkyCoord(ra=45.76103592*u.degree, dec=1.66188041*u.degree, distance=147.80520028*u.pc,
            pm_ra_cosdec = -30.50401334*u.mas/u.year, pm_dec = 33.25870675*u.mas/u.year)
    print(star)

    print(f'Current separation: {star.separation_3d(center)}')

    rv = [-5, 0 ,2.0, 4]*u.km/u.second
    times = [-24e6, -20e6, -10e6, -1e6, 0]*u.year

    test_coeval = coeval(star, center, times, rv)

    print(test_coeval)