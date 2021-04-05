import numpy as np
import pandas as pd
import astropy.units as u


def rms(x):
    return np.sqrt((x**2).sum()/len(x))

from scipy.stats import chi2

class star():
    def __init__(self, gstar, source_id):
        self.A = 4.740470463496208 # converts PM, omegabar to km/s
        self.source_id = source_id
        self.rarad = np.radians(gstar.ra)
        self.decrad=np.radians(gstar.dec)
        self.distance = 1000.0/gstar.parallax
        self.rv = gstar.radial_velocity
        self.rv_error = gstar.radial_velocity_error
        star.gstar = gstar
        self.dof = 3 if np.isfinite(gstar.radial_velocity) else 2
        self.R = self.get_R(gstar, self.dof==3)

        self.xyz, self.d_xyz, self.d_xyz_covar, self.pm_error, self.pm_covar, self.tang_v, self.tang_v_covar=self.eq_to_cartesian(gstar, self.dof==3)

    def get_R(self, gstar, three_d):
        rarad = self.rarad; decrad=self.decrad

        R = np.array([[-np.sin(rarad), -np.sin(decrad)*np.cos(rarad), np.cos(decrad)*np.cos(rarad)],
                    [ np.cos(rarad), -np.sin(decrad)*np.sin(rarad), np.cos(decrad)*np.sin(rarad)],
                    [ 0,              np.cos(decrad),               np.sin(decrad)]
                    ])
        #the transpose of R should equal its inverse
        assert np.allclose(R.T,np.linalg.inv(R))
        if not three_d:
            #chop off last column
            R = R[:,:2].copy()

        return R

    def get_pm_jacobian(self, gstar, three_d):

        A = self.A

        pm_jacobian = np.array([[-gstar.pmra*A/gstar.parallax**2,  A/gstar.parallax, 0,                0],
                                [-gstar.pmdec*A/gstar.parallax**2, 0,                A/gstar.parallax, 0],
                                [0,                              0,                0,               1]
                            ])
        if not three_d:
            pm_jacobian = pm_jacobian[:2,:3].copy()

        return pm_jacobian

    def eq_to_cartesian(self, gstar, three_d):
        """
        returns cartesian coords, velocity and velocity covariance for the given star
        """
        A = 4.740470463496208
        rv = gstar.radial_velocity if three_d else 0
        rv_error = gstar.radial_velocity_error if three_d else 0

        R = self.R
        xyz = self.distance*np.array([np.cos(self.decrad)*np.cos(self.rarad), np.cos(self.decrad)*np.sin(self.rarad),np.sin(self.decrad)]).reshape(-1,1)
        tang_v = np.array([[A*gstar.pmra/gstar.parallax], [A*gstar.pmdec/gstar.parallax], [rv]])
        if not three_d:
            tang_v = tang_v[:2]
        d_xyz = R.dot(tang_v)

        #get the covariance matrix of d_xyz
        # {parallax, pmra, pmdec, rv}; rv not correlated with anything
        pm_corr = np.array([ [1,                       gstar.parallax_pmra_corr,   gstar.parallax_pmdec_corr,   0],
                            [gstar.parallax_pmra_corr, 1,                          gstar.pmra_pmdec_corr,      0],
                            [gstar.parallax_pmdec_corr, gstar.pmra_pmdec_corr,     1,                          0],
                            [0,                        0,                        0,                          1]
                            ])
        if three_d:
            pm_error = np.diag(np.array([gstar.parallax_error, gstar.pmra_error, gstar.pmdec_error, rv_error]))
        else:
            pm_error = np.diag(np.array([gstar.parallax_error, gstar.pmra_error, gstar.pmdec_error]))
            pm_corr = pm_corr[:3,:3]

        pm_covar = pm_error.dot(pm_corr.dot(pm_error))

        pm_jacobian = self.get_pm_jacobian(gstar, three_d)
        tang_v_covar = pm_jacobian.dot(pm_covar.dot(pm_jacobian.T))
        d_xyz_covar = R.dot(tang_v_covar.dot(R.T))

        return xyz, d_xyz, d_xyz_covar, pm_error, pm_covar, tang_v, tang_v_covar

    def compare_center_motion(self, center_motion, conf):
        chi_val = chi2.ppf(conf, self.dof)

        # get center's expected motion at star's position and calculate difference
        exp_motion = self.R.T.dot(center_motion['d_xyz'])
        delta_tang_v = self.tang_v - exp_motion
        # get the covariance of the exp motion at the position:
        cen_covar = self.R.T.dot(center_motion['d_xyz_covar'].dot(self.R))
        Sigma_inv = np.linalg.inv(self.tang_v_covar+cen_covar)

        #chi square stat:
        c = delta_tang_v.T.dot(Sigma_inv.dot(delta_tang_v))[0,0]

        #H_0: velocities are same; H_1: velocities are different
        self.velocity_consistent = c < chi_val

        return self.velocity_consistent


class perryman():
    def __init__(self, star_df, init_members):
        self.objs = star_df.copy()
        
        self.n_stars = len(star_df)
        
        #conversion factor A to map (pm,omega_bar) to km/s
        #pm in mas/year, omega_bar in mas
        seconds_per_year = (1*u.year).to(u.second)
        km_per_pc = (1*u.pc).to(u.km)
        rad_per_mas = (1.0*u.mas).to(u.radian)
        pc_per_mas_plx = 1000*u.pc/1*u.mas

        A = (rad_per_mas*pc_per_mas_plx*km_per_pc/seconds_per_year).to_value()
        self.A = A
        
        if type(init_members) is list:
            
            idlist = set(self.objs.index)
            members = pd.Series(False, index=self.objs.index)
            common_members = idlist.intersection(set(init_members))
            members[common_members] = True
            self.init_members = np.array(members)
            self.missing_members = list(set(init_members)-idlist)

        elif type(init_members) is dict:
            
            ra = init_members['ra']
            dec = init_members['dec']
            radius = init_members['radius']
            dist = init_members.pop('distance', None)
            if dist is None:
                dist_ok = True
            else:
                #better be a tuple of (minpc, maxpc)
                maxplx = 1000/dist[0]
                minplx = 1000/dist[1]
                dist_ok = np.logical_and(self.objs.parallax >= minplx, self.objs.parallax <= maxplx)
                
            seps_ok = self.ang_separation(ra, dec) <= radius
            
            members = np.logical_and(seps_ok, dist_ok)
            self.init_members = np.array(members)
            self.missing_members = []
            
        else:
            print(f'init_members argument must be a list or dict, not a {type(init_members)}')
        
        print(f'Model initialized with {self.init_members.sum()} members; Missing members: {len(self.missing_members)}')
        
        self.stars = [star(s,id) for id,s in star_df.iterrows()]
        self.objs_xyz = np.array([s.xyz for s in self.stars])
        self.objs_d_xyz = np.array([s.d_xyz for s in self.stars])
        
        
    def fit(self,conf = 0.95, max_dist=100, maxiter=100):
        

        print('iterating')
        iter = 0
        is_member_this_iter = self.init_members
        self.nmembers_iter = np.full(maxiter, np.nan)
        while iter < maxiter:

            was_member_last_iter = is_member_this_iter
            self.nmembers_iter[iter] = was_member_last_iter.sum()
            center_pos = self.calculate_center_pos(was_member_last_iter)
            center_motion = self.calculate_center_motion(was_member_last_iter)
            
            is_member_this_iter = self.calculate_membership(center_pos, center_motion,
                                                            conf, max_dist=max_dist)
            
            if not np.any(np.logical_xor(was_member_last_iter,is_member_this_iter)):
                break
            iter += 1
                
        print(f'Iterations remaining: {maxiter - iter}, number of members: {is_member_this_iter.sum()}')
        
        self.center_pos = center_pos
        self.center_motion = center_motion
        self.members = is_member_this_iter
        
        return center_pos, center_motion, is_member_this_iter
    
    def calculate_center_pos(self, member_list):

        xyz_m = self.objs_xyz[member_list]
        xyz = xyz_m.mean(axis=0)
        #eq_coords = self.to_spherical(xyz)
        #R = self.get_R(eq_coords['ra'], eq_coords['dec'])
        
        ret_dict = {'xyz':xyz} #,
                #    'eq_coords':eq_coords,
                #    'R':R}

        return ret_dict
    
    def calculate_center_motion(self, member_list):
        # means of the members
        
        d_xyz_m = self.objs_d_xyz[member_list]
        d_xyz = d_xyz_m.mean(axis=0)

        d_xyz_covar = np.cov(d_xyz_m.reshape(-1,3),rowvar=False)

        return {'d_xyz':d_xyz, 'd_xyz_covar':d_xyz_covar}
    
    def calculate_center_distance(self, cen_xyz):

        xyz_m = self.objs_xyz
        cen_dist_xyz = xyz_m - cen_xyz[np.newaxis,:]
        
        cen_dist = np.sqrt((cen_dist_xyz**2).sum(axis=1)).reshape(-1)
        
        return cen_dist
    
    def calculate_bc_velocity(self, cen_vel=None):
        
        sqrt_three_halves = np.sqrt(3.0/2.0)
        
        if cen_vel is None:
            cv = np.array([0.0,0.0])
        else:
            cv = cen_vel
        delta_vel = self._tangental_velocity - cv[np.newaxis,:]
        
        vel = np.sqrt((delta_vel**2).sum(axis=1))*sqrt_three_halves
        
        return vel
    
    def calculate_membership(self, center_pos, center_motion, conf, max_dist):

        velocity_consistent = np.array([s.compare_center_motion(center_motion, conf) for s in self.stars])
        cen_dist = self.calculate_center_distance(center_pos['xyz'])
        
        is_member = np.logical_and(velocity_consistent, cen_dist <= max_dist)
        
        return is_member
    
    def ang_separation(self, alpha1, delta1, alpha2=None, delta2=None):
        arcsec_per_degree = 3600

        delta1rad = np.radians(delta1)
        alpha1rad = np.radians(alpha1)
        
        if alpha2 is None and delta2 is None:
            alpha2rad = np.radians(self.objs.ra)
            delta2rad = np.radians(self.objs.dec)
        else:
            delta2rad = np.radians(delta2)
            alpha2rad = np.radians(alpha2)

        cos_theta = np.sin(delta1rad)*np.sin(delta2rad)+np.cos(delta1rad)*np.cos(delta2rad)*np.cos(alpha1rad-alpha2rad)
        thetarad = np.arccos(cos_theta)
        theta = np.degrees(thetarad)

        return theta*arcsec_per_degree

    def get_info(self, star_i:int):
        inp =self.objs.iloc[star_i]
        ret_dict = {
            'input': {'ra':inp.ra, 'dec':inp.dec,'plx':inp.parallax,
                    'pmra':inp.pmra, 'pmdec':inp.pmdec,
                    'pmra_error':inp.pmra_error, 'pmdec_error': inp.pmdec_error},
            'R': self._R[star_i],
            'PM_covar': self._Covariance[star_i],
            'tangental_velocity': self._tangental_velocity[star_i],
            'tangental_velocity_covar': self._tangental_velocity_error_covar[star_i],
            'space_pos': self.objs_xyz[star_i],
            'space_velocity': self.objs_dxyz[star_i],
            'space_velocity_covar':self.objs_dxyz_covar[star_i]
        }
        return ret_dict

    def plot(self, fig, pm_percentile=(10,90)):
        gs = fig.add_gridspec(4,3, width_ratios=[5,5,4])
        radec_ax = fig.add_subplot(gs[:,0])
        pm_ax = fig.add_subplot(gs[:,1])
        iter_ax = fig.add_subplot(gs[3,2])
        cen_dist_hist = fig.add_subplot(gs[2,2])
        vel_dist = fig.add_subplot(gs[1,2])
        vel_hist = fig.add_subplot(gs[0,2])
        
        pmra_percentile = np.percentile(self.objs.pmra, pm_percentile)
        pmdec_percentile = np.percentile(self.objs.pmdec, pm_percentile)
        
        members = self.objs[self.members]
        nonmembers = self.objs[~self.members]
        
        radec_ax.scatter(nonmembers.dec, nonmembers.ra, color='lightgrey', s=1,label='Noncandidates')        
        radec_ax.scatter(members.dec, members.ra, color='red', s=1,label='Candidates')
        radec_ax.invert_xaxis()
        radec_ax.grid()
        radec_ax.legend()
        radec_ax.set_title('Postions in RA/Dec')
        radec_ax.set_xlabel('Right Ascension (degree)')
        radec_ax.set_ylabel('Declination (degree)')
        
        pm_ax.scatter(nonmembers.pmra, nonmembers.pmdec, s=1,  color='lightgrey', label='Noncandidates')
        pm_ax.scatter(members.pmra, members.pmdec, color='red',s=2, label='Candidates')
        pm_ax.grid()
        pm_ax.legend()
        pm_ax.set_title('Proper Motions')
        pm_ax.set_xlabel('PM in Right Ascension (mas/year)')
        pm_ax.set_ylabel('PM in Declination (mas/year)')
        pm_ax.set_xlim(pmra_percentile)
        pm_ax.set_ylim(pmdec_percentile)
        
        iter_ax.plot(self.nmembers_iter, label=f'N Candidates: {self.members.sum()}')
        iter_ax.set_title('Number of Candidates by Iteration')
        iter_ax.set_xlabel('Iteration')
        iter_ax.set_ylabel('# Candidates')
        iter_ax.grid(axis='y')
        iter_ax.legend()
        
        cen_dist = self.calculate_center_distance(self.center_pos['xyz'])
        med = np.median(cen_dist[self.members])
        cen_dist_hist.hist(cen_dist[self.members], bins=100, color='red', alpha=0.5)
        cen_dist_hist.axvline(med,ls=':', lw=4, color='blue', label=f'Median: {med:.1f} pc')
        cen_dist_hist.set_title('Distribution of Distance from Center')
        cen_dist_hist.set_ylabel('Number of Stars')
        cen_dist_hist.set_xlabel('Distance from Center (pc)')
        cen_dist_hist.legend()
        
        bc_vel = self.calculate_bc_velocity(self.center_motion['tangental_velocity'])
        
        vel_hist.hist(bc_vel[self.members], bins=100, color='red', alpha=0.5)
        med = np.median(bc_vel[self.members])
        vel_hist.set_ylabel('Number of Stars')
        vel_hist.axvline(med,ls=':', lw=4, color='blue', label=f'Median: {med:.1f} km/s')
        vel_hist.set_xlabel('Est. Space Velocity (km/s)')
        vel_hist.set_title('Distribution of Estimated Space Velocity')
        vel_hist.legend()
        
        vel_dist.scatter(cen_dist[self.members], bc_vel[self.members], s=1, color='red')
        vel_dist.set_ylabel('bc velocity (km/s)')
        vel_dist.set_xlabel('Distance from Center (pc)')
        vel_dist.set_title('Barycentric Velocity v. Distance from Center')

def get_R(star_dict):
    A = 4.740470463496208
    star = pd.Series(star_dict)

    rarad = np.radians(star.ra); decrad=np.radians(star.dec)

    R = np.array([[-np.sin(rarad), -np.sin(decrad)*np.cos(rarad), np.cos(decrad)*np.cos(rarad)],
                  [ np.cos(rarad), -np.sin(decrad)*np.sin(rarad), np.cos(decrad)*np.sin(rarad)],
                  [ 0,              np.cos(decrad),               np.sin(decrad)]
                ])
    #the transpose of R should equal its inverse
    assert np.allclose(R.T,np.linalg.inv(R))
    return R

def get_pm_jacobian(star_dict):

    A = 4.740470463496208
    star=pd.Series(star_dict)
    pm_jacobian = np.array([[-star.pmra*A/star.parallax**2,  A/star.parallax, 0,                0],
                            [-star.pmdec*A/star.parallax**2, 0,                A/star.parallax, 0],
                            [0,                              0,                0,               1]
                        ])

    
    return pm_jacobian

def eq_to_cartesian(self, star):
    """
    returns cartesian coords, velocity and velocity covariance for the given star
    """
    A = 4.740470463496208
    rv = star.rv if np.isfinite(star.radial_velocity) else 0
    rv_error = star.rv_error if np.isfinite(star.radial_velocity_error) else 0

    R = get_R({'ra':star.ra, 'dec':star.dec,'parallax':star.parallax})
    xyz = R.dot(np.array([[0], [0], [1000/star.parallax]]))
    tang_v = np.array([[A*star.pmra/star.parallax], [A*star.pmdec/star.parallax], [rv]])
    d_xyz = R.dot(tang_v)

    #get the covariance matrix of d_xyz
    # {parallax, pmra, pmdec, rv}; rv not correlated with anything
    pm_corr = np.array([ [1,                       star.parallax_pmra_corr,   star.parallax_pmdec_corr,   0],
                         [star.parallax_pmra_corr, 1,                          star.pmra_pmdec_corr,      0],
                         [star.parallax_pmdec_corr, star.pmra_pmdec_corr,     1,                          0],
                         [0,                        0,                        0,                          1]
                        ])
    pm_error = np.diag(np.array([star.parallax_error, star.pmra_error, star.pmdec_error, rv_error]))
    pm_covar = pm_error.dot(pm_corr.dot(pm_error))

    pm_jacobian = get_pm_jacobian({'pmra':star.pmra, 'pmdec':star.pmdec, 'parallax':star.parallax})
    tang_v_covar = pm_jacobian.dot(pm_covar.dot(pm_jacobian.T))
    d_xyz_covar = R.dot(tang_v_covar.dot(R.T))

    return xyz, d_xyz, d_xyz_covar, pm_error, pm_covar, tang_v, tang_v_covar, pm_jacobian, R

def cartesian_to_eq(xyz, d_xyz, d_xyz_covar):
    A = 4.740470463496208
    #ra, dec and distance(r)
    r = np.sqrt((xyz[:,0]**2).sum())
    delta = np.arctan(xyz[2,0]/np.sqrt(xyz[0,0]**2+xyz[1,0]**2))
    alpha = np.arctan2(xyz[1,0], xyz[0,0])
    alpha = np.where(alpha<0, alpha+2.0*np.pi, alpha)
    ra = np.rad2deg(alpha); dec = np.rad2deg(delta); parallax = 1000/r

    print(f'RA: {ra}, Dec: {dec}, parallax: {parallax}')

    R = get_R({'ra':ra, 'dec':dec, 'parallax':parallax})
    # R.T is same as inverse(R)
    tang_v =  R.T.dot(d_xyz)
    pmra = tang_v[0,0]*parallax/A
    pmdec = tang_v[1,0]*parallax/A
    rv = tang_v[2,0]

    pm_jacobian = get_pm_jacobian({'pmra':pmra, 'pmdec':pmdec, 'parallax':parallax})
    # unwind the to_cartesian tranform to get tang_v covar
    tang_v_covar = R.dot(d_xyz_covar.dot(R.T))
    #now get the pm_covar
    pm_covar = pm_jacobian.T.dot(tang_v_covar.dot(pm_jacobian))
    #pull apart the covar matrix
    # See: https://math.stackexchange.com/questions/186959/correlation-matrix-from-covariance-matrix/300775

    pm_err = np.sqrt(np.diag(pm_covar))
    Dinv = np.diag(1/pm_err) #note Dinv.T == Dinv
    pm_corr = Dinv.dot(pm_covar.dot(Dinv))

    ret_dict = {'ra': ra, 'dec': dec,
                'parallax': parallax, 'parallax_error': pm_err[0],
                'pmra':pmra, 'pmra_error': pm_err[1],
                'pmdec': pmdec, 'pmdec_error':pm_err[2],
                'radial_velocity':rv, 'radial_velocity_error':pm_err[3],
                'pmra_pmdec_corr':pm_corr[1,1],
                'parallax_pmra_corr': pm_corr[0,1],
                'parallax_pmdec_corr': pm_corr[0,2]}

    return pd.Series(ret_dict)


        
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    #matplotlib inline

    from matplotlib.ticker import FormatStrFormatter
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    import sys
    sys.path.append('./src')

    from data_queries import  getClusterInfo, getGAIAKnownMembers
    from coeval import coeval
    from gaiastars import gaiastars as gs,gaiadr2xdr3

    import astropy.units as u
    from astropy.coordinates import SkyCoord

    import os
    print(os.getcwd())

        #trumpler meta data
    trumpler_df = pd.DataFrame([
        ['Trumpler14', '10:43:55.4','-59:32:16', 2.37,0.15, 264, -6.58, 0.06, 2.185, 0.084],
        ['Trumpler15', '10:44:40.8', '-59:22:10', 2.36, 0.09, 320, np.nan, np.nan, np.nan, np.nan],
        ['Trumpler16', '10:45:10.6', '-59:42:28', 2.32,0.12, 320, -6.931,0.063, 2.612, 0.058]
    ], columns=['ClusterName','ra', 'dec','distance','disterr','radius','pm_ra_cosdec','pm_ra_cosdec_error','pm_dec','pm_dec_error']
    ).set_index('ClusterName')
    tc = SkyCoord(ra=trumpler_df.ra, dec=trumpler_df.dec,
        pm_ra_cosdec = list(trumpler_df.pm_ra_cosdec)*u.mas/u.year,
        pm_dec = list(trumpler_df.pm_dec)*u.mas/u.year,
        unit=(u.hourangle, u.deg),
        distance = list(trumpler_df.distance)*u.kpc)
    
    trumpler_coords = {}
    for i, cl in enumerate(trumpler_df.index):
        trumpler_coords[cl]=tc[i]

    print(trumpler_coords)

    carina_known_members = pd.read_csv('./data/carina_members.csv', comment='#')

    errorcolumns = [
    'ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error','dr2_radial_velocity_error',
    'ra_dec_corr', 'ra_parallax_corr','ra_pmra_corr', 'ra_pmdec_corr',
    'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
    'parallax_pmra_corr', 'parallax_pmdec_corr',
    'pmra_pmdec_corr'
    ]
    #add table to query to get the ruwe parameter
    fixme = gs(name='fixme')
    fixme.add_table_columns(errorcolumns)

    
    carina_members={}
    #for cl in cluster_names:
    for cl, cluster in carina_known_members.groupby('Cluster'):
        known_members = list(cluster['Gaia Number'])
        print(f'Fetching {cl}')
        carina_members[cl]  = gs(name = cl, description=f'{cl} sources from Shull table from Gaia eDR3')
        carina_members[cl].from_source_idlist(known_members,schema='gaiaedr3', query_type='sync')

    #carina_cluster_names = list(carina_members.keys())
    # can only deal with the Trumplers at the momemnt
    carina_cluster_names = ['Trumpler14', 'Trumpler15', 'Trumpler16']

    for cl in carina_cluster_names:
        center=trumpler_coords[cl]
        coords = carina_members[cl].get_coords()
        cen_dists = center.separation_3d(coords).to_value(u.pc)
        cen_seps = center.separation(coords).to_value(u.arcsecond)
        carina_members[cl].objs['DistanceFromCenter'] = cen_dists
        carina_members[cl].objs['SeparationFromCenter'] = cen_seps

    print('\n--- Min/Max distance of Known Members from Cluster Center ---\n')
    for cl in carina_cluster_names:
        print(f'Cluster: {cl}, Min distance: {carina_members[cl].objs.DistanceFromCenter.min():.2f} pc, Max distance: {carina_members[cl].objs.DistanceFromCenter.max():.2f} pc' )
    print('\n--- Min/Max separation of Known Members from Cluster Center ---\n')
    for cl in carina_cluster_names:
        print(f'Cluster: {cl}, Min Separation: {carina_members[cl].objs.SeparationFromCenter.min():.2f} as, Max Separation: {carina_members[cl].objs.SeparationFromCenter.max():.2f} as' )

    from gaiastars import from_pickle
    carina_search_results = from_pickle(f'./data/carina_search_results')

    t14 = perryman(carina_search_results.objs, list(carina_members['Trumpler14'].objs.index))

    pos=t14.calculate_center_pos(t14.init_members)
    print(pos)

    #t14_pos2, t14_motion2, t14_member2 = t14.fit(maxiter=150, max_dist=50)

    star_i = 32332
    print(f'\n Info for the {star_i}\'th')
    info = t14.get_info(32332)
    for inf in info:
        print(f'\n------------ {inf} ------------')
        print(info[inf])

    s = t14.objs.iloc[star_i]
    xyz, d_xyz, d_xyz_covar = eq_to_cartesian(None,s)
    

    print(cartesian_to_eq(xyz, d_xyz, d_xyz_covar))   