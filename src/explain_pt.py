import numpy as np
import pandas as pd
import astropy.units as u


def explain_pt(star, center, center_radius):
    df = pd.DataFrame([('Star', star.ra, star.dec, star.pmra, star.pmdec),
                        ('Cluster', center.ra.degree, center.dec.degree, center.pm_ra_cosdec.value, center.pm_dec.value)   ],
        columns=['Which', 'ra', 'dec', 'pm_ra_cosdec', 'pm_dec']).set_index('Which')

    print('Input:')
    print(df)

    print()
    colat_star = np.radians(90-star.dec);
    colat_center = np.radians(90-center.dec.degree)
    polar_angle = np.radians(star.ra - center.ra.degree)
    print(f'Colatitude Star (radian): {colat_star:.4f}')
    print(f'Colatitude Center (radian): {colat_center:.4f}')
    print(f'Polar Angle, Star-to-Center (radian): {polar_angle:.4f}')

    print()
    print('Step 1: Calculate Theta, angle from star to center wrt. Meridian')
    print('----------------------------------------------------------------')

    print('Step 1a: Calculate Great Circle Distance Star-to-Center (radian)')
    print('\tcos(star_cen_dist) = cos(colat_star)*cos(colat_center) + sin(colat_star)*sin(colat_center)*cos(polar_angle)')
    print(f'\tcos(star_cen_dist) = cos({colat_star:.4f})*cos({colat_center:.4f}) + sin({colat_star:.4f})*sin({colat_center:.4f})*cos({polar_angle:.4f})')
    cos_star_cen_dist =  np.cos(colat_star)*np.cos(colat_center) + np.sin(colat_star)*np.sin(colat_center)*np.cos(polar_angle)
    print(f'\tcos(star_cen_dist) = {cos_star_cen_dist:.4f}')
    star_cen_dist = np.arccos(cos_star_cen_dist)
    print(f'\tstar_cen_dist = {star_cen_dist:.4f} radian')

    print('Step 1b: Calculate theta: angle from star to center wrt. meridian')
    print('\tcos(colat_center) = cos(star_cen_dist)*cos(colat_star) + sin(star_cen_dist)*sin(colat_star)*cos(theta)')
    print('\tSolve for theta:')
    print('\tcos(theta) = (cos(colat_center)-cos(star_cen_dist)*cos(colat_star))/(sin(star_center_dist)*sin(colat_star))')
    print(f'\tcos(theta) = (cos({colat_center:.4f})-cos({star_cen_dist:.4f})*cos({colat_star:.4f}))/(sin({star_cen_dist:.4f})*sin({colat_star:.4f}))')
    cos_theta = (np.cos(colat_center)-np.cos(star_cen_dist)*np.cos(colat_star))/(np.sin(star_cen_dist)*np.sin(colat_star))
    theta = np.arccos(cos_theta)
    print(f'\tcos(theta) = {cos_theta:.4f}')
    print(f'\ttheta = {theta:.4f} radian')
    print()

    print('Step 2: Calculate gamma: Angle from Star to outer edge of Center')
    print('----------------------------------------------------------------')
    print('Let cen_rad be center radius in radians:')
    cen_rad = np.radians(center_radius)
    print(f'Center Radius: {center_radius} degrees = {cen_rad:.4f} radians')
    print('Angle "r" between arc(star<->center) and arc(center<->outer_edge) = 90 degrees => cos(r) = 0.0')
    print('Step 2a: Calculate hypot: length of arc connecting star to outer edge of center:')
    print('\tcos(hypot) = cos(star_cen_dist)*cos(cen_rad) + sin(star_cen_dist)*sin(cen_rad)*cos(r)')
    print('\tcos(hypot) = cos(star_cen_dist)*cos(cen_rad) since cos(r)=0')
    cos_hypot = np.cos(star_cen_dist)*np.cos(cen_rad)
    hypot = np.arccos(cos_hypot)
    print(f'\tcos(hypot) = cos({star_cen_dist:.4f})*cos({cen_rad:.4f})')
    print(f'\tcos(hypot) = {cos_hypot:.4f}')
    print(f'\thypot = {hypot:.4f} radian')
    print('Step 2b: calculate gamma')
    print('\tcos(cen_rad) = cos(hypot)*cos(star_cen_dist) + sin(hypot)*sin(star_cen_dist)*cos(gamma)')
    print('\tSolve for gamma:')
    print('\tcos(gamma) = (cos(cen_rad)-cos(hypot)*cos(star_cen_dist))/(sin(hypot)*sin(star_cen_dist))')
    print(f'\tcos(gamma) = (cos({cen_rad:.4f})-cos({hypot:.4f})*cos({star_cen_dist:.4f}))/(sin({hypot:.4f})*sin({star_cen_dist:.4f}))')
    cos_gamma = (np.cos(cen_rad)-np.cos(hypot)*np.cos(star_cen_dist))/(np.sin(hypot)*np.sin(star_cen_dist))
    gamma = np.arccos(cos_gamma)
    print(f'\tcos(gamma) = {cos_gamma:.4f}')
    print(f'\tgamma = {gamma:.4f}')
    print()

    print('Step 3: Calculate Direction of Star Proper Motion in Center Reference Frame')
    print('---------------------------------------------------------------------------')

    pm_ra_cosdec = star.pmra - center.pm_ra_cosdec.to_value(u.mas/u.year)
    pm_dec = star.pmdec - center.pm_dec.to_value(u.mas/u.year)
    print(f'PM_RA_COSDEC (mas/year): Star: {star.pmra:.2f}, Center:{center.pm_ra_cosdec.value:.2f}, Difference: {pm_ra_cosdec:.2f}  ')
    print(f'PM_DEC (mas/year): Star: {star.pmdec:.2f}, Center:{center.pm_dec.value:.2f}, Difference: {pm_dec:.2f}  ')
    print(f'\tphi = arctan2(pm_ra_cosdec, pm_dec)')
    print(f'\tphi = arctan2({pm_ra_cosdec:.2f}, {pm_dec:.2f})')
    phi = np.remainder(np.arctan2(pm_ra_cosdec, pm_dec)+2*np.pi, 2*np.pi)
    print(f'\tphi = {phi:.4f} radian')
    print('\tNeed phi\', opposite direction of phi')
    print('\tphi\' = phi - pi')
    print(f'\tphi\' = {phi:.4f} - {np.pi:.4f}')
    phi_prime = np.remainder(phi + np.pi, 2.0*np.pi)
    print(f'\tphi\' = {phi_prime:.4f}')
    print()

    print('Step 4: Calculate points_to')
    print('---------------------------')
    print('\tPoints_to = theta - gamma <= phi\' <= theta + gamma')
    print(f'\tPoints_to = {theta:.4f} - {gamma:.4f} <= {phi_prime:.4f} <= {theta:.4f} + {gamma:.4f}')
    upper = theta+gamma; lower=theta-gamma
    pt = (lower <= phi_prime) & (phi_prime <= upper)
    print(f'\tPoints_to = {lower:.4f} <= {phi_prime:.4f} <= {upper:.4f}')
    print(f'\tPoints_to = {pt}')