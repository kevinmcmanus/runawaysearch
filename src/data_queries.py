<<<<<<< HEAD
from locate_cluser_outliers.src.gaiastars import gaiastars as gs
=======
#from locate_cluser_outliers.src.gaiastars import *
>>>>>>> 5e5c56bbad55098d47e3ec94d436ffc33017bce0
from astroquery.simbad import Simbad
from astropy.time import Time

import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity
from astroquery.gaia import Gaia
import astropy.coordinates as coord
from astropy.table import QTable, Table, vstack

import pandas as pd
import numpy as np
import pickle
import os

#this is where functions for searching SIMBAD, GAIA known members, and SDSS or other catalogs
#clusters = ["Pleiades"]

#function for querying simbad with given cluster name
def querySIMBAD(objnames, formatGaia=False):
    """
    Returns SIMBAD catalog info for one or more object names
    Arguments:
        objnames: str (single name), list of strs (multiple names) or dict (names from key values)
    """
    if isinstance(objnames,str):
        ids = [ objnames ]
    elif isinstance(objnames, list):
        ids = objnames
    elif isinstance(objnames, dict):
        ids = list(objnames.keys())
    else:
        raise TypeError(f'Invalid argument type; Valid types are str, list and dict')


    #set up simbad search
    sim = Simbad()
    sim.add_votable_fields('parallax', 'pm','velocity','typed_id')
    #make table of data queried from given cluster name
    sim_table = vstack([sim.query_object(id) for id in ids], join_type = 'exact')
    #turn into usable table
    cluster_data = Table(sim_table['TYPED_ID', 'PLX_VALUE', 'PLX_PREC','RA', 'RA_PREC', 'DEC', 'DEC_PREC', 
                                   'PMRA', 'PMDEC', 'RVZ_RADVEL', 'RVZ_ERROR'])
    if formatGaia:
        name_mapper = objnames if isinstance(objnames, dict) else None
        return formatSIMBADtoGAIA(cluster_data, name_mapper=name_mapper)
    return(cluster_data)


def formatSIMBADtoGAIA(simbad_table, name_mapper=None):
    """
    the simbad table has different collumn names and coordinates than GAIA, 
    this function converts the SIMBAD data table into the GAIA format
    """
    if name_mapper is not None:
        if not isinstance(name_mapper, dict):
            raise TypeError('name_mapper argurment must be a dict')


    my_table = simbad_table.copy() # so we don't clobber the argument table

    #change the TYPED_ID to a regular ol' string (gotta be a better way to do this):
    my_table['TYPED_ID'] = [c.decode('utf-8') for c in my_table['TYPED_ID']]

    #fix up the column names
    my_table.rename_column('TYPED_ID','typed_id')
    my_table.rename_column('PLX_VALUE','parallax')
    my_table.rename_column('PLX_PREC', 'parallax_error')
    my_table.rename_column('RA', 'ra')
    my_table.rename_column('RA_PREC','ra_error')
    my_table.rename_column('DEC', 'dec')
    my_table.rename_column('DEC_PREC','dec_error')
    my_table.rename_column('PMRA', 'pmra')
    my_table.rename_column('PMDEC', 'pmdec')
    my_table.rename_column('RVZ_RADVEL','radial_velocity')
    my_table.rename_column('RVZ_ERROR', 'rv_error')

    #ditch the masked arrays
    my_table = my_table.filled()

    #add cluster column which is either the typed name or the mapped name
    if name_mapper is not None:
        #for names not mapped, substitute the typed_id
        my_table['cluster']=[name_mapper.get(tid, tid) for tid in my_table['typed_id']]
    else:
        my_table['cluster'] = my_table['typed_id']

    #add index on cluster
    my_table.add_index('cluster')
    
    #tack on sky coordinate objects for each
    my_table['coords'] = \
        SkyCoord(ra = my_table['ra'],
            dec = my_table['dec'], unit = (u.hour, u.deg),
            obstime = 'J2000',  #simbad returns J2000 coords
            distance = coord.Distance(parallax=Quantity(my_table['parallax'])),
            pm_ra_cosdec = my_table['pmra'],
            pm_dec = my_table['pmdec'],
        radial_velocity = my_table['radial_velocity']).apply_space_motion(new_obstime=Time(2015.5,format='decimalyear'))
    
    return(my_table)

#function for getting data on the GAIA known cluster members as defined by this paper http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2018A%26A...616A..10G
def getGAIAKnownMembers(name_mapper=None):
    """
    returns known member list from Gaia paper: http://simbad.u-strasbg.fr/simbad/sim-ref?bibcode=2018A%26A...616A..10G (Table1a: Nearby Open Clusters)
    
    Returns:
        Pandas dataframe indexed by SourceID, and list of cluster names retrieved.
    """
    known_members = pd.read_csv('ftp://cdsarc.u-strasbg.fr/pub/cats/J/A+A/616/A10/tablea1a.dat',
                      delim_whitespace=True,
                      header=None, index_col=None,
                      names = ['SourceID', 'Cluster', 'RAdeg', 'DEdeg', 'Gmag', 'plx', 'e_plx'])
    known_members.set_index('SourceID', inplace=True)
    
    cluster_names = known_members.Cluster.unique()
    
    #not sure how to deal with name mapping back to SIMBAD names
    #name_mapper = {'Pleiades': 'Pleiades'}
    #members['SimbadCluster'] = members.Cluster.apply(lambda c:name_mapper[c])
    
    #make skycoords objects for each member, using gaia epoch of 2015.5
    known_members['coords']=SkyCoord(ra = np.array(known_members.RAdeg)*u.degree,
        dec = np.array(known_members.DEdeg)*u.degree,
        obstime = Time(2015.5,format='decimalyear'),  #Gaia ref epoch is 2015.5
        distance = coord.Distance(parallax=Quantity(np.array(known_members.plx)*u.mas)))
    
    return(known_members, cluster_names)


#this function will add various distances to the known members table, plot is a boolean to know if the result should be converted into a dimentionless scalar
def getDistances(plot, cluster_data, known_members, cluster_name):
    #add column for distance of each member to cluster center
    known_members['dist_c3d']=[s[1].coords.separation_3d(cluster_data.loc[s[1].SimbadCluster]['coords']) for s in known_members.iterrows()]
    
    #if you want to plot you need it to be a dimentionless scalar
    if(plot == True):
        known_members['dist_c3d'] = Quantity(members.dist_c3d, unit=u.pc).valueS
    
    #get distance of each member from sun
    known_members['r_est'] = [s.distance.value for s in members.coords]
    return(known_members)


if __name__ == '__main__':
    name_mapper = {
        'Hyades': 'Hyades',
        'Coma Berenices Cluster': 'ComaBer',
        'Pleiades': 'Pleiades',
        'Praesepe': 'Praesepe',
        'alpha Per': 'alphaPer',
        'IC 2391': 'IC2391',
        'IC 2602': 'IC2602',
        'Blanco 1': 'Blanco1',
        'NGC 2451A': 'NGC2451'
    }
 

    print('\n------------------Test querySIMBAD ----------------')
    simbad_info = querySIMBAD(name_mapper)
    print(simbad_info)
    orig_colnames = simbad_info.colnames

    print('\n------------------Test formatSIMBADtoGAIA ----------------')
    cluster_info = formatSIMBADtoGAIA(simbad_info)
    print(cluster_info)
    print(cluster_info.loc['Pleiades']['coords'])
    # did we clobber simbad_info?
    assert simbad_info.colnames == orig_colnames

    print('\n------------------Test Combined Query ----------------')
    cluster_info2 = querySIMBAD(name_mapper, formatGaia=True)
    print(cluster_info2.loc['Pleiades']['coords'])
    print(cluster_info2)

    print('\n------------------Test GetGAIAKnownMembers ----------------')
    cluster_members, cluster_names = getGAIAKnownMembers()
    #how many members in each cluster?
    member_counts = cluster_members.reset_index().groupby('Cluster').count()['SourceID']
    print(member_counts)

