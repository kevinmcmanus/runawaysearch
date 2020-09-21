from locate_cluser_outliers.src.gaiastars import gaiastars as gs
from astropy.time import Time

import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity
from astroquery.gaia import Gaia
import astropy.coordinates as coord
from astropy.table import QTable, Table, vstack
#this is where functions for calculating magnitude, color index, and whatever other values we need 

#this function will add various distances to the known members table, plot is a boolean to know if the result should be converted into a dimentionless scalar
def getDistancesKnownMembers(plot, cluster_data, known_members, cluster_name):
    #add column for distance of each member to cluster center
    known_members['dist_c3d']=[s[1].coords.separation_3d(cluster_data.loc[s[1].SimbadCluster]['coords']) for s in known_members.iterrows()]
    
    #if you want to plot you need it to be a dimentionless scalar
    if(plot == True):
        known_members['dist_c3d'] = Quantity(members.dist_c3d, unit=u.pc).valueS
    
    #get distance of each member from sun
    known_members['r_est'] = [s.distance.value for s in members.coords]
    return(known_members)

# def getDistancesCandidateMembers(candidates, ):
    