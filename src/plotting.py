# <<<<<<< points_to

#general file for all plotting needs (keeps the main program neat)


#from locate_cluser_outliers.src.gaiastars import gaiastars as gs
#from locate_cluster_outliers.src.data_quieries import *
# =======
#from locate_cluser_outliers.src.gaiastars import gaiastars as gs
#from locate_cluster_outliers.src.data_quieries import *
# >>>>>>> master
import matplotlib.pyplot as plt

#general file for all plotting needs (keeps the main program neat
#plotting known members in RA/DEC 
# def plotKnownMembers(known_members):
# #plotting candidate members in RA/DEC
# def plotCandidates(candidate_members):
# #plotting all possible members in RA/DEC
# def plotAllMembers(known_members, candidate_members):
# #plotting distances of members from sun or from cluster center
# def plotDistances(distance_to_plot):
# def plotHrDiagrams(color_index, abs_mag):

# <<<<<<< points_to
def make_square_axes(ax):
    """Make an axes square in screen units.

    Should be called after plotting.
    """
    ax.set_aspect(1 / ax.get_data_ratio())

import numpy as np
from matplotlib.lines import Line2D 

def getTangentPoints(center, radius, pt):
    """
    returns 4 points ((x,y) tuples); two on the circle centered at center,
    the line between which is perpendicular to the line connecting center and pt
    The other two lie at radius on pt; lines connecting the two points go through pt and are
    tangent to the encircled center.
    """
    
    #first move frame of reference to center
    pt_x = pt[0]-center[0]
    pt_y = pt[1]-center[1]
    cen_dist = np.sqrt(pt_x**2+pt_y**2)
    
    #get theta of line connecting center to pt
    theta = np.arctan2(pt_y, pt_x)
    theta = theta+2*np.pi if theta <0 else theta

    #get phi angle from centerline to tangent
    phi = np.arctan2(radius, cen_dist) #always positive
    
    #get the thetas of the perps
    theta_up = theta+ np.pi/2.0
    theta_down = theta - np.pi/2.0
    #print(f'Theta: {theta}, Theta_up: {theta_up}, Theta_down: {theta_down}')
    
    #convert polar coords to cartesian
    rel_cen_up = (radius*np.cos(theta_up), radius*np.sin(theta_up))
    rel_cen_down = (radius*np.cos(theta_down), radius*np.sin(theta_down))
    #move back to orig ref frame
    pt_cen_up = (rel_cen_up[0]+center[0], rel_cen_up[1]+center[1])
    pt_cen_down = (rel_cen_down[0]+center[0], rel_cen_down[1]+center[1])

    #points on the opposite side of pt
    s = radius*3 # to make the tails bigger
    rel_pt_up = (s*np.cos(theta+phi), s*np.sin(theta+phi))
    rel_pt_down   = (s*np.cos(theta-phi), s*np.sin(theta-phi))
    #move back to orig ref frame
    pt_pt_up = (rel_pt_up[0]+pt_x+center[0], rel_pt_up[1]+pt_y+center[1])
    pt_pt_down = (rel_pt_down[0]+pt_x+center[0], rel_pt_down[1]+pt_y+center[1])
 
    return pt_cen_up, pt_pt_up, pt_cen_down, pt_pt_down

def plot_points_to(ptr, cluster, ax):
    """
    Plots a points_to record on ax.
    Arguments:
        ptr: points to record (row of dataframe) returned by gaiastars.points_to()
        cluster: gaiastars object, presumably the centerpoint of the points_to analysis
        ax: Matplotlib.Axes object - where to put the plot
    """

    #local routine to put on a data table
    def add_ptr_table(ax, ptr):
        """
        puts two data tables on ax, data from points_to record ptr
        """

        #directions not returned on the ptr record:
        cenPMDir = np.arctan2(ptr.CenPMDec, ptr.CenPMRA)
        cenPMDir = cenPMDir + 2*np.pi if cenPMDir <0 else cenPMDir
        objPMDir = np.arctan2(ptr.ObjPMDec, ptr.ObjPMRA)
        objPMDir = objPMDir + 2*np.pi if objPMDir <0 else objPMDir


        tbl = ax.table(
            rowLabels = ['RA (Deg)','Dec (Deg)','PM RA (mas/yr)', 'PM Dec (mas/yr)', 'PM Dir (Radians)', 'Radius (Deg)'],
            colLabels = ['Center', 'Object','Differential'],        
            
            cellText = [
                        [f'{ptr.CenRA:.2f}',    f'{ptr.ObjRA:.2f}',    f'{ptr.ObjRelRA:.2f}'],
                        [f'{ptr.CenDec:.2f}',   f'{ptr.ObjDec:.2f}',   f'{ptr.ObjRelDec:.2f}'],
                        [f'{ptr.CenPMRA:.4f}',  f'{ptr.ObjPMRA:.4f}',  f'{ptr.ObjRelPMRA:.4f}'],
                        [f'{ptr.CenPMDec:.4f}', f'{ptr.ObjPMDec:.2f}', f'{ptr.ObjRelPMDec:.2f}'],
                        [f'{cenPMDir:.2f}',     f'{objPMDir:.2f}',     f'{ptr.ObjRelPMDir:.2f}'],
                        [f'{ptr.CenRad:.2f}',   '',                    '']
                    ],
            bbox=[0.0, -0.50, 1.0, 0.4])

        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)

        #plot the summary table to the right
        tbl2 = ax.table(
            colLabels = ['Measure', 'Value', 'Units'],
            cellText = [
                ['Phi', f'{ptr.Phi:.3f}', 'Radian'],
                ['Theta', f'{ptr.Theta:.3f}', 'Radian'],
                ['ObjDistCen', f'{ptr.ObjDistCen:.3f}','Degree'],
                ['UpBound',f'{ptr.Upper:.3f}','Radian'],
                ['LowBound',f'{ptr.Lower:.3f}', 'Radian'],
                ['Inbounds', f'{ptr.Inbounds}',''],
                ['ObjDistSun', f'{ptr.ObjDistSun:.1f}','parsec'],
                ['WithinDist', f'{ptr.WithinDist}', ''],
                ['PointsTo', f'{ptr.PointsTo}','']
            ],
            colWidths=[0.4, 0.3,0.3],
            bbox=[1.05, 0.50, 0.5, 0.5])
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(12)

        return #from add_ptr_table

    # main line of plot_points_to

    # line from the center to the object
    ax.plot([ptr.CenRA, ptr.ObjRA],[ptr.CenDec, ptr.ObjDec], color='black', label='Center Line')

    # plot the object as a big star
    ax.scatter(ptr.ObjRA, ptr.ObjDec, marker='*', s=500, color='red', label='Star')

    #plot the known cluster members and draw circle around them
    ax.scatter(cluster.objs.ra, cluster.objs.dec,s=1, color='grey', alpha=0.4, label='Known Members')
    clust = plt.Circle((ptr.CenRA, ptr.CenDec), ptr.CenRad, edgecolor='r', fc='None', ls=':')
    ax.add_artist(clust)
    
    #plot cone to star:
    pt_cen_up, pt_pt_up,  pt_cen_down, pt_pt_down = getTangentPoints((ptr.CenRA, ptr.CenDec),
                                                            ptr.CenRad, (ptr.ObjRA, ptr.ObjDec) )
    ax.plot([pt_pt_down[0], pt_cen_up[0]], [pt_pt_down[1], pt_cen_up[1]], color='green', linestyle='dashed', label='Upper/Lower Limit')
    ax.plot([pt_pt_up[0], pt_cen_down[0]], [pt_pt_up[1], pt_cen_down[1]], color='green', linestyle='dashed')
                                    
    
    #plot the pm vectors greatly exagerated
    scale = 1e6
    mas_per_degree = 3.6e6

    # of the object
    ax.arrow(ptr.ObjRA, ptr.ObjDec, ptr.ObjPMRA*scale/mas_per_degree,
             ptr.ObjPMDec*scale/mas_per_degree, color='blue', head_width=1)

    # of the center
    a1 = ax.arrow(ptr.CenRA, ptr.CenDec, ptr.CenPMRA*scale/mas_per_degree,
            ptr.CenPMDec*scale/mas_per_degree, color='blue', head_width=1)

    # of the differential
    a2 = ax.arrow(ptr.ObjRA, ptr.ObjDec, ptr.ObjRelPMRA*scale/mas_per_degree,
            ptr.ObjRelPMDec*scale/mas_per_degree, color='red', head_width=1)

    #pm_lines = [Line2D([0], [0], color='Blue'), Line2D([0],[0], color='red')]
    pm_handles = [a1, a2]
    pm_labels = ['Proper Motion','Differential Motion']
    #add to the legend
    handles, labels = ax.get_legend_handles_labels()
    handles += pm_handles
    labels += pm_labels

    #data table
    add_ptr_table(ax, ptr)

    ax.set_xlim(0,90)
    ax.set_ylim(0,90)
    ax.set_xlabel('Right Ascension (degree)')
    ax.set_ylabel('Declination (degree)')
    ax.set_title(f'Points_to Analysis, GAIA Source ID: {ptr.name}')
    ax.legend(handles, labels, shadow=True)
    ax.grid()
    
# =======

# >>>>>>> master
