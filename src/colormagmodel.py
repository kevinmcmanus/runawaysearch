import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde

class cm_model():
    def __init__(self, name, description=None):
        self.name = name
        self.descrition = description
        self.kde = None
        
    def fit(self,colors, mags):
        valid_values = np.logical_and(np.isfinite(colors), np.isfinite(mags))
        BP_RP = np.array(colors[valid_values])
        M_G = np.array(mags[valid_values])
        self.kde = kde.gaussian_kde([BP_RP, M_G])
        kde_val = self.kde([BP_RP, M_G])
        
        kde_val_i = np.argsort(kde_val)
        
        self.BP_RP = BP_RP[kde_val_i]
        self.M_G = M_G[kde_val_i]
        self.kde_val = kde_val[kde_val_i]
        
    def predict(self, colors, mags, threshold=0.9):
        nobs = self.kde.n
        nobs_in = int(nobs*threshold)
        #min score that includes nobs_in observations
        thresh = np.flip(self.kde_val)[nobs_in]
        
        scores = self.kde([colors, mags])
        scores = np.where(np.isfinite(scores),scores,0.0)
        
        return scores >= thresh
    
    def plot(self, **kwargs):
        ax = kwargs.pop('ax',None)
        title = kwargs.pop('title', self.name)
        nbins = kwargs.pop('nbins',300)
        if ax is None:
            yax = plt.subplot(111)
        else:
            yax = ax

        xi, yi = np.mgrid[self.BP_RP.min():self.BP_RP.max():nbins*1j, self.M_G.min():self.M_G.max():nbins*1j]
        zi = self.kde(np.vstack([xi.flatten(), yi.flatten()]))
        
        pcm = yax.pcolormesh(xi, yi, zi.reshape(xi.shape))
        
        if not yax.yaxis_inverted():
            yax.invert_yaxis()
        yax.set_title(title)
        yax.set_ylabel(r'$M_G$',fontsize=14)
        yax.set_xlabel(r'$G_{BP}\ -\ G_{RP}$', fontsize=14)
            
        return pcm
