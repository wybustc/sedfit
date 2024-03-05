#NOTE The procedure were used to apply the SMC extinction correction 
from astropy.table import Table 
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
import os 

PATH_ext=PATH_to_your_extinction_file 
def SMC_ext(wave=None,flux=None,Av=None, ebv=None, Rv=2.74):
    #NOTE the unit of wave shuould be 'Angstrom' 

    if Av is None: 
        Av=ebv*Rv 

    data=Table.read(PATH_ext,format='ascii') 
    lam,A_lam=data['lam']*1e4,data['smc']*Av

    f1=interp1d(lam,A_lam) 
    A_wave=f1(wave) 

    # plt.plot(lam,A_lam) 
    # plt.plot(wave,A_wave)
    # plt.show() 

    flux=flux*10**(-0.4*A_wave)
    return flux 

# import numpy as np 
# fcorr=SMC_ext(wave=np.linspace(2000,5000,1000),flux=np.ones_like(1000),Av=0.9)
# plt.plot(np.linspace(2000,5000,1000), fcorr) 
# plt.show()
    

