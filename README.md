# sedfit
flexible multi-band sed fitting

# Installation 
Requirements: Python3,and some neccessary packages including numpy, astropy, scipy, extinction, sfdmap, matplotlib, all installable with pip 
```
git clone https://github.com/wybustc/sedfit
```

# How to run 
You can run the code as follow
```
first you should transform your multiband LCs to the format of:
LC={'band': 'mjd':{}, 'mag':{}, 'mer':{}, 'flux':{}, 'ferr':{}, 'msy':'AB'}
band name should be in the format of the filter name in ./filters
mag(mer)/flux(ferr), one of them is okay
msy, the system of magnitude, 'AB' or 'vega'  
```
```
then you can fit multibands LCs as follow
from SEDfit import sedfit
lcfit=sedfit(LC)
lcfit.proSED(band_include='all',mjd_base='Swift-U',show=False) #producing the sed at every epoch
lcfit.add_blackbodySED() #the model to describe intrinsic SED, such blackbody, powerlaw, see SEDmodel.py
lcfit.apply_HostExt()#Optional, considering the host galaxy extinction
lcfit.apply_GalExt(ebv=mw_ebv)#If the data have been done for galactic extinction correction, skip this
lcfit.apply_filter()#Optional, if you would like to consider the transimission curve of filter, add this
lcfit.fitting(MC=False,show_fit=True,show_evl=True) #show_fit=True, show the best fit for every epoch, show_evl=True, show the evolution behavior of every key parameter of the SED model.
```

