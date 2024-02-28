import numpy as np 
import astropy.units as u 
import astropy.constants as const 
from library import flux2L, Mag2Abs
from sfdmap import ebv as ebv_value 
from extinction import calzetti00,remove,fitzpatrick99
from copy import deepcopy 
import matplotlib.pyplot as plt 

class  magflux():
    def __init__(self,LC,makeup=True,redshift=None, mw_ext=True, mw_ebv=None, ra_dec=None,K_correction=True):
        #NOTE please add the part to do the K-correction 
        assert (mw_ext==False) | (mw_ebv is not None) | (ra_dec is not None), 'must provide mw_ebv or ra_dec when doing MW extinction correction'
        self.LC=deepcopy(LC) 
    #    print(LC)
        self.redshift=redshift 

        self.mw_ext=mw_ext 
        if self.mw_ext: 
            if mw_ebv is not None: self.mw_ebv=mw_ebv 
            else: self.mw_ebv=ebv_value(ra_dec,unit='degree')

        #The library of zero points  for different magnitudes systen and different bands, retrieved frm the SVO 
        # There are two class of values in SVO, one is called 'specified' and the other is 'calculated', what's the difference? 
        #And Which one is better? it seems only for vlaues of WISE, we use the specified value
        self.fzero={'Swift-U': [3520.88, 3.63407e-9, 9.05581e-9, 3.631e-9] , #lam_effective, Vega, AB, ST, zeropoint in unit of erg/s/cm^2
                    'Swift-B': [4345.28, 6.47878e-9, 5.75381e-9, 3.631e-9] ,
                    'Swift-V': [5411.45, 3.72561e-9, 3.69824e-9, 3.631e-9] ,
                    'Swift-UVW1': [2681.67, 4.14563e-9, 1.6344e-8, 3.631e-9] ,
                    'Swift-UVW2': [2083.95, 5.36274e-9, 2.57862e-8,3.631e-9] , 
                    'Swift-UVM2': [2245.03, 4.67904e-9, 2.15706e-8,3.631e-9] , 
                    'ATLAS-o': [6629.82, 1.93068e-9, 2.38902e-9, 3.631e-9] , 
                    'ATLAS-c': [5182.42, 4.00286e-9, 3.89323e-9, 3.631e-9] ,
                    'ZTF-g': [4746.48, 5.2673e-9, 4.75724e-9 ,3.631e-9], 
                    'ZTF-r': [6366.38, 2.23049e-9,2.64344e-9 ,3.631e-9], 
                    'ZTF-i': [7829.03, 1.1885e-9, 1.75867e-9 ,3.631e-9], 
                    'Gaia-G': [5822.39,2.50386e-9,2.81581e-9 ,3.631e-9],  #NOTE Gaia3.G
                    'WISE-W1': [33526,8.1787e-12,9.59502e-11,3.631e-9], #NOTE for lam_effective,Vega is specified value ,and for AB,ST ,it is calculated vlaue 
                    'WISE-W2': [46028,2.415e-12	,5.10454e-11,3.631e-9], #Similar as W1
                    'sdss-u': [3608.04, 3.75079e-9, 8.60588e-9, 3.631e-9], #NOTE f_lam *const.c/lam_eff**2 != 3631 ,infact, f_lam*const.c/lam_ref**2= 3631 
                    'sdss-g': [4671.78, 5.45476e-9, 4.92255e-9, 3.631e-9], #NOTE what lam value is more proper? 
                    'sdss-r': [6141.12, 2.49767e-9, 2.85425e-9, 3.631e-9], 
                    'sdss-i': [7457.89, 1.38589e-9, 1.94038e-9, 3.631e-9],
                    'sdss-z': [8922.78, 8.38585e-10, 1.35994e-9, 3.631e-9],
                    'sdss-rp': [6201.71, 2.41892e-9 , 2.78937e-9 , 3.631e-9],
                    'sdss-ip': [7672.59, 1.2626e-9, 1.82728e-9, 3.631e-9],
                    'wfst-u': [3561.7 , None       , 8.58088890e-09,3.631e-9],
                    'wfst-g': [4763.4 , None       , 4.79747640e-09,3.631e-9], 
                    'wfst-r': [6205.7 , None       , 2.82660679e-09,3.631e-9], 
                    'wfst-i': [7530.7 , None       , 1.91944759e-09,3.631e-9], 
                    'wfst-z': [8704.5 , None       , 1.43667751e-09,3.631e-9], 
                    'wfst-w': [6121.5 , None       , 2.90490039e-09,3.631e-9],
                    'swope-u':[3648.35, 3.84229e-9 , 8.4092e-9	   ,3.631e-9], #LCO/e2v.u 
                    'swope-g':[4716.67, 5.2927e-9  , 4.81058e-9    ,3.631e-9], #LCO/e2v.g 
                    'swope-r':[6180.78, 2.44417e-9 , 2.81004e-9    ,3.631e-9], #LCO/e2v.r
                    'swope-i':[7603.67, 1.30237e-9 , 1.8652e-9     ,3.631e-9], #LCO/CSP2.i, LCO/e2v.i have same value as LCO/e2v.r. There may be an error for that. For other band (u,g,r), LCO/CSP2 have same value with LCO/e2v, hence we use LCO/CSP2.i here. 
                    'KeplerCam-u':[3514.35, 3.46237e-9, 8.97193e-9, 3.631e-9],
                    'KeplerCam-g':[4735.89, 5.25575e-9, 4.7765e-9, 3.631e-9 ],
                    'KeplerCam-r':[6198.96, 2.41644e-9, 2.78319e-9,3.631e-9 ],
                    'KeplerCam-i':[7645.77, 1.27628e-9, 1.8399e-9, 3.631e-9], 
                    'KeplerCam-z':[10023.96, 5.8977e-10, 1.05072e-9, 3.631e-9],
                    'ASASSN-g': [4671.78, 5.45476e-9, 4.92255e-9, 3.631e-9], # NOTE the data use sloan g 
                    'ASASSN-V': [5467.57, 3.5833e-9 , 3.59667e-9, 3.631e-9], # NOTE the data use Generic/Johnson.V
                    }
        if makeup:
            self.magflux()
#lam_eff defined  as  Integrate(lam*filter(lam) *vega_spec(lam) *dlam) / Integrate( filter(lam) *vega_spec(lam) *dlam)
#lam_mean defined as  Integrate(lam*filter(lam)  *dlam) / Integrate( filter(lam)  *dlam)
#lam_pivot or lam_ref defined as  sqrt( Integrate( filter(lam)*dlam ) / Integrate( filter(lam)/lam**2*dlam ) )
# NOTE for the wfst the lam_eff defined as Integrate(lam*filter(lam)  *dlam) / Integrate( filter(lam)  *dlam)
    def magflux(self): 
    #    a=input('##### Press enter to do the Magflux #####')
        ind_zero={'vega':1, 'AB': 2, 'ST':3}
        # print('hello', self.LC.keys())
        for band in self.LC.keys(): 
            lc=self.LC[band] 
            zero=self.fzero[band][ind_zero[lc['msy']]]* u.Unit('erg/s/cm^2/AA') 
            lc['lam'] = self.fzero[band][0] *u.AA #NOTE did all the bands need this ?
            lc['mag_zero']=zero
            if ('mag' in lc.keys() ) & ('flux' in lc.keys()): 
                pass
            elif 'mag' in lc.keys():
                lc['flux']= 10**(-0.4*lc['mag']) *zero
                lc['ferr']= np.log(10)/2.5* lc['flux']* lc['mer'] 
            elif 'flux' in lc.keys(): 
                try: 
                    lc['mag']= -2.5*np.log10( lc['flux']/zero ) 
                    lc['mer']= 2.5/np.log(10)*lc['ferr']/lc['flux'] 
                except: 
                    #NOTE for AB magnitude, we can covert 3631Jy to zero in erg/s/cm^2/AA directly?
                    lc['flux']= (lc['flux']* const.c/lc['lam']**2).to(u.Unit('erg/s/cm^2/AA'))
                    lc['ferr']= (lc['ferr']* const.c/lc['lam']**2).to(u.Unit('erg/s/cm^2/AA'))
                    lc['mag'] = -2.5*np.log10(lc['flux']/zero) 
                    lc['mer'] = 2.5/np.log(10)*lc['ferr']/lc['flux'] 
            if self.redshift is not None: 
                lc['redshift']=self.redshift 
                lc['lamLumi']=flux2L( lc['flux']*lc['lam'],self.redshift)
                lc['lamLerr']=flux2L( lc['ferr']*lc['lam'],self.redshift) 
                lc['absMag'] =Mag2Abs(lc['mag'], self.redshift)
                lc['absMer'] =lc['mer']
                #NOTE we can also add a convertation to absolute magnitude 

            if self.mw_ext:  #NOTE if True, do the Galactic extinction correction. check this part
                mcorr=fitzpatrick99(np.array([lc['lam'].to(u.AA).value]),3.1*self.mw_ebv)[0]
                fcorr=10**(0.4*mcorr)
                for key in ['flux','ferr','lamLumi','lamLerr']: 
                    lc[key]=lc[key]*fcorr 
                for key in ['absMag', 'mag']: 
                    lc[key]=lc[key]-mcorr 
            
            self.LC[band]=lc 

        return self.LC     
        
    def getinfo(self,bands,inf='mag_zero', mag_sys='AB'): 
        if   (inf=='mag_zero') & (mag_sys=='AB')  : ind=2 
        elif (inf=='mag_zero') & (mag_sys=='vega'): ind=1
        elif  inf=='lam_eff'                      : ind=0 
        else: raise Exception('Only support mag_zero, lam_eff') 

        infs=[] 
        for band in bands: 
            infs.append(self.fzero[band][ind]) 
        
        return np.array(infs)

    def plotlc(self, gap_frac={}, fmt='mag',MJD0=0, xscale=None,yscale=None,title=None ): 
        #gap_frac, is a dict which should contains the multiply factor using in the case of fmt=='flux'/'lumi' 
        #                                          or the gap using the case of  fmt=='mag' 
        #MJD0, use as an reference mjd 
        # ax=plt.axes() 
        plt.xlabel('MJD') 
        if fmt=='mag': 
            plt.gca().invert_yaxis() 
            plt.ylabel('Magnitude')
        elif fmt=='flux': 
            plt.ylabel(r'Flux ($erg\,s^{-1}\,cm^{-2}\,{AA}^{-1} $)') 
        elif fmt=='lumi': 
            plt.ylabel(r'Luminoisty ($erg\,s^{-1}$)')

        if xscale is not None: plt.xscale(xscale) 
        if yscale is not None: plt.yscale(yscale) 

        for band in self.LC.keys(): 
            lc =self.LC[band] 
            ind=lc['upperlimits'] 
            if fmt=='mag': 
                gap=gap_frac.get(band, 0)
                pl=plt.errorbar(lc['mjd'][ind==False]-MJD0, lc['mag'][ind==False]+gap,yerr=lc['mer'][ind==False], fmt='o',label=band ) 
                plt.errorbar(lc['mjd'][ind]-MJD0, lc['mag'][ind], fmt='v', color=pl[0].get_color() ) 
            elif fmt=='flux': 
                frac=gap_frac.get(band, 1) 
                pl=plt.errorbar(lc['mjd'][ind==False]-MJD0, lc['flux'][ind==False]*frac, yerr=lc['ferr'][ind==False]*frac,fmt='o', label=band) 
                plt.errorbar(lc['mjd'][ind]-MJD0, lc['flux'][ind]*frac, yerr=lc['ferr'][ind]*frac, fmt='v', color=pl[0].get_color()) 

            elif fmt=='lumi': 
                frac=gap_frac.get(band, 1) 
                pl=plt.errorbar(lc['mjd'][ind==False]-MJD0, lc['lamLumi'][ind==False]*frac, yerr=lc['lamLerr'][ind==False]*frac,fmt='o', label=band) 
                plt.errorbar(lc['mjd'][ind]-MJD0, lc['lamLumi'][ind]*frac, yerr=lc['lamLerr'][ind]*frac, fmt='v', color=pl[0].get_color())
        plt.legend() 
        if title is not None: 
            plt.title(title)
        plt.show() 
            

# wfst   = np.array([3561.7, 4763.4, 6205.7, 7530.7, 8704.5, 6121.5] ) 
# fzero  =3631* 1e-23 # erg/s/cm2/Hz 
# fzero_lam= fzero* const.c.value/ (wfst*1e-10)/ wfst
# print(fzero_lam)  
# fzero_lam= [8.58088890e-09 ,4.79747640e-09 , 2.82660679e-09, 1.91944759e-09 ,1.43667751e-09 ,2.90490039e-09]
