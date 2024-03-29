import numpy as np 
import mpfit 
import astropy.constants as const 
import astropy.units as u 
from astropy.table import Table 
from magflux import magflux 
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d 
from binLC import binlc 
# from sfdmap import ebv as ebv_value
from astropy.coordinates import SkyCoord
# from extinction import calzetti00,apply,remove,fitzpatrick99
from library import flux2L,MonteC 
import os, json 
from SEDmodel import SEDmodel 
#import multiprocessing as multip 
from pathos.multiprocessing import Pool 
from copy import deepcopy 

class sedfit(magflux,SEDmodel):
    '''Different methods for the photometric SED fitting, such as blackbody''' 
    def __init__(self,LC,redshift=None, mw_ebv=None, spec_fit=False, path_savedir=None):
        # super().__init__(LC,redshift=redshift,mw_ext=False) 
        # super(magflux, self).__init__()
        magflux.__init__(self,deepcopy(LC),redshift=redshift,mw_ext=False) 
        SEDmodel.__init__(self)

        self.redshift=redshift
    #    self.pars=[] 
        self.spec_fit=spec_fit 
        self.path_savedir=path_savedir
        if self.path_savedir is not None:
            if not os.path.exists(self.path_savedir): 
                os.mkdir(self.path_savedir) 
                os.mkdir(os.path.join(self.path_savedir, 'epoch_fit'))
            else: 
                if not os.path.exists(os.path.join(self.path_savedir, 'epoch_fit')):
                    os.mkdir(os.path.join(self.path_savedir, 'epoch_fit'))

    @staticmethod
    def processLC(mjd_base, mjd,flux,ferr,band=None, method='interp',kind='linear',gapmax=None,binmax=None,binmin=None,show=True,path_savedir=None ): 
        #NOTE, the input mjd,flux,ferr should be dimensionless value 
        #In this function, we can get the an estimate at mjd_base from the light curve (mjd,flux,ferr)
        #kind is the method used in the interpolate processing 
        #gapmax, the maximum gap allowed for the separation between mjd_base and mjd
        #The bin region should <= binmax*2 if binmin is not None 
        #The bin region should >= binmin if binmin is not None 
        #NOTE maybe we can support that the binmax or binmin are an array 
        if method=='interp':
            #NOTE NOTE 如果某一个波段没有数据，插值会报错
            if gapmax is not None:
                f=interp1d( mjd, mjd, fill_value='extrapolate', kind='nearest')
                ind=abs( mjd_base-f(mjd_base) )<gapmax #NOTE maybe we should define that the mjd_pre-mjd_max <gapmax
                # ind=(mjd_base>mjd_min) & (mjd_base<mjd_max)
                # mjd_pre=interp1d(mjd,mjd, fill_value='', kind='previous')(mjd_base) 
                # mjd_nex=interp1d(mjd,mjd, fill_value='', kind='next'    )(mjd_base) 
                # ind=(mjd_pre-mjd_nex)<gapmax
                mjd_base=mjd_base[ind] 

            flux1d=interp1d(mjd,flux,fill_value='extrapolate',kind=kind) 
            ferr1d=interp1d(mjd,ferr,fill_value='extrapolate',kind='nearest')

            mjd_pre = interp1d(mjd,ferr,fill_value='extrapolate',kind='previous')(mjd_base)
            mjd_nex = interp1d(mjd,ferr,fill_value='extrapolate',kind='next')(mjd_base)
            ferr_pre= interp1d(mjd,ferr,fill_value='extrapolate',kind='previous')(mjd_base)
            ferr_nex= interp1d(mjd,ferr,fill_value='extrapolate',kind='next')(mjd_base) 
            (mjd_base-mjd_pre)
             
            flux_base=flux1d(mjd_base)
            ferr_base=ferr1d(mjd_base)
            if (path_savedir is not None) | show: 
                plt.plot(mjd,flux,color='gray',alpha=0.6)
                plt.errorbar(mjd     , flux     , yerr=ferr               , color='blue',alpha=0.6,fmt='o',fillstyle='none') 
                plt.errorbar(mjd_base, flux_base, yerr=ferr_base , fmt='s', color='k') 
                plt.title(str(band)+' '+ method)
                if path_savedir is not None: plt.savefig(os.path.join(path_savedir, 'proSED', '%s_%s.png'%(band,method)))
                if show: plt.show()
                plt.close()  
            return mjd_base,flux_base,ferr_base
        elif method=='bin': 
            if binmin  is not None: 
                mjd_base_new=np.array([mjd_base[0]]) 
                for mjd_grid in mjd_base: 
                    if mjd_grid>= mjd_base_new[-1]+binmin: mjd_base_new=np.append(mjd_base_new,mjd_grid) 
                mjd_base =mjd_base_new 

            bl=binlc(mjd,flux,ferr,bin1=False,show=show, path_savedir=os.path.join(path_savedir,'proSED', '%s_%s.png'%(band,method))) 
            bl.nodes_low=np.append( mjd_base[0]-(mjd_base[1]-mjd_base[0])/2, (mjd_base[0:-1]+mjd_base[1:])/2    ) 
            bl.nodes_up =np.append((mjd_base[0:-1]+mjd_base[1:])/2, mjd_base[-1]+ (mjd_base[-1]-mjd_base[-2])/2 )
            if binmax is not None: 
                ind=(mjd_base-bl.nodes_low)>binmax
                bl.nodes_low[ind]= mjd_base[ind]-binmax 
                ind=(bl.nodes_up-mjd_base) >binmax 
                bl.nodes_up[ind] = mjd_base[ind]+binmax 
            bl.binning(prod_nodes=False)  #NOTE check the bin procedure to see whether this is true or not?
            ind=bl.bN>0 #NOTE check this 
            mjd_base=mjd_base[ind]; flux_base=bl.bflux[ind]; ferr_base=bl.bferr[ind] #NOTE check this
            return mjd_base,flux_base,ferr_base 
        else: 
            raise Exception('Now only two methods are supported: interp and bin. But input methods is %s'%method)

                 

    def proSED(self,NP_min=2,band_include=['Swift-U','Swift-B','Swift-V','Swift-UVW1','Swift-UVW2','Swift-UVM2'],mjd_base='Swift-U',gapmin=None,gapmax=None, method='interp',kind='linear', binmin=None, binmax=None, show=True,mjd_min=None, mjd_max=None): 
        #bands_include: the bands used to produce the sed 
        #mjd_base: the baseline of mjd used to produce the SED 
        #gapmin: if the two mjds diffrence less than gapmin, we will remain only the first one 
        #method: the method used to produce the sed: interpolate, or bin . Maybe we can also use resampling method? 
        #NP_min, for a sed acquired at one phase, we require the number of bands at least Np_min 
        if band_include.__class__ is str :
            if (band_include =='all') :  band_include=list(self.LC.keys()) 
            else: raise Exception('band_include should be all when it is a string') 

        if mjd_base.__class__ is str:  #mjd_base should be in band_include or 'all' 
            mjd_base={mjd_base:[[-1e10, 1e10]] }    # convert to dict format to avoid repeating 
        if mjd_base.__class__ is dict: #mjd_base should be in format of {"key":[range1, range2]}
            mjd_grid=[] 
            for key in mjd_base.keys(): 
                if '+' in key:
                    mjd_key=[] 
                    for band in key.split('+'): mjd_key=mjd_key+list(self.LC[band]['mjd'])
                    mjd_key=np.array(mjd_key)
                elif key in band_include: 
                    mjd_key=self.LC[key]['mjd']
                elif key =='all':  
                    mjd_key=[]  
                    for band in band_include:  mjd_key=mjd_key+list(self.LC[band]['mjd']) 
                    mjd_key=np.array(mjd_key) 
                else: raise Exception('The key of mjd_base should be in band_include or all')

                mjd_key_new=[] 
                for region in mjd_base[key]: 
                    ind=(mjd_key> region[0]) & (mjd_key<region[1]) 
                    mjd_key_new=mjd_key_new+ list( mjd_key[ind]) 
                mjd_grid=mjd_grid+ mjd_key_new  
            mjd_base= np.sort( np.array( list(set(mjd_grid)) ) ) # remove the same mjd and sort 
        elif (mjd_base.__class__ == np.ndarray) | (mjd_base.__class__== list): 
            pass 
        else: raise Exception('The class of mjd_base is %s but should be np.ndarray,list,str,or dict'%mjd_base.__class__ ) 

        if mjd_min is not None: mjd_base=mjd_base[mjd_base>mjd_min] 
        if mjd_max is not None: mjd_base=mjd_base[mjd_base<mjd_max]
        if len(mjd_base)==0: raise Exception('The length of mjd_base is zero, please check the orignal mjd_base or mjd_min/max')

        #produce the sed 
        self.baseLC={} 
        for band in band_include: 
            mjd_base_band,flux_base_band,ferr_base_band=self.processLC(mjd_base,self.LC[band]['mjd'], self.LC[band]['flux'].to(u.Unit('erg/s/cm^2/AA')).value, 
                                                          self.LC[band]['ferr'].to(u.Unit('erg/s/cm^2/AA')).value, 
                                                          band=band,binmin=binmin,binmax=binmax, gapmax=gapmax, method=method, show=show,kind=kind) 
            self.baseLC[band]={'mjd':mjd_base_band, 'flux': flux_base_band, 'ferr':ferr_base_band,'lam':self.LC[band]['lam'].to(u.AA).value} 
            #NOTE for the baseline LC, all the variables are dimensionless 
        
        self.sed={}
        self.mjd_base=[] 
        for i in  range(len(mjd_base)):
            self.sed[mjd_base[i]]={'lam':[], 'flux':[], 'ferr':[], 'band':[],'band_index':[]} 
            for band in band_include: 
                ind=np.where(self.baseLC[band]['mjd']==mjd_base[i])[0] 
                if len(ind)==0: continue 
                if len(ind)>1 : raise Exception('The length should be one') 
                self.sed[mjd_base[i]]['lam'].append(self.baseLC[band]['lam']) 
                self.sed[mjd_base[i]]['flux'].append(self.baseLC[band]['flux'][ind][0]) 
                self.sed[mjd_base[i]]['ferr'].append(self.baseLC[band]['ferr'][ind][0]) 
                self.sed[mjd_base[i]]['band'].append(band) 
                self.sed[mjd_base[i]]['band_index'].append(band_include.index( band) )
            if len(self.sed[mjd_base[i]]['lam'])<NP_min:  #NOTE, for a sed, there must be at least two bands 
                del self.sed[mjd_base[i]]
            else: 
                self.mjd_base.append(mjd_base[i])
        
        # self.mag_zero=[]
        # for band in band_include: 
        #     self.mag_zero.append( self.LC[band]['mag_zero'].to(u.Unit('erg/s/cm^2/AA')).value )
        # self.mag_zero=np.array(self.mag_zero)
        self.mag_zero=self.getinfo(band_include,inf='mag_zero', mag_sys='AB') #NOTE this mag_Zero were use to convert the filter-passed flux to monochromatic flux 
                                                                       #since we only use AB magnitude in the convertion, here we should use the 'AB' system
                                                                        #The unit here should be 'erg/s/cm^2/AA'
        
        #convert the list to np.array 
        for mjd_key in self.sed.keys():
            for key in ['lam','flux','ferr']: 
                self.sed[mjd_key][key]=np.array(self.sed[mjd_key][key]) 
        
        self.band_include=band_include 
        if show: 
            for mjd_key in self.sed.keys():
                # print(self.sed[mjd_key]['lam'] )
                # print(self.sed[mjd_key]['flux'] )
                # print(self.sed[mjd_key]['ferr'])
                # print(self.sed[mjd_key]['band'])
                plt.errorbar(self.sed[mjd_key]['lam'],self.sed[mjd_key]['flux'],yerr=self.sed[mjd_key]['ferr'],fmt='o')
            plt.show()     

    def color_evl(self,band1,band2,show=True):
        # produce the color evolution of band1, band2 
        mjds=[]; colors=[]; color_errs=[] 
        for mjd in self.sed.keys(): 
            if ( band1 in self.sed[mjd]['band'] ) & (band2 in self.sed[mjd]['band']):
                ind1=self.sed[mjd]['band'].index(band1) 
                ind2=self.sed[mjd]['band'].index(band2) 
                flux1=self.sed[mjd]['flux'][ind1]
                ferr1=self.sed[mjd]['ferr'][ind1] 
                flux2=self.sed[mjd]['flux'][ind2]
                ferr2=self.sed[mjd]['ferr'][ind2] 

                colors.append( -2.5* np.log10( flux1/flux2) )   # both flux are in same unit.  
                color_errs.append( 2.5/np.log(10) *  ( (ferr1/flux1 )**2+ (ferr2/flux2)**2 )**0.5 ) 
                mjds.append(mjd) 
        mjds=np.array(mjds); colors=np.array(colors); color_errs=np.array(color_errs) 

        if show: 
            plt.errorbar(mjds, colors, yerr=color_errs, fmt='o') 
            plt.title('%s-%s'%(band1,band2)) 
            plt.show() 
        return mjds, colors, color_errs 

    def params2physical(self,params, perror,redshift=None,scale=1): 
        # In this function, we will convert the fitting parameters to the physical of model 
        # params, the best-fit paramters
        # perror, the error estimated for the best-fit parameters 
        Npar=0 ; physical={} 
        for model in self.sedmodel: 
            if  'BlackBody' in model : 
                fluxbb,ferrbb,Lumi_bb,Lerr_bb, R, R_err,Temp,Terr =SEDmodel.Blackbody_PhyicalParams(params[Npar],params[Npar+1],
                                                                                          sca_err=perror[Npar],T_err=perror[Npar+1]
                                                                                          ,redshift=redshift,scale=scale)
                # Temp=params[Npar+1]*(1+redshift) if redshift is not None else params[Npar+1] NOTE we have do the redshift in the above function 
                # Terr=perror[Npar+1]*(1+redshift) if redshift is not None else perror[Npar+1] 
                physical[model]={'lumi':[Lumi_bb,Lerr_bb], 'flux':[fluxbb, ferrbb], 'Tbb':[Temp, Terr], 'Rbb':[R, R_err]}
                Npar=Npar+2
            elif 'NonGrayBB' in model: 
                pass 
            elif 'Powerlaw' in model:
                physical[model]={'scale':[params[Npar],perror[Npar]],'index':[params[Npar+1],perror[Npar+1]],'flux':[np.nan,np.nan],'lumi':[np.nan,np.nan]} 
                Npar=Npar+2
            else: 
                raise Exception('The models supported now do not include %s'%model)

        if len(self.sedmodel)==1: 
            physical['total']=physical[model] 
        else: 
            physical['total']={'flux':[0,0],'lumi':[0,0]} #flux and lumi should be given for every model components
            for key in physical.keys(): 
                physical['total']['flux'][0]+=physical[key]['flux'][0] 
                physical['total']['flux'][1] =(physical['total']['flux'][1]**2+ physical[key]['flux'][1]**2)**0.5 
                physical['total']['lumi'][0]+=physical[key]['lumi'][0] 
                physical['total']['lumi'][1] =(physical['total']['lumi'][1]**2+ physical[key]['lumi'][1]**2)**0.5 
        
        if 'HostExt' in self.fitmodel: 
            physical['HostExt']={'ebv':[params[-1], perror[-1]] }#NOTE the ebv should be the last param

        return physical 
 
    def sedfit_nofilter(self,lam,flux,ferr, fit_method='MPFIT',show=False,mjd=None): 
        y_mean=np.mean( flux ) #normalise factor used to avoid digital problem 
        f_mean=np.mean( self.sedfunc(lam, self.p_ini) ) 
        if fit_method=='MPFIT':
            def fitFunc(p, fjac=None, xval=lam, yval=flux/y_mean, errval=ferr/y_mean): 
                # the yval unit here should be erg/s/cm^2/AA/y_mean 
                yf=self.sedfunc(xval, p) 
                yf=yf/f_mean  # To normalise the fitting data 
                return [0, (yval-yf)/errval]

            res=mpfit.mpfit(fitFunc, parinfo=self.pars)
            if not res.errmsg=='': raise Exception('MPFIT return errmsg: %s'%res.errmsg) 
            if res.perror is None: res.perror=np.nan*np.ones_like(res.params) #NOTE NOTE NOTE, why this case ? 
            if (self.path_savedir is not None) | show: #NOTE, maybe we also should plot the sub-components
                yfit_lam=self.sedfunc(lam, res.params) / f_mean *y_mean
            #    print('manually-chi2',sum((flux-yfit_lam)**2/ferr**2)/(len(flux)-2)) 
                xfit=np.arange(min(lam),max(lam), 0.1)  
                # xfit=np.arange(1000, max(lam),0.1) #NOTE for test        
                yfit=self.sedfunc(xfit, res.params) / f_mean *y_mean 
                plt.plot(xfit,yfit)  
                if self.spec_fit: 
                    plt.plot(lam,flux)  
                else:
                    plt.errorbar(lam, yfit_lam  , fmt='go',alpha=0.3)
                    plt.errorbar(lam, flux, ferr, fmt='o') 
                ymin =np.min([np.min(flux-ferr),np.min(yfit)]) 
                ymax =np.max([np.max(flux+ferr),np.max(yfit)]) 
                # ymin =np.min(flux-ferr) 
                # ymax =np.max(flux+ferr)
                plt.ylim(ymin, ymax)
                plt.title(str(mjd)+' '+str(res.fnorm/res.dof))
                if self.path_savedir is not None: 
                    plt.savefig(os.path.join(self.path_savedir, 'epoch_fit', '%0.3f.png'%mjd))
                if show: plt.show() 
                plt.close() 
        #NOTE 1/f_mean*y_mean is not same among each sed fitting, so parsing it is neccessary to get true luminosity and flux 
        return res, self.params2physical(res.params, res.perror,scale=1/f_mean*y_mean, redshift=self.redshift), 1/f_mean*y_mean 

    def sedfit_filter(self,lam,flux,ferr,bands,wave,filter_data,fzero_lam, fit_method='MPFIT', show=False, mjd=None): 
        y_mean=np.mean( flux ) #normalise factor used to avoid digital problem 
        f_mean=np.mean( self.sedfunc(bands,self.p_ini, wave,filter_data, fzero_lam) ) 
        if fit_method=='MPFIT': 
            def fitFunc(p, fjac=None, xval=bands, yval=flux/y_mean, errval=ferr/y_mean): 
                # the yval unit here should be erg/s/cm^2/AA/y_mean 
                yf=self.sedfunc(xval,p, wave,filter_data, fzero_lam)
                yf=yf/f_mean  # To normalise the fitting data 
                return [0, (yval-yf)/errval]
            res=mpfit.mpfit(fitFunc, parinfo=self.pars)
            if not res.errmsg=='': raise Exception('MPFIT return errmsg: %s'%res.errmsg) 
            if res.perror is None: res.perror=np.nan*np.ones_like(res.params) #NOTE NOTE NOTE, why this case ? 

            if (self.path_savedir is not None) | show:      
                yfit=self.sedfunc(bands,res.params, wave,filter_data, fzero_lam) / f_mean *y_mean 
                ysed=self.sedfunc_beforeFilter(wave, res.params)  / f_mean * y_mean
                ax=plt.axes() 
                ind=np.argsort(lam)
                plt.plot(lam[ind],yfit[ind]) 
                plt.plot(wave, ysed,label='sed before filter') 
                plt.errorbar(lam, yfit,       fmt='s',label='best-fit')  
                plt.errorbar(lam, flux, ferr, fmt='o',label='data')
                plt.title(str(mjd)+' '+str(res.fnorm/res.dof))
                ymin=np.min([flux-ferr, yfit-ferr]) 
                ymax=np.max([flux+ferr, yfit+ferr])
                plt.ylim(ymin, ymax)
                ax.legend() 
                ax=ax.twinx() 
                for frac,band in zip(filter_data,bands): 
                    ax.plot(wave,frac,label=band,alpha=0.4) 
                ax.legend() 
                if self.path_savedir is not None: 
                    plt.savefig(os.path.join(self.path_savedir, 'epoch_fit', '%0.3f.png'%mjd))
                if show:  plt.show() 
                plt.close() 
        #NOTE 1/f_mean*y_mean is not same among each sed fitting, so parsing it is neccessary to get true luminosity and flux 
        return res, self.params2physical(res.params, res.perror,scale=1/f_mean*y_mean, redshift=self.redshift), 1/f_mean*y_mean 
    
    def process_MCres(self,result_list,show=True,show_fit_MC=True,lam=None,flux=None,ferr=None ): 
        params_MC=[];  physical_MC={} 
        for model in result_list[0][1].keys(): 
            physical_MC[model]={} 
            for key in result_list[0][1][model].keys(): 
                physical_MC[model][key]=[] 
        if show_fit_MC: xval=np.arange(min(lam), max(lam), 0.1) 
        for res in result_list: 
            params_MC.append(res.params) 
            for model in res[1].keys():
                for key in res[1][model].keys(): 
                    physical_MC[model][key].append(res[1][model][key]) 
            if show_fit_MC: 
                yf=self.sedfunc(xval, res.params)*res[2] 
                plt.plot(xval,yf,color='green',alpha=0.3) 
        if show_fit_MC: 
            plt.errorbar(lam, flux, yerr=ferr, fmt='o') 
            plt.show() 
        
        params_MC=np.array(params_MC) 
        params=[] ;perror=[] 
        for i in range(len(params_MC[0])): 
            params.append( np.mean(params_MC[:,i]) ) 
            perror.append( np.std(params_MC[:,i],ddof=1) )
        
        physical={} 
        for model in physical_MC.keys(): 
            physical[model]={} 
            for key in physical_MC[model].keys(): 
                physical_MC[model][key]=np.array(physical_MC[model][key])
                physical[model][key]   =[np.mean(physical_MC[model][key][:,0] ), np.std(physical_MC[model][key][:,0],ddof=1)]
                if show: 
                    plt.hist(physical_MC[model][key][:,0],bins=20, density=True) 
                    plt.show() 
        return params, perror,physical, params_MC, physical_MC 


        
    def fitting(self,MC=False,MC_number=1000,MC_pool_number=6, fit_method='MPFIT',show_evl=True, show_fit=False): 
        #MC,wheter to do MC or not? 
        #MC_number, the MC_number 
        #fit_method: ['MPFIT', 'Curve-fit','calculate'] 
        self.fitParams=[]
        self.chi2= [] 
        self.mjd_sed  =[] 
        self.normalise_scale=[] 
        self.Physical ={'total':{'flux':[],'lumi':[] }}
        
        for model in self.sedmodel: 
            if 'BlackBody' in model: self.Physical[model]={'lumi':[], 'Rbb':[], 'Tbb':[], 'flux':[]}
            if 'Powerlaw'  in model: self.Physical[model]={'scale':[], 'index':[]}
        if 'HostExt' in self.fitmodel: self.Physical['HostExt']={'ebv':[]}
        if self.consider_filter: 
            self.wave_filter,self.filter_data=self.load_filter(band_include=self.band_include)
        for mjd in self.sed.keys(): 
            self.mjd_sed.append(mjd) 
            lam,flux,ferr=self.sed[mjd]['lam'],self.sed[mjd]['flux'],self.sed[mjd]['ferr'] 
            if not self.consider_filter: 
                if not MC: 
                    res, physical, scale= self.sedfit_nofilter(lam,flux,ferr, show=show_fit,mjd=mjd)
                    self.fitParams.append(res.params) 
                    self.chi2.append(res.fnorm/res.dof) 
                    self.normalise_scale.append(scale) 
                    for model in self.Physical.keys(): 
                        for key in self.Physical[model].keys():
                            self.Physical[model][key].append(physical[model][key]) 
                else: 
                    pool=Pool(processes=MC_pool_number)
                    iterable= [] 
                    for i in range(MC_number):
                        flux0 = MonteC(flux,ferr)  
                        iterable.append([lam,flux0,ferr])
                    result_list=pool.starmap_async(func=self.sedfit_nofilter, iterable=iterable  ).get()
                    params, perror, physical,params_MC, physical_MC=self.process_MCres(result_list) # process the MC results to the normal format 
                    for model in self.Physical.keys(): 
                        for key in self.Physical[model].keys():
                            self.Physical[model][key].append(physical_MC[model][key]) 
            else: 
                bands=self.sed[mjd]['band']
                fzero_lam=self.mag_zero[ self.sed[mjd]['band_index'] ]
                filter_data=self.filter_data[ self.sed[mjd]['band_index'] ]
                # filter_data=self.load_filter(band_include=self.band_include[np.array(self.sed[mjd][band_index])])
                if not MC: 
                    res, physical, scale= self.sedfit_filter(lam,flux,ferr, bands,self.wave_filter, filter_data, fzero_lam, show=show_fit,mjd=mjd)
                    self.fitParams.append(res.params) 
                    self.chi2.append(res.fnorm/res.dof) 
                    self.normalise_scale.append(scale)
                    for model in self.Physical.keys(): 
                        for key in self.Physical[model].keys():
                            self.Physical[model][key].append(physical[model][key]) 

      #  a=input(self.mjd_sed[0].__class__)
        self.mjd_sed=np.array(self.mjd_sed) 
        self.normalise_scale=np.array(self.normalise_scale)
        self.chi2=np.array(self.chi2)
        for model in self.Physical.keys():
            for key in self.Physical[model].keys(): 
                self.Physical[model][key]=np.array(self.Physical[model][key]) 

        if (self.path_savedir is not None) | show_evl:  
            if 'BlackBody' in self.fitmodel: 
                for key in ['flux', 'lumi', 'Tbb','Rbb']:
                    ymin=[]; ymax=[] 
                    for model in self.Physical.keys(): 
                        if (not 'BlackBody' in model) : continue   
                        plt.errorbar(self.mjd_sed, self.Physical[model][key][:,0],yerr=self.Physical[model][key][:,1],label='%s-%s'%(model,key),fmt='o')
                        ymin.append( np.min(self.Physical[model][key][:,0])- np.std( self.Physical[model][key][:,0]))
                        ymax.append( np.max(self.Physical[model][key][:,0])+ np.std( self.Physical[model][key][:,0]))
                    #   plt.errorbar(self.mjd_sed, self.Physical[model][key][:,1],yerr=self.Physical[model][key][:,1],label='%s-%sErr'%(model,key),fmt='o')
                    plt.ylim(np.min(ymin), np.max(ymax));plt.legend()
                    if self.path_savedir is not None: plt.savefig(os.path.join(self.path_savedir, 'BlackB_%s_evolution.png'%key))
                    if show_evl: plt.show() 
                    plt.close() 
            if 'Powerlaw' in self.fitmodel: 
                for key in ['scale','index']:
                    ymin=[]; ymax=[] 
                    for model in self.Physical.keys():
                        if (not 'Powerlaw' in model) :continue 
                        plt.errorbar(self.mjd_sed, self.Physical[model][key][:,0],yerr=self.Physical[model][key][:,1],label='%s-%s'%(model,key),fmt='o')
                        ymin.append( np.min(self.Physical[model][key][:,0])- np.std( self.Physical[model][key][:,0]))
                        ymax.append( np.max(self.Physical[model][key][:,0])+ np.std( self.Physical[model][key][:,0]))
                    plt.ylim(np.min(ymin), np.max(ymax));plt.legend()
                    if self.path_savedir is not None: plt.savefig(os.path.join(self.path_savedir, 'Powlaw_%s_evolution.png'%key))
                    if show_evl: plt.show()
                    plt.close() 
            if 'HostExt' in self.fitmodel:
                plt.errorbar(self.mjd_sed, self.Physical['HostExt']['ebv'][:,0], yerr=self.Physical['HostExt']['ebv'][:,1],label='HostExt-ebv',fmt='o' )
                plt.title('HostExt-Ebv');plt.xlabel('MJD') ;plt.ylabel('EBV')
                plt.ylim(np.min(self.Physical['HostExt']['ebv'][:,0])-np.std(self.Physical['HostExt']['ebv'][:,0]), 
                         np.max(self.Physical['HostExt']['ebv'][:,0])+np.std(self.Physical['HostExt']['ebv'][:,0]))
                if self.path_savedir is not None: plt.savefig(os.path.join(self.path_savedir, 'HostEBV_evolution.png')) #NOTE EBV need to be a fixed value 
                if show_evl: plt.show() 
                plt.close()
        if self.path_savedir is not None: 
            self.save_results() 


    def save_results(self, fmt='json'): 
        if fmt=='json':
            def convert_numpy_to_list(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_to_list(value) for key, value in obj.items()}
                else:
                    return obj
            sed= convert_numpy_to_list(deepcopy(self.sed)) 
            with open(os.path.join(self.path_savedir,'sed.json'),'w') as fp: 
                json.dump(sed, fp) 
            param=deepcopy(self.Physical) 
            param['chi2']=self.chi2 
            param['nomalise_scale']=self.normalise_scale
            param['mjd_sed']=self.mjd_sed 
            param= convert_numpy_to_list(param) 
            with open(os.path.join(self.path_savedir,'res.txt'),'w') as fp: 
                json.dump(param, fp) 
        elif fmt=='table': 
            pass   









    # def proSED(self,bands_include=['Swift-U','Swift-B','Swift-V','Swift-UVW1','Swift-UVW2','Swift-UVM2'],band_base='Swift-U',show=True):  
    #     #covert the LC from format of {'band':{'mjd':[], 'flux':[], 'ferr':[] }} 
    #     #              to   format of {'mjd': {'lam':[], 'flux':[], 'ferr':[], 'bands':[]}} 
    #     mjd=self.LC[band_base]['mjd']
    #     sed={} 
    #     for i in range(len(mjd)):
    #         sed[mjd[i]]={'lam':[],'flux':[], 'ferr':[], 'bands':[]} 
    #         for band in bands_include: 
    #             lc=self.LC[band] 
    #             if band==band_base: 
    #                 lam,flux,ferr=lc['lam'][i], lc['flux'][i], lc['ferr'][i] 
    #             else: 
    #                 if i==0:
    #                     ind=( lc['mjd']>(mjd[i]- (mjd[i+1]-mjd[i])/2) ) & (lc['mjd']< (mjd[i+1]+mjd[i])/2) 
    #                 elif i==len(mjd)-1:
    #                     ind=( lc['mjd']>(mjd[i]+mjd[i-1])/2) & (lc['mjd']< (mjd[i]+(mjd[i]-mjd[i-1])/2) ) 
    #                 else: 
    #                     ind=( lc['mjd']>(mjd[i]+mjd[i-1])/2) & (lc['mjd']< (mjd[i+1]+mjd[i])/2) 
    #                 flux,ferr=self.binN(lc['flux'][ind], lc['ferr'][ind]) 
    #             #    print(flux,ferr) 
    #                 lam =lc['lam'][0] 
    #             for key,data in zip(['lam','flux','ferr','bands'],[lam,flux,ferr,band]): #NOTE NOTE why here need a index zero?
    #                 sed[mjd[i]][key].append(data) 
    #         for key in ['lam','flux','ferr']:
    #      #       print(sed[mjd[i]][key])
    #             datas=[]
    #             for data in sed[mjd[i]][key]:
    #                 datas.append(data.value)
    #             sed[mjd[i]][key]=np.array(datas) #NOTE change this  
    #             print(datas)

    #     self.sed=sed 
    #     if show: 
    #         #NOTE NOTE we also should plot the SED together
    #         for band in bands_include:
    #             lc=self.LC[band]  
    #             plt.errorbar(lc['mjd'],lc['flux'],yerr=lc['ferr'],fmt='o',alpha=0.3) 

    #         ylim=plt.ylim() 
    #         color=['g','pink']
    #         x=np.arange(mjd[0],mjd[-1], 0.01)
    #         for i in range( len(mjd) ): 
    #             if i==0:ind=( x> (mjd[i]-(mjd[i+1]-mjd[i])/2) ) & (x< (mjd[i+1]+mjd[i])/2) 
    #             elif i==len(mjd)-1: ind=( x >(mjd[i]+mjd[i-1])/2) & (x< (mjd[i]+(mjd[i]-mjd[i-1])/2) ) 
    #             else:   ind=( x> (mjd[i]+mjd[i-1])/2          ) & (x< (mjd[i+1]+mjd[i])/2) 
    #             plt.fill_between(x,ylim[0],ylim[1],where=ind,facecolor=color[int(math.fmod(i,2))])
    #             plt.errorbar(np.ones_like(sed[mjd[i]]['flux'])*mjd[i], sed[mjd[i]]['flux'],yerr=sed[mjd[i]]['ferr'],fmt='s')
    #         plt.show() 
#     def __init__(self,wave=None, freq=None, flux=None, ferr=None, redshift=None, mw_ebv=None):
#         assert (wave is not None ) & (freq is not None), 'wave or freq must be given' 
        
#         if freq is not None: self.freq=(const.c/wave).to(u.Hz)  #
#         else:                self.freq=freq.to(u.Hz) 

#         try:
#             self.flux=flux.to(u.Unit('erg/s/cm^2/Hz')) 
#             self.ferr=ferr.to(u.Unit('erg/s/cm^2/Hz')) 
#         except: 
#             self.flux=(flux*const.c/self.freq**2).to(u.Unit('erg/s/cm^2/Hz')) 
#             self.ferr=(ferr*const.c/self.freq**2).to(u.Unit('erg/s/cm^2/Hz')) 
        
#     def blackbody(self):
#         def residual(p, fjac=None, xval=self.freq, yval=self.flux, errval=self.ferr): 
#             yf=p[0]*2*xval**3*h/c**2*(num.e**(h*xval/k/p[1])-1)**(-1)
#             yf_mean=num.mean(yf);yf=yf/yf_mean
#             return [0,(yval - yf)/errval] 
#         self.func=residual
#         self.pars=[{'value':1,   'limited':[1,0],'limits':[0,0],'parname':'scale'},\
#                    {'value':1000,'limited':[1,0],'limits':[0,0],'parname':'Temperature'}]
#         def BB():
#             pass 


        
        

# def fit_blackbody(f1,f1_err,f2,f2_err,v1=299792458.0/4722.74e-10*u.Hz,v2=299792458.0/6339.61e-10*u.Hz,show=True):
# 	print('\033[1;31m NOTE: when you input redshift corrected flux , you also need to input redshift-corrected frequecy v1 and v2 because the default is in observed frame\033[0m')
# 	f1=f1.to('erg/s/cm^2/Hz').value;f1_err=f1_err.to('erg/s/cm^2/Hz').value;f2=f2.to('erg/s/cm^2/Hz').value;f2_err=f2_err.to('erg/s/cm^2/Hz').value
# 	v1=v1.to('Hz').value;v2=v2.to('Hz').value
# 	h=const.h.to('J*s').value;k=const.k_B.to('J/K').value;c=const.c.to('km/s').value	
# 	x=num.array([v1,v2]);y=num.array([f1,f2]);yerr=num.array([f1_err,f2_err])
# 	mean=num.mean(y);y=y/mean;yerr=yerr/mean
# #	print(f1,f2)
# 	def residuals0(p,fjac=None, xval=x, yval=y, errval=yerr): #The effect of the parameter 'fjac'
# 		yf=p[0]*2*xval**3*h/c**2*(num.e**(h*xval/k/p[1])-1)**(-1) # *(c/xval)**-1.7
# 		yf_mean=num.mean(yf);yf=yf/yf_mean
	
# 		return [0,(yval - yf)/errval]
# 	par=[{'value':1,   'limited':[1,0],'limits':[0,0],'parname':'scale'},\
# 		 {'value':1000,'limited':[1,0],'limits':[0,0],'parname':'Temperature'}]
	
# 	res=mpfit.mpfit(residuals0, parinfo=par,quiet=True)
# 	p=res.params
# 	#Bv1=2*v1**3*h/c**2*(num.e**(h*v1/k/T)-1)**(-1)
# 	#Bv2=2*v2**3*h/c**2*(num.e**(h*v2/k/T)-1)**(-1)
# 	yf=p[0]*2*x**3*h/c**2*(num.e**(h*x/k/p[1])-1)**(-1) # * (c/x)**-1.7
# 	yf_mean=num.mean(yf)

# 	res.params[0]=res.params[0]*mean/yf_mean
# 	res.perror[0]=res.perror[0]*mean/yf_mean

# #	p=res.params
# #	yf=p[0]*2*x**3*h/c**2*(num.e**(h*x/k/p[1])-1)**(-1)
# 	if show:
# 		plt.errorbar(x,y*mean,yerr=yerr*mean,fmt='o')
# 		plt.plot(x,yf/yf_mean*mean)
# 		plt.show()

# 	fluxbb=res.params[0]*2*(k*res.params[1])**4*(num.pi)**4/15/h**3/c**2*u.Unit('erg/s/cm^2')
# #	ferrbb=( (res.perror[0]/res.params[0])**2+ (4*res.perror[1]/res.params[1])**2 )**0.5*fluxbb #NOTE when you use this ,check it 
# #	print('scale_err:',res.perror[0]/res.params[0])
# #	
# #	bb=models.BlackBody(temperature=res.params[1]*u.K,scale=1)
# #	print(fluxbb,bb.bolometric_flux)
# 	return x,yf,res,fluxbb#,ferrbb