import astropy.constants as const 
import astropy.units     as u 
from sfdmap import ebv as ebv_value
from astropy.coordinates import SkyCoord
from extinction import calzetti00,apply,remove,fitzpatrick99
from library import flux2L,L2flux 
from smc import SMC_ext 
import numpy as np 
import os 
from astropy.table import Table 
from scipy.interpolate import interp1d 
from scipy.integrate import trapz

class SEDmodel():
    def __init__(self,redshift=None): 
        def sedfunc(lam,p): 
            return 0 
        self.sedfunc=sedfunc 
        self.fitmodel='' 
        self.sedmodel=[]
        self.pars =[] 
        self.consider_filter=False #In default, 
        self.p_ini=[]
        self.redshift=redshift

    def add_blackbodySED(self, T_limits=[0,0], T_limited=[1,0], T_ini=10000, T_fixed=False): 
        #NOTE add ini_parameter for other parameter, e.g.,scale, please change this in the follow sentence self.p_ini=.....
        N=len(self.pars) 
        sedfunc=self.sedfunc 
        def modify_sedfunc(lam,p):
            lam0=1e-10*lam #convert the AA to m . Since the lam would be used in another function, we let the converted value to be lam0
            h=6.62607015e-34; c=299792458.0; k=1.380649e-23 #units in  SI
            yf= p[N]* 2*h*c**2/lam0**5 * 1/( np.e**(h*c/lam0/k/p[N+1]) - 1) 
            return sedfunc(lam,p)+yf #here we cannot use the self.sedfunc directly, otherwise the self.sedfunc will be identified as the one when we do the MPFIT,instead when we do the add_blacbody 

        pars=[{'value':1,   'limited':[1,0]    ,'limits':[0,0]      ,'parname':'scale'},\
              {'value':T_ini,  'limited':T_limited,'limits':T_limits,'parname':'Temperature'}]
        if T_fixed: pars[1]['fixed']=1
        self.pars=self.pars+pars  #NOTE if we want to add multiple BB, do we need to consider that some BB due to the initial parameters maybe have too small values and so that the mpfit couldn't give proper result 
        self.p_ini=self.p_ini+[1,T_ini]
        self.fitmodel=self.fitmodel+'BlackBody' 
        self.sedmodel.append('BlackBody%d'%(len(self.sedmodel)+1) ) 
        self.sedfunc=modify_sedfunc 
        
    def add_absorbedBB(self, T_limits=[0,0], T_limited=[1,0], T_ini=10000, T_fixed=False, alpha=1, lam_cut=3000): 
        #NOTE add ini_parameter for other parameter, e.g.,scale, please change this in the follow sentence self.p_ini=.....
        N=len(self.pars) 
        sedfunc=self.sedfunc 
        def modify_sedfunc(lam,p):
            lam0=1e-10*lam #convert the AA to m . Since the lam would be used in another function, we let the converted value to be lam0
            h=6.62607015e-34; c=299792458.0; k=1.380649e-23 #units in  SI
            yf= p[N]* 2*h*c**2/lam0**5 * 1/( np.e**(h*c/lam0/k/p[N+1]) - 1) 
            yf[lam<lam_cut]=yf[lam<lam_cut]*(lam[lam<lam_cut]/lam_cut)**alpha
            return sedfunc(lam,p)+yf #here we cannot use the self.sedfunc directly, otherwise the self.sedfunc will be identified as the one when we do the MPFIT,instead when we do the add_blacbody 

        pars=[{'value':1,   'limited':[1,0]    ,'limits':[0,0]      ,'parname':'scale'},\
              {'value':T_ini,  'limited':T_limited,'limits':T_limits,'parname':'Temperature'}]
        if T_fixed: pars[1]['fixed']=1
        self.pars=self.pars+pars  #NOTE if we want to add multiple BB, do we need to consider that some BB due to the initial parameters maybe have too small values and so that the mpfit couldn't give proper result 
        self.p_ini=self.p_ini+[1,T_ini]
        self.fitmodel=self.fitmodel+'BlackBody' 
        self.sedmodel.append('BlackBody%d'%(len(self.sedmodel)+1) ) 
        self.sedfunc=modify_sedfunc 

    def add_PowerlawSED(self,index_limits=[0,0], index_limited=[1,0], index_ini=2): 
        N=len(self.pars) 
        sedfunc=self.sedfunc

        def modify_sedfunc(lam,p): 
            return sedfunc(lam,p)+p[N]*(lam/1000)**-p[N+1]*1e15  #NOTE for the blackbody component, when scale=1, T=10000, the value of sedfunc is about 1e13-1e14, however for the powerlaw, if scale=1, index=2, the value is about 1e-2-1e-1, so we add a scale factor 1e15 
        pars=[{'value':1        ,'limited':[1,0]        , 'limits':[0,0]       , 'parname':'scale'},
              {'value':index_ini,'limited':index_limited, 'limits':index_limits, 'parname':'index'}] 
        self.pars =self.pars+pars 
        self.p_ini=self.p_ini+[1,index_ini]
        self.sedfunc=modify_sedfunc
        self.fitmodel=self.fitmodel+'Powerlaw'
        self.sedmodel.append( 'Powerlaw%d'%(len(self.sedmodel)+1) )
        
    def add_nonGreyBBSED(self, T_limits=[0,0], T_limited=[1,0], T_ini=1000): 
        N=len(self.pars) 
        sedfunc=self.sedfunc 
        def modify_sedfunc(lam,p):
            lam0=lam*1e-10
            h=6.62607015e-34; c=299792458.0; k=1.380649e-23
            yf= p[N]* 2*h*c**2/lam0**5 * 1/( np.e**(h*c/lam0/k/p[N+1]) - 1) #NOTE Maybe we can change to the SI units 
            return sedfunc(lam,p)+yf* lam**-1.7

        pars=[{'value':1    , 'limited':[1,0]    ,'limits':[0,0]   ,'parname':'scale'},\
              {'value':T_ini, 'limited':T_limited,'limits':T_limits,'parname':'Temperature'}]
        self.sedfunc=modify_sedfunc
        self.pars=self.pars+pars 
        self.fitmodel=self.fitmodel+'NonGrayBB' 
        self.sedmodel.append('NonGreyBB%d'%(len(self.sedmodel)+1) )
        self.p_ini=self.p_ini+[1,T_ini]

    def add_starlight(self, wave, starlight,scale_limits=[0,0], scale_limited=[1,0],scale_ini=1): 
        starlight=starlight/np.median(starlight)*1e14 #The template of the starlight
        N=len(self.pars) 
        sedfunc=self.sedfunc 
        def modify_sedfunc(lam,p): 
            stars=interp1d(wave,starlight)(lam)  
            return sedfunc(lam,p)+stars*p[N] 
        pars=[{'value':scale_ini    , 'limited':scale_limited    ,'limits':scale_limits   ,'parname':'scale'} ]
        self.sedfunc=modify_sedfunc
        self.pars=self.pars+pars 
        self.fitmodel=self.fitmodel+'Starlight' 
        self.p_ini=self.p_ini+[scale_ini]

            
    def apply_HostExt(self, ebv_ini=0.0,ebv_limited=[1,0],ebv_limits=[0,0], ebv_fixed=False,  ext_curve='smc'): 
        #NOTE when we do this we should do the exintion at restframe 
        assert self.redshift is not None, 'redshift is neccessary to apply host extinction'
        N=len(self.pars) 
        sedfunc=self.sedfunc 
        if ext_curve=='smc':
            def modify_sedfunc(lam,p): 
                fcorr=SMC_ext(wave=lam/(1+self.redshift), flux=np.ones_like(lam), ebv=p[N]) #NOTE here (1+self.redshift)
                return sedfunc(lam,p)*fcorr 
        elif ext_curve=='mw':
            def modify_sedfunc(lam,p): 
                fcorr=apply(fitzpatrick99(np.array(lam/(1+self.redshift),dtype='float64'),3.1*p[N]), np.ones_like(lam)) #NOTE here (1+self.redshift)
                return sedfunc(lam,p)*fcorr 
        else: 
            raise Exception('Now only extinction curve from smc and mw were supported ')
        pars=[{'value':ebv_ini   , 'limited':ebv_limited    ,'limits':ebv_limits   ,'parname':'HostEbV'} ] 
        if ebv_fixed: pars[0]['fixed']=1
        self.pars=self.pars+pars 
        self.sedfunc=modify_sedfunc 
        self.fitmodel='[%s]*HostExt'%self.fitmodel 
        self.p_ini=self.p_ini+[0.0]

    def apply_GalExt(self, ebv=None, ra_dec=None): 
        assert (ebv is not None) | (ra_dec is not None),'must provide ebv or the coordinates of the sources to do the galactic extinction'  
        if ebv is None:
            ebv=ebv_value(ra_dec,unit='degree')
        sedfunc=self.sedfunc
        def modify_sedfunc(lam, p):
            fcorr=apply(fitzpatrick99(np.array(lam,dtype='float64'),3.1*ebv), np.ones_like(lam)) #NOTE the lam unit?
            return sedfunc(lam,p)*fcorr 
        self.sedfunc=modify_sedfunc 
        self.fitmodel  ='[%s]*GalExt'%self.fitmodel

    def load_filter(self, band_include=[] ): 
        #NOTE do we should consider the vega magnitude? 
        #NOTE or we can covert it to the AB magnitude directly?
        path_filter=r"D:\sample\filters"
        wave_min=99999
        wave_max=0 
        for band in band_include: 
            if not os.path.join(path_filter, '%s.dat'%band) : 
                raise Exception('Filter transmission not found \n'%band) 
            data=Table.read(os.path.join(path_filter, '%s.dat'%band),format='ascii') 
            # except: data=Table.read(os.path.join(path_filter, '%s.txt'%band),format='ascii') 
            wave_min=min(min(data['col1']),wave_min) 
            wave_max=max(max(data['col1']),wave_max) 

        wave=np.arange(wave_min,wave_max, 1) #1AA-separated grid  
        filter_data=[] 
        for band in band_include: 
            data=Table.read(os.path.join(path_filter, '%s.dat'%band),format='ascii') 
            #except: data=Table.read(os.path.join(path_filter, '%s.txt'%band),format='ascii')
            frac=interp1d(data['col1'],data['col2'], fill_value=0, bounds_error=False, kind='linear') 
            filter_data.append( frac(wave) )
        
        return wave, np.array( filter_data)

    def apply_filter(self): 
        #apply filter transimission on the sed 
        sedfunc=self.sedfunc
        self.sedfunc_beforeFilter=self.sedfunc
        def modify_sedfunc(bands,p, wave,filter_data, fzero_lam): 
            zero =3631*1e-23* const.c.value/(wave*1e-10)/wave #The unit is erg/s/cm^2/AA, the magnitude must be AB system ?
            h_cmgs=const.h.cgs.value ; c_cmgs=const.c.cgs.value
            zero =zero / (h_cmgs *c_cmgs / wave/1e-8) # The unit is photon/s/cm^2/AA
            fzero=trapz(zero*filter_data, x=wave)
            flam =sedfunc(wave,p) 
            flam =flam / (h_cmgs *c_cmgs/ wave/1e-8) 
            fband=trapz(flam*filter_data, x=wave) /fzero * fzero_lam

            # print('hello', fzero)
            # print('hello-1', fzero_lam)
            return fband    
        self.sedfunc=modify_sedfunc 
        self.fitmodel  ='[%s]*Filter'%self.fitmodel
        self.consider_filter=True

    @staticmethod
    def Blackbody_PhyicalParams(sca, T, sca_err=0, T_err=0,unit=1e10*u.Unit('erg/s/cm^2'),redshift=None,scale=1):
        #BB= sca* 2*hc/lam^5 * 1/(e^{hc/lam/k/T}-1) 
        #Input: A, the scale of the blackbody
        #       T,the Temperature of the blackbody
        #       unit, defualt 1e10*u.Unit('erg/s/cm^2'), The original flux unit is erg/s/cm/AA  *m ~1e10 erg/s/cm^2
        #Return: The total flux of the blackbody erg/s/cm^2
        #        The total luminosity of the blackbody  erg/s 
        #        The radius of the blackbody   cm 
        h=6.62607015e-34; c=299792458.0; k=1.380649e-23
        fluxbb=sca*2*(k*T)**4*(np.pi)**4/15/h**3/c**2* unit *scale
        ferrbb=( (sca_err/sca)**2+ (4*T_err/T)**2 )**0.5*fluxbb 
        if redshift is not None: 
            Lumi_bb=flux2L(fluxbb, redshift)  # NOTE the unit of Lumi_bb is erg
            Lerr_bb=flux2L(ferrbb, redshift)  #      the unit ..............erg
            T=T*(1+redshift) ; T_err=T_err*(1+redshift)
        else: 
            Lumi_bb=np.nan; Lerr_bb=np.nan 

        sigma=5.67e-5 #Stephan-bolthzman constant, the unit is in cm,g,s 
        R    =  (Lumi_bb/sigma/T**4/4/np.pi)**0.5 #NOTE  here we should use T=T*(1+z)
        R2_err=  sca_err/sca*R**2
        R_err =  0.5*R2_err/R
        
        fluxbb=fluxbb.to(u.Unit('erg/s/cm^2')).value
        ferrbb=ferrbb.to(u.Unit('erg/s/cm^2')).value 
        Lerr_bb=1/np.log(10)* Lerr_bb/Lumi_bb
        Lumi_bb=np.log10(Lumi_bb) 

        return fluxbb,ferrbb,Lumi_bb,Lerr_bb, R, R_err ,T ,T_err

    @staticmethod
    def PhysicalParams_Blackbody(T, R,redshift=None,sca_unit=1e10*u.Unit('erg/s/cm^2')): 
        #NOTE T in unit K, R in unit cm 
        #sca_unit, 1e10*u.Unit('erg/s/cm^2'),in this case, the unit of flam is erg/s/cm^2/AA, and the 1e10 is correction between AA and m 
        assert redshift is not None, 'redshift is necessary to covert lumi to flux '
        sigma=5.67e-5 #Stephan-bolthzman constant, the unit is in cm,g,s 
        lumi= 4*np.pi*R**2 * sigma*T**4 
        print('#######A blackbody with Luminosity %s at redshift  %s'%(lumi,redshift) ) 
        flux= L2flux(lumi*u.Unit('erg/s'), redshift) * u.Unit('erg/s/cm^2')
        h=6.62607015e-34; c=299792458.0; k=1.380649e-23
        T=T/(1+redshift) #NOTE we do the redshift here, because the fitting of sed were at observed frame, hence the following integral should used the observed temperature
        sca=flux/ ( 2*(k*T)**4*(np.pi)**4/15/h**3/c**2)  
    
        return sca.to(sca_unit).value,T


    # @staticmethod
    # def sedfunc_filter1(sedfunc,p, bands,filter_data, fzero_lam={} ): 
    #     fbands=[] 
    #     for band in bands: 
    #         wave
    #         flam=sedfunc(wave,p) 
    #         fband=integrate.trapz(flam*frac,x=wave) # NOTE the unit?  
    #         fzero=integrate.trapz(3631*frac* const.c.value/wave**2,x=wave) *1e-23*1e10 #NOTE the unit? 
    #     #    mag=-2.5*np.log( fband/fzero)
    #     #    mag_lam=-2.5*np.log(flam/fzero_lam) 
    #         fband= fband/fzero*fzero_lam[band]   
    #         fbands.append(fband) 
    #  #  print('hellllllo', fbands)
    #     return np.array(fbands) 
    # def IntegrateE(self,p):
    #     lam=np.arange(1,1e6,0.1) 
    #     sed=self.sedfunc(lam, p) 
    #     flux=sum(sed*0.1)*u.Unit('erg/s/cm^2')
    #     Lbb =flux2L(flux,self.redshift) 
    #     return Lbb