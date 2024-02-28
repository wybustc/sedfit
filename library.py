import numpy as np
from astropy.cosmology import FlatLambdaCDM


def MonteC(flux_dbsp,err_dbsp,model='normal'):
	flux=np.ones_like(flux_dbsp)
	if model=='normal':
		for i in range(0,len(flux_dbsp)):
			flux[i]=np.random.normal(flux_dbsp[i],err_dbsp[i])
	return flux
  
def flux2L(flux,redshift, H0_used=70):
	#Hubble constant
	cosmo=FlatLambdaCDM(H0=H0_used*u.km/u.s/u.Mpc,Om0=0.3,Tcmb0=2.725*u.K)
	dl=cosmo.luminosity_distance(redshift)
	luminosity=flux*4*np.pi*dl**2
	luminosity=luminosity.to('erg/s').value #the unit is 'erg/s'
	return luminosity
def L2flux(lumi, redshift): 
	cosmo=FlatLambdaCDM(H0=70*u.km/u.s/u.Mpc,Om0=0.3,Tcmb0=2.725*u.K)
	dl=cosmo.luminosity_distance(redshift)
	flux=lumi/ (4*np.pi*dl**2 )
	flux=flux.to('erg/s/cm^2').value
	return flux 
def DL(redshift):
	cosmo=FlatLambdaCDM(H0=70*u.km/u.s/u.Mpc,Om0=0.3,Tcmb0=2.725*u.K)
	dl=cosmo.luminosity_distance(redshift)
	return dl
def Mag2Abs(mag, redshift): 
	dl=DL(redshift)
	dl=dl.to(u.pc).value
	mag_abs=mag-5*num.log10(dl/10)
	return mag_abs  
