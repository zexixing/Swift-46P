import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from pyvectorial import pyvm
from tools import *

def filt_para(filt):
    para = {'v': {'fwhm': 769, 'zp': 17.89, 'zp_err': 0.013, 'cf': 2.61e-16, 'cf_err': 2.4e-18},
            'b': {'fwhm': 975, 'zp': 19.11, 'zp_err': 0.016, 'cf': 1.32e-16, 'cf_err': 9.2e-18},
            'u': {'fwhm': 785, 'zp': 18.34, 'zp_err': 0.020, 'cf': 1.5e-16, 'cf_err': 1.4e-17},
            'uw1': {'fwhm': 693, 'zp': 17.49, 'zp_err': 0.03, 'cf': 4.3e-16, 'cf_err': 2.1e-17, 'rf': 0.1375},
            'um2': {'fwhm': 498, 'zp': 16.82, 'zp_err': 0.03, 'cf': 7.5e-16, 'cf_err': 1.1e-17},
            'uw2': {'fwhm': 657, 'zp': 17.35, 'zp_err': 0.04, 'cf': 6.0e-16, 'cf_err': 6.4e-17}
            }
    return para[filt]

def cr2mag(cr, cr_err, filt):
    """use zero points to transfer counts to mag
    """
    zp = filt_para(filt)['zp']
    mag = zp - 2.5*np.log10(cr)
    mag_err_1 = 2.5*cr_err/(np.log(10)*cr)
    mag_err_2 = filt_para(filt)['zp_err']
    mag_err = np.sqrt(mag_err_1**2 + mag_err_2**2)
    return mag, mag_err

def cr2equi_flux(cr, cr_err, filt):
    # read ea
    ea_path = get_path('../data/auxil/arf_'+filt+'.fits')
    ea_data = fits.open(ea_path)[1].data
    ea_wave = (ea_data['WAVE_MIN']+ea_data['WAVE_MAX'])/20. # A to nm
    ea_area = ea_data['SPECRESP']
    delta_wave = ea_wave[1] - ea_wave[0]
    factor = 0
    for i in range(len(ea_wave)):
        factor += ea_area[i]*ea_wave[i]*delta_wave*1e8*5.034116651114543
    equi_flux = cr/(factor*10) # 10: nm to A
    equi_flux_err = cr_err/(factor*10)
    snr = equi_flux/equi_flux_err
    return equi_flux, equi_flux_err, snr

def cr2sb(cr, cr_err, filt, solid_angle):
    """convert count rate to surface brightness
    unit: W m-2 sr-1

    cr to flux density: Poole et al. 2008
    flux den to flux: flux den*FWHM
    flux to sf: flux*factors/solid_angle
    factors: 1 arcsec2 = 2.35e-11 sr
             1 erg s-1 = 1e-7W
             1 cm2 = 1e-4m2
    """
    fwhm = filt_para(filt)['fwhm']
    cf = filt_para(filt)['cf']
    cf_err = filt_para(filt)['cf_err']
    factors = 1. #TODO:
    sb = cr*cf*fwhm*factors/solid_angle
    sb_err_1 = cr_err*cf
    sb_err_2 = cf_err*cr
    sb_err = (fwhm*factors/solid_angle)*np.sqrt(sb_err_1**2 + sb_err_2**2)
    return sb, sb_err

def cr2flux(cr, cr_err, filt):
    fwhm = filt_para(filt)['fwhm']
    cf = filt_para(filt)['cf']
    cf_err = filt_para(filt)['cf_err']
    flux = cr*cf*fwhm*(4/2.35)
    flux_err = fwhm*np.sqrt(np.power(cr_err*cf,2)+np.power(cf_err*cr,2))*(4/2.35)
    return flux, flux_err

def mag_sb_flux_from_spec(spec_name, filt):
    """use effective area and theoretical spectra
    to calculate apparent magnitude
    """
    # C2: cr2flux = 1.676235658983609e-14 
    # read spectra
    spec_path = get_path('../data/auxil/'+spec_name)
    spec_wave = np.loadtxt(spec_path)[:, 0]
    spec_flux = np.loadtxt(spec_path)[:, 1]#*2.720E-4 # flux moment to irradiance
    if np.min(spec_wave)<1000:
        spec_wave = spec_wave*10
    # read ea
    ea_path = get_path('../data/auxil/arf_'+filt+'.fits')
    ea_data = fits.open(ea_path)[1].data
    ea_wave = (ea_data['WAVE_MIN']+ea_data['WAVE_MAX'])/2#0. # A to nm
    ea_area = ea_data['SPECRESP']
    # interpolate ea to cater for spec
    ea = interpolate.interp1d(ea_wave, ea_area, fill_value='extrapolate')
    spec_ea = ea(spec_wave)
    #plt.plot(spec_wave,spec_flux/np.max(spec_flux))
    #plt.plot(spec_wave,spec_ea/np.max(spec_ea))
    #plt.plot(spec_wave,spec_ea)
    #plt.show()
    wave_min = max([np.min(spec_wave), np.min(ea_wave)])
    wave_max = min([np.max(spec_wave), np.max(ea_wave)])
    spec = np.c_[np.c_[spec_wave, spec_flux.T], spec_ea.T]
    spec_reduce = spec[spec[:,0]>wave_min, :]
    spec_reduce = spec_reduce[spec_reduce[:,0]<wave_max, :]
    spec = spec_reduce[spec_reduce[:,2]>0, :]
    # integral
    delta_wave = spec[2, 0] - spec[1, 0]
    cr = 0
    for i in range(len(spec)):
        cr += spec[i, 0]*spec[i, 1]*spec[i, 2]*delta_wave*1e7*5.034116651114543 #10^8 for Kurucz
    #print(np.sum(spec_flux)*(spec_wave[1]-spec_wave[0])/cr)
    # cr to mag
    return cr, cr2mag(cr, 0, filt), cr2sb(cr, 0, filt, 1.), cr2flux(cr, 0, filt)

#mag_sb_flux_from_spec('2019-07-15_emission_models_OH.txt', 'uw1')
#mag_sb_flux_from_spec('PSG_spectra_46P_data.txt', 'v')

def flux2num(flux, flux_err, g_name, mean_r, mean_rv, mean_delta):
    # load g factor file
    g_path = get_path('../docs/'+g_name)
    g_file = np.loadtxt(g_path, skiprows=3)
    helio_v_list = g_file[:, 0]
    g_1au_list = (g_file[:,1]+g_file[:,2]+g_file[:,3])*1e-16
    # interpolate
    g_1au = interpolate.interp1d(helio_v_list, g_1au_list, 
                                 kind='cubic', fill_value='extrapolate')
    # return num
    lumi = flux*4*np.pi*np.power(au2km(mean_delta)*1000*100, 2)
    lumi_err = flux_err*4*np.pi*np.power(au2km(mean_delta)*1000*100, 2)
    mean_g = g_1au(mean_rv)/np.power(mean_r, 2)
    num = lumi/mean_g
    num_err = lumi_err/mean_g
    return num, num_err

def num_assu(wvm_name, aperture, delta=False, start=False, scale=0.502, profile=False, rh=1):
    if not start:
        start = 0.
    if wvm_name == 'pyvm':
        dis, col_den = pyvm(rh, delta)
        dis = [i*1000*100 for i in dis]
    else:
        # load col density from wvm file
        dis = []
        col_den = []
        wvm_path = get_path('../docs/'+wvm_name)
        wvm_file = open(wvm_path)
        wvm_file_lines = wvm_file.readlines()
        wvm_file.close()
        for line in wvm_file_lines[52:70]: #[14:25]
            line = line[:-1]
            line = line.split()
            line = [float(i) for i in line]
            dis.append(line[0]*1000*100)
            dis.append(line[2]*1000*100)
            dis.append(line[4]*1000*100)
            dis.append(line[6]*1000*100)
            col_den.append(line[1])
            col_den.append(line[3])
            col_den.append(line[5])
            col_den.append(line[7])
    if profile == True:
        return np.array(dis)/(1000*100), np.array(col_den)
    # interpolate
    if delta==False:
        delta = wvm_file_lines[2].split()
        delta = float(delta[13])
    dis2col = interpolate.interp1d(dis, col_den, 
                                   kind='quadratic', fill_value='extrapolate') 
    if not aperture:
        aperture = wvm_file_lines[72].split()
        aperture = float(aperture[3])
    step_num = int((aperture-start)*100)#10000.
    start = au2km(as2au(start*scale, delta))*1000*100.
    aperture = au2km(as2au(aperture*scale, delta))*1000*100. # pixel to cm
    dis_list = np.linspace(start, aperture, step_num)
    dis_list = (dis_list[1:]+dis_list[:-1])/2.
    step = dis_list[1]-dis_list[0]
    col_list = dis2col(dis_list)
    print('model:',start/1e5,aperture/1e5)
    # integral
    num_assu = np.sum(2*np.pi*dis_list*step*col_list)
    return num_assu

def num2q(num, num_err, wvm_name, scale, delta=False, aperture=False, if_show = True, start=False):
    """ covert number to production rate
        from web vectoria model
    """
    # get assumed num of the model within the aperture
    num_model = num_assu(wvm_name, aperture, delta, start, scale)
    print(num_model)
    print(num/num_model)
    dis_model, col_den_model = num_assu(wvm_name, aperture, start, profile=True)
    dis2col = interpolate.interp1d(dis_model, col_den_model,
                                   kind='quadratic', fill_value='extrapolate')
    # readin the assumed Q_H2O
    if wvm_name == 'pyvm':
        q_assu = 1.e28
    else:
        wvm_path = get_path('../docs/'+wvm_name)
        wvm_file = open(wvm_path)
        wvm_file_lines = wvm_file.readlines()
        wvm_file.close()
        q_assu = wvm_file_lines[6].split()
        q_assu = float(q_assu[4])
    # ratio -> actual Q_H2O
    q = (q_assu/num_model)*num
    q_err = (q_assu/num_model)*num_err
    if if_show == True:
        print('water production rate: '+str(q)+' +/- '+str(q_err))
    return q, q_err, dis2col,num/num_model

def num2q_fit(aper_list, num_list, num_err_list, wvm_name, 
              aperture=False, if_show = True, start=False, delta=1., rh=1.):
    if wvm_name == 'pyvm':
        q_assu = 1.e+28
        dis_model, col_den_model = num_assu(wvm_name, aperture, delta=delta, start=start, profile=True, rh=rh)
    else:
        # get assumed num of the model within the aperture
        dis_model, col_den_model = num_assu(wvm_name, aperture, start=start, profile=True)
        # readin the assumed Q_H2O
        wvm_path = get_path('../docs/wvm/'+wvm_name)
        wvm_file = open(wvm_path)
        wvm_file_lines = wvm_file.readlines()
        wvm_file.close()
        q_assu = wvm_file_lines[6].split()
        q_assu = float(q_assu[4])
    # fitting
    dis2col = interpolate.interp1d(dis_model, col_den_model, 
                                   kind='quadratic', fill_value='extrapolate')
    from scipy import optimize
    def func(x,ratio):
        return ratio*dis2col(x)
    popt,pcov=optimize.curve_fit(func,aper_list,num_list,[1],sigma=num_err_list,absolute_sigma=True)
    # ratio -> actual Q_H2O
    para = popt[0]
    para_err = np.sqrt(np.diag(pcov))[0]
    print(para,para_err)
    q = q_assu*para
    q_err = q_assu*para_err #TODO:
    if if_show == True:
        print('water production rate: '+str(q)+' +/- '+str(q_err))
    return q, q_err, dis2col, para

def mag2afr(horizon_id, mag_v, mag_v_err, aperture, obs_log_name, phase_name, phase_corr):
#def mag2afr(horizon_id, mag_v, mag_v_err, aperture, r, delta, phase, phase_name, phase_corr):
    
    obs_log_path = get_path('../docs/'+obs_log_name)
    obs_log = pd.read_csv(obs_log_path, sep=' ', 
                          index_col=['FILTER'])
    obs_log = obs_log[['HELIO', 'HELIO_V', 'START', 'END', 'OBS_DIS']]
    obs_log = obs_log.loc['V']
    if obs_log.index.name == 'FILTER':
        start = obs_log['START'].iloc[0]
        end = obs_log['END'].iloc[-1]
    else:
        start = obs_log['START']
        end = obs_log['END']
    dt = Time(end)-Time(start)
    mid_time = Time(start)+1/2*dt
    obj = Horizons(id=horizon_id,
                   location='@swift',
                   epochs=mid_time.jd)
    eph = obj.ephemerides()[0]
    r = eph['r']
    delta = eph['delta']
    phase = eph['alpha']
    
    # afr
    mag_sun = -26.75
    rou = au2km(as2au(aperture,delta))*1000
    factor = np.power(2*au2km(delta)*r*1000,2)/rou
    afr = np.power(10,0.4*(mag_sun-mag_v))*factor
    afr_err = 0.4*np.log(10)*np.power(10,0.4*(mag_sun-mag_v))*mag_v_err*factor
    # phase effect correction
    phase_path = get_path('../docs/'+phase_name)
    phase_file = np.loadtxt(phase_path)
    deg = phase_file[:,0]
    corr_coef = phase_file[:,1]
    pha_corr = interpolate.interp1d(deg, corr_coef, fill_value='extrapolate')
    corr_coef = pha_corr(phase)
    if phase_corr == True:
        afr = afr/corr_coef
        afr_err = afr_err/corr_coef
    else:
        pass
    return afr, afr_err

def den2colden(wvm_name):
    # load col density from wvm file
    col_dis = []
    col = []
    den_dis = []
    den = []
    wvm_path = get_path('../docs/'+wvm_name)
    wvm_file = open(wvm_path)
    wvm_file_lines = wvm_file.readlines()
    wvm_file.close()
    delta = wvm_file_lines[2].split()
    delta = float(delta[13])
    for line in wvm_file_lines[52:70]: #[14:25]
        line = line[:-1]
        line = line.split()
        line = [float(i) for i in line]
        col_dis.append(line[0]*1000*100)
        col_dis.append(line[2]*1000*100)
        col_dis.append(line[4]*1000*100)
        col_dis.append(line[6]*1000*100)
        col.append(line[1])
        col.append(line[3])
        col.append(line[5])
        col.append(line[7])
    #col_dis = np.array(col_dis)/(1000*100)
    for line in wvm_file_lines[14:25]:
        line = line[:-1]
        line = line.split()
        line = [float(i) for i in line]
        den_dis.append(line[0]*1000*100)
        den_dis.append(line[2]*1000*100)
        den_dis.append(line[4]*1000*100)
        den_dis.append(line[6]*1000*100)
        den.append(line[1])
        den.append(line[3])
        den.append(line[5])
        den.append(line[7])
    dx = 0.23E+03*1000*100
    den_dis.append(den_dis[-1]+dx)
    den_dis.append(den_dis[-1]+dx)
    den.append(0)
    den.append(0)
    #den_dis = np.array(den_dis)/(1000*100)
    den_profile = interpolate.interp1d(den_dis, den, fill_value='extrapolate')
    col_predict = []
    for r in col_dis:
        colden = 0
        for x in np.arange(0.23E+03*1000*100, (0.97E+06+0.23E+03)*1000*100, dx):
            dist = np.sqrt(r**2+x**2)
            colden += den_profile(dist)*dx
        col_predict.append(2*colden)
    plt.plot(np.array(col_dis)/(1000*100),col_predict,'r-',label='predicted profile from density(cm**-3)')
    plt.plot(np.array(col_dis)/(1000*100),col,'b-',label='profile(cm**-2) from wvm file')
    plt.legend()
    plt.xlabel('km')
    plt.ylabel('cm**-2')
    plt.title('WVM')
    plt.ylim(-0.1e13,2.7e13)
    plt.xlim(-2000,65000)
    plt.show()

def c2rates(wvm_name,scale,r,delta):
    # load colden profile from wvm model
    dis = []
    col_den = []
    wvm_path = get_path('../docs/wvm/'+wvm_name)
    wvm_file = open(wvm_path)
    wvm_file_lines = wvm_file.readlines()
    wvm_file.close()
    for line in wvm_file_lines[52:70]: #[14:25]
        line = line[:-1]
        line = line.split()
        line = [float(i) for i in line]
        dis.append(line[0])
        dis.append(line[2])
        dis.append(line[4])
        dis.append(line[6])
        col_den.append(line[1])
        col_den.append(line[3])
        col_den.append(line[5])
        col_den.append(line[7])
    dis = np.array(dis)
    col_den = np.array(col_den)
    # convert unit: km->pix, #/cm^2->#/pix
    pix2km = au2km(as2au(scale, delta))
    pix2cm2 = (pix2km*1000*100)**2
    col_den = pix2cm2*col_den*0.4
    # col_den to flux
    g_factor = 4.5e-13/np.power(r, 2)
    lumi = col_den*g_factor
    flux = lumi/(4*np.pi*(au2km(delta)*1000*100)**2)
    # flux to count rate
    cr = flux/3.352471317967212e-13
    # sensitivity correction
    #cr = cr*(1-0.01*13.913)
    c2 = interpolate.interp1d(dis,cr,fill_value='extrapolate')
    bkg = c2(100000)
    dis = dis/pix2km
    c2 = interpolate.interp1d(dis,cr,fill_value='extrapolate')
    #fig = plt.figure()
    #plt.plot(dis,cr)
    #plt.show()
    return c2, bkg

def f2l(f,r):
    # f: erg/cm2/s, r: au
    r_cm = au2km(r)*1e5
    return 4*np.pi*r_cm*r_cm*f

#c2rates('26_wvm_c2.txt',0.502,1.1365363279,0.18963365494949)
#c2rates('14_wvm_c2.txt',0.502,1.055386343318,0.07952451430935)