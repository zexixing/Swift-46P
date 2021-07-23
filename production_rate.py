from tools import *
from aper_phot import *
from before_stack import *
from conversion import *
from profile_q import *
#from cr_to_flux import *
import matplotlib.pyplot as plt
from itertools import combinations
import os


# define parameters (obtain z)
def obtainObs(obs_log_name, horizon_id, epoch):
    #epoch = obs_log_name.split('_')[0]
    # plate scale
    # scale_p5 = [1,2,4,6,8,10,12,14,16,17,19,21,23,25]
    scale_1 = ['0','3_scale1','5_scale1','7_scale1','9_scale1','11_scale1','13_scale1',
               '15_scale1','18_scale1','20_scale1','22_scale1','24_scale1','26_scale1']
    if epoch in scale_1:
        scale = 1.004
    else:
        scale = 0.502
    obs_log_path = get_path('../docs/obs_log/'+obs_log_name)
    obs_log = pd.read_csv(obs_log_path, sep=' ', index_col=['FILTER'])
    obs_log = obs_log[['START', 'END', 'EXP_TIME']]
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
    obj2 = Horizons(id=horizon_id,epochs=mid_time.jd)
    ele = obj2.elements()[0]
    rh = eph['r']
    rhv = eph['r_rate']
    delta = eph['delta']
    phase = eph['alpha']
    exp_v = 0
    exp_uw1 = 0
    # exp time
    obs_log_v = obs_log.loc['V']
    obs_log_uw1 = obs_log.loc['UVW1']
    try:
        exp_v_list = list(obs_log_v['EXP_TIME'])
        exp_v = sum(exp_v_list)
    except TypeError:
        exp_v = obs_log_v['EXP_TIME']
    try:
        exp_uw1_list = list(obs_log_uw1['EXP_TIME'])
        exp_uw1 = sum(exp_uw1_list)
    except TypeError:
        exp_uw1 = obs_log_uw1['EXP_TIME']
    tp = ele['Tp_jd']
    dtp = mid_time.jd - tp
    obs_table = {'epoch':epoch,'scale':scale,
                 'start':start, 'end':end, 
                 'midtime':mid_time, 'tp': dtp,
                 'rh':rh, 'rhv':rhv, 'delta':delta,
                 'phase':phase,'exp_v':exp_v,'exp_uw1':exp_uw1}
    return obs_table

def obtainRadius(radius_km, delta, scale):
    arcs = (radius_km*3600*360)/(149597870.7*2*np.pi*delta)
    return int(arcs/scale)

def obtainBkg(bg_reg_name):
    bg = load_reg_list(bg_reg_name,'reg/')
    bg_x = bg[1]
    bg_y = bg[0]
    bg_r = bg[2]
    bg_center = list(zip(bg_x, bg_y))
    #reg2bg_bri(img_name, bg_method, bg_center, bg_r, count_method, mask_img = False)
    return bg_r, bg_center
'''
def obtainPara(comet_name, epoch, delta, scale):
    # horizon_id
    img_name_uw1 = epoch+'_stack_uw1.fits'
    img_name_v = epoch+'_stack_uvv.fits'
    spec_name_sun = 'sun_1A.txt'
    spec_name_OH = '2019-07-15_emission_models_OH.txt'
    src_center = (2000, 2000)
    bg_reg_name = epoch+'_sub_bg.reg'
    mask_name = epoch+'_mask.reg'
    wvm_name = epoch+'_wvm.txt'
    phase_name = 'phase_correction.txt'
    src_r_uvw1 = obtainRadius(100000, delta, scale)
    src_r_v = src_r_uvw1/10
    z = #--TODO:
    #img_name = epoch+'_sub_red'+str(int(red))+'.fits'
    #err2_name = epoch+'_sub_red'+str(int(red))+'_err.fits'
    bg_r, bg_center = obtainBkg(bg_reg_name)
'''
def howMask(ifmask):
    if ifmask == True:
        mask_img = mask_region(img_name_uw1, mask_name,'stack/')
        src_method = 'azim_median'
    else:
        mask_img = False
        src_method = 'total_mean'
    return mask_img, src_method

# aperture photometry
def aperPhot():
    #--TODO: 做了三遍测光（减少遍数）；提高每一遍的速度
    #mask_img, src_method = howMask(ifmask)
    result_uw1 = aper_phot(img_name_uw1, 'uw1', 
                           src_center, src_r_uw1,
                           bg_center, bg_r,
                           src_method, 'multi_mean',
                           2, mask_img, 0., 'stack/',ext)
    cr_uw1, cr_uw1_err = result_uw1[0]
    snr_uw1 = result_uw1[1]
    mag_uw1, mag_uw1_err = result_uw1[2]
    bg_cr_uw1, bg_cr_uw1_err = result_uw1[3]
    #bg_cr_uw1 = 0####0.0013638746855089488*0.15
    bg_cr_uw1_err = bg_cr_uw1

    result_v = aper_phot(img_name_v, 'v', 
                         src_center, src_r_uw1,
                         bg_center, bg_r,
                         src_method, 'multi_mean',
                         2, mask_img, 0., 'stack/',ext)
    cr_v, cr_v_err = result_v[0]
    snr_v = result_v[1]
    mag_v, mag_v_err = result_v[2]
    bg_cr_v, bg_cr_v_err = result_v[3]
    #bg_cr_v = 0.0048569324053546305*0.25*2
    bg_cr_v_err = bg_cr_v

    flux_uw1, flux_uw1_err = flux_ref_uw1(spec_name_sun,
                                          spec_name_OH,
                                          cr_uw1, cr_uw1_err,
                                          cr_v, cr_v_err, red)

    #flux_v, flux_v_err = flux_ref_v(spec_name_sun, 
    #                                spec_name_OH, 
    #                                cr_uw1, cr_uw1_err, 
    #                                cr_v, cr_v_err, red)
    
    result_v = aper_phot(img_name_v, 'v', 
                         src_center, src_r_v,
                         bg_center, bg_r,
                         src_method, 'multi_mean',
                         int(src_r_v/2.), mask_img, 0., 'stack/',ext)

    cr_v, cr_v_err = result_v[0]
    snr_v = result_v[1]
    mag_v, mag_v_err = result_v[2]
    bg_cr_v, bg_cr_v_err = result_v[3]
    #bg_cr_v = 0.0048569324053546305*0.25*2
    bg_cr_v_err = bg_cr_v

    flux_v, flux_v_err = flux_ref_v(spec_name_sun, 
                                    spec_name_OH, 
                                    cr_uw1, cr_uw1_err, 
                                    cr_v, cr_v_err, red)
    lumi_v = flux_v*4*np.pi*np.power(au2km(delta)*1000*100, 2)
    lumi_v_err = flux_v_err*4*np.pi*np.power(au2km(delta)*1000*100, 2)
    phot_dict = {'cr_uw1':cr_uw1,'cr_uw1_err':cr_uw1_err,
                 'snr_uw1':snr_uw1,
                 'mag_uw1':mag_uw1,'mag_uw1_err':mag_uw1_err,
                 'bg_cr_uw1':bg_cr_uw1,'bg_cr_uw1_err':bg_cr_uw1_err,
                 'flux_uw1':flux_uw1,'flux_uw1_err':flux_uw1_err,
                 'flux_v':flux_v,'flux_v_err':flux_v_err,
                 'cr_v':cr_v,'cr_v_err':cr_v_err,
                 'snr_v':snr_v,
                 'mag_v':mag_v,'mag_v_err':mag_v_err,
                 'bg_cr_v':bg_cr_v,'bg_cr_v_err':bg_cr_v_err,
                 'lumi_v':lumi_v,'lumi_v_err':lumi_v_err}
    return phot_dict

# obtain vectorial model
def obtainWvm(phot_dict):
    pass
    # placeholder for future wvm model codes

# determine reddening
# calculate OH & active area

def img2q(img_name):
    #mask_img, src_method = howMask(ifmask)
    # count rate profile
    aper_list_km, aper_list_pix, cr, cr_err = sur_bri_profile(scale, epoch, red,
                                                              ext, delta,
                                                              src_center, src_r_uw1,
                                                              bg_center, bg_r,
                                                              start, step, mask_img,
                                                              relative_path='sub/',
                                                              refcorr=refcorr,img_name=img_name,
                                                              star_clip=star_clip)[:4]
    #aper_list = au2km(as2au(aper_list*scale, delta))
    # total counts
    # unit conversion
    flux = 1.2750906353215913e-12*cr
    flux_err = 1.2750906353215913e-12*cr_err
    lumi = flux*4*np.pi*np.power(au2km(delta)*1000*100, 2)
    lumi_err = flux_err*4*np.pi*np.power(au2km(delta)*1000*100, 2)
    #--TODO: how to calculate g-factor in flux2num()
    num, num_err = flux2num(flux, flux_err,
                            '../data/auxil/fluorescenceOH.txt',
                            rh, rhv, delta)
    g_factor = lumi/num
    dr = (aper_list_km[1]-aper_list_km[0])*1e5
    #num_total, num_total_err = flux2num(1.275e-12*np.sum(cr*2*np.pi*aper_list_km*1e5*dr), 0,
    #                                    '../data/auxil/fluorescenceOH.txt',
    #                                    rh, rhv, delta)
    #num_total = np.sum(num[150:-50]*2*np.pi*aper_list_km[150:-50]*1e5*dr)
    num_total = np.sum(num*2*np.pi*aper_list_km*1e5*dr)
    flux_total = np.sum(flux*2*np.pi*aper_list_km*1e5*dr)
    flux_total_err = np.sum(flux_err*2*np.pi*aper_list_km*1e5*dr)
    print('OH flux:',flux_total,flux_total_err)
    num_total_err = np.sum(num_err*2*np.pi*aper_list_km*1e5*dr)
    # read wvm and interpolate & fitting
    #q, q_err, dis2col, ratio = num2q_fit(aper_list_km, num, num_err, wvm_name, 
    #                                    aperture=src_r_uw1, if_show = True, start=start,delta=delta,rh=rh) #--TODO: check
    #q, q_err, dis2col, ratio = num2q_fit(aper_list_km[:-100], num[:-100], num_err[:-100], wvm_name,
    #                                     aperture=1.5*aper_list_pix[-100]-0.5*aper_list_pix[-101],
    #                                     if_show = False, start=start)#1.5*aper_list_pix[150]-0.5*aper_list_pix[151])
    q, q_err, dis2col, ratio = num2q(num_total, num_total_err, wvm_name, scale, delta=delta, aperture=src_r_uw1,#1.5*aper_list_pix[-100]-0.5*aper_list_pix[-101], 
                                     if_show = False, start=start)
    # active area
    active_area = q/z
    active_area_err = q_err/z
    r_least = np.sqrt(active_area/(4*np.pi))
    r_least_err = active_area_err/(4*np.sqrt(np.pi*active_area))
    src_r_km = au2km(as2au(src_r_uw1*scale,delta))
    # print
    q_dict = {'src_r':src_r_uw1,'src_r_km':src_r_km,
              'cr':cr,'cr_err':cr_err,
              'aper_list_km':aper_list_km,'aper_list_pix':aper_list_pix,
              #'bg_bri':bg_bri,'bg_bri_err':bg_bri_err, #~~TODO:
              'g_factor':g_factor,
              'num':num,'num_err':num_err,
              'q':q,'q_err':q_err,
              'active_area':active_area/1e10,
              'active_area_err':active_area_err/1e10,
              'r_least':r_least/1e5,'r_least_err':r_least_err/1e5,
              'ratio':ratio,'model':dis2col,
              'lumi':lumi,'lumi_err':lumi_err,
              'flux_oh':flux_total,'flux_oh_err':flux_total_err}
    return q_dict

# if the minimum squared residual exists
def ifMultiple(norm_list):
    from scipy import signal
    from functools import reduce
    multiple = False
    min_ind_list = signal.argrelextrema(norm_list, np.less)
    min_ind_list = min_ind_list[min_ind_list!=ind]
    if len(min_ind_list) > 0: #less_equal
        multiple = True
        fluc_list = []
        max_ind_list = signal.argrelextrema(norm_list, np.greater)
        # if the multiple values are caused by fluctuations
        for i in min_ind_list:
            max_value_list = []
            try:
                max_value_list.append(max(max_ind_list[max_ind_list<i]))
            except ValueError:
                pass
            try:
                max_value_list.append(min(max_ind_list[max_ind_list>i]))
            except ValueError:
                pass
            if len(max_value_list)!=0:
                fluc = min(np.array(max_value_list)-norm_list[i])
                if fluc < (max(norm_list)-min(norm_list))/10:
                    fluc_list.append(True)
        if len(fluc_list)==len(min_ind_list) and bool(reduce(lambda x,y:x*y,a))==True:
            multiple=False
    return multiple

# obtain reddening       
def howRed(red_list=np.linspace(0,25,26),chatter=1, if_default=False):
    default_red = 15
    if if_default == True:
        # return reddening, if fitted
        return default_red, False
    # determine reddening
    red_list = np.linspace(0,25,26)
    norm_list = []
    print('reddening will be determined from the range: ', red_list)
    for red in red_list:
        if chatter == 2:
            print('current position: '+str(red))
        img_name = epoch+'_sub_red'+str(int(red))+'.fits'
        #err2_name = epoch+'_sub_red'+str(int(red))+'_err.fits'
        img_path = get_path('../docs/sub/'+img_name)
        #err2_path = get_path('../docs/sub/'+err2_name)
        if os.path.exists(img_path):
            os.remove(img_path)
        fits_sub(img_name_uw1, img_name_v, epoch, red, rh, delta, scale, 'stack/',
                 'stack/',wvm_name=wvm_c2_name,exp=True,smooth=smooth,coicorr=coicorr,c2corr=c2corr)
        q = img2q(img_name)['q']
        #step_num = int(src_r_uw1/2.)
        aper_list_km, aper_list_pix, sur_bri_list, err_list = \
            sur_bri_profile(scale, epoch, red,
                            ext, delta,
                            src_center, src_r_uw1,
                            bg_center, bg_r,
                            start, 2., mask_img,
                            relative_path='sub/')[:4] #--TODO: to be updated 2021-4-27
        colden_meas, colden_err = \
            colden_profile(sur_bri_list, 
                           err_list, 
                           rh,rhv,delta)
        colden_model = sur_bri_model(wvm_name, 
                                     aper_list_km, 
                                     q)
        norm_r = norm_cd(colden_meas, colden_model)
        if chatter == 2:
            print('corresponding result: '+str(norm_r)+'\n')
        norm_list.append(norm_r)
        os.remove(img_path)
    # judge for norm_list
    # if the point is at ends
    print('reddening was determined from the range: ', red_list)
    print('corresponding determination factors are: ',norm_list)
    ind = np.where(norm_list==np.min(norm_list))
    red = ind
    if_fit = True
    if ind[0][0] == 0 or ind[0][0] == 25:
        raise Warning('The minimum point is around the ends of the reddeing range.')
        red = default_red
        if_fit = False
    # if there are multiple minimum points / seek valley
    multiple = ifMultiple(norm_list)
    if multiple == True:
        raise Warning('There are multiple minimum values')
        red = default_red
        if_fit = False
    return red, if_fit

# pipeline
#obtainObs(obs_log_name, horizon_id, epoch)
#obtainPara(comet_name, epoch)

#aperPhot(epoch)
#obtainWvm()

#howRed(chatter=1, if_default=False)
#imageSub(red_default=False)
#img2q(img_name)


#for epoch in range(1,27):
# define global parameters
print('###################################')
#obs_log_name = '25'+'_obs-log_46P.txt'
epoch = 'jan'#'26'
obs_log_name = epoch+'_obs-log_46P.txt'
horizon_id = 90000548#90000546
obs_table = obtainObs(obs_log_name, horizon_id, epoch)
#epoch = obs_table['epoch']
delta = obs_table['delta']
scale = obs_table['scale']
rh = obs_table['rh']
rhv = obs_table['rhv']
print(obs_table)
img_name_uw1 = epoch+'_uw1.fits'
img_name_v = epoch+'_uvv.fits'
spec_name_sun = 'sun_1A.txt'
spec_name_OH = '2019-07-15_emission_models_OH.txt'
src_center = (2000, 2000)
bg_reg_name = epoch+'_bg.reg'
mask_name = epoch+'_mask.reg'
wvm_name = 'wvm/'+epoch+'_wvm.txt' #'pyvm'
wvm_c2_name = 'wvm/'+epoch+'_wvm_c2.txt'
phase_name = 'phase_correction.txt'
ext={0:'img',1:'exp'}
src_r_uw1 = obtainRadius(70000, delta, scale)
#src_r_uw1 = obtainRadius(30000, delta, scale)
src_r_v = obtainRadius(10000, delta, scale)
z = 3.233E+17#3.106E+17#6.278E+16--TODO:
bg_r, bg_center = obtainBkg(bg_reg_name)
ifmask = False
if epoch in ['1','2','4','6','8','10','12','14','16','nov','dec']:
    smooth = True 
else:
    smooth = False
coicorr=False
c2corr=False
sencorr=False
star_clip=False
mask_img, src_method = howMask(ifmask)
if scale == 1.004:
    start = 6
    step = 2
elif scale == 0.502:
    start = 50
    step = 5
    #src_r_uw1 = 1070
print('start: '+str(start)+';  step: '+str(step))

#red, if_fit = howRed(chatter=2)
red = 0
if_fit = False


phot_dict = aperPhot()

ext={0:'img',1:'err2',2:'exp'}
#red, if_fit = howRed(chatter=1)
img_name = epoch+'_sub_red'+str(int(red))+'.fits'
#err2_name = epoch+'_sub_red'+str(int(red))+'_err.fits'
refcorr=True

fits_sub(img_name_uw1, img_name_v, epoch, red, rh, delta, scale, 
         'stack/',wvm_name=wvm_c2_name,exp=True,smooth=smooth,coicorr=coicorr,c2corr=c2corr)

q_dict = img2q(img_name)

# print
print('\n')
print('#---------- EPOCH: '+str(epoch)+' ----------')
print('DELTA (au): '+str(delta)+';  Rh (au): '+str(obs_table['rh']))
print('PLATE SCALE (arcsec/pixel): '+str(scale))
print('MIDTIME (UT): '+str(obs_table['midtime'])+' ( JD: '+str(obs_table['midtime'].jd)+')')
print('EXPOSURE (s): '+'UVW1: '+str(round(obs_table['exp_uw1'],2))+' V: '+str(round(obs_table['exp_v'],2)))
print('\n')
print( 'UW1:\n'
      +'COUNT RATE (cts/s): '+str(phot_dict['cr_uw1'])+' +/- '+str(phot_dict['cr_uw1_err'])+'\n'
      +'SNR: '+str(phot_dict['snr_uw1'])+'\n'
      +'MAGNITUDE (mag): '+str(phot_dict['mag_uw1'])+' +/- '+str(phot_dict['mag_uw1_err'])+'\n'
      +'BACKGROUND CR (cts/s/arcsec2): '+str(phot_dict['bg_cr_uw1'])+' +/- '+str(phot_dict['bg_cr_uw1_err'])+'\n'
      +'FLUX from reflection (erg/s/cm2): '+str(phot_dict['flux_uw1'])+' +/- '+str(phot_dict['flux_uw1_err']))
print( 'V:\n'
      +'COUNT RATE (cts/s): '+str(phot_dict['cr_v'])+' +/- '+str(phot_dict['cr_v_err'])+'\n'
      +'SNR: '+str(phot_dict['snr_v'])+'\n'
      +'MAGNITUDE (mag): '+str(phot_dict['mag_v'])+' +/- '+str(phot_dict['mag_v_err'])+'\n'
      +'BACKGROUND CR (cts/s/arcsec2): '+str(phot_dict['bg_cr_v'])+' +/- '+str(phot_dict['bg_cr_v_err'])+'\n'
      +'FLUX (erg/s/cm2): '+str(phot_dict['flux_v'])+' +/- '+str(phot_dict['flux_v_err'])+'\n'
      +'LUMINOSITY (erg/s): '+str(phot_dict['lumi_v'])+' +/- '+str(phot_dict['lumi_v_err'])
      +'\n')

print('\n')
print('REDDENING (%): '+str(int(red))+';   if fitted: '+str(if_fit))
print('APERTURE (pixel): '+str(q_dict['src_r'])+';   (km): '+str(q_dict['src_r_km']))
#print('COUNT RATE (cts/s): '+str(q_dict['cr'])+' +/- '+str(q_dict['cr_err']))
#print('BACKGROUND SB (cts/s/arcsec2): '+str(q_dict['bg_bri'])+' +/- '+str(q_dict['bg_bri_err']))
#print('g-factor (erg s-1 mol-1): '+str(q_dict['g_factor']))
#print('NUMBER (mol): '+str(q_dict['num'])+' +/- '+str(q_dict['num_err']))
print('Water production rate (mol/s): '+str(q_dict['q'])+' +/- '+str(q_dict['q_err']))
print('Active area(km2): '+str(q_dict['active_area'])+' +/- '+str(q_dict['active_area_err']))
print('Radiu(km): '+str(q_dict['r_least'])+' +/- '+str(q_dict['r_least_err'])+'\n')
print('\n')

# plot
model = q_dict['model']
ratio = q_dict['ratio']
#q0 = 7.55e27
#print(ratio)
#ratio = ratio*q0/q_dict['q']
#print(ratio)
import pandas as pd
result_df = pd.DataFrame.from_dict({'radius_km':q_dict['aper_list_km'],'column_density_data':q_dict['num'],'column_density_data_error':q_dict['num_err'],'column_density_model':model(q_dict['aper_list_km'])*ratio})
result_df.to_csv('/Users/zexixing/Research/swift46P/'+epoch+'.csv')
plt.plot(q_dict['aper_list_km'],q_dict['num'],color='b')
plt.fill_between(q_dict['aper_list_km'],q_dict['num']+q_dict['num_err'],q_dict['num']-+q_dict['num_err'],color='b',alpha=0.3)
plt.plot(q_dict['aper_list_km'],model(q_dict['aper_list_km'])*ratio,color='r')
plt.xlabel('km')
plt.ylabel('column density (/cm2)')
#plt.yscale('log')
#plt.xscale('log')
plt.title('epoch '+epoch+'   '+str(obs_table['midtime'])+'\n'+\
          'red.='+str(int(red))+'% '+\
          'rh='+str(round(rh,2))+'au '+\
          'delta='+str(round(delta,2))+'au ')
          #'ap.='+str(q_dict['src_r'])+'pix')
plt.show()

def compareKV(col_name, cr_name, dist_km_ZX, col_ZX, dist_pix_ZX, cr_ZX, col_MD, scale, delta):
    col_path = get_path('../docs/'+col_name)
    cr_path = get_path('../docs/'+cr_name)
    dist_col = np.loadtxt(col_path,delimiter=',', skiprows=1)
    dist_cr = np.loadtxt(cr_path,delimiter=',', skiprows=1)
    dist_km = dist_col[:,0]
    col = dist_col[:,1]
    dist_pix = dist_cr[:,0]
    cr = dist_cr[:,1]
    plt.plot(dist_km,col,'y-',label='KV')
    plt.plot(dist_km_ZX, col_ZX, 'b-', label='ZX')
    plt.title('km - column density')
    pix2cm2_factor = np.power(au2km(as2au(scale, delta))*1e5, 2)
    #plt.plot(dist_pix,cr,'y-',label='KV')
    #plt.plot(dist_pix_ZX, cr_ZX*pix2cm2_factor, 'b-', label='ZX')
    #plt.title('pixel - count rates')
    plt.legend()
    plt.show()

#compareKV('CD_coi_corrected_reddening_zero.csv', 'Count_rate_with_sky_correction.csv', 
#          q_dict['aper_list_km'], q_dict['num'], 
#          q_dict['aper_list_pix'], q_dict['cr'], 
#          model(q_dict['aper_list_km'])*ratio,
#          scale, delta)



    
    