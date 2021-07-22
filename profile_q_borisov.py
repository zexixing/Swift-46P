import numpy as np
from tools import *
from aper_phot import *
from conversion import *
from cr_to_flux import *

def fits_sub(img_uw1_name, img_uvv_name, mon, r):
    img_uw1_data = load_img(img_uw1_name)
    img_uvv_data = load_img(img_uvv_name)
    img_uw1_header = load_header(img_uw1_name)
    exposure_uw1 = float(load_header(img_uw1_name)['EXPTIME'])
    exposure_uvv = float(load_header(img_uvv_name)['EXPTIME'])
    beta = 0.09276191501510327
    beta = reddening_correct(r)*beta
    sub_data = img_uw1_data/exposure_uw1 \
               - beta*(img_uvv_data/exposure_uvv)
    sub_data_err = img_uw1_data/np.power(exposure_uw1,2) + img_uvv_data*np.power(beta/exposure_uvv,2)
    img_name = mon+'_sub_red'+str(int(r))+'.fits'
    err2_name = mon+'_sub_red'+str(int(r))+'_err.fits'
    output_path = get_path('../docs/'+img_name)
    output_path_err = get_path('../docs/'+err2_name)
    hdu = fits.PrimaryHDU(sub_data)
    hdr = hdu.header
    hdr['TELESCOP'] = img_uw1_header['TELESCOP']
    hdr['INSTRUME'] = img_uw1_header['INSTRUME']
    hdr['COMET'] = img_uw1_header['COMET']
    hdr['XPOS'] = img_uw1_header['XPOS']
    hdr['YPOS'] = img_uw1_header['YPOS']
    #hdr['MID_T_UV'] = img_uw1_header['MID_TIME']
    #hdr['MID_T_V'] = load_header(img_uvv_name)['MID_TIME']
    hdu.writeto(output_path)
    hdu_err = fits.PrimaryHDU(sub_data_err)
    hdu_err.writeto(output_path_err)

def sur_bri_profile(img_name, horizon_id,
                    err2_name, obs_log_name,
                    src_center, src_r,
                    bg_center, bg_r,
                    start, step_num, mask_img):
    count_rate_list = []
    pixel = []
    err_list = []    
    sur_bri_list = []
    aper_list = []
    step = (src_r-start)/step_num
    for i in range(0, step_num):
        r_i = (i*step+start,(i+1)*step+start)
        result = donut_ct(img_name,
                          src_center, r_i,
                          'median', mask_img)
        bg_bri = reg2bg_bri(img_name, 'multi', 
                            bg_center, bg_r, 
                            'mean', mask_img)[0]
        aper_list.append((r_i[0]+r_i[1])/2.)
        count_rate_list.append(result[0]-bg_bri*result[1])
        pixel.append(result[1])
        result_err = donut_ct(err2_name,
                              src_center, r_i,
                              'mean', mask_img)
        src_err = (np.sqrt(result_err[0])/result_err[1])*result[1]*1.2533
        bg_bri_err, bg_pixel = multi_circle_ct(err2_name, bg_center, bg_r, 'mean', mask_img)
        bg_bri_err = np.array(bg_bri_err)
        bg_pixel = np.array(bg_pixel)
        bg_bri_err = np.sqrt(np.sum(bg_bri_err/np.power(bg_pixel,2)))/len(bg_pixel)
        bg_err = bg_bri_err*result[1]
        err = np.sqrt(np.power(src_err,2)+np.power(bg_err,2))
        err_list.append(err)
    count_rate_list = np.array(count_rate_list)
    obs_log_path = get_path('../docs/'+obs_log_name)
    obs_log = pd.read_csv(obs_log_path, sep=' ', index_col=['FILTER'])
    obs_log = obs_log[['START', 'END']]
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
    mean_delta = eph['delta']
    pix2cm2_factor = np.power(au2km(as2au(1, mean_delta))*1e5, 2)
    cm2_list = np.array(pixel)*pix2cm2_factor
    sur_bri_list = count_rate_list/cm2_list
    err_list = np.array(err_list)/cm2_list
    aper_list = [au2km(as2au(i, mean_delta)) for i in aper_list]
    return aper_list, sur_bri_list, err_list

def colden_profile(sur_bri_list, err_list, obs_log_name, horizon_id):
    flux_list = 1.2750906353215913e-12*sur_bri_list
    err_list = 1.2750906353215913e-12*err_list
    colden_list = []
    colden_err_list = []
    for i in range(len(flux_list)):
        num, num_err = flux2num(flux_list[i], err_list[i],
                                'fluorescenceOH.txt',
                                obs_log_name,
                                'both_ends', horizon_id, False)
        colden_list.append(num)
        colden_err_list.append(num_err)
    return np.array(colden_list), np.array(colden_err_list)

def sur_bri_model(wvm_name, aper_list, q):
    rou = []
    colden = []
    wvm_path = get_path('../docs/'+wvm_name)
    wvm_file = open(wvm_path)
    wvm_file_lines = wvm_file.readlines()
    wvm_file.close()
    for line in wvm_file_lines[52:70]:
        line = line[:-1]
        line = line.split()
        line = [float(i) for i in line]
        rou.append(line[0])
        rou.append(line[2])
        rou.append(line[4])
        rou.append(line[6])
        colden.append(line[1])
        colden.append(line[3])
        colden.append(line[5])
        colden.append(line[7])
    rou = np.array(rou)
    colden = np.array(colden)
    q_model = wvm_file_lines[6].split()
    q_model = float(q_model[4])
    colden = (q/q_model)*colden
    wvm_model = interpolate.interp1d(rou, colden, fill_value='extrapolate')
    colden_list = wvm_model(aper_list)
    return colden_list

def norm(colden_meas, colden_model):
    return np.sum(np.power(colden_meas-colden_model, 2))/len(colden_meas)

def crop(img_name, mask_name, bg_reg_name, delta, g, x=100):
    # read in and crop mask data
    mask_img = mask_region(img_name, mask_name)
    mask_crop = mask_img[1000-x:999+x,1000-x:999+x]
    # read in and crop img data
    img_data = load_img(img_name)
    img_crop = img_data[1000-x:999+x,1000-x:999+x]
    # cr to column den
    bg = load_reg_list(bg_reg_name)
    bg_x = bg[1]
    bg_y = bg[0]
    bg_r = bg[2]
    bg_center = list(zip(bg_x, bg_y))
    bg_bri = reg2bg_bri(img_name, 'multi',
                        bg_center, bg_r,
                        'mean', mask_img)[0]
    img_crop = img_crop - bg_bri # net cr
    img_crop = 1.2750906353215913e-12*img_crop # cr 2 flux
    img_crop = img_crop*4*np.pi*np.power(au2km(delta)*1000*100, 2) # flux 2 lumi
    img_crop = img_crop/g # lumi 2 num
    pix2cm2_factor = np.power(au2km(as2au(1, delta))*1e5, 2)
    img_crop = img_crop/pix2cm2_factor # num per pixel 2 per cm2
    # return cropped img and mask
    return img_crop, mask_crop

def wvm_img(delta, wvm_name, q, x=100):
    aper_list = []
    wvm_data = np.zeros((2*x-1,2*x-1))
    for i in range(0,2*x-1):
        for j in range(0, 2*x-1):
            d = get_dist((i,j),(x-1,x-1))
            # arcsec to km
            d = au2km(as2au(d, delta))
            aper_list.append(d)
    wvm_crop = sur_bri_model(wvm_name, np.array(aper_list), q)
    wvm_crop = wvm_crop.reshape((2*x-1,2*x-1))
    return wvm_crop

def sub_img(img_name, mask_name, bg_reg_name, wvm_name, save_name, q, delta, g, x=100):
    img_crop, mask_crop = crop(img_name, mask_name, bg_reg_name, delta, g, x)
    wvm_crop = wvm_img(delta, wvm_name, q, x)
    net_crop = (img_crop - wvm_crop)/1e11
    net_crop = net_crop*mask_crop
    output_path = get_path('../docs/'+save_name)
    hdu = fits.PrimaryHDU(net_crop)
    hdu.writeto(output_path)
'''
sub_img('sep_sub_red0.fits', 'sep_mask.reg', 
        'sep_sub_bg.reg', 'sep_wvm.txt', 
        'sep_img-data_red0.fits', 
        -2.655764475272607e+26, 3.09822744350658, 3.712554616374067e-16, x=100)

img_name =  'feb_sub_red13.fits'
mask_name = 'feb_mask.reg'
bg_reg_name = 'feb_sub_bg.reg'
img_data = load_img(img_name)
img_header = load_header(img_name)
mask_img = mask_region(img_name, mask_name)
#exposure = float(load_header(img_name)['EXPTIME'])
bg = load_reg_list(bg_reg_name)
bg_x = bg[1]
bg_y = bg[0]
bg_r = bg[2]
bg_center = list(zip(bg_x, bg_y))
bg_bri = reg2bg_bri(img_name, 'multi', 
                    bg_center, bg_r, 
                    'mean', mask_img)[0]
img_data = img_data - bg_bri
output_path = get_path('../nasa_media_press/'+'OH_06_feb.fits')
hdu = fits.PrimaryHDU(img_data)
hdr = hdu.header
hdr['TELESCOP'] = img_header['TELESCOP']
hdr['INSTRUME'] = img_header['INSTRUME']
hdr['COMET'] = img_header['COMET']
hdr['XPOS'] = img_header['XPOS']
hdr['YPOS'] = img_header['YPOS']
#hdr['MID_T_UV'] = img_uw1_header['MID_TIME']
hdu.writeto(output_path)
'''