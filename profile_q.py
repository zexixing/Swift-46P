import numpy as np
from tools import *
from aper_phot import *
from conversion import *
from cr_to_flux import *
from remove_smear import gaussBlur
from scipy.stats import norm
import math

def fits_sub(img_uw1_name, img_uvv_name, ind, r, rh, delta, scale, 
             relative_path='', wvm_name = False,
             exp=False, smooth=False, coicorr=True, c2corr=True):
    img_uw1_data = load_img(img_uw1_name,relative_path,ext=0)
    img_uvv_data = load_img(img_uvv_name,relative_path,ext=0)
    img_uw1_header = load_header(img_uw1_name,relative_path)
    beta = 0.09276191501510327
    #beta = 0.18
    beta = reddening_correct(r)*beta
    if exp:
        exp_uw1 = load_img(img_uw1_name,relative_path, ext = 1)
        exp_uvv = load_img(img_uvv_name,relative_path, ext = 1)
        exp_data = exp_uw1+exp_uvv
        exp_data = exp_data/np.max(exp_data)
    else:
        exp_uw1 = float(load_header(img_uw1_name,relative_path)['EXPTIME'])
        exp_uvv = float(load_header(img_uvv_name,relative_path)['EXPTIME'])
        #if c2corr:
        #    c2rate = c2(dist_data)
        #    img_uvv_data = img_uvv_data-c2rate*exp_uvv_data
    img_uw1_rate = img_uw1_data/exp_uw1
    img_uvv_rate = img_uvv_data/exp_uvv
    #sub_data_err = img_uw1_data/np.power(exp_uw1,2) + img_uvv_data*np.power(beta/exp_uvv,2)
    #sub_data_err_uvw1 = error_prop('sub', 0, np.sqrt(img_uw1_data), 0, 0.001364*0.15*exp_uw1*0.3)/exp_uw1
    #sub_data_err_v = error_prop('sub', 0, np.sqrt(img_uvv_data),0, 0.004857*0.25*exp_uvv*0.3)/exp_uvv*beta
    sub_data_err = np.power(error_prop('sub', img_uw1_data/exp_uw1, np.sqrt(img_uw1_data)/exp_uw1, 
                            beta*img_uvv_data/exp_uvv, beta*np.sqrt(img_uvv_data)/exp_uvv),2)
    if coicorr:
        coicorr_uw1 = coincidenceCorr(img_uw1_rate, scale)
        coicorr_uvv = coincidenceCorr(img_uvv_rate, scale)
        img_uw1_rate = img_uw1_rate*coicorr_uw1
        img_uvv_rate = img_uvv_rate*coicorr_uvv
        sub_data_err = np.power(error_prop('sub', img_uw1_rate*coicorr_uw1, np.sqrt(img_uw1_data)*coicorr_uw1/exp_uw1, 
                                           img_uvv_rate*coicorr_uvv*beta, np.sqrt(img_uvv_data)*coicorr_uvv/exp_uvv*beta),2) # FIXME:
    if c2corr:
        center = (2000,2000)
        cen_pix = [center[1]-1, center[0]-1]
        pos_data = np.argwhere(img_uvv_data!=np.nan) #(2000,2000)
        dist_data = np.sqrt(np.sum(np.power(pos_data-cen_pix,2),axis=1)).reshape(img_uvv_data.shape)
        c2,bkg_c2 = c2rates(wvm_name,scale,rh,delta)
        c2rate = c2(dist_data)
        img_uvv_rate = img_uvv_rate-c2rate+bkg_c2
        #plt.imshow(img_uvv_rate,vmin=0,vmax=0.06)
        #plt.show()
    if smooth:
        img_uw1_rate = gaussBlur(img_uw1_rate, 2, 9, 9, "symm")
        img_uvv_rate = gaussBlur(img_uvv_rate, 2, 9, 9, "symm")
    sub_data = img_uw1_rate - beta*(img_uvv_rate)
    #sub_data = img_uvv_rate*exp_uvv
    #plt.imshow(sub_data,vmin=0,vmax=50)
    #plt.show()
    img_name = ind+'_sub_red'+str(int(r))+'_new.fits.gz'
    #err2_name = ind+'_sub_red'+str(int(r))+'_err.fits'
    output_path = get_path('../docs/sub/'+img_name)
    #output_path_err = get_path('../docs/sub/'+err2_name)
    if os.path.exists(output_path):
        os.remove(output_path)
    hdr = fits.Header()
    hdr['TELESCOP'] = img_uw1_header['TELESCOP']
    hdr['INSTRUME'] = img_uw1_header['INSTRUME']
    #hdr['COMET'] = img_uw1_header['COMET']
    hdr['XPOS'] = img_uw1_header['XPOS']
    hdr['YPOS'] = img_uw1_header['YPOS']
    hdr['EXT'] = ('img','[image]')
    hdr1 = fits.Header()
    hdr1['EXT'] = ('err2','[error square map]')
    if exp == True:
        hdr2 = fits.Header()
        hdr2['EXT'] = ('exp','[exposure map]')
    #hdr['MID_T_UV'] = img_uw1_header['MID_TIME']
    #hdr['MID_T_V'] = load_header(img_uvv_name)['MID_TIME']
    hdu_img = fits.PrimaryHDU(sub_data, header=hdr)
    hdu_err = fits.ImageHDU(sub_data_err, header=hdr1)
    if exp == False:
        hdul = fits.HDUList([hdu_img,hdu_err])
    else:
        hdu_exp = fits.ImageHDU(exp_data,header=hdr2)
        hdul = fits.HDUList([hdu_img,hdu_err,hdu_exp])
    hdul.writeto(output_path)
    os.system('mv '+output_path+' '+output_path[:-3])
    #hdu_err.writeto(output_path_err)
    

#fits_sub('26_uw1.fits', '26_uvv.fits', '26', 0, 1.1365363279, 0.18963365494949, 0.502, 
#         relative_path='stack/', wvm_name = '26_wvm_c2.txt',
#          exp=True, smooth=True, coicorr=True, c2corr=True)

'''
def sur_bri_profile(img_name, scale,
                    ext, delta, 
                    src_center, src_r,
                    bg_center, bg_r,
                    start, step, mask_img,
                    relative_path='',
                    chatter=0):
    # initialize
    step_num = math.floor((src_r-start)/step)
    err_list = []    
    sur_bri_list = []
    # read in bkg 
    if (bg_center != False) and (bg_r != False):
        bg_bri = reg2bg_bri(img_name, 'multi',
                            bg_center, bg_r,
                            'mean', mask_img,
                            relative_path=relative_path)[0] #~~TODO:
    # aper list
    aper_list = np.linspace(start,start+step*step_num,step_num+1)
    aper_list = (aper_list[1:]+aper_list[:-1])/2.
    # img data
    img_data, cen_pix, i_range, j_range = \
        limit_loop(img_name, src_center, start+step*step_num, relative_path=relative_path, ext=ext)
    if isinstance(ext,int) == True:
        img_data = img_data - bg_bri #~~TODO:
        exp_data = np.ones(img_data.shape)
        err2_data = np.zeros(np.shape(img_data))
        bg_bri_err = 0
    else:
        dic_data = img_data
        img_data = dic_data['img'] - bg_bri
        data_keys = list(dic_data.keys())
        if 'err2' in data_keys:
            err2_data = dic_data['err2']
            if (bg_center != False) and (bg_r != False):
                bg_bri_err, bg_pixel = multi_circle_ct(err2_data, bg_center, bg_r, 'mean', mask_img,relative_path=relative_path)
                bg_bri_err = np.array(bg_bri_err)
                bg_pixel = np.array(bg_pixel)
                bg_bri_err = np.sqrt(np.sum(bg_bri_err/np.power(bg_pixel,2)))/len(bg_pixel) 
        else:
            err2_data = np.zeros(np.shape(img_data))
            bg_bri_err = 0
        if 'exp' in data_keys:
            exp_data = dic_data['exp']
        else:
            exp_data = np.ones(img_data.shape)
    # mask img
    if isinstance(mask_img, bool):
        if mask_img == False:
            mask_img = np.ones(img_data.shape)
    # get median profile
    import itertools
    import pandas as pd
    import functools
    pos_pix_list = list(itertools.product(j_range,i_range))
    pos_pix_arr = np.array(pos_pix_list)
    dist_list = np.sqrt(np.sum(np.power(pos_pix_arr-src_center,2),axis=1))
    index_list = ((np.array(dist_list)-start)/step).astype(int)
    image_list = img_data[i_range[0]:(i_range[-1]+1),j_range[0]:(j_range[-1]+1)].flatten()
    exp_list = exp_data[i_range[0]:(i_range[-1]+1),j_range[0]:(j_range[-1]+1)].flatten()
    err2_list = err2_data[i_range[0]:(i_range[-1]+1),j_range[0]:(j_range[-1]+1)].flatten()
    mask_list = mask_img[i_range[0]:(i_range[-1]+1),j_range[0]:(j_range[-1]+1)].flatten()
    data = {'distance': dist_list,
            'position': pos_pix_list,
            'index': index_list,
            'image': image_list,
            'err2': err2_list,
            'mask': mask_list,
            'exp':exp_list,}
            #'aper': aper_list}
    data = pd.DataFrame(data)
    data = data[data['mask']==1]
    data = data[data['exp']==np.max(exp_list)]
    data = data[data['distance']>start]
    data = data[data['distance']<start+step*step_num]
    data_group = data.groupby('index')
    #median_list = {}
    median_list = []
    err2sum_list = []
    pixel_mask = []
    for name,group in data_group:
        #median_list[name] = np.median(group['image'])
        median_list.append(np.median(group['image']))
        err2sum_list.append(np.sum(group['err2']))
        pixel_mask.append(len(group))
    median_list = np.array(median_list)
    err_per_pixel_list = np.sqrt(np.array(err2sum_list))/np.array(pixel_mask)*1.2533
    if len(median_list)<len(aper_list):
        aper_list = aper_list[:len(median_list)]
        if chatter == 1:
            print('Aperture is so large that the blank area is included.')
            print('Blank area has been cut.')
    # conversion
    pix2cm2_factor = np.power(au2km(as2au(1, delta))*1e5, 2)
    median_list = median_list/pix2cm2_factor
    err_per_cm2_list = err_per_pixel_list/pix2cm2_factor
    aper_list = au2km(as2au(aper_list*scale, delta))
    return aper_list, median_list, err_per_cm2_list
'''
def map2list(img_data, index_data, err2_data, mask_map,err='std'):
    image_list = img_data[mask_map.mask==False]
    index_list = index_data[mask_map.mask==False]
    err2_list = err2_data[mask_map.mask==False]
    median_list = []
    err2sum_list = []
    pixel_mask = []
    std_list = []
    med_data = np.zeros(img_data.shape)
    shape_data = np.zeros(img_data.shape)
    for ind in np.arange(np.min(index_list),np.max(index_list)+1):
        median_unit = image_list[index_list==ind]
        median_unit = median_unit[np.logical_not(np.isnan(median_unit))]
        err2_unit = err2_list[index_list==ind]
        err2_list = err2_list[np.logical_not(np.isnan(err2_list))]
        median_list.append(np.median(median_unit))
        err2sum_list.append(np.sum(err2_unit))
        pixel_mask.append(len(err2_unit))
        std_list.append(np.std(median_unit))
        med_data[index_data==ind] = np.median(median_unit)
        shape_data[mask_map.mask==False] = 1
        shape_data[mask_map.mask==True] = np.nan
    median_list = np.array(median_list)
    med_data = med_data*shape_data
    if err == 'median':
        err_per_pixel_list = np.sqrt(np.array(err2sum_list))/np.array(pixel_mask)*1.2533
    elif err == 'std':
        err_per_pixel_list = std_list
    else:
        print('Check the error method')
    return median_list,pixel_mask,err_per_pixel_list,med_data

def sur_bri_profile(scale, epoch, red,
                    ext, delta, 
                    src_center, src_r,
                    bg_center, bg_r,
                    start, step, mask_img,
                    relative_path='',
                    chatter=0,smooth=False,
                    coicorr=False,rate_img=True,
                    sencorr=False,refcorr=False,
                    img_name=False,star_clip=False,
                    err_method='std'):
    # initialize
    if start == False:
        start = 0
    step_num = math.floor((src_r-start)/step)
    err_list = []    
    sur_bri_list = []
    # img data 
    if (isinstance(img_name, np.ndarray) == True) or (isinstance(img_name, str) == True):
        img_name = img_name
    else:
        img_name = epoch+'_sub_red'+str(int(red))+'.fits'
    img_data, cen_pix, i_range, j_range = \
        limit_loop(img_name, src_center, start+step*step_num, relative_path=relative_path, ext=ext)
    if isinstance(ext,int) != True:
        dic_data = img_data
        img_data = dic_data['img']
    if coicorr:
        if rate_img:
            coicorr = coincidenceCorr(img_data, scale)
        elif isinstance(ext,int):
            exp_data = float(load_header(img_name,relative_path)['EXPTIME'])
            coicorr = coincidenceCorr(img_data/exp_data, scale)
        elif 'exp' in dic_data.keys():
            exp_data = dic_data['exp']
            coicorr = coincidenceCorr(img_data/exp_data, scale)
        else:
            raise('not found exposure map')
        img_data = img_data*coicorr
    if sencorr:
        img_data = img_data/(1-0.01*13.913)
    if smooth:
        img_data = gaussBlur(img_data, 2, 9, 9, "symm")
    # read in bkg 
    if (bg_center != False) and (bg_r != False):
        bg_bri = reg2bg_bri(img_data, 'multi',
                            bg_center, bg_r,
                            'mean', mask_img,
                            relative_path=relative_path)[0] #~~TODO:
        #bg_uvv = 0.0038431776735597814*0.5
        #bg_uvv =  0.0048569324053546305*0.25*2
        #bg_uw1 = 0.000979366426149368*0.4
        #bg_uw1 = 0.0013638746855089488*0.15*2
        #bg_uvv = 0.
        #bg_uw1 = 0.
        bg_uvv = 0.006277 - 0.002923481399193406
        bg_uw1 = 0.000532 + 0.0002636321005411446
        beta = 0.09276191501510327
        beta = reddening_correct(red)*beta
        bg_bri = bg_uw1 - beta*bg_uvv # extrapolated bkg
    #bg_bri = 0.000379584 # Borisov
    else:
        bg_bri=0
    #bg_bri = 0 # no bkg subtraction
    img_data = img_data - bg_bri #~~TODO:
    # aper list
    aper_list = np.linspace(start,start+step*step_num,step_num+1)
    aper_list = (aper_list[1:]+aper_list[:-1])/2.
    # img data
    if isinstance(ext,int) == True:
        exp_data = np.ones(img_data.shape)
        err2_data = np.zeros(np.shape(img_data))
        bg_bri_err = 0
    else:
        data_keys = list(dic_data.keys())
        if 'err2' in data_keys:
            err2_data = dic_data['err2']
            if (bg_center != False) and (bg_r != False):
                #bg_bri_err, bg_pixel = multi_circle_ct(err2_data, bg_center, bg_r, 'mean', mask_img,relative_path=relative_path)
                #bg_bri_err = np.array(bg_bri_err)
                #bg_pixel = np.array(bg_pixel)
                #bg_bri_err = np.sqrt(np.sum(bg_bri_err/np.power(bg_pixel,2)))/len(bg_pixel) 
                bg_bri_err = bg_bri*1.0
            else:
                #bg_bri_err = 0
                bg_bri_err = bg_bri*1.0
            err2_data = err2_data + bg_bri_err**2
        else:
            err2_data = np.zeros(np.shape(img_data))
            bg_bri_err = 0
        if 'exp' in data_keys:
            exp_data = dic_data['exp']
        else:
            exp_data = np.ones(img_data.shape)
    # mask img
    if isinstance(mask_img, bool):
        if mask_img == False:
            mask_img = np.ones(img_data.shape)
    # other maps
    pos_data = np.argwhere(img_data!=np.nan) #(2000,2000)
    dist_data = np.sqrt(np.sum(np.power(pos_data-cen_pix,2),axis=1)).reshape(img_data.shape) 
    index_data = np.ceil((np.array(dist_data)-start)/step)-1 
    index_data[np.where(index_data==-1)]=0
    vec_data = pos_data-cen_pix
    vec_data_x = vec_data[:,0].reshape(img_data.shape)
    vec_data_y = vec_data[:,1].reshape(img_data.shape)
    vec_data_45 = np.abs(vec_data_x)-np.abs(vec_data_y)
    # mask
    import numpy.ma as ma
    mask_map = ma.masked_where(mask_img==0,img_data)
    mask_map = ma.masked_where(np.isnan(img_data),mask_map)
    if refcorr:
        if epoch in ['1','2','4','6','8','10','nov']: 
            mask_map = ma.masked_where(vec_data_x<0,mask_map) #epoch nov
            mask_map = ma.masked_where(vec_data_y>0,mask_map) #epoch nov
            #mask_map = ma.masked_where(vec_data_45<0,mask_map) #epoch nov
        elif epoch in ['12','14','16','dec']:
            mask_map = ma.masked_where(vec_data_x<0,mask_map) #epoch dec
            mask_map = ma.masked_where(vec_data_45<0,mask_map) #epoch dec
        elif epoch in ['17','18','19','20','21','22','23','24','25','26','jan']:
            mask_map = ma.masked_where(vec_data_y<0,mask_map) # epoch jan
        else:
            pass
    if start != 0:
        mask_map = ma.masked_where(dist_data<=start,mask_map)
    else:
        pass
    mask_map = ma.masked_where(dist_data>start+step*step_num,mask_map)
    mask_map = ma.masked_where(exp_data!=np.max(exp_data),mask_map)
    
    #plt.imshow(img_data,vmin=0,vmax=0.008)
    #plt.show()
    #shape_data = np.zeros(img_data.shape)
    #shape_data[mask_map.mask==False] = 1
    #shape_data[mask_map.mask==True] = np.nan
    #plt.imshow(img_data*shape_data,vmin=0,vmax=0.03)
    #theta = np.arange(0,2*np.pi,0.02)
    #r_peak = np.array([450,635,716,745])
    #plt.plot(np.outer(np.cos(theta),r_peak)+1999,np.outer(np.sin(theta),r_peak)+1999,color='y')
    #plt.show()

    # get median profile
    median_list,pixel_mask,err_per_pixel_list,med_data = map2list(img_data, index_data, err2_data, mask_map,err=err_method)
    if star_clip:
        sub_img = img_data - med_data
        sub_list = sub_img[mask_map.mask==False]
        (mu_clip, sigma_clip) = norm.fit(sub_list)
        mask_map = ma.masked_where(sub_img>(mu_clip+3*sigma_clip),mask_map)
        mask_map = ma.masked_where(sub_img<(mu_clip-3*sigma_clip),mask_map)
        #from scipy.stats import norm
        #n, bins, patches = plt.hist(sub_list,100, facecolor='green', alpha=0.75)
        #y = norm.pdf( bins, mu, sigma)
        #plt.plot(bins, y, 'r--', linewidth=2)
        median_list,pixel_mask,err_per_pixel_list,med_data = map2list(img_data, index_data, err2_data, mask_map,err=err_method)
        #shape_data = np.zeros(img_data.shape)
        #shape_data[mask_map.mask==False] = 1
        #shape_data[mask_map.mask==True] = np.nan
        #plt.imshow(img_data*shape_data,vmin=0,vmax=0.008)
        #plt.show()
    if len(median_list)<len(aper_list):
        aper_list = aper_list[:len(median_list)]
        if chatter == 1:
            print('Aperture is so large that the blank area is included.')
            print('Blank area has been cut.')
    # conversion
    aper_list_km = au2km(as2au(aper_list*scale, delta))
    pix2cm2_factor = np.power(au2km(as2au(scale, delta))*1e5, 2)
    median_pixel_list = median_list
    median_list = median_list/pix2cm2_factor
    err_per_cm2_list = err_per_pixel_list/pix2cm2_factor
    #plt.plot(aper_list,median_list)
    #plt.fill_between(aper_list,np.array(median_list)+np.array(err_per_cm2_list),
    #                 np.array(median_list)-np.array(err_per_cm2_list),color='k',alpha=0.3)
    #plt.show()

    '''
    # ----- test omega-----
    plt.imshow(index_data,vmin=0,vmax=10)
    plt.show()
    area_me = np.array(pixel_mask)*pix2cm2_factor
    th1=np.linspace(start,start+step*step_num,step_num+1)[:-1]
    th2=np.linspace(start,start+step*step_num,step_num+1)[1:]
    omega_real = 2*np.pi*(np.cos(th1*scale*np.pi/(3600*180))-np.cos(th2*scale*np.pi/(3600*180)))
    pixel_should = np.pi*(th2**2-th1**2)
    if len(median_list)<len(aper_list):
        omega_real = omega_real[:len(median_list)]
        pixel_should = pixel_should[:len(median_list)]
    area_real = omega_real*np.power(au2km(delta)*1e5,2)
    plt.plot(aper_list*scale/60,area_me,'b')
    plt.plot(aper_list*scale/60,area_real,'r')
    plt.show()
    print(aper_list,th1,th2,pixel_should,pixel_mask)
    plt.plot(aper_list*scale/60,pixel_mask,'b')
    plt.plot(aper_list*scale/60,pixel_should,'r')
    plt.show()
    '''
    return aper_list_km, aper_list, median_list, err_per_cm2_list, median_pixel_list, err_per_pixel_list

#sur_bri_profile(0.502, '26', 0,
#                {0: 'img', 1: 'err2', 2: 'exp'}, 0.1896322876609, 
#                (2000, 2000), 1013,
#                [(714.44365, 2093.9035), (1819.4318, 768.43878)], [20.0, 34.84456],
#                0, 5, False,
#                relative_path='sub/',
#                chatter=0,smooth=False,
#                coicorr=False,rate_img=True,
#                sencorr=False,refcorr=False,
#                img_name='26_sub_red0_new.fits',star_clip=True)


def colden_profile(sur_bri_list, err_list, rh, rhv, delta):
    flux_list = 1.2750906353215913e-12*sur_bri_list
    err_list = 1.2750906353215913e-12*err_list
    colden_list = []
    colden_err_list = []
    for i in range(len(flux_list)):
        num, num_err = flux2num(flux_list[i], err_list[i],
                                '../data/auxil/fluorescenceOH.txt',
                                rh,rhv,delta)
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

def norm_cd(colden_meas, colden_model):
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

def coifile():
    dir_path = get_path('../docs/stack/')
    epoch_list = np.arange(1,27,dtype='int')
    epoch_list = [str(i) for i in epoch_list]
    for filt in ['uvv','uw1']:
        for index in epoch_list:
            if index in ['1','2','4','6','8','10','12','14','16']:
                name = index+'_'+filt+'_img.fits'
            else:
                name = index+'_'+filt+'.fits'
            print(name)
            file_path = dir_path+name
            hdul = fits.open(file_path)
            scale = float(hdul[0].header['PLATESCL'])
            coi_map = coincidenceCorr(hdul[0].data/hdul[1].data, scale)
            hdr = fits.Header()
            hdr['EPOCH'] = name.split('_')[0]
            hdr['FILTER'] = hdul[0].header['FILTER']
            hdr['MOVECORR'] = 'NO'
            hdu_coi = fits.PrimaryHDU(coi_map,header=hdr)
            hdul = fits.HDUList([hdu_coi])
            output_path = get_path('../docs/coi_map/'+index+'_coi-map_'+filt+'_noMoveCorr'+'.fits.gz')
            hdul.writeto(output_path)
            os.system('mv '+output_path+' '+output_path[:-3])
    for filt in ['uvv','uw1']:
        for index in ['1','2','4','6','8','10','12','14','16']:
            name = index+'_'+filt+'.fits'
            print(name)
            file_path = dir_path+name
            hdul = fits.open(file_path)
            scale = float(hdul[0].header['PLATESCL'])
            coi_map = coincidenceCorr(hdul[0].data/hdul[1].data, scale)
            hdr = fits.Header()
            hdr['EPOCH'] = name.split('_')[0]
            hdr['FILTER'] = hdul[0].header['FILTER']
            hdr['MOVECORR'] = 'YES'
            hdu_coi = fits.PrimaryHDU(coi_map,header=hdr)
            hdul = fits.HDUList([hdu_coi])
            output_path = get_path('../docs/coi_map/'+index+'_coi-map_'+filt+'_MoveCorr'+'.fits.gz')
            hdul.writeto(output_path)
            os.system('mv '+output_path+' '+output_path[:-3])


def combineEvtImg():
    epoch_list = ['1','2','4','6','8','10','12','14','16']
    dir_path = get_path('../docs/stack/')
    for epoch in epoch_list:
        for filt in ['uvv','uw1']:
            print(epoch, filt)
            name_evt = epoch+'_'+filt+'.fits'
            name_img = epoch+'_'+filt+'_img.fits'
            evt_path = get_path(dir_path+name_evt)
            img_path = get_path(dir_path+name_img)
            evt = fits.open(evt_path)
            img = fits.open(img_path)
            # write
            hdr = img[0].header
            hdr['MAP'] = 'IMAGE MAP'
            hdu_img = fits.PrimaryHDU(img[0].data,header=hdr)
            hdr = fits.Header()
            hdr['MAP'] = 'IMAGE EXPOSURE'
            hdu_exp_img = fits.ImageHDU(img[1].data,header=hdr)
            hdr = evt[0].header
            hdr['MAP'] = 'EVENT MAP'
            hdu_evt = fits.ImageHDU(evt[0].data,header=hdr)
            hdr = fits.Header()
            hdr['MAP'] = 'EVENT EXPOSURE'
            hdu_exp_evt = fits.ImageHDU(evt[1].data,header=hdr)    
            hdr = fits.Header()  
            hdr['MAP'] = 'BLURRED EVENT MAP'
            smear = gaussBlur(evt[0].data, 2, 9, 9, "symm")
            hdu_smr = fits.ImageHDU(smear,header=hdr)
            hdul = fits.HDUList([hdu_img,hdu_exp_img,hdu_evt,hdu_exp_evt,hdu_smr])
            output_path = get_path('../docs/combine/'+epoch+'_'+filt+'.fits.gz')
            hdul.writeto(output_path)
            os.system('mv '+output_path+' '+output_path[:-3])

#combineEvtImg()          


#coifile()
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

'''
ext={0:'img',1:'err2', 2:'exp'}
delta=0.07952451430935
src_center=(2000,2000)
#src_center=(2300,2300)
src_r_uw1 = 600
bg_center = [(1753.3279,1892.642)]
bg_r = [100]
start = 12
mask_img = False
epoch = '14'
aper_list_img, sur_bri_list_img, err_list = \
    sur_bri_profile(epoch+'_img_sub_red0.fits', 0.502,
                    ext, delta,
                    src_center, src_r_uw1,
                    bg_center, bg_r,
                    start, 4., mask_img,
                    relative_path='sub/')
aper_list_evt, sur_bri_list_evt, err_list = \
    sur_bri_profile(epoch+'_evt_sub_red0.fits', 0.502,
                    ext, delta,
                    src_center, src_r_uw1,
                    bg_center, bg_r,
                    start, 4., mask_img,
                    relative_path='sub/')
aper_list_smooth, sur_bri_list_smooth, err_list = \
    sur_bri_profile(epoch+'_smooth_sub_red0.fits', 0.502,
                    ext, delta,
                    src_center, src_r_uw1,
                    bg_center, bg_r,
                    start, 4., mask_img,
                    relative_path='sub/',smooth=True)
plt.plot(aper_list_img,sur_bri_list_img,'r-',label='img')
plt.plot(aper_list_evt,sur_bri_list_evt,'b-',label='evt')
plt.plot(aper_list_smooth,sur_bri_list_smooth,'g-',label='evt(smooth)')
plt.title('OH')
plt.ylabel('counts/s')
plt.xlabel('km')
plt.legend()
plt.show()
'''
'''
red = 0
for i in range(1,27):
    epoch = str(int(i))
    print(epoch)
    if epoch in ['1','2','4','6','8','10','12','14','16']:
        smooth = True
    else:
        smooth = False
    fits_sub(epoch+'_uw1.fits', epoch+'_uvv.fits', epoch, red, 0.502, 'stack/',exp=True,smooth=smooth,coicorr=True)
'''
'''
ext={0:'img',1:'err2', 2:'exp'}
src_center=(2000,2000)
src_r_uw1 = 1500
bg_r = False#[40]
start = 12
mask_img = False

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    import matplotlib as mpl
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

n=13

for i in range(1,27):
    epoch = str(int(i))
    print(epoch)
    img = ''
    obs_log_name = epoch+'_obs-log_46P.txt'
    obs_log_path = get_path('../docs/obs_log/'+obs_log_name)
    img_set = pd.read_csv(obs_log_path, sep=' ')
    delta_list = img_set['OBS_DIS']
    r_list = img_set['HELIO']
    rv_list = img_set['HELIO_V']
    rh = (np.min(r_list)+np.max(r_list))/2
    rhv = (np.min(rv_list)+np.max(rv_list))/2
    delta = (np.min(delta_list)+np.max(delta_list))/2
    if i in np.arange(1,12):
        mon = 'Nov'
        color = 'r'
        if i == 1:
            bg_center = [(1154.0975,2469.1257)]
        else:
            bg_center = [(827.17389,2528.5664)]
    elif i in np.arange(12,17):
        mon = 'Dec'
        color = 'b'
        bg_center = [(3025.0348,1184.4508)]
        if i in [12,14,16]:
            img = ''
    elif i in np.arange(17,27):
        mon = 'Jan'
        color = 'yellow'
        bg_center = [(1013.4503,2229.0942)]
    else:
        pass
    if i in np.arange(1,14):
        c1='r'
        c2='b'
    else:
        c1='b'
        c2='yellow'
    bg_center = False
    aper_list_km, aper_list_pix, sur_bri_list, err_list = \
        sur_bri_profile(epoch+'_sub_red0.fits', 0.502,
                        ext, delta,
                        src_center, src_r_uw1,
                        bg_center, bg_r,
                        start, 4., mask_img,
                        relative_path='sub/')
    colden_list, err_list = colden_profile(sur_bri_list, err_list, rh, rhv, delta)
    if i in [1,14,25]:
        label = mon
        plt.plot(aper_list_km, colden_list, color=color, label=label)
    else:
        plt.plot(aper_list_km, colden_list, color=color)#colorFader(c1,c2,i/n-np.ceil(i/n)+1)
plt.title('OH')
plt.ylabel('counts/s')
plt.xlabel('km')
plt.legend()
plt.show()
'''