import numpy as np
from tools import *
from aper_phot import *
from conversion import *
from cr_to_flux import *
from profile_q import fits_sub
from remove_smear import gaussBlur
import math

def morphology(img_name, scale,
               ext, delta, 
               src_center, src_r,
               bg_center, bg_r,
               start, step, mask_img, mid_time,
               relative_path='',
               chatter=0,smooth=False,
               coicorr=False,rate_img=True,):
    # initialize
    if start == False:
        start = 0
    step_num = math.floor((src_r-start)/step)
    err_list = []    
    sur_bri_list = []
    # img data
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
    if smooth:
        img_data = gaussBlur(img_data, 2, 9, 9, "symm")
    # read in bkg 
    if (bg_center != False) and (bg_r != False):
        bg_bri = reg2bg_bri(img_data, 'multi',
                            bg_center, bg_r,
                            'mean', mask_img,
                            relative_path=relative_path)[0] #~~TODO:
    bg_uvv = 0.0038431776735597814
    bg_uw1 = 0.000979366426149368
    beta = 0.09276191501510327
    beta = reddening_correct(0)*beta
    bg_bri = bg_uw1 - beta*bg_uvv # extrapolated bkg
    bg_bri = 0 # no bkg subtraction
    #bg_bri = 0.000379584 # Borisov
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
    #mask_map = ma.masked_where(vec_data_x<0,mask_map)
    #mask_map = ma.masked_where(vec_data_y>0,mask_map)
    #mask_map = ma.masked_where(vec_data_45<0,mask_map)
    if start != 0:
        mask_map = ma.masked_where(dist_data<=start,mask_map)
    else:
        pass
    mask_map = ma.masked_where(dist_data>start+step*step_num,mask_map)
    mask_map = ma.masked_where(exp_data!=np.max(exp_data),mask_map)
    # get median profile
    image_list = img_data[mask_map.mask==False]
    index_list = index_data[mask_map.mask==False]
    median_list = []
    pixel_mask = []
    med_data = np.zeros(img_data.shape)
    shape_data = np.zeros(img_data.shape)
    for ind in np.arange(np.min(index_list),np.max(index_list)+1):
        median_unit = image_list[index_list==ind]
        med_data[index_data==ind] = np.median(median_unit) #median
    shape_data[mask_map.mask==False] = 1
    shape_data[mask_map.mask==True] = np.nan
    mor_data = (img_data/med_data)*shape_data
    mor_data = mor_data[500:3499,500:3499]
    epoch = img_name.split('_')[0]
    output_path = get_path('../docs/morphology/'+epoch+'.png')
    #plt.imshow(img_data*shape_data,vmin=0,vmax=0.02)
    #plt.imshow(med_data)
    plt.imshow(mor_data,vmin=0.75,vmax=2.7)
    #plt.plot(900,1500,'kx',MarkerSize=2)
    plt.title('Epoch: '+epoch+' ('+str(mid_time)+')')
    #plt.savefig(output_path,dpi=300)
    plt.show()
        #median_list.append(np.median(median_unit))
        #err2sum_list.append(np.sum(err2_unit))
        #pixel_mask.append(len(err2_unit))
    #median_list = np.array(median_list)
    #err_per_pixel_list = np.sqrt(np.array(err2sum_list))/np.array(pixel_mask)*1.2533
    #if len(median_list)<len(aper_list):
    #    aper_list = aper_list[:len(median_list)]
    #    if chatter == 1:
    #        print('Aperture is so large that the blank area is included.')
    #        print('Blank area has been cut.')
    # conversion
    #pix2cm2_factor = np.power(au2km(as2au(scale, delta))*1e5, 2)
    #median_list = median_list/pix2cm2_factor
    #err_per_cm2_list = err_per_pixel_list/pix2cm2_factor
    #aper_list_km = au2km(as2au(aper_list*scale, delta))
    #return aper_list_km, aper_list, median_list, err_per_cm2_list

#epoch_list = [1,2,4,6,8,10,12,14,16,
#              17,18,19,20,21,22,23,24,25,26]
epoch_list = [2,4,6,8,10]


#fits_sub('12_uw1.fits', '12_uvv.fits', '12', 0, 1.055, 0.080, 0.502, 
#         relative_path='stack/', wvm_name = '1_wvm_c2.txt',
#          exp=True, smooth=True, coicorr=True, c2corr=True)


epoch_list = [12]
for epoch in epoch_list:
    epoch = str(epoch)
    print(epoch)
    ext={0:'img',1:'err2',2:'exp'}
    horizon_id = 90000546
    obs_log_name = 'obs_log/'+epoch+'_obs-log_46P.txt'
    delta, rh, mid_time = get_orbit(obs_log_name, horizon_id)
    file_name = epoch+'_sub_red0.fits'
    morphology(file_name, 0.502,
               ext, delta, 
               (2000,2000), 2000,
               False, False,
               0, 4, False, mid_time,
               relative_path='sub/',
               chatter=0,smooth=False,
               coicorr=False,rate_img=True)