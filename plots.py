from tools import *
from aper_phot import *
from before_stack import *
from conversion import *
from profile_q import *
#from cr_to_flux import *
import matplotlib.pyplot as plt
from itertools import combinations
import os

def mask_map(img_name, scale,
             ext, delta, 
             src_center, src_r,
             bg_center, bg_r,
             start, step, mask_img,
             relative_path='',
             chatter=0):
    step_num = math.floor((src_r-start)/step)
    img_data, cen_pix, i_range, j_range = \
        limit_loop(img_name, src_center, start+step*step_num, relative_path=relative_path, ext=ext)
    if (bg_center != False) and (bg_r != False):
        bg_bri = reg2bg_bri(img_name, 'multi',
                            bg_center, bg_r,
                            'mean', mask_img,
                            relative_path=relative_path)[0]
    if isinstance(ext,int) == True:
        img_data = img_data - bg_bri
        exp_data = np.ones(img_data.shape)
        err2_data = np.zeros(np.shape(img_data))
        bg_bri_err = 0
        data_keys = ['img']
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
    if isinstance(mask_img, bool):
        if mask_img == False:
            mask_img = np.ones(img_data.shape)
    import numpy.ma as ma
    pos_y, pos_x = np.where(img_data)
    dist_data = np.sqrt(np.power(pos_y-src_center[0],2)+np.power(pos_x-src_center[1],2))
    dist_data = dist_data.reshape(img_data.shape)
    mask_map = ma.masked_where(mask_img==0,img_data)
    mask_map = ma.masked_where(dist_data<=start,mask_map)
    mask_map = ma.masked_where(dist_data>=start+step*step_num,mask_map)
    mask_map = ma.masked_where(exp_data!=np.max(exp_data),mask_map)
    mask_y, mask_x = np.where(mask_map.mask)
    # 
    plt.imshow(img_data)
    plt.plot(mask_x,mask_y,'ws',MarkerSize=1,alpha=0.7)
    plt.show()
    #data_group = data.groupby('index')
'''
bg = load_reg_list('0_bg.reg','reg/')
bg_center = list(zip(bg[1], bg[0]))
bg_r = bg[2]
mask_map('0_uw1.fits', 0.502,
         {0:'img',1:'exp'}, 0.13328992265821, 
         (2000,2000), 2060,
         bg_center, bg_r,
         6, 10, False,
         relative_path='stack/',
         chatter=0)

bg = load_reg_list('1_bg.reg','reg/')
bg_center = list(zip(bg[1], bg[0]))
bg_r = bg[2]
sur_bri_profile('1_uw1.fits', 0.502,
                {0:'img',1:'exp'}, 0.13328992265821, 
                (2000,2000), 2060,
                bg_center, bg_r,
                6, 10, False,
                relative_path='stack/',
                chatter=0)
'''
def oh2uw1(uw1_name,oh_name,bg_reg_name,
           src_center,src_r,delta,scale):
    bg = load_reg_list(bg_reg_name,'reg/')
    bg_center = list(zip(bg[1], bg[0]))
    bg_r = bg[2]
    print('hi')
    aper_list, uw1_list, uw1_err_list = \
        sur_bri_profile(uw1_name, scale,
                        {0:'img',1:'exp'}, delta, 
                        src_center, src_r,
                        bg_center, bg_r,
                        6, 2, False,
                        relative_path='stack/',
                        chatter=0)
    uw1_exp = float(load_header(uw1_name,relative_path='stack/')['EXPTIME'])
    uw1_list = uw1_list/uw1_exp
    print('hihi')
    aper_list, oh_list, oh_err_list = \
        sur_bri_profile(oh_name, scale,
                        {0:'img',1:'err2',2:'exp'}, delta, 
                        src_center, src_r,
                        bg_center, bg_r,
                        6, 2, False,
                        relative_path='sub/',
                        chatter=0)
    print('hihihi')
    short = min(len(oh_list),len(uw1_list))
    ratio_list = oh_list[:short]/uw1_list[:short]
    #plt.plot(aper_list[:short],ratio_list*100,label='(uw1-b*v)/uw1')
    plt.plot(aper_list[:short],oh_list[:short]/np.max(oh_list[:short]),label='oh')
    plt.plot(aper_list[:short],uw1_list[:short]/np.max(uw1_list[:short]),label='uw1')
    #plt.ylabel('(uw1-b*v)/uw1 (%)')
    plt.xlabel('km')
    plt.legend()
    plt.title('epoch:1')
    plt.show()


#oh2uw1('1_uw1.fits','1_sub_red0.fits','1_bg.reg',
#       (2000,2000),2060,0.13328992265821,0.502)

def compBorisov():
    horizon_id = 90004430#90000545
    img_name_uw1 = '0_uw1.fits'
    src_center = (1000, 1000)
    bg_reg_name = '0_bg.reg'
    mask_name = '0_mask.reg'
    wvm_name = '0_wvm.txt'
    phase_name = 'phase_correction.txt'
    ext={0:'img',1:'err2'}#,2:'exp'}
    src_r_uw1 = 67
    z = 6.278E+16
    bg = load_reg_list('0_bg.reg','reg/')
    bg_center = list(zip(bg[1], bg[0]))
    bg_r = bg[2]
    ifmask = True
    mask_img = mask_region(img_name_uw1, mask_name,'stack/')
    start = 6
    step = 2

    #red, if_fit = howRed(chatter=2)
    red = 14
    if_fit = False
    result_uw1 = aper_phot(img_name_uw1, 'uw1', 
                           src_center, src_r_uw1,
                           bg_center, bg_r,
                           'azim_median', 'multi_mean',
                           2, mask_img, 0., 'stack/')
    print(result_uw1)
    
    from aper_phot_borisov import aper_phot_bori
    result_uw1_bori = aper_phot_bori('stack/'+img_name_uw1, 'uw1',
                                     src_center, src_r_uw1,
                                     bg_center, bg_r,
                                     'azim_median', 'multi_mean',
                                     step=2, mask_img = mask_img, start = 0.)
    print(result_uw1_bori)
    


compBorisov()
