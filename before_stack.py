from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.jplhorizons import Horizons
from scipy.ndimage import rotate
from os import listdir
from tools import *
from _mypath import thisdir
import pandas as pd
import numpy as np
import tarfile
import os

def untar(folder_path, file_name):
    """untar zip file"""
    file_path = folder_path + file_name
    tar = tarfile.open(file_path)
    names = tar.getnames()
    for name in names:
        tar.extract(name, folder_path)
    tar.close()

def untar_obs(folder_name):
    folder_path = get_path('../data/'+folder_name+'/')
    obs_list = listdir(folder_path)
    for obs in obs_list:
        if obs[0] != '.':
            if (os.path.isfile(folder_path+obs) == True and 
                os.path.exists(folder_path+obs[:-4]) == False):
                untar(folder_path, obs)
                os.remove(folder_path+obs)

def make_obs_log(horizon_id,
                 folder_name, 
                 map_type, 
                 output_name):
    """creat an observing log file before all work,
    to provide non-data info for every extension.
    
    Inputs: 1. Horizon ID of the object, string;
            2. name of the folder containing dirs of every observation, string;
            3. 'sk' or 'rw' or 'ex', expected to be 'sk', string;
            4. ouput name of the newly generated obs log, string
            
    Outputs: No, only generate an obs log file
    """
    # initiate quantity names. w: write; r: read; v: value
    # make the code easier to be modified or appended
    obs     = {"w": "OBS_ID"}
    ext     = {"w": "EXTENSION"}
    start_t = {"w": "START",
               "r": "DATE-OBS"}
    end_t   = {"w": "END",
               "r": "DATE-END"}
    mid_t   = {"w": "MIDTIME"}
    exp     = {"w": "EXP_TIME",
               "r": "EXPOSURE"}
    fil     = {"w": "FILTER",
               "r": "FILTER"}
    pa      = {"w": "PA",
               "r": "PA_PNT"}    
    helio   = {"w": "HELIO",
               "r": "r"}
    helio_v = {"w": "HELIO_V",
               "r": "r_rate"}
    obs_dis = {"w": "OBS_DIS",
               "r": "delta"}
    phase   = {"w": "PHASE",
               "r": "alpha"}
    ra      = {"w": "RA",
               "r": "RA"}
    dec     = {"w": "DEC",
               "r": "DEC"}
    px      = {"w": "PX"}
    py      = {"w": "PY"}
    
    # initiate a log file
    output_path = get_path('../docs/'+output_name)
    f = open(output_path, 'w')
    #f.write(  obs["w"]     + ' ' 
    #        + ext["w"]     + ' ' 
    #        + start_t["w"] + ' ' 
    #        + end_t["w"]   + ' ' 
    #        + mid_t["w"]   + ' ' 
    #        + exp["w"]     + ' ' 
    #        + fil["w"]     + ' '
    #        + pa["w"]      + ' ' 
    #        + helio["w"]   + ' '
    #        + helio_v["w"] + ' ' 
    #        + obs_dis["w"] + ' ' 
    #        + phase["w"]   + ' '
    #        + ra["w"]      + ' '
    #        + dec["w"]     + ' '
    #        + px["w"]      + ' '
    #        + py["w"]      + '\n'
    #        )
    
    # start a loop iterating over observations
    input_path = get_path('../data/'+folder_name+'/')
    obs_list = os.listdir(input_path)
    obs_list = [obs for obs in obs_list if obs[0] != '.']
    obs_list.sort()
    for obs_id in obs_list:
        obs["v"] = obs_id
        # find the only *sk* file for every obs
        map_dir_path = os.path.join(input_path, 
                                    obs_id+'/uvot/image/')
        map_file_list = os.listdir(map_dir_path)
        map_files = [x for x in map_file_list 
                    if (map_type in x)]
        if len(map_files) == 0:
            break
        else:
            for map_file in map_files:
                map_file_path = os.path.join(map_dir_path, 
                                             map_file)
                # get the num of extensions to
                # start a loop interating over extensions
                hdul = fits.open(map_file_path)
                ext_num = len(hdul) - 1
                for ext_id in range(1, 1 + ext_num):
                    # read observing info from header of every extension
                    # and put the info into the log file
                    ext["v"]     = ext_id
                    ext_header   = hdul[ext_id].header
                    start_t["v"] = Time(ext_header[start_t["r"]])
                    end_t["v"]   = Time(ext_header[end_t["r"]])
                    exp["v"]     = ext_header[exp["r"]]
                    dt           = end_t["v"] - start_t["v"]
                    mid_t["v"]   = start_t["v"] + 1/2*dt
                    fil["v"]     = ext_header[fil["r"]]
                    pa["v"]      = ext_header[pa["r"]]
                    mode = ext_header['DATAMODE']
                    if mode == 'IMAGE':
                        mode = 'image'
                    elif mode == 'EVENT':
                        mode = 'event'
                    #f.write(  obs["v"]           + ' '
                    #        + f'{int(ext["v"])}' + ' '
                    #        + f'{start_t["v"]}'  + ' '
                    #        + f'{end_t["v"]}'    + ' '
                    #        + f'{mid_t["v"]}'    + ' '
                    #        + f'{exp["v"]}'      + ' '
                    #        + fil["v"]           + ' '
                    #        + f'{pa["v"]}'       + ' ')
                    
                    # read ephmerides info from Horizon API
                    # and put the info into the log file
                    obj = Horizons(id=horizon_id, 
                                location='@swift',
                                epochs=mid_t["v"].jd)
                    eph = obj.ephemerides()[0]
                    helio["v"] = eph[helio["r"]]
                    helio_v["v"] = eph[helio_v["r"]]
                    obs_dis["v"] = eph[obs_dis["r"]]
                    phase["v"] = eph[phase["r"]]
                    ra["v"] = eph[ra["r"]]
                    dec["v"] = eph[dec["r"]]
                    elong = eph['elong']
                    #f.write(  f'{helio["v"]}'    + ' '
                    #        + f'{helio_v["v"]}'  + ' '
                    #        + f'{obs_dis["v"]}'  + ' '
                    #        + f'{phase["v"]}'    + ' '
                    #        + f'{ra["v"]}'       + ' '
                    #        + f'{dec["v"]}'      + ' ')
                    w = WCS(ext_header)
                    px["v"], py["v"] = w.wcs_world2pix( ra["v"], dec["v"], 1)
                    #f.write(  f'{px["v"]}'       + ' '
                    #        + f'{py["v"]}'       + '\n')
                    if int(obs["v"])>=94318001 and int(obs["v"])<=94410002:
                        epoch = str(int(1))
                    elif int(obs["v"])>=94421001 and int(obs["v"])<=94430001:
                        epoch=str(int(2))
                    else:
                        epoch=str(int(3))
                    use = '$\\times$'
                    if epoch == '1' or epoch == '2':
                        if fil["v"] == 'V' or fil["v"] == 'UVW1':
                            if mode=='event':
                                use ='$\surd$'
                    else:
                        if fil["v"] == 'V' or fil["v"] == 'UVW1':
                            use ='$\surd$'
                    f.write(obs["v"]+' & '+epoch+' & '+str(int(ext_id))+' & '+mode+' & '+\
                            f'{start_t["v"]}'+' & '+f'{end_t["v"]}'+' & '+\
                            f'{exp["v"]:.1f}'+' & '+f'{ra["v"]:.3f}'+' & '+f'{dec["v"]:.3f}'+' & '+\
                            f'{helio["v"]:.3f}'+' & '+f'{obs_dis["v"]:.3f}'+' & '+f'{elong:.3f}'+' & '+\
                            fil["v"]+' \\\\'+'\n')
    f.close()


def remove_refl():
    raw_path = get_path('../data/46P_raw_uvot/00094319001/uvot/image/sw00094319001uw1_rw.img.gz')
    raw = fits.open(raw_path)
    refl_path = get_path('../data/donuts/w1.fits')
    refl = fits.open(refl_path)
    output_path = get_path('../docs/refl/00094319001/uvot/image/sw00094319001uw1_rw.img.gz')
    os.system('rm '+output_path)
    raw[1].data = refl[1].data
    raw.writeto(output_path)
    #os.system('mv '+output_path+' '+output_path[:-3]+'.fits')

#remove_refl()

def set_coord(image_array, target_index,
              size):
    """To shift a target on an image 
    into the center of a new image;
    
    The size of the new image can be given 
    but have to ensure the whole original
    image is included in the new one.
    
    Inputs: array of an original image, 2D array
            original coordinate values of the target, array shape in [r, c]
            output size, tuple of 2 elements
    Outputs: array of the shifted image in the new coordinate, 2D array
    """
    # interpret the size and create new image
    try:
        half_row, half_col = size
    except:
        print("Check the given image size!")
    new_coord = np.zeros((2*half_row - 1, 
                          2*half_col - 1))
    # shift the image, [target] -> [center]
    def shift_r(r):
        return int(r+(half_row-1- target_index[0]))#.astype(int)
    def shift_c(c):
        return int(c+(half_col-1- target_index[1]))#.astype(int)
    for r in range(image_array.shape[0]):
        for c in range(image_array.shape[1]):
            new_coord[shift_r(r), 
                      shift_c(c)] = image_array[r, c]
    #r = range(image_array.shape[0])
    #c = range(image_array.shape[1])
    #from itertools import product
    #g = np.array(list(product(r,c)))
    #new_coord[shift_r(g[:,0]),shift_c(g[:,1])] = image_array[g[:,0],g[:,1]]
    # reture new image
    return new_coord

def rescale(inp,src_center,r=999):
    # upsample
    outp = np.zeros(inp.shape)
    if src_center[0] == src_center[1]:
        cen_p = src_center[0] - 1
    else:
        raise('check src_center')
    low = cen_p - r
    up = cen_p + r
    def new_pos(i,c):
        return (i-c)*2+c
    low_new = new_pos(low,cen_p)
    up_new = new_pos(up,cen_p)
    outp[low_new:up_new+1:2,low_new:up_new+1:2]=\
        inp[low:up+1,low:up+1]
    # interpolate
    outp_i = outp[low_new:up_new+1:2]
    outp[low_new+1:up_new:2]=(outp_i[:2*r]+outp_i[1:])/2
    outp_j = outp[:,low_new:up_new+1:2]
    outp[:,low_new+1:up_new:2]=(outp_j[:,:2*r]+outp_j[:,1:])/2
    return outp

def rescale_fits(img_name,output_name,src_center):
    img_path = get_path('../docs/'+img_name)
    img_hdul = fits.open(img_path)
    rescale_img = rescale(img_hdul[0].data,src_center)/4
    rescale_exp = rescale(img_hdul[1].data,src_center)
    hdr = img_hdul[0].header
    hdr['PLATESCL'] = ('0.502','arcsec/pixel')
    hdu_img = fits.PrimaryHDU(rescale_img,header=hdr)
    hdu_exp = fits.ImageHDU(rescale_exp)
    hdul = fits.HDUList([hdu_img,hdu_exp])
    output_path = get_path('../docs/stack/'+output_name)
    hdul.writeto(output_path)
    os.system('mv '+output_path+' '+output_path[:-3]+'.fits')
'''
#epoch = ['5','7','9','11','18','20','22','24','26']
epoch = ['13','15']
for i in epoch:
    rescale_fits('stack/'+i+'_img_scale1_uw1.fits',i+'_uw1.gz',(2000,2000))
    rescale_fits('stack/'+i+'_img_scale1_uvv.fits',i+'_uvv.gz',(2000,2000))
'''
def stack_image(obs_log_name, filt, size, output_name,rescale_list=False):
    '''sum obs images according to 'FILTER'
    
    Inputs:
    obs_log_name: the name of an obs log in docs/
    filt: 'uvv' or 'uw1' or 'uw2'
    size: a tuple
    output_name: string, to be saved in docs/
    
    Outputs:
    1) a txt data file saved in docs/
    2) a fits file saved in docs/
    '''
    # load obs_log in DataFrame according to filt
    print(obs_log_name)
    obs_log_path = get_path('../docs/obs_log/'+obs_log_name)
    img_set = pd.read_csv(obs_log_path, sep=' ',
                          index_col=['FILTER'])
    img_set = img_set[['OBS_ID', 'EXTENSION',
                       'PX', 'PY', 'PA', 'EXP_TIME',
                       'END', 'START']]
    if filt == 'uvv':
        img_set = img_set.loc['V']
    elif filt == 'uw1':
        img_set = img_set.loc['UVW1']
    elif filt == 'uw2':
        img_set = img_set.loc['UVW2']
    elif filt == 'um2':
        img_set = img_set.loc['UVM2']
    elif filt == 'uuu':
        img_set = img_set.loc['U']
    #---transfer OBS_ID from int to string---
    img_set['OBS_ID']=img_set['OBS_ID'].astype(str)
    img_set['OBS_ID']='000'+img_set['OBS_ID']
    # create a blank canvas in new coordinate
    stacked_img = np.zeros((2*size[0] -1,
                            2*size[1] -1))
    stacked_exp = np.zeros((2*size[0] -1,
                            2*size[1] -1))
    # loop among the data set, for every image, shift it to center the target, rotate and add to the blank canvas
    exp = 0
    for i in range(len(img_set)):
        #print(i)
        #---get data from .img.gz---
        if img_set.index.name == 'FILTER':
            img_now = img_set.iloc[i]
            exp += img_now['EXP_TIME']
            if_break = False
        else:
            img_now = img_set
            exp = img_now['EXP_TIME']
            if_break = True
        print(img_now['OBS_ID'])
        img_path = get_path(img_now['OBS_ID'], 
                            filt, to_file=True)
        exp_path = get_path(img_now['OBS_ID'],
                            filt, to_file=True, map_type='ex')
        img_hdu = fits.open(img_path)[img_now['EXTENSION']]
        exp_hdu = fits.open(exp_path)[img_now['EXTENSION']]
        img_data = img_hdu.data.T # .T! or else hdul PXY != DS9 PXY
        exp_data = exp_hdu.data.T
        #---shift the image to center the target---
        new_img = set_coord(img_data, 
                            np.array([img_now['PX']-1,
                                      img_now['PY']-1]),
                            size)
        new_exp = set_coord(exp_data,
                            np.array([img_now['PX']-1,
                                      img_now['PY']-1]),
                            size)
        #---rotate the image according to PA to
        #---eliminate changes of pointing
        #---this rotating step may be skipped---
        #new_img = rotate(new_img, 
        #                 angle=img_now['PA'],
        #                 reshape=False,
        #                 order=1)
        #---sum modified images to the blank canvas---
        if rescale_list==False:
            stacked_img = stacked_img + new_img
            stacked_exp = stacked_exp + new_exp
        elif i in rescale_list:
            stacked_img = stacked_img + rescale(new_img,size)/4
            stacked_exp = stacked_exp + rescale(new_exp,size)
        elif i not in rescale_list:
            stacked_img = stacked_img + new_img
            stacked_exp = stacked_exp + new_exp
        if if_break:
            break
    #stacked_exp = stacked_exp/np.max(stacked_exp)
    #stacked_exp = np.where((stacked_exp>0)*(stacked_exp<1),0.5,stacked_exp)
    # get the summed results and save in fits file
    output_path = get_path('../docs/stack/'+output_name)#+
                          #'_'+filt+'.fits')
    if img_set.index.name == 'FILTER':
        dt = Time(img_set.iloc[-1]['END']) - Time(img_set.iloc[0]['START'])
        mid_t = Time(img_set.iloc[0]['START']) + 1/2*dt
    else:
        dt = Time(img_set['END']) - Time(img_set['START'])
        mid_t = Time(img_set['START']) + 1/2*dt
    hdr = fits.Header()
    hdr['TELESCOP'] = img_hdu.header['TELESCOP']
    hdr['INSTRUME'] = img_hdu.header['INSTRUME']
    hdr['FILTER'] = img_hdu.header['FILTER']
    hdr['COMET'] = obs_log_name.split('_')[0]+' '+obs_log_name.split('_')[-1][:-4]
    hdr['PLATESCL'] = ('0.502','arcsec/pixel')#1.004
    hdr['XPOS'] = f'{size[0]}'
    hdr['YPOS'] = f'{size[1]}'
    hdr['EXPTIME'] = (f'{exp}', '[seconds]')
    hdr['MID_TIME'] = f'{mid_t}'
    hdu_img = fits.PrimaryHDU(stacked_img,header=hdr)
    hdu_exp = fits.ImageHDU(stacked_exp)
    hdul = fits.HDUList([hdu_img,hdu_exp])
    hdul.writeto(output_path)
    os.system('mv '+output_path+' '+output_path[:-3])

def stack_image_simple(img_name_list, size, output_name, midtime):
    stacked_img = np.zeros((2*size[0] -1,
                            2*size[1] -1))
    stacked_exp = np.zeros((2*size[0] -1,
                            2*size[1] -1))
    exp = 0
    time_list = []
    for img_name in img_name_list:
        img_path = get_path('../docs/stack/'+img_name)#smear
        img_hdu = fits.open(img_path)
        stacked_img += img_hdu[0].data
        stacked_exp += img_hdu[1].data
        exp += float(img_hdu[0].header['EXPTIME'])
    hdr = fits.Header()
    hdr['TELESCOP'] = 'SWIFT'
    hdr['INSTRUME'] = 'UVOT'
    hdr['FILTER'] = 'V'
    hdr['COMET'] = '46P'
    hdr['PLATESCL'] = ('0.502','arcsec/pixel')
    hdr['XPOS'] = '2000'
    hdr['YPOS'] = '2000'
    hdr['MID_TIME'] = midtime
    hdr['EXPTIME'] = (f'{exp}', '[seconds]')
    hdu_img = fits.PrimaryHDU(stacked_img,header=hdr)
    hdu_exp = fits.ImageHDU(stacked_exp)
    hdul = fits.HDUList([hdu_img,hdu_exp])
    output_path = get_path('../docs/stack/'+output_name)
    hdul.writeto(output_path)
    os.system('mv '+output_path+' '+output_path[:-3])

def read_midtime(img_name):
    img_path = get_path('../docs/stack/'+img_name)
    img = fits.open(img_path)
    return img[0].header['MID_TIME']


#stack_image('141p_obs-log_1.txt', 'uw1', (2000,2000),  '1_uw1.fits.gz')
#stack_image_simple(['00094421002_uvv_19.fits', '00094425002_uvv_18.fits', '00094429002_uvv_16.fits'], 
#                   (2000,2000), 'epoch2_stack_uvv.fits.gz', '2018-12-13T12:31:05.000')
#stack_image_simple(['00094405003_uvv.fits','00094406005_uvv.fits'], 
#                   (2000,2000), '10_uvv.fits.gz', read_midtime('10_uvv_img.fits'))
'''
stack_image('17'+'_obs-log_46P.txt', 'uw1', (2000,2000), '17'+'_uw1.fits.gz',[0])
stack_image('19'+'_obs-log_46P.txt', 'uw1', (2000,2000), '19'+'_uw1.fits.gz',[0])
stack_image('21'+'_obs-log_46P.txt', 'uw1', (2000,2000), '21'+'_uw1.fits.gz',[0])
stack_image('23'+'_obs-log_46P.txt', 'uw1', (2000,2000), '23'+'_uw1.fits.gz',[0])
'''
'''
def stack46p():
    #large = [1,2,4,6,8,10,12,14,16,17,19,21,23,25]
    #small = [3,5,7,9,11,13,15,18,20,22,24,26]
    small = [13,15]#18,20,22,24,26]
    large = [25]
    
    uw2 = [10,11,12,13,14,15,16]
    um2 = [1]
    uuu = [1,2,3,4,5,6,7,8,9,10,11,24,25,26]
    
    epoch1 = [1,2,3,4,5,6,7,8,9,10,11]
    epoch2 = [12,13,14,15,16]
    epoch3 = [17,18,19,20,21,22,23,24,25,26]
    

    for i in small:
        ind = str(i)
        print(i)
        #stack_image(ind+'_obs-log_46P.txt', 'uvv', (2000,2000), ind+'_uvv.fits.gz')
        #stack_image(ind+'_obs-log_46P.txt', 'uw1', (2000,2000), ind+'_uw1.fits.gz')
        if i in uw2:
            stack_image(ind+'_obs-log_46P.txt', 'uw2', (2000,2000), ind+'_uw2_img_scale1.fits.gz')
        if i in um2:
            stack_image(ind+'_obs-log_46P.txt', 'um2', (2000,2000), ind+'_um2_img_scale1.fits.gz')
        if i in uuu:
            stack_image(ind+'_obs-log_46P.txt', 'uuu', (2000,2000), ind+'_uuu_img_scale1.fits.gz')
    
    #epoch1 = [1,2,3,4,5,6,7,8,9,10,11]
    #print('hi')
    #stack_image('epoch1_obs-log_large_46P.txt', 'uuu', (2000,2000), 'epoch1_0.5_stack_uuu.fits.gz')
    #stack_image('epoch1_obs-log_large_46P.txt', 'uw2', (2000,2000), 'epoch1_0.5_stack_uw2.fits.gz')
    #print('hihi')
    #stack_image('epoch3_obs-log_large_46P.txt', 'uuu', (2000,2000), 'epoch3_0.5_stack_uuu.fits.gz')
    #stack_image('epoch3_obs-log_large_46P.txt', 'uw2', (2000,2000), 'epoch3_0.5_stack_uw2.fits.gz')
    
stack46p()
'''

#stack_image('dec'+'_obs-log_Garradd.txt', 'uvv', (2000,2000), 'garradd_dec'+'_uvv.fits.gz',[0])
#stack_image('dec'+'_obs-log_Garradd.txt', 'uw1', (2000,2000), 'garradd_dec'+'_uw1.fits.gz',[0])
'''
filt='uw1'

img_name_list = [1,2,4,6,8,10]
midtime = '2018-11-29T16:11:43.500'
output_name = 'nov_'+filt+'.fits.gz'

img_name_list = [12,14,16]
midtime = '2018-12-13T12:32:22.500'
output_name = 'dec_'+filt+'.fits.gz'

img_name_list = [17,18,19,20,21,22,23,24,25,26]
output_name = 'jan_'+filt+'.fits.gz'
midtime = '2019-01-12T20:29:05.500'

img_name_list = [str(int(i))+'_'+filt+'.fits' for i in img_name_list]

stack_image_simple(img_name_list, (2000,2000), output_name, midtime)
'''