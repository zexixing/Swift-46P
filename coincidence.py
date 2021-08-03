from profile_q import sur_bri_profile
from aper_phot import coincidenceCorr
from astropy.io import fits
from remove_bkg import zoomImg
import matplotlib.pyplot as plt
from tools import *
from remove_smear import *
from pathlib import Path
from astropy.time import Time
from scipy.optimize import curve_fit
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
import stsci.imagestats as imagestats

def filter_conv(filt):
    filt_dict = {'V':'uvv', 'U':'uuu', 'UVW1':'uw1', 'UVW2':'uw2', 'UVM2':'um2'}
    return filt_dict[filt]

def obtainRadius(radius_km, delta, scale):
    arcs = (radius_km*3600*360)/(149597870.7*2*np.pi*delta)
    return int(arcs/scale)

# read obsid list
def read_obslist():
    pass

def save_evt_slice(obsid,filt,epoch):
    id_horizons = 90000548#90000546
    size = 2000
    stacked_img = np.zeros((2*size -1,
                            2*size -1))
    stacked_exp = np.zeros((2*size -1,
                            2*size -1))
    hdulist_img = []
    hdulist_exp = []
    time_list, x_list, y_list, time_array, x_comet, y_comet, exp_data, x_comet_exp, y_comet_exp, num, exp, hd = \
        stack_info(obsid, filt, id_horizons)
    for i in range(0,num):
        print(i+1)
        slice_img = sliceImg(time_list, x_list, y_list, x_comet, y_comet, time_array, i, size, filt, num, exp)
        slice_exp = sliceExp(exp_data,x_comet_exp,y_comet_exp,size,num,i)
        stacked_img += slice_img
        stacked_exp += slice_exp
        hdu_slice_img = fits.ImageHDU(slice_img)
        hdu_slice_exp = fits.ImageHDU(slice_exp)
        hdulist_img.append(hdu_slice_img)
        hdulist_exp.append(hdu_slice_exp)
    output_name_img = '16_'+obsid+'_'+filt+'_img.fits.gz'#'_'+str(int(num))+'.fits.gz'
    output_name_exp = '16_'+obsid+'_'+filt+'_exp.fits.gz'#'_'+str(int(num))+'.fits.gz'
    output_path_img = get_path('../docs/evt_slice/'+output_name_img)
    output_path_exp = get_path('../docs/evt_slice/'+output_name_exp)
    hdr = fits.Header()
    dt = Time(hd['DATE-OBS']) - Time(hd['DATE-END'])
    mid_t = Time(hd['DATE-OBS']) + 1/2*dt
    hdr['TELESCOP'] = 'Swift'
    hdr['INSTRUME'] = 'UVOT'
    hdr['FILTER'] = filt
    hdr['OBSID'] = obsid
    hdr['PLATESCL'] = ('0.502','arcsec/pixel')
    hdr['XPOS'] = f'{size}'
    hdr['YPOS'] = f'{size}'
    hdr['EXPTIME'] = (f'{exp}', '[seconds]')
    hdr['MID_TIME'] = f'{mid_t}'
    hdr['EPOCH'] = epoch
    hdr['EXTE_NUM'] = f'{num}'
    hdu_img = fits.PrimaryHDU(stacked_img,header=hdr)
    hdu_exp = fits.PrimaryHDU(stacked_exp,header=hdr)
    hdul_img = fits.HDUList([hdu_img]+hdulist_img)
    hdul_exp = fits.HDUList([hdu_exp]+hdulist_exp)
    hdul_img.writeto(output_path_img)
    hdul_exp.writeto(output_path_exp)
    os.system('mv '+output_path_img+' '+output_path_img[:-3])
    os.system('mv '+output_path_exp+' '+output_path_exp[:-3])
'''
'00094318002','uvv',1
'00094319001','uw1',1
'00094319004','uvv',1
'00094320002','uw1',1
'00094381002','uw1',2
'00094381003','uvv',2
'00094382004','uw1',2
'00094382005','uvv',2
'00094387002','uw1',4
'00094387003','uvv',4
'00094388004','uw1',4
'00094388005','uvv',4
'00094393002','uw1',6
'00094393003','uvv',6
'00094394004','uw1',6
'00094394005','uvv',6
'00094399002','uw1',8
'00094399003','uvv',8
'00094400004','uw1',8
'00094400005','uvv',8
'00094405002','uw1',10
'00094405003','uvv',10
'00094406004','uw1',10
'00094406005','uvv',10
'''
#save_evt_slice('00094421002','uvv',12)    
#save_evt_slice('00094422001','uw1',12)
#save_evt_slice('00094425002','uvv',14)
#save_evt_slice('00094426001','uw1',14)
#save_evt_slice('00094429002','uvv',16)
#save_evt_slice('00094430001','uw1',16)
    
def stack_slice_part():
    size = 2000
    stacked_img = np.zeros((2*size -1,
                            2*size -1))
    stacked_exp = np.zeros((2*size -1,
                            2*size -1))
    hdul_img = fits.open(get_path('../docs/evt_slice/00094421002_uw1_img.fits'))
    hdul_exp = fits.open(get_path('../docs/evt_slice/00094421002_uw1_exp.fits'))
    for i in range(1,17):
        print(i)
        stacked_img += hdul_img[i].data
        stacked_exp += hdul_exp[i].data
    hdr = hdul_img[0].header
    exp = float(hdr['EXPTIME'])*4./6.
    hdr['EXPTIME'] = (f'{exp}', '[seconds]')
    hdu_img = fits.PrimaryHDU(stacked_img,header=hdr)
    hdu_exp = fits.ImageHDU(stacked_exp)
    hdul = fits.HDUList([hdu_img,hdu_exp])
    output_path = get_path('../docs/evt_slice/12_uw1_stack_img.fits.gz')
    hdul.writeto(output_path)
    os.system('mv '+output_path+' '+output_path[:-3])

#stack_slice_part()

# read info for every obs (scale, pos, ifSmooth)
def read_extension(epoch,obsid,ext):
    # read in obs-log
    obs_log_name = epoch+'_obs-log_46P.txt'
    obs_log_path = get_path('../docs/obs_log/'+obs_log_name)
    img_set = pd.read_csv(obs_log_path, sep=' ')
    img_set = img_set[['OBS_ID', 'EXTENSION','FILTER',
                       'PX', 'PY', 'PA', 'EXP_TIME','OBS_DIS',
                       'END', 'START']]
    img_set['OBS_ID']=img_set['OBS_ID'].astype(str)
    img_set['OBS_ID']='000'+img_set['OBS_ID']
    ext_series = img_set[(img_set['OBS_ID']==obsid)&(img_set['EXTENSION']==ext)].squeeze()
    filt = filter_conv(ext_series['FILTER'])
    ext_series['FILTER'] = filt
    # read in data and its features
    obs_path = '../data/46P_raw_uvot/'+obsid+'/uvot/'
    evt_path = Path(get_path(obs_path)+'event')
    if evt_path.exists() and int(epoch)<17:
        name_img = epoch+'_'+obsid+'_'+filt+'_img.fits'
        name_exp = epoch+'_'+obsid+'_'+filt+'_exp.fits'
        path_img = get_path('../docs/evt_slice/'+name_img)
        path_exp = get_path('../docs/evt_slice/'+name_exp)
        img = fits.open(path_img)
        exp = fits.open(path_exp)
        ext_series['data'] = img[ext].data
        ext_series['data_exp'] = exp[ext].data
        ext_series['scale'] = 0.502
        ext_series['type'] = 'event'
        ext_series['PX'] = 2000
        ext_series['PY'] = 2000
    else:
        data_path = get_path(obs_path+'image/sw'+obsid+filt+'_sk.img.gz')
        ext_series['data'] = fits.open(data_path)[ext].data
        exp_path = get_path(obs_path+'image/sw'+obsid+filt+'_ex.img.gz')
        ext_series['data_exp'] = fits.open(exp_path)[1].data
        ext_series['type'] = 'image'
        if evt_path.exists():
            ext_series['scale'] = 0.502
        else:
            ext_series['scale'] = 1.004
    return ext_series

#read_extension('12','00094422001',1)

# stack with coincidenceCorr
# ~TODO: img created by event: there is only 0 or 1 in a single 
#        slice -> cannot apply coi corr because of too much noise
#        ... get coi-factor from every smoothed slice?
# get spatial profile (cts/s/pixel, after zoom)

def check_donut(obsid,ext):
    #donut = fits.open('/Users/zexixing/Research/swift46P/data/donuts/mod8_uvv.fits')
    #data = donut[1].data
    path = get_path('../docs/donut/'+obsid+'_donut_sk.fits')
    data = fits.open(path)[ext].data
    evt_path = Path(get_path('../data/46P_raw_uvot/'+obsid+'/uvot/event'))
    if not evt_path.exists():
        data = zoomImg(data,'larger')
    imgstats = imagestats.ImageStats(data[data!=0],nclip=0)
    mean = imgstats.mean
    sig = imgstats.stddev
    bkg = 0.0037
    data = (data/mean)*bkg
    data[data==0] = np.nan
    result = sur_bri_profile(0.502,'donut',0,
                             1, 1,
                             (1024,1024), 1447,
                             False, False,
                             0, 5, False,
                             coicorr=False,rate_img=True, smooth=False,
                             img_name=data) # can be any: epoch, red, ext=int
    r_pixel = result[1]
    crppixel = result[4]
    #plt.plot(r_pixel,crppixel)
    #plt.show()
    return r_pixel, crppixel

#check_donut()

def donut_rw(obsid, filt):
    donut = fits.open('/Users/zexixing/Research/swift46P/data/donuts/mod8_'+filt+'.fits')
    data = donut[1].data
    # read in data and its features
    obs_path = '../data/46P_raw_uvot/'+obsid+'/uvot/'
    evt_path = Path(get_path(obs_path)+'event')
    if not evt_path.exists():
        data = zoomImg(data,'smaller')
    if evt_path.exists() and int(epoch)<17:
        pass
    else: # Jan: only write for images instead of events
        data_path = get_path(obs_path+'image/sw'+obsid+filt+'_rw.img.gz')
        hdul = fits.open(data_path)
        n = len(hdul)
        hdr0 = hdul[0].header
        hdu_img = fits.PrimaryHDU(hdul[0].data,header=hdr0)
        hdulist = [hdu_img]
        for ext in range(1,n):
            hdu_ext = fits.ImageHDU(data,header=hdul[ext].header)
            hdulist += [hdu_ext]
        hdul_output = fits.HDUList(hdulist)
        output_path = get_path('../docs/donut/'+obsid+'_'+filt+'_donut_rw.fits.gz')
        hdul_output.writeto(output_path)
        os.system('mv '+output_path+' '+output_path[:-3])

#donut_rw('00094463002', 'uvv')

def donut_rw2sky(obsid, filt):
    infile = get_path('../docs/donut/'+obsid+'_'+filt+'_donut_rw.fits')
    #infile = get_path('../data/46P_raw_uvot/'+obsid+'/uvot/image/sw'+obsid+filt+'_rw.img.gz')
    outfile = get_path('../docs/donut/'+obsid+'_'+filt+'_donut_sk.fits.gz')
    attfile = get_path('../data/46P_raw_uvot/'+obsid+'/auxil/sw'+obsid+'sat.fits.gz')
    hdr = fits.open(infile)[0].header
    ra = hdr['RA_PNT']
    dec = hdr['DEC_PNT']
    roll = hdr['PA_PNT']
    os.system('swiftxform infile='+infile+' outfile='+outfile+\
              ' to=SKY attfile='+attfile+' teldeffile=caldb method=AREA bitpix=-32'+\
              ' ra='+str(ra)+' dec='+str(dec)+' roll='+str(roll))
    os.system('mv '+outfile+' '+outfile[:-3])

#donut_rw2sky('00094463002', 'uvv')

def rm_donut(obsid,ext):
    path = get_path('../docs/donut/'+obsid+'_donut_sk.fits')
    data = fits.open(path)[ext].data
    evt_path = Path(get_path('../data/46P_raw_uvot/'+obsid+'/uvot/event'))
    if not evt_path.exists():
        data = zoomImg(data,'larger')
    imgstats = imagestats.ImageStats(data[data!=0],nclip=0)
    mean = imgstats.mean
    sig = imgstats.stddev
    return data/mean

def read_rate(epoch, obsid, ext):
    ext_dict = read_extension(epoch,obsid,ext)
    data = ext_dict['data']
    data_exp = ext_dict['data_exp']
    scale = ext_dict['scale']
    img_type = ext_dict['type']
    #---log---
    px = ext_dict['PX']
    py = ext_dict['PY']
    delta = ext_dict['OBS_DIS']
    # profile
    src_center = (px, py)
    data_exp = data_exp.astype('float')
    data_exp[np.where(data_exp==0.)]=np.nan
    rate = data/data_exp
    if scale == 1.004:
        rate = zoomImg(rate,'larger')
        scale = 0.502
        src_center = (px*2, py*2)
    smooth=False
    if float(epoch)<17:
        smooth = True
    return rate, scale, src_center, smooth, delta

def get_crprofile_for_ext(epoch, obsid, ext, ifdonut=False, ifcoi=True):
    # read data
    rate, scale, src_center, smooth, delta = read_rate(epoch, obsid, ext)
    if ifcoi == True:
        coicorr = coincidenceCorr(rate, scale)
        rate = rate*coicorr
    donut = rm_donut(obsid,ext)
    if ifdonut == True:
        bkg = 0.0037
        #rate = rate/donut[:,1:-1] - bkg
        rate = rate - bkg*donut[:,1:-1]
        #plt.imshow(donut[:,1:-1],vmin=0.8,vmax=1.3)
        #plt.show()
    plt.imshow(rate,vmin=0,vmax=0.05)
    theta = np.arange(0,2*np.pi,0.02)
    r_peak = np.array([640,1040])
    plt.plot(np.outer(np.cos(theta),r_peak)+src_center[0]-1,
                      np.outer(np.sin(theta),r_peak)+src_center[1]-1,color='y')
    plt.show()
    src_r = obtainRadius(100000, delta, scale)
    re_coi = sur_bri_profile(scale,epoch,0,
                             ext, delta,
                             src_center, src_r,
                             False, False,
                             0, 5, False,
                             coicorr=False,rate_img=True, smooth=smooth,
                             img_name=rate) # can be any: epoch, red, ext=int
    r_pixel = re_coi[1]
    crppixel_coi = re_coi[4]
    #result = sur_bri_profile(scale,epoch,0,
    #                         ext, delta,
    #                         src_center, src_r,
    #                         False, False,
    #                         0, 5, False,
    #                         coicorr=False,rate_img=True, smooth=smooth,
    #                         img_name=rate) # can be any: epoch, red, ext=int
    #crppixel = result[4]
    #plt.plot(np.log(r_pixel),np.log(crppixel),'b-')
    #plt.plot(np.log(r_pixel),np.log(crppixel_coi),'r-')
    #plt.show()
    return r_pixel, crppixel_coi


#get_crprofile_for_ext('14', '00094425002', 1)
#get_crprofile_for_ext('26', '00094463002', 1)
#get_crprofile_for_ext('25', '00094461002', 1)

gauss_1D_kernel = Gaussian1DKernel(2.5)

def func(x, a, b, c):
    #return convolve(a*np.power(x,-b)+c,gauss_1D_kernel)
    return a*np.power(x,-b)+c

def func2(x, a, b):
    return a-b*x

def fitting(aper_list, signal_list, method='linear'):
    if method == 'linear':
        f = func
        initial = np.array([2,1, 0.0])
        ranges = (np.array([-np.inf,1,-np.inf]),
                  np.array([np.inf,1.3,np.inf]))
    elif method == 'log':
        f = func2
        initial = np.array([0.7,1])
        ranges = (np.array([-np.inf,-np.inf]),
                  np.array([np.inf,np.inf]))
    else:
        raise ValueError('Check the method!')
    popt, pcov = curve_fit(f, aper_list, signal_list, 
                           p0=initial,
                           bounds=ranges)
    #bkg = func(src_r, *popt)
    return popt

# fitting the outer part of spatial profile (what parameters?)
def fit_outer(epoch, obsid, ext, drop_min, drop_max_num, ifdonut=True, method='linear'):
    # a, b, c, ..., d, drop_min, ..., e, drop_max, f, g, ...
    # get [drop_min, ..., e]
    r_pixel, crppixel_coi = get_crprofile_for_ext(epoch, obsid, ext, ifdonut=ifdonut)
    if not drop_min:
        index_list = np.arange(0, len(crppixel_coi))
        drop_min = np.min(index_list[crppixel_coi<=0.01])
    else: pass
    drop_max = len(r_pixel)-drop_max_num
    r_pixel_fit = r_pixel[drop_min:drop_max]
    crppixel_coi_fit = crppixel_coi[drop_min:drop_max]
    crppixel_coi_fit = np.append(crppixel_coi_fit[(r_pixel_fit<640)],crppixel_coi_fit[r_pixel_fit>1040])
    r_pixel_fit = np.append(r_pixel_fit[(r_pixel_fit<640)],r_pixel_fit[r_pixel_fit>1040])
    if method == 'linear':
        popt = fitting(r_pixel_fit, crppixel_coi_fit, method='linear')
        print(popt)
        crppixel_coi_model = func(r_pixel, *popt)
        #r_pixel_notdonut, crppixel_notdonut = get_crprofile_for_ext(epoch, obsid, ext, ifdonut=False)
        plt.plot(r_pixel, crppixel_coi, 'b-')
        plt.plot(r_pixel, crppixel_coi_model, 'r-')
        plt.plot(r_pixel, crppixel_coi_model-popt[2], 'r--', alpha=0.3)
        #plt.plot(r_pixel_notdonut, crppixel_notdonut, 'b--')
        plt.vlines(x=r_pixel[drop_min],ymin=-0.25,ymax=1.75)
        plt.vlines(x=r_pixel[drop_max],ymin=-0.25,ymax=1.75)
        #plt.ylim(0.004,0.011)
        #plt.xlim(300,1300)
        plt.show()
    elif method == 'log':
        bkg = 0
        popt = fitting(np.log(r_pixel_fit), np.log(crppixel_coi_fit-bkg), method='log')
        print(popt)
        #plt.plot(np.log(r_pixel), np.log(crppixel_coi), 'b-')
        #plt.plot(np.log(r_pixel), np.log(crppixel_coi_model), 'r-')
        #plt.plot(np.log(r_pixel), np.log(crppixel_coi_model-popt[2]), 'r--', alpha=0.3)
        #plt.plot(np.log(r_pixel_donut), np.log(crppixel_donut), 'b--')
        plt.vlines(x=np.log(r_pixel[drop_min]),ymin=-10,ymax=3)
        plt.vlines(x=np.log(r_pixel[drop_max]),ymin=-10,ymax=3)
        crppixel_coi_model = func2(np.log(r_pixel), *popt)
        plt.plot(np.log(r_pixel), np.log(crppixel_coi-bkg),'b-')
        plt.plot(np.log(r_pixel), crppixel_coi_model,'r-')
        plt.show()
    return r_pixel, crppixel_coi, drop_min, drop_max, crppixel_coi_model, popt

#fit_outer('26', '00094463002', 1, drop_min=False, drop_max_num=50, ifdonut=False, method='log')
#fit_outer('26', '00094464001', 1, drop_min=False, drop_max_num=20)
# use the fitted results to get coi-factor for inner part

def correct_inner(epoch, obsid, ext, drop_min, drop_max_num, ifdonut, method):
    start = 0
    step =5
    r_pixel, crppixel_coi, drop_min, drop_max, crppixel_coi_model, popt = \
        fit_outer(epoch, obsid, ext, drop_min, drop_max_num, ifdonut, method)
    r_pixel, crppixel = get_crprofile_for_ext(epoch, obsid, ext, ifdonut=True, ifcoi=False)
    coifactor = crppixel_coi_model/crppixel
    rate, scale, src_center, smooth, delta = read_rate(epoch, obsid, ext)
    pos_data = np.argwhere(rate!=np.nan)
    dist_data = np.sqrt(np.sum(np.power(pos_data-src_center,2),axis=1)).reshape(rate.shape)
    index_data = np.ceil((np.array(dist_data)-start)/step)-1 
    index_data[np.where(index_data==-1)]=0
    index_data[np.isnan(rate)] = 0
    index_data = index_data.astype(int)
    coifactor_img = coifactor[index_data]
    coifactor_img[np.isnan(rate)] = np.nan
    rate_corr = rate*coifactor_img
    plt.imshow(rate_corr,vmin=0.002,vmax=0.04)
    plt.show()


#correct_inner('26', '00094463002', 1, drop_min=False, drop_max_num=50, ifdonut=True, method='linear')
    
# compare with directly corrected image (subtract)

# stack all extensions

def makeplot(epoch, obsid, ext):
    r_pixel_notdonut, crppixel_notdonut = get_crprofile_for_ext(epoch, obsid, ext, ifdonut=False)
    index_list = np.arange(0, len(crppixel_notdonut))
    drop_min = np.min(index_list[crppixel_notdonut<=0.01])
    plt.plot(r_pixel_notdonut, crppixel_notdonut,'b-')
    plt.vlines(x=r_pixel_notdonut[drop_min],ymin=-10,ymax=10,colors='k',alpha=0.3)
    plt.ylim(0,0.04)
    plt.xlim(0,1440)
    bkg_r, bkg_sig = check_donut(obsid,ext)
    plt.plot(bkg_r,bkg_sig)
    plt.show()


#makeplot('26', '00094463002', 1)