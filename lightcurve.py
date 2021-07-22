from scipy.optimize import curve_fit
from tools import *
from aper_phot import phot_ct, load_header
from conversion import cr2mag
from profile_q import sur_bri_profile
import numpy as np
import matplotlib.pyplot as plt

def km2pix(km, reso, delta):
    au = km/(1.5e8)
    rad = au/delta
    arcs = rad*(3600*180/np.pi)
    pix = arcs/reso
    return pix

def func(x, a, b, c):
    return a*np.power(x,-b)+c

def bkg_fit(aper_list, signal_list):
    popt, pcov = curve_fit(func, aper_list, signal_list, 
                           p0=np.array([200,0.4,5]),
                           bounds=(np.array([-np.inf,-np.inf,-np.inf]),
                                   np.array([np.inf,np.inf,np.inf])))
    #bkg = func(src_r, *popt)
    return popt

def read_reso(ind):
    set1 = [1,2,4,6,8,10,12,14,16,17,19,21,23,25]
    set2 = [3,5,7,9,11,13,15,18,20,22,24,26]
    if ind in set1:
        return 0.5
    elif ind in set2:
        return 1
    else:
        pass
#TODO: use exp map to update mask img 
def bkg_get(ind):
    src_r = 1300
    reso = 0.502#read_reso(ind)
    ind = str(int(ind))
    img_name = ind+'_uvv.fits'
    obs_log_name = 'obs_log/'+ind+'_obs-log_46P.txt'
    err2_name = False
    horizon_id = 90000545
    delta = get_orbit(obs_log_name, horizon_id)[0]
    src_center = (2000,2000)
    #src_r = 500
    start = 64
    step = 4
    mask_img = False
    bg_center = False
    bg_r = False
    relative_path='stack/'
    ext={0:'img',1:'exp'}
    #ext={0:'img',1:'err2',2:'exp'}
    smooth = False
    if ind in [1,2,4,6,8,10,12,14,16]:
        smooth = True
    epoch = str(int(ind))
    red = 0
    exp = float(load_header(img_name,relative_path)['EXPTIME'])
    result = sur_bri_profile(
                 reso, ind, red,
                 ext, delta, 
                 src_center, src_r,
                 bg_center, bg_r,
                 start, step, mask_img,
                 relative_path,
                 chatter=0,smooth=True,
                 coicorr=True,rate_img=False,
                 img_name=img_name)
    aper_list = result[1]
    sig_list = result[2]/exp
    pix2cm2_factor = np.power(au2km(as2au(reso, delta))*1e5, 2)
    sig_list = sig_list*pix2cm2_factor
    popt= bkg_fit(aper_list, sig_list)
    print(popt)
    sig_model = func(result[1], *popt)
    bkg1 = func(1500, *popt)
    bkg2 =  func(km2pix(70000, reso, delta), *popt)
    bkg3 =  func(km2pix(100000, reso, delta), *popt)
    plt.plot(aper_list,sig_list,'b-')
    plt.plot(result[1],sig_model,'r--')
    #plt.title(str(int(ind)))
    #plt.show()
    return aper_list, sig_list, sig_model, bkg1, bkg2, bkg3, popt, exp

bkg_fit1 = []
bkg_fit2 = []
bkg_fit3 = []
fig = plt.figure()
#for ind in [2,4,6,8,10]:
for ind in [17]:#,18,19,20,21,22,23,24,25,26]:
    print(ind)
    bkg_ind = bkg_get(ind)
    bkg_fit1.append(bkg_ind[3])
    bkg_fit2.append(bkg_ind[4])
    bkg_fit3.append(bkg_ind[5])
bkg_fit1 = np.array(bkg_fit1)
bkg_fit2 = np.array(bkg_fit2)
bkg_fit3 = np.array(bkg_fit3)
print(bkg_fit1)
print(bkg_fit2)
print(bkg_fit3)
print(np.mean(bkg_fit1),np.mean(bkg_fit2),np.mean(bkg_fit3),np.std(bkg_fit3))
plt.show()



def bkg_plot():
    i = 24
    fig, ax = plt.subplots(nrows=2, ncols=2)
    f = open(get_path('epoch_25-26.txt'),'w')
    for row in ax:
        for col in row:
            i += 1
            print(i)
            reso = read_reso(i)
            if i < 27:
                aper_list, sig_list, sig_model, bkg1, bkg2, popt = bkg_get(i)
                f.write(str(int(i))+' '+str(bkg1)+' '+str(bkg2)+' '
                        +str(popt[0])+' '+str(popt[1])+' '+str(popt[2])+'\n')
                col.plot(aper_list, sig_list, 'b-')
                col.plot(aper_list, sig_model, 'r--')
                col.set_title('Epoch:'+str(int(i))+' Bkg:'+str(round(bkg1,2))+' resol:'+str(reso))
                col.set_ylabel('median counts')
                col.set_xlabel('pixel')
    f.close()
    plt.savefig(get_path('epoch_25-26.png'))
    plt.show()

# bkg_plot()

def write_orbit():
    bkg_file = get_path('../docs/'+'epoch_17-26.txt')
    f = open(bkg_file, 'r')
    horizon_id = 90000545
    peri_t = 2458465.5
    lines = []
    for line in f.readlines():
        line = line.strip()
        ind = line.split(' ')[0]
        obs_log_name = ind+'_obs-log_46P.txt'
        delta, rh, time = get_orbit(obs_log_name, horizon_id)
        dt = time.jd - peri_t
        line = line+' '+str(dt)+' '+str(rh)+' '+str(delta)+'\n' 
        lines.append(line)
    f.close()
    whole_file = get_path('../docs/'+'epoch_17-26_orbit.txt')
    with open(whole_file,'a+') as f:
        for line in lines:
            f.write(line)
    f.close()

# write_orbit()

def read_bkg():
    #return a dic
    # ind, bkg1, bkg2, a, b, c, [in a*np.power(x,-b)+c] time, rh, delta
    # read data
    bkg_file = get_path('../docs/epoch_17-26.txt')
    data = np.loadtxt(bkg_file)
    ind_list = data[:,0]
    bkg_list = data[:,1]
    bkg2_list = data[:,2]
    a_list = data[:,3]
    b_list = data[:,4]
    c_list = data[:,5]
    time_list = data[:,6]
    rh_list = data[:,7]
    delta_list = data[:,8]
    # make dic
    dic = {}
    for i in range(0,len(ind_list)):
        dic_ind = {}
        dic_ind['bkg'] = bkg_list[i]
        dic_ind['bkg2'] = bkg2_list[i]
        dic_ind['a'] = a_list[i]
        dic_ind['b'] = b_list[i]
        dic_ind['c'] = c_list[i]
        dic_ind['time'] = time_list[i]
        dic_ind['rh'] = rh_list[i]
        dic_ind['delta'] = delta_list[i]
        dic[ind_list[i]] = dic_ind
    return dic

def plot_bkg():
    dic = read_bkg()
    ind_list = np.arange(17,27)
    bri_list = []
    bri_1_list = []
    time_list = []
    time_1_list = []
    for ind in ind_list:
        img_name = str(int(ind))+'_stack_uvv.fits'
        dic_ind = dic[ind]
        bkg = dic_ind['bkg']/(read_reso(ind))**2
        time = dic_ind['time']
        exposure = float(load_header(img_name)['EXPTIME'])
        bkg_bri = bkg/exposure
        if read_reso(ind) == 1:
            bri_1_list.append(bkg_bri)
            time_1_list.append(time)
        bri_list.append(bkg_bri)
        time_list.append(time)
    print(np.mean(np.array(bri_list)))
    plt.plot(time_list,bri_list,'bo')
    plt.show()

#plot_bkg()


def lc_plot():
    # read time, rh, delta, bkg1, bkg2, a, b, c (q)
    dic = read_bkg()
    # bkg_dic = read_dic()
    ind_list = np.arange(17,27)
    mag_list = []
    time_list = []
    for ind in ind_list:
        # aper phot
        reso = read_reso(ind)
        horizon_id = 90000545
        obs_log_name = str(int(ind))+'_obs-log_46P.txt'
        img_name = str(int(ind))+'_stack_uvv.fits'
        center = (2000,2000)
        dic_ind = dic[ind]
        delta = dic_ind['delta']
        rh = dic_ind['rh']
        #bkg = dic_ind['bkg']
        time = dic_ind['time']
        time_list.append(time)
        src_r = km2pix(10000, reso, delta)
        count, pixel = circle_ct(img_name, center, src_r, method='mean', mask_img = False)
        exposure = float(load_header(img_name)['EXPTIME'])
        # remove bkg
        #bkg_rate = bkg/exposure
        #bkg_rate = 0.01527646247859759*(read_reso(ind))**2
        bkg_rate = 0.01109430972462222*(read_reso(ind))**2
        cr = count/exposure - bkg_rate*pixel
        # 1
        if ind == 17:
            q1 = 1
            delta1 = delta
            rh1 = rh
        # get mag
        mag = cr2mag(cr, 0., 'v')[0] #TODO:
        # mag correction
        ### F ~ Q/(rh^2*delta)
        q = 1
        mag = mag-2.5*np.log10((q1/q)*(delta/delta1)*np.power(rh/rh1,2))
        mag_list.append(mag)
    return time_list, mag_list

#time_list, mag_list = lc_plot()
#print(time_list)
#print(mag_list)
#plt.plot(time_list, mag_list, 'ko')
#plt.xlabel('time to perihelion (days)')
#plt.ylabel('V mag')
#plt.show()



    
