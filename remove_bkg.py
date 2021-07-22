from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.jplhorizons import Horizons
from scipy.ndimage import rotate
import stsci.imagestats as imagestats
from os import listdir
from tools import *
from _mypath import thisdir
import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import tarfile
import os

def getidlist(epoch='bkg_jan',filt='uvv'):
    # epoch: 'bkg_jan'
    # filt: 'uw1' or 'uvv' or 'all'
    datadir = '/Users/zexixing/Research/swift46P/data/'+epoch+'/'
    idlist_all = os.listdir(datadir)
    idlist_all = [i for i in idlist_all if i[:3]=='000']
    if filt == 'all':
        return idlist_all
    idlist = []
    for obsid in idlist_all:
        obsdir = datadir+obsid+'/uvot/image'
        filelist = os.listdir(obsdir)
        if ('sw'+obsid+filt+'_rw.img.gz' in filelist) or \
            ('sw'+obsid+filt+'_rw.img' in filelist):
            idlist.append(obsid)
    return idlist

def getDet(epoch='bkg_jan', filt='uvv'):
    # v list & uw1 list
    datadir = '/Users/zexixing/Research/swift46P/data/'+epoch+'/'
    docsdir = '/Users/zexixing/Research/swift46P/docs/'+epoch+'/'
    idlist_all = getidlist(epoch=epoch,filt=filt)
    for obsid in idlist_all:
        infile = datadir+obsid+'/uvot/image/sw'+obsid+filt+'_rw.img.gz'
        outfile = docsdir+'sw'+obsid+filt+'_dt.img'
        attfile = datadir+obsid+'/auxil/sw'+obsid+'sat.fits.gz'
        # conversion
        os.system('swiftxform infile='+infile+' outfile='+outfile+' attfile='+attfile\
                +' method=AREA to=DET teldeffile=caldb ra=0 dec=0')

#getDet(epoch='bkg_jan', filt='uw1')

def checkImg(epoch='bkg_jan',filt='uvv',img='det'):
    idlist = getidlist(epoch=epoch,filt=filt)
    for obsid in idlist:
        if img=='det':
            t = fits.open('/Users/zexixing/Research/swift46P/docs/'+epoch+'/sw'+obsid+filt+'_dt.img')
        elif img == 'raw':
            t = fits.open('/Users/zexixing/Research/swift46P/data/'+epoch+'/'+obsid+'/uvot/image/sw'+obsid+filt+'_rw.img.gz')
        elif img == 'sky':
            t = fits.open('/Users/zexixing/Research/swift46P/data/'+epoch+'/'+obsid+'/uvot/image/sw'+obsid+filt+'_sk.img.gz')
        else:
            print('please check the path and img type')
        if os.path.exists('/Users/zexixing/Research/swift46P/data/'+epoch+'/'+obsid+'/uvot/event'):
            ifevent = True
        else:
            ifevent = False
        print(obsid+': ', np.shape(t[1].data), ' event: ', ifevent)
        t.close()

def clipDet(epoch='bkg_jan',obsid='00000000000',data=None):
    #path = '/Users/zexixing/Research/swift46P/data/bkg_jan/00085932004/uvot/image/sw00085932004uvv_rw.img.gz'
    #path = '/Users/zexixing/Research/swift46P/data/bkg_jan/00085932004/uvot/image/sw00085932004uvv_sk.img.gz'
    #path = '/Users/zexixing/Research/swift46P/docs/bkg_jan/sw00085932004uvv_dt.img'
    #path = '/Users/zexixing/Research/swift46P/data/46P_raw_uvot/00094318002/uvot/image/sw00094318002uvv_rw.img.gz'
    #path = '/Users/zexixing/Research/swift46P/data/46P_raw_uvot/00094318002/uvot/image/sw00094318002uvv_sk.img.gz'
    #path = '/Users/zexixing/Research/swiftUVOT/data/Borisov_raw_latedec/00012976002/uvot/image/sw00012976002uvv_rw.img.gz'
    #path = '/Users/zexixing/Research/swiftUVOT/data/Borisov_raw_latedec/00012977003/uvot/image/sw00012977003uw1_rw.img.gz'
    #path = '/Users/zexixing/Research/swiftASTER/data/dembowska/00091027001/uvot/image/sw00091027001ugu_dt.img'
    #path = '/Users/zexixing/Downloads/sw00012343001/uvot/image/sw00012343001um2_sk.img.gz'
    #path = '/Users/zexixing/Downloads/swtest/00037812001/uvot/image/sky_uvv.img'
    if isinstance(data,np.ndarray):
        img = data
    else:
        path = '/Users/zexixing/Research/swift46P/docs/'+epoch+'/sw'+obsid+'uw1_dt.img'
        img = fits.open(path)
        exp = img[1].header['EXPOSURE']
        img = img[1].data/exp
    imgstats = imagestats.ImageStats(img,nclip=3)
    mean = imgstats.mean
    sig = imgstats.stddev
    print(mean)
    clip=mean+3*sig
    clip_map = ma.masked_where(img>clip,img)
    shape_data = np.zeros(img.shape)
    shape_data[clip_map.mask==False]=1
    #plt.imshow(img,vmin=0,vmax=2)
    #plt.show()
    #print(mean,sig,clip)
    #plt.imshow(img*shape_data,vmin=-0.03,vmax=0.05)
    #plt.show()
    #plt.imshow(clip_map.mask,vmin=0,vmax=1)
    #plt.show()
    return img, clip_map.mask

#clipDet()

def coaddDet(epoch,filt,rm_list):
    # get median/mean image --> fringe
    idlist = getidlist(epoch=epoch,filt=filt)
    for obsid_rm in rm_list:
        try:    idlist.remove(obsid_rm)
        except:    pass
    img_all = []
    mask_all = []
    for obsid in idlist:
        img,mask = clipDet(epoch,obsid)
        #img = fits.open('/Users/zexixing/Research/swift46P/docs/bkg_jan/sw'+obsid+'uw1_dt.img')
        #exp = img[1].header['EXPOSURE']
        #img = img[1].data/exp
        #mask = np.zeros(img.shape)
        img_all.append(img)
        mask_all.append(mask)
    img_all = np.array(img_all)
    mask_all = np.array(mask_all)
    arr=ma.masked_array(img_all,mask_all)
    med = ma.mean(arr,axis=0)
    plt.imshow(med.data,vmin=0,vmax=0.003)
    plt.show()
    return med.data, med.mask


#coaddDet(epoch='bkg_jan',filt='uw1',rm_list=['00035259001','00035259002'])

#def rotateDet():
#    pass 

def zoomImg(data,zoom):
    # zoom: larger or smaller or auto
    # zoom for twice
    # -------
    # judge zoom
    data = np.array(data)
    row, col = data.shape
    if zoom == 'auto':
        if (row < 1500) and (col < 1500):
            zoom = 'larger'
        elif (row > 1700) and (col >1700):
            zoom = 'same'
            return data
    # check zoom
    if zoom == 'smaller':
        if (row%2)!=0 or (col%2)!=0:
            assert ValueError('number of row and column should be even numbers')
    # build array and fill
    if zoom == 'larger':
        new = np.zeros((row*2,col*2))
        new[0: row*2   :2,   0: col*2   :2]=data[0:row,0:col]
        new[0: row*2   :2,   1:(col*2+1):2]=data[0:row,0:col]
        new[1:(row*2+1):2,   0: col*2   :2]=data[0:row,0:col]
        new[1:(row*2+1):2,   1:(col*2+1):2]=data[0:row,0:col]
        return new/4
    elif zoom == 'smaller':
        row = int(row)
        col = int(col)
        new = np.zeros((int(row/2),int(col/2)))
        new[0:int(row/2),0:int(col/2)] += data[0:row  :2, 0:col  :2]
        new[0:int(row/2),0:int(col/2)] += data[0:row  :2, 1:col+1:2]
        new[0:int(row/2),0:int(col/2)] += data[1:row+1:2, 0:col  :2]
        new[0:int(row/2),0:int(col/2)] += data[1:row+1:2, 1:col+1:2]
        return new
    elif zoom == 'same':
        return data

def readDonut(filt):
    path = '/Users/zexixing/Research/swift46P/data/donuts/mod8_'+filt+'.fits'
    data = fits.open(path)[1].data
    exp = 1000.
    return data/exp

def rmDonut(epoch,obsid,filt):
    path = '/Users/zexixing/Research/swift46P/data/'+epoch+'/'+obsid+'/uvot/image/sw'+obsid+filt+'_rw.img.gz'
    hdul = fits.open(path)
    data = hdul[1].data
    data = zoomImg(data,'auto')
    exp = hdul[1].header['EXPOSURE']
    print(exp)
    donut = readDonut(filt)
    data_rmdonut = data/exp-donut
    return data_rmdonut


#d = rmDonut('46P_raw_uvot','00094319001','uw1')
##clipDet(data=d)
#plt.imshow(d,vmin=-0.02,vmax=0.04)
#plt.show()

def meanBkg(epoch,filt):
    idlist = getidlist(epoch=epoch,filt=filt)
    for obsid_rm in ['00035259001','00035259002']:
        try:    idlist.remove(obsid_rm)
        except:    pass
    datadir = '/Users/zexixing/Research/swift46P/data/'+epoch+'/'
    mean_list = []
    for obsid in idlist:
        datapath = datadir+obsid+'/uvot/image/sw'+obsid+filt+'_sk.img.gz'
        exppath = datadir+obsid+'/uvot/image/sw'+obsid+filt+'_ex.img.gz'
        data = fits.open(datapath)[1].data
        exp = fits.open(exppath)[1].data
        data = zoomImg(data/exp,'auto')
        exp = zoomImg(exp,'auto')
        #exp = np.floor(exp/np.max(exp))
        data_list = data[exp==np.max(exp)]
        imgstats = imagestats.ImageStats(data_list,nclip=3)
        mean = imgstats.mean
        sig = imgstats.stddev
        mean_list.append(mean)
        print(obsid,': ',mean,'+/-',sig,' #',len(data_list))
    mean_list = np.array(mean_list)
    bkg = imagestats.ImageStats(mean_list,nclip=0)
    print(bkg.mean, bkg.stddev)

#print('----UW1----')
#meanBkg('bkg_jan','uw1')
#print('----V----')
#meanBkg('bkg_jan','uvv')