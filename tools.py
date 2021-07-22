from _mypath import thisdir
from os import listdir
from scipy import interpolate
import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons
import os

def get_path(*rela_path, to_file=False, map_type='sk'):
    """get an absolute path
    if to_file=True, the path will link to a .img.gz file according to inputted OBS_ID and FILTER
    if to_file=False, the path will link to any file/dir according to the rela_path
    
    Inputs: 
    1> 'OBS_ID', 'FILTER'('uw1', 'uw2', 'uvv') 
    2> 'rela_path', to_file=F
    Outpus: 
    absolute path, string
    """
    if to_file == True:
        data_path = '../data/'
        data_abs_path = os.path.join(thisdir, data_path)
        folders = [one for one in listdir(data_abs_path) if 'raw' in one]#raw
        for folder in folders:
            folder_abs_path = ''.join([data_abs_path, folder, '/'])
            obs_list = listdir(folder_abs_path)
            obs_list = [obs for obs in obs_list if obs[0] != '.']
            if rela_path[0] in obs_list:
                break
        map_path = '/uvot/image/'
        file_name = 'sw' + rela_path[0] + rela_path[1] + '_'+map_type+'.img.gz'
        file_path = ''.join([folder_abs_path, rela_path[0], map_path, file_name])
        return file_path
    else:
        return os.path.join(thisdir, rela_path[0])

def get_orbit(obs_log_name, horizon_id):
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
    mean_rh = eph['r']
    return mean_delta, mean_rh, mid_time

def error_prop(method, x, x_err, y, y_err):
    if method == 'sum' or method == 'sub':
        return np.sqrt(np.power(x_err,2)+np.power(y_err,2))
    elif method == 'mul':
        err_1 = np.power(x_err*y,2)
        err_2 = np.power(y_err*x,2)
        return np.sqrt(err_1+err_2)
    elif method == 'div':
        err_1 = np.power(x_err/y,2)
        err_2 = np.power(x*y_err/np.power(y,2),2)
        return np.sqrt(err_1+err_2)

def as2au(arcsec, dis_au):
    return dis_au*2*np.pi*arcsec/(3600.*360)
    
def au2km(au):
    return 149597870.7*au

def integ(x, y, step_num=False):
    if step_num:
        f = interpolate.interp1d(x, y, fill_value='extrapolate')
        x = np.linspace(x[0], x[-1], num=step_num)
        y = f(x)
    result = 0
    for i in range(len(x)):
        result += x[i]*y[i]
    return result