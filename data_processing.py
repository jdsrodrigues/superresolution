import warnings
warnings.simplefilter("ignore", UserWarning)
import iris
import numpy as np
from scipy import interpolate
import os

datapath = '/data/users/jrodrigu/test_final/01102013-30092014/'
locations = ['azores/','australia/','alaska/','oklahoma/'] 
surfaces = [0.0300, 0.0299, 0.008, 0.314] #km

datapath_um = '/data/users/jrodrigu/tmp/'
filename_um = "20200101T0000Z_glm_pa000.pp"
file_um  = datapath_um + filename_um


# UM vertical grid
cube = iris.load_cube(file_um, 'air_pressure')
altitude = cube.coord("level_height")
um_altitude = altitude[altitude.points<12700].points

# Intermediate grid with 50m levels
new_altitude = np.arange(5,12800,50)

def unmask(X, high_h):
    x = X.data
    new_x = x.filled(np.nan)
    return new_x

def regrid(column, og_h, inter_h, low_h): 
    # regrid to intermediate resolution
    high_h = og_h * 1000 #metres
    func = interpolate.interp1d(high_h, column, kind = "linear")
    inter_column = func(inter_h)
    # regrid to low resolution
    func_low = interpolate.interp1d(inter_h, inter_column, kind = "linear")
    low_column = func_low(low_h)
    return (inter_column, low_column) 

def process(i):
    inter_lst,low_lst = [],[]
    files = os.listdir(datapath + locations[i])
    for j in range(len(files)):
        file_in = datapath + locations[i] + files[j]
        for f in range(48):
            k=f*30
            T = iris.load_cube(file_in, ['air_temperature'])[k]
            q = iris.load_cube(file_in, ['specific_humidity'])[k]  
            P = iris.load_cube(file_in, ['air_pressure'])[k]       
            og_altitude = T.coord('altitude').points - surfaces[i]
        
            high_altitude = og_altitude[og_altitude<13]
            
            T_unmsk = unmask(T, og_altitude)[:len(high_altitude)] 
            q_unmsk = unmask(q, og_altitude)[:len(high_altitude)] 
            P_unmsk = unmask(P, og_altitude)[:len(high_altitude)] 
            
            if np.logical_not(np.isnan(T_unmsk).any()):
                
                # interpolate to new grids
                T_inter, T_low = regrid(T_unmsk, high_altitude, new_altitude, um_altitude) 
                q_inter, q_low = regrid(q_unmsk, high_altitude, new_altitude, um_altitude) 
                P_inter, P_low = regrid(P_unmsk, high_altitude, new_altitude, um_altitude)
        
                inters = np.array([T_inter, q_inter, P_inter]) 
                lows  = np.array([T_low, q_low, P_low])
                
                inter_lst.append(inters)
                low_lst.append(lows)
                
    inter_arr = np.array(inter_lst)
    low_arr   = np.array(low_lst)
    
    return inter_arr, low_arr
            
inter_azor, low_azor = process(0)
inter_aust, low_aust = process(1)
inter_alsk, low_alsk = process(2)
inter_oklh, low_oklh = process(3)

shuffler_azor = np.random.permutation(len(low_azor))
shuffler_aust = np.random.permutation(len(low_aust))
shuffler_alsk = np.random.permutation(len(low_alsk))
shuffler_oklh = np.random.permutation(len(low_oklh))

len_min = np.min([len(inter_azor), len(inter_aust), len(inter_alsk), len(inter_oklh)])

new_low_azor   = low_azor[shuffler_azor][:len_min]
new_inter_azor = inter_azor[shuffler_azor][:len_min]

new_low_aust   = low_aust[shuffler_aust][:len_min]
new_inter_aust = inter_aust[shuffler_aust][:len_min]

new_low_alsk   = low_alsk[shuffler_alsk][:len_min]
new_inter_alsk = inter_alsk[shuffler_alsk][:len_min]

new_low_oklh   = low_oklh[shuffler_oklh][:len_min]
new_inter_oklh = inter_oklh[shuffler_oklh][:len_min]

new_inter_arr = np.concatenate((new_inter_azor, new_inter_aust, new_inter_alsk, new_inter_oklh))
new_low_arr   = np.concatenate((new_low_azor, new_low_aust, new_low_alsk, new_low_oklh))

np.save("/data/users/jrodrigu/data_processed/final/INTER_TRAIN", new_inter_arr)
np.save("/data/users/jrodrigu/data_processed/final/LOW_TRAIN", new_low_arr)

np.save("/data/users/jrodrigu/data_processed/final/um_h", um_altitude)
np.save("/data/users/jrodrigu/data_processed/final/ml_h", new_altitude)