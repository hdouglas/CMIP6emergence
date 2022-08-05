#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 20:57:11 2021

@author: 1001883
"""

# Douglas et al. 2022 Script 2 for the CMIP5 series of data
# Script for multi-model average analysis  - regridding
# Load signal and noise files calculated using the Frame et al 2017 methodology 
# and combine into single netCDF files on a common grid.
# This version uses all ensembles and just includes those models with all the SSPs 
# used in Douglas et al. 2022.

# Inputs: multi-ensemble netcdf files (S, N, S/N) from script 01 for all models
# Outputs: multi-model netcdf files with all models, on a common grid

import os
import numpy as np
import xarray as xr
import xesmf as xe  

###############################
####----USER-CHANGEABLE----####

homedir = '/Volumes/GeoCCRI_01/users-data/douglahu/CMIP5_data'

# Set the parameters of interest:
variable_id = 'tas'
table_id = 'Amon'
scen_short = ['rcp26','rcp45','rcp85']

# Set the multi-ensemble member averaging type
avgType = 'median' # 'mean'

years = list(np.arange(1990,2101,1)) #Check this aligns with the years from the previous script

POI = [2040,2060] # Period of interest over which signal and SN are averaged

# Set the common gridsize
dlat = 2.5 # Size (in degrees) of latitude of each grid cell
dlon = 2.5 # Size (in degrees) of longitude of each grid cell
# Set the interpolation method for regridding
interpMethod = 'bilinear' # 'patch' # 'conservative' 

####----END USER-CHANGEABLE----####
###################################

# Get a list of available sources
os.chdir(homedir+'/'+variable_id+'/'+table_id)
dirList = np.sort(os.listdir())

validDirList = set() # Set of directories with the right files

for thisDir in dirList:
    if thisDir == '.DS_Store':
        print('Skipping .DS_Store')
    else:
        os.chdir(thisDir)
        fileList = os.listdir()
        
        try:
            fileList.index('dea2022_allEns_signal_'+scen_short[0]+'_allYears.nc')
            fileList.index('dea2022_allEns_signal_'+scen_short[1]+'_allYears.nc')
            fileList.index('dea2022_allEns_signal_'+scen_short[2]+'_allYears.nc')
            validDirList.add(thisDir)
        except:
            pass
        os.chdir('..')

validDirList = list(validDirList) 
print(validDirList)
print(len(validDirList),' valid models')

for thisRCP in scen_short:
    print(thisRCP)
    
    # Set up parameters for regridding to common gridsize
    lat = np.arange(-90+dlat/2, 90, dlat)
    lon = np.arange(0+dlon/2, 360, dlon)
    ds_out = xr.Dataset({'lat': (['lat'], lat),
                         'lon': (['lon'], lon),
                        }
                       )
    
    # Set up empty DataArrays for storing the signal and noise for all models   
    mdCall = validDirList # List of model names
    
    noise_models = xr.DataArray(data=0.0, dims=['lat','lon','model'], 
                                coords=[lat,lon,mdCall], name='noise') 
    signal_models = xr.DataArray(data=0.0, dims=['lat','lon','model'], 
                                 coords=[lat,lon,mdCall], name='signal') 
    SN_models = xr.DataArray(data=0.0, dims=['lat','lon','model'], 
                             coords=[lat,lon,mdCall], name='SN')
    signal_models_allYears = xr.DataArray(data=0.0, dims=['lat','lon','model','year'], 
                                          coords=[lat,lon,mdCall,years], name='signal') 
    SN_models_allYears = xr.DataArray(data=0.0, dims=['lat','lon','model','year'], 
                                      coords=[lat,lon,mdCall,years], name='SN')
    
    # Load the relevant files and regrid to a common grid
    os.chdir(homedir+'/'+variable_id+'/'+table_id)
    for thisDir in validDirList:
        os.chdir(thisDir)
        print(thisDir)
        
        # Load signal, noise, and signaltonoise
        noiseDset = xr.open_dataset('dea2022_allEns_noise_'+thisRCP+'.nc')

        noiseArr = xr.open_dataarray('dea2022_allEns_noise_'+thisRCP+'.nc')
        signalArr = xr.open_dataarray('dea2022_allEns_signal_'+str(POI[0])+'-'+str(POI[1])+'_'+thisRCP+'.nc')
        SN_Arr = xr.open_dataarray('dea2022_allEns_SN_'+str(POI[0])+'-'+str(POI[1])+'_'+thisRCP+'.nc')
        signalArr_allYears = xr.open_dataarray('dea2022_allEns_signal_'+thisRCP+'_allYears.nc')
        SN_Arr_allYears = xr.open_dataarray('dea2022_allEns_SN_'+thisRCP+'_allYears.nc')
        
        # Get a multi-ensemble average
        if avgType == 'median':
            noiseAvg = noiseArr.median('variant', keep_attrs=True, skipna=True)
            signalAvg = signalArr.median('variant', keep_attrs=True, skipna=True)
            SN_Avg = SN_Arr.median('variant', keep_attrs=True, skipna=True)
            signalAvg_allYears = signalArr_allYears.median('variant', keep_attrs=True, skipna=True)
            SN_Avg_allYears = SN_Arr_allYears.median('variant', keep_attrs=True, skipna=True)
        elif avgType == 'mean':
            noiseAvg = noiseArr.mean('variant', keep_attrs=True, skipna=True)
            signalAvg = signalArr.mean('variant', keep_attrs=True, skipna=True)
            SN_Avg = SN_Arr.mean('variant', keep_attrs=True, skipna=True)
            signalAvg_allYears = signalArr_allYears.mean('variant', keep_attrs=True, skipna=True)
            SN_Avg_allYears = SN_Arr_allYears.mean('variant', keep_attrs=True, skipna=True)
        else:
            print('Specify mean or median for avgType')
            exit
        
        # Regrid the data using xESMF
        
        # Use the noise dataset to provide the common dimensions
        regridder = xe.Regridder(noiseDset, ds_out, interpMethod, periodic=True) 
        regridder  # print basic regridder information.
        
        noiseOut = regridder(noiseAvg)
        signalOut = regridder(signalAvg)
        SN_Out = regridder(SN_Avg)
        signalOut_allYears = regridder(signalAvg_allYears)
        SN_Out_allYears = regridder(SN_Avg_allYears)
        
        print('Max longitude, in: ',np.max(noiseDset.lon.values),', out:',np.max(noiseOut.lon.values))
        
        #Reorder the dimensions as necessary:
        signalOut_allYears = signalOut_allYears.transpose('lat','lon','year')
        SN_Out_allYears = SN_Out_allYears.transpose('lat','lon','year')
        
        #Save in the DataArrays we set up earlier 
        noise_models.loc[dict(model=thisDir)]=noiseOut.values 
        signal_models.loc[dict(model=thisDir)]=signalOut.values 
        SN_models.loc[dict(model=thisDir)]=SN_Out.values 
        signal_models_allYears.loc[dict(model=thisDir)]=signalOut_allYears.values 
        SN_models_allYears.loc[dict(model=thisDir)]=SN_Out_allYears.values 
            
        # Navigate back to the parent directory
        os.chdir('..')
        #os.chdir(homedir+'/'+variable_id+'/'+table_id)
        
    # Save the multi-model data arrays
    try:
        os.mkdir('01_MM_output')
    except: 
        pass
    os.chdir('01_MM_output')
    
    noise_models.to_netcdf(path='MM_dea2022_noise_'+thisRCP+'.nc')
    signal_models.to_netcdf(path='MM_dea2022_signal_'+str(POI[0])+'-'+str(POI[1])+'_'+thisRCP+'.nc')
    SN_models.to_netcdf(path='MM_dea2022_SN_'+str(POI[0])+'-'+str(POI[1])+'_'+thisRCP+'.nc')
    signal_models_allYears.to_netcdf(path='MM_dea2022_signal_'+thisRCP+'_allYears.nc')
    SN_models_allYears.to_netcdf(path='MM_dea2022_SN_'+thisRCP+'_allYears.nc')
