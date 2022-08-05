#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 20:57:11 2021

@author: 1001883
"""

# Douglas et al. 2022 Script 2
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

homedir = '/Volumes/GeoCCRI_01/users-data/douglahu/CMIP6_data'

# Set the parameters of interest:
variable_id = 'tas'
table_id = 'Amon'
scen_short = ['ssp119','ssp126','ssp245','ssp370','ssp585']

# Set the multi-ensemble member averaging type
avgType = 'median' # 'mean'

years = list(np.arange(1990,2101,1)) #Check this aligns with the years from the previous script

POI = [2040,2060] # Period of interest over which signal and SN are averaged

# Set the common gridsize
dlat = 2.5 # Size (in degrees) of latitude of each grid cell
dlon = 2.5 # Size (in degrees) of longitude of each grid cell
# Set the interpolation method for regridding
interpMethod = 'bilinear' # 'patch' # 'conservative' 

# Toggle to filter the models to just the subset with ECS in the CMIP5 range
ECSfilter = False # True
# If True, you'll then have to use the alternate versions of subsequent scripts

# List of models with ECS within CMIP5 range
ECSdirList = ['IPSL-CM6A-LR','KACE-1-0-G','EC-Earth3-Veg','TaiESM1','EC-Earth3',
              'CNRM-CM6-1-HR','GFDL-CM4','ACCESS-ESM1-5','SAM0-UNICON','MCM-UA-1-0',
              'CMCC-CM2-SR5','CAS-ESM2-0','BCC-ESM1','AWI-CM-1-1-MR','MRI-ESM2-0',
              'GISS-E2-1-H','NorCPM1','BCC-CSM2-MR','FGOALS-f3-L','MPI-ESM1-2-LR',
              'MPI-ESM1-2-HR','MPI-ESM-1-2-HAM','FGOALS-g3','GISS-E2-1-G','MIROC-ES2L',
              'MIROC6','GFDL-ESM4','NorESM2-LM','NorESM2-MM','GISS-E2-2-G','CAMS-CSM1-0']

####----END USER-CHANGEABLE----####
###################################

# Get a list of available sources
os.chdir(homedir+'/'+variable_id+'/'+table_id)
dirList = np.sort(os.listdir())

validDirList = set() # Set of directories with the right files
dirList119 = set() # Separate set for ssp119 as it has fewer valid models

for thisDir in dirList:
    if thisDir == '.DS_Store':
        print('Skipping .DS_Store')
    else:
        os.chdir(thisDir)
        fileList = os.listdir()
        
        try:
            fileList.index('dea2022_allEns_signal_'+scen_short[1]+'_allYears.nc')
            fileList.index('dea2022_allEns_signal_'+scen_short[2]+'_allYears.nc')
            fileList.index('dea2022_allEns_signal_'+scen_short[3]+'_allYears.nc')
            fileList.index('dea2022_allEns_signal_'+scen_short[4]+'_allYears.nc')
            validDirList.add(thisDir)
        except:
            pass
        
        try:
            fileList.index('dea2022_allEns_signal_'+scen_short[0]+'_allYears.nc')
            dirList119.add(thisDir)
        except:
            pass        
        
        os.chdir('..')

# If using the ECS filter, only include those models in the list we specified
if ECSfilter:
    validDirList = validDirList & set(ECSdirList)
    dirList119 = dirList119 & set(ECSdirList)

validDirList = list(validDirList) 
print(len(validDirList),' valid models')
dirList119 = list(dirList119) 
print(len(dirList119),' models for ssp119')




for thisSSP in scen_short:
    print(thisSSP)
    
    # Set up parameters for regridding to common gridsize
    lat = np.arange(-90+dlat/2, 90, dlat)
    lon = np.arange(0+dlon/2, 360, dlon)
    ds_out = xr.Dataset({'lat': (['lat'], lat),
                         'lon': (['lon'], lon),
                        }
                       )
    
    # Set up empty DataArrays for storing the signal and noise for all models
    if thisSSP == 'ssp119':
        dirListRange = dirList119
        mdCall = dirList119 # List of model names
    else:
        dirListRange = validDirList
        mdCall = validDirList # List of model names
            
    noise_models = xr.DataArray(data=0.0, dims=['lat','lon','model'], coords=[lat,lon,mdCall], 
                                name='noise') 
    signal_models = xr.DataArray(data=0.0, dims=['lat','lon','model'], coords=[lat,lon,mdCall], 
                                 name='signal') 
    SN_models = xr.DataArray(data=0.0, dims=['lat','lon','model'], coords=[lat,lon,mdCall], 
                             name='SN')
    signal_models_allYears = xr.DataArray(data=0.0, dims=['lat','lon','model','year'], 
                                          coords=[lat,lon,mdCall,years], name='signal') 
    SN_models_allYears = xr.DataArray(data=0.0, dims=['lat','lon','model','year'], 
                                      coords=[lat,lon,mdCall,years], name='SN')
    
    # Convert the zero values to NaN to avoid accidentally skewing averages
    SN_models_allYears = SN_models_allYears.where(SN_models_allYears>0)
    
    # Load the relevant files and regrid to a common grid
    os.chdir(homedir+'/'+variable_id+'/'+table_id)
    
    for thisDir in dirListRange:
        os.chdir(thisDir)
        print(thisDir)
        
        # Load signal, noise, and signaltonoise
        noiseDset = xr.open_dataset('dea2022_allEns_noise_'+thisSSP+'.nc')

        noiseArr = xr.open_dataarray('dea2022_allEns_noise_'+thisSSP+'.nc')
        signalArr = xr.open_dataarray('dea2022_allEns_signal_'+str(POI[0])+'-'+str(POI[1])+'_'+thisSSP+'.nc')
        SN_Arr = xr.open_dataarray('dea2022_allEns_SN_'+str(POI[0])+'-'+str(POI[1])+'_'+thisSSP+'.nc')
        signalArr_allYears = xr.open_dataarray('dea2022_allEns_signal_'+thisSSP+'_allYears.nc')
        SN_Arr_allYears = xr.open_dataarray('dea2022_allEns_SN_'+thisSSP+'_allYears.nc')
        
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
    
    if ECSfilter:
        noise_models.to_netcdf(path='MM_dea2022_ECSfilter_noise_'+thisSSP+'.nc')
        signal_models.to_netcdf(path='MM_dea2022_ECSfilter_signal_'+str(POI[0])+'-'+str(POI[1])+'_'+thisSSP+'.nc')
        SN_models.to_netcdf(path='MM_dea2022_ECSfilter_SN_'+str(POI[0])+'-'+str(POI[1])+'_'+thisSSP+'.nc')
        signal_models_allYears.to_netcdf(path='MM_dea2022_ECSfilter_signal_'+thisSSP+'_allYears.nc')
        SN_models_allYears.to_netcdf(path='MM_dea2022_ECSfilter_SN_'+thisSSP+'_allYears.nc')
    else:
        noise_models.to_netcdf(path='MM_dea2022_noise_'+thisSSP+'.nc')
        signal_models.to_netcdf(path='MM_dea2022_signal_'+str(POI[0])+'-'+str(POI[1])+'_'+thisSSP+'.nc')
        SN_models.to_netcdf(path='MM_dea2022_SN_'+str(POI[0])+'-'+str(POI[1])+'_'+thisSSP+'.nc')
        signal_models_allYears.to_netcdf(path='MM_dea2022_signal_'+thisSSP+'_allYears.nc')
        SN_models_allYears.to_netcdf(path='MM_dea2022_SN_'+thisSSP+'_allYears.nc')
