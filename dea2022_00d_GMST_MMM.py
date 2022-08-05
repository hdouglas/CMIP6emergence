#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 20:57:11 2021

@author: 1001883
"""

# Douglas et al. 2022 optional script for MMM GMST (needed for GWL comparisons)

# Inputs: multi-ensemble netcdf files from script 00c for all models
# Outputs: multi-model gmst netcdf files with all models, on a common grid

import os
import numpy as np
import xarray as xr

homedir = '/Volumes/GeoCCRI_01/users-data/douglahu/CMIP6_data'

# Set the parameters of interest:
variable_id = 'tas'
table_id = 'Amon'
scen_short = ['ssp119','ssp126','ssp245','ssp370','ssp585']

# Set the multi-ensemble member averaging type
avgType = 'median' # 'mean'

# Get a list of available sources
os.chdir(homedir+'/'+variable_id+'/'+table_id)
dirList = np.sort(os.listdir())

validDirList = set() # Set of directories with the right files
dirList119 = set()

for thisDir in dirList:
    if thisDir == '.DS_Store':
        print('Skipping .DS_Store')
    else:
        os.chdir(thisDir)
        fileList = os.listdir()
        
        try:
            fileList.index('dea2022_allEns_gmst_'+scen_short[1]+'_allYears.nc')
            fileList.index('dea2022_allEns_gmst_'+scen_short[2]+'_allYears.nc')
            fileList.index('dea2022_allEns_gmst_'+scen_short[3]+'_allYears.nc')
            fileList.index('dea2022_allEns_gmst_'+scen_short[4]+'_allYears.nc')
            validDirList.add(thisDir)
        except:
            pass
        
        try:
            fileList.index('dea2022_allEns_gmst_'+scen_short[0]+'_allYears.nc')
            dirList119.add(thisDir)
        except:
            pass        
        
        os.chdir('..')

validDirList = list(validDirList) 
print(validDirList)
print(len(validDirList),' valid models')
dirList119 = list(dirList119) 
print(len(dirList119),' models for ssp119')

for thisSSP in scen_short:
    print(thisSSP)
    
    # Set up empty DataArray for storing the signal and noise for all models
    years = list(np.arange(1990,2101,1))

    if thisSSP == 'ssp119':
        dirListRange = dirList119
        mdCall = dirList119
    else:
        dirListRange = validDirList
        mdCall = validDirList
        
    gmst_models = xr.DataArray(data=0.0, dims=['model','year'], coords=[mdCall,years], name='gmst') 

    # Load the data and store in the dataarray
    os.chdir(homedir+'/'+variable_id+'/'+table_id)
    
    for thisDir in dirListRange:
        os.chdir(thisDir)
        print(thisDir)

        # Load the array
        gmstArrOrig = xr.open_dataarray('dea2022_allEns_gmst_'+thisSSP+'_allYears.nc')
        # Filter to just the years we want
        gmstArr = gmstArrOrig.loc[dict(year=years)]
        
        # Get a multi-ensemble average
        if avgType == 'median':
            gmstAvg = gmstArr.median('variant', keep_attrs=True, skipna=True)

        elif avgType == 'mean':
            gmstAvg = gmstArr.mean('variant', keep_attrs=True, skipna=True)

        else:
            print('Specify mean or median for avgType')
            exit
        
        gmst_models.loc[dict(model=thisDir)]=gmstAvg.values 
            
        os.chdir('..')
        
    # Save the multi-model data arrays
    try:
        os.mkdir('01_MM_output')
    except: 
        pass
    os.chdir('01_MM_output')
       
    gmst_models.to_netcdf(path='MM_dea2022_gmst_'+thisSSP+'.nc')
