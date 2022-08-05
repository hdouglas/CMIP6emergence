#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 20:57:11 2021

@author: 1001883
"""

# Douglas et al. 2022 Script 1
# Generate noise, signal, and SN files using the Frame2017 approach. Save a dataframe
# that includes all ensembles for a given model. 

# Inputs: raw netcdf files from the CMIP6 database
# Outputs: netcdf files of signal, noise, signal-to-noise for each model and scenario,
# with all ensemble members included.

import os
import numpy as np
import xarray as xr

from dea2022_fn_gmst import gmst_calc 

###############################
####----USER-CHANGEABLE----####

# Set your home directory for where the data are stored
homedir = '/Volumes/GeoCCRI_01/users-data/douglahu/CMIP6_data'

# Set the parameters of interest:
variable_id = 'tas'
table_id = 'Amon'
scen_short = ['ssp119','ssp126','ssp245','ssp370','ssp585']
years = np.arange(1990,2101,1) # Change if you want coarser temporal resolution

# Set whether or not to use the use_cftime option in xr.open_dataset
# (this avoids issues if the dates go past 2262, but might slow performance. 
usingCFtime = True

####----END USER-CHANGEABLE----####
###################################

# Get a list of available sources
os.chdir(homedir+'/'+variable_id+'/'+table_id)
dirList = np.sort(os.listdir())

# Loop over all scenarios
for thisSSP in scen_short:
    print(thisSSP)
    for thisDir in dirList:
        # On Macs, hidden files can trip up the loop
        if thisDir == '.DS_Store':
            print('Skipping .DS_Store')
            
        else:
            os.chdir(thisDir)
            print(thisDir)
            fileList = os.listdir()
            try: 
                fileList.index('dea2022_allEns_gmst_'+thisSSP+'_allYears.nc_dummy')
                print('Already processed. Skipping.')
            except:
                
                try:
                    fileList.index('piControl')
                    fileList.index('historical')
                    fileList.index(thisSSP)
                    
                    # Just calculate the gmst, then save for later regridding 
                    
                    # Get the ensemble list to input to the signal-to-noise function
                    ensembleList = np.sort(os.listdir(homedir+'/'+variable_id+'/'+table_id+'/'+thisDir+'/'+thisSSP))
                    if ensembleList[0] == '.DS_Store':
                        ensembleList = ensembleList[1:]
                    
                    # Set up dataArrays for storing the gmst
                    # every ensemble member. Dimensions: year, variant
                    gmst_allYears = xr.DataArray(data=0.0, dims=['variant','year'],
                                                 coords=[ensembleList,years])
                    
                    for thisVariant in ensembleList:
                        print(thisVariant)
                        print('Trying the function. Wish me luck!')
                        data = gmst_calc(thisDir, thisSSP, 2100, thisVariant)
                        print('It worked!')
                        
                        # For datasets that cut off before 2100 (e.g. ssp370-lowNTCF and one or two
                        # that end on 31/12/2099), add in nan placeholders.
                        Ymax = np.max(data['tasAverageRel'].year.values) # What's the last year in 'data'?
                        if Ymax < 2100:
                            print('End date <2100. Padding with nan')
                            Yextra = years[years>Ymax] # Extra years we need to fill
                            fillArr = xr.DataArray(data=np.nan, dims=['year'], coords=[Yextra])
                            fillSet = xr.Dataset({'tasAverage':fillArr, 'tasAverageRel':fillArr})
                            data = data.merge(fillSet)
                            print('Success')
                        
                        print('cp-a1')
                        gmst_allYears.loc[dict(variant=thisVariant)] = data['tasAverageRel'].loc[dict(year=years)]
                        print('cp-a2')
                    # Save the output
                    print('cp-b1')
                    gmst_allYears.loc[dict(year=years)].to_netcdf(path='dea2022_allEns_gmst_'+thisSSP+'_allYears.nc')
                    print('cp-b2')
                except:
                    pass
            os.chdir(homedir+'/'+variable_id+'/'+table_id)

