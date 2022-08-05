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

from dea2022_fn_signoise import signoise_calc 

###############################
####----USER-CHANGEABLE----####

# Set your home directory for where the data are stored
homedir = '/Volumes/GeoCCRI_01/users-data/douglahu/CMIP6_data'

# Set the parameters of interest:
variable_id = 'tas'
table_id = 'Amon'
framework = 'Frame2017' # 'globrollavg' #'Hawkins2020'
scen_short = ['ssp119','ssp126','ssp245','ssp370','ssp585']
years = np.arange(1990,2101,1) # Change if you want coarser temporal resolution
POI = [2040,2060] # Period of interest over which signal and SN are averaged
 
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
                fileList.index('dea2022_allEns_signal_'+str(POI[0])+'-'+str(POI[1])+'_'+thisSSP+'.ncdummy')
                print('Already processed. Skipping.')
            except:
                
                try:
                    fileList.index('piControl')
                    fileList.index('historical')
                    fileList.index(thisSSP)
                    
                    # Just calculate the signal, noise, and signaltonoise, then save these 
                    # files for later regridding 
                    
                    # Get the ensemble list to input to the signal-to-noise function
                    ensembleList = np.sort(os.listdir(homedir+'/'+variable_id+'/'+table_id+'/'+thisDir+'/'+thisSSP))
                    if ensembleList[0] == '.DS_Store':
                        ensembleList = ensembleList[1:]
                    
                    # Need to load one file to know the lat & lon sizes. Use historical, because
                    # that's what's used if there's a conflict between historical and SSP grids.
                    dum_ensList = np.sort(os.listdir(homedir+'/'+variable_id+'/'+table_id+'/'+thisDir+'/historical'))
                    if dum_ensList[0] == '.DS_Store':
                        dum_ensList = dum_ensList[1:]
                        
                    dum_grid = np.sort(os.listdir(homedir+'/'+variable_id+'/'+table_id+'/'+thisDir+
                                       '/historical/'+dum_ensList[0]))[-1]
                    dum_file = np.sort(os.listdir(homedir+'/'+variable_id+'/'+table_id+'/'+thisDir+
                                       '/historical/'+dum_ensList[0]+'/'+dum_grid))[-1]
                    dummy = xr.open_dataset(homedir+'/'+variable_id+'/'+table_id+
                                              '/'+thisDir+'/historical/'+dum_ensList[0]+
                                              '/'+dum_grid+'/'+dum_file, use_cftime=usingCFtime)
                    try:
                        print(len(dummy.lat))
                    except:
                        print('Renaming latitude and longitude')
                        dummy=dummy.rename({'latitude':'lat','longitude':'lon'})
                    lat = dummy.lat
                    lon = dummy.lon
                    
                    # Set up dataArrays for storing the signal, noise , etc. for 
                    # every ensemble member. Dimensions: lat, lon, (year), variant
                    yearsArr = xr.DataArray(data=np.zeros(len(years)),dims='year',coords=[years])
                    ensArr = xr.DataArray(data=np.zeros(len(ensembleList)),dims='variant',coords=[ensembleList])
                    
                    noiseArr = xr.DataArray(data=0.0, dims=['lat','lon','variant'],
                                            coords=[lat,lon,ensembleList])
                    signalArr = xr.DataArray(data=0.0, dims=['lat','lon','variant'],
                                            coords=[lat,lon,ensembleList])
                    SN_Arr = xr.DataArray(data=0.0, dims=['lat','lon','variant'],
                                            coords=[lat,lon,ensembleList])
                    signalArr_allYears = xr.DataArray(data=0.0, dims=['lat','lon','variant','year'],
                                            coords=[lat,lon,ensembleList,years])
                    SN_Arr_allYears = xr.DataArray(data=0.0, dims=['lat','lon','variant','year'],
                                            coords=[lat,lon,ensembleList,years])
                    
                    for thisVariant in ensembleList:
                        print(thisVariant)
                        #print('Trying the function. Wish me luck!')
                        data = signoise_calc(thisDir, thisSSP, framework, POI[0], POI[1], thisVariant)
                        #print('It worked!')
                        
                        # For datasets that cut off before 2100 (e.g. ssp370-lowNTCF and one or two
                        # that end on 31/12/2099), add in nan placeholders.
                        Ymax = np.max(data['signal'].year.values) # What's the last year in 'data'?
                        if Ymax < 2100:
                            print('End date <2100. Padding with nan')
                            Yextra = years[years>Ymax] # Extra years we need to fill
                            lat = data['signal'].lat
                            lon = data['signal'].lon
                            zeroArr = np.zeros([len(lat),len(lon),len(Yextra)])
                            zeroArr=np.nan
                            fillArr = xr.DataArray(data=zeroArr, dims=['lat','lon','year'], 
                                                   coords=[lat,lon,Yextra])
                            fillSet = xr.Dataset({'signal':fillArr, 'signaltonoise':fillArr})
                            data = data.merge(fillSet)
                        
                        noiseArr.loc[dict(variant=thisVariant)] = data['noise']
                        signalArr.loc[dict(variant=thisVariant)] = data['POIsignal'] 
                        SN_Arr.loc[dict(variant=thisVariant)] = data['POIsignaltonoise'] 
                        signalArr_allYears.loc[dict(variant=thisVariant)] = data['signal'].loc[dict(year=years)].transpose('lat','lon','year')
                        SN_Arr_allYears.loc[dict(variant=thisVariant)] = data['signaltonoise'].loc[dict(year=years)].transpose('lat','lon','year')
                    
                    # Save the signal etc. for both all years and the Period of Interest
                    # Note that "all years" is 2010, 2015, 2020, 2025, etc.
                    noiseArr.to_netcdf(path='dea2022_allEns_noise_'+thisSSP+'.nc')
                    signalArr.to_netcdf(path='dea2022_allEns_signal_'+str(POI[0])+'-'+str(POI[1])+'_'+thisSSP+'.nc')
                    SN_Arr.to_netcdf(path='dea2022_allEns_SN_'+str(POI[0])+'-'+str(POI[1])+'_'+thisSSP+'.nc')
                    signalArr_allYears.loc[dict(year=years)].to_netcdf(path='dea2022_allEns_signal_'+thisSSP+'_allYears.nc')
                    SN_Arr_allYears.loc[dict(year=years)].to_netcdf(path='dea2022_allEns_SN_'+thisSSP+'_allYears.nc')
                    
                except:
                    pass
            os.chdir(homedir+'/'+variable_id+'/'+table_id)

