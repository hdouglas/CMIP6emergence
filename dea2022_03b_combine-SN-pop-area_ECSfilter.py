#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 16:01:02 2021

@author: 1001883
"""

# Douglas et al. 2022 Script 3
# Load and combine the signal-to-noise (S/N) with the gridded SSP population data,
# and land area.

# Inputs: multi-model netcdf S/N files, complted outputs from population processing
# scripts.
# Outputs: netcdfs with population and area exposed to different S/N thresholds
# (multi-model average)

import xarray as xr
import numpy as np
import xesmf as xe

###############################
####----USER-CHANGEABLE----####

avgType = 'median' # 'mean'
interpMethod = 'bilinear' # 'patch' # 'conservative' #  Set the interpolation method for regridding

# Path to gridded population dataset
popPath = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/popGridSSP_SEDAC.nc'
# Path to area dataset
areaPath = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/landAreas_0.25deg.nc'
# Path to multi-model S/N netcdfs
SNpath = '/Volumes/GeoCCRI_01/users-data/douglahu/CMIP6_data/tas/Amon/01_MM_output/'
# Path for outputs
outPath = '/Volumes/GeoCCRI_01/users-data/douglahu/Scripts/FEA2017/'

popGridSSP4D = xr.open_dataarray(popPath)
scenarios = popGridSSP4D.scenario.values
print(scenarios)
years = popGridSSP4D.year.values
print(years)

scen_short = ['ssp119','ssp126','ssp245','ssp370','ssp585']
# If you edit the population dataset, ensure that these scenarios align with scen_short
scen_long = [scenarios[0], scenarios[0], scenarios[1], scenarios[2], scenarios[4]]

# Set the S/N thresholds for assessing when populations cross these
thresholds = [1,2,3,5]

####----END USER-CHANGEABLE----####
###################################

# Set up empty DataArrays for storing the signal-to-noise for all scenarios 
SN_SSP4D = xr.DataArray(data=0.0, dims=['lat','lon','year','scenario'], 
                      coords=[popGridSSP4D.lat,popGridSSP4D.lon,popGridSSP4D.year,scen_short], 
                      name='SN') 
SN_SSP4D_16 = SN_SSP4D.copy(deep=True)
SN_SSP4D_84 = SN_SSP4D.copy(deep=True)

i = 0
for thisSSP in scen_short:
    print(thisSSP)
    SN_models = xr.open_dataarray(SNpath+'MM_dea2022_ECSfilter_SN_'+thisSSP+'_allYears.nc')
    
    # Multi-model averaging
    if avgType == 'median':
        SN_Avg = SN_models.median('model', keep_attrs=True, skipna=True)
        SN_16 = SN_models.quantile(0.16, 'model', keep_attrs=True, skipna=True)
        SN_84 = SN_models.quantile(0.84, 'model', keep_attrs=True, skipna=True)
    elif avgType == 'mean':
        SN_Avg = SN_models.mean('model', keep_attrs=True, skipna=True)
    else:
        print('Specify mean or median for avgType')
        exit
    
    # Some tricky things: the population data don't span all latitudes, and shouldn't 
    # be averaged across gridpoints. So instead we start with the SN data, and then
    # upscale and truncate it to match the population data.
    
    # Regrid SN data to the population grid with xESMF   
    ds_in = SN_Avg.to_dataset(promote_attrs=True)
    
    thisPop = popGridSSP4D.loc[dict(scenario = scen_long[i])]
    lat = thisPop.lat.values
    lon = thisPop.lon.values
    ds_out = xr.Dataset({'lat': (['lat'], lat), 'lon': (['lon'], lon)})
    
    regridder = xe.Regridder(ds_in, ds_out, interpMethod, periodic=True) 
    regridder  # print basic regridder information.
    
    SN_Avg = SN_Avg.transpose('lat','lon','year') # These need to be in the right order
    SN_Out = regridder(SN_Avg)
    SN_Out16 = regridder(SN_16)
    SN_Out84 = regridder(SN_84)
    
    #Reorder the dimensions as necessary:
    SN_Out = SN_Out.transpose('lat','lon','year')
    SN_SSP4D.loc[dict(scenario=thisSSP)]=SN_Out.values
    
    if avgType == 'median':   
        SN_Out16 = SN_Out16.transpose('lat','lon','year')
        SN_Out84 = SN_Out84.transpose('lat','lon','year')    
        SN_SSP4D_16.loc[dict(scenario=thisSSP)]=SN_Out16.values
        SN_SSP4D_84.loc[dict(scenario=thisSSP)]=SN_Out84.values
    
    i+=1

# Now we have dataarrays for the population and SN data with matching dimensions :)

# We're going to reuse the median regridded dataset in the 5th script, so save that here
SN_SSP4D.to_netcdf(path=outPath+'dea2022_ECSfilter_SN_SSP4D.nc')

#-----Time of Emergence (TOE) data-----#
# We want to produce cumulative distribution plots showing when different proportions
# of the world's population cross different signal-to-noise thresholds. Note that
# global population is different between scenarios.

# Create an empty dataarray with dimensions years, scenarios, and thresholds. 
# This will store the global population with signal-to-noise above a range of thresholds
# across the years, under the different scenarios.
 
pop_exposed = xr.DataArray(data=0.0, dims=['year','scenario','threshold'], 
                            coords=[years,scen_short,thresholds], name='persons')
pop_exposed_16 = xr.DataArray(data=0.0, dims=['year','scenario','threshold'], 
                            coords=[years,scen_short,thresholds], name='persons')
pop_exposed_84 = xr.DataArray(data=0.0, dims=['year','scenario','threshold'], 
                            coords=[years,scen_short,thresholds], name='persons')

# Loop across scenarios, thresholds, and years. Use the .where command to filter 
# the population data based on where the corresponding SN data exceed the thresholds. 

for i in range(len(scen_short)):
    thisSSP = scen_short[i]
    print(thisSSP)
    thisScenario = scen_long[i]
    for thisThreshold in thresholds:
        print(thisThreshold)
        for thisYear in years:
            thisPop = popGridSSP4D.loc[dict(year=thisYear, scenario=thisScenario)]
            thisSN = SN_SSP4D.loc[dict(year=thisYear, scenario=thisSSP)]
            pop_SN_over = thisPop.where(thisSN>thisThreshold)
            pop_exposed.loc[dict(year=thisYear, scenario=thisSSP, 
                                  threshold=thisThreshold)]=int(np.sum(pop_SN_over))
            
            thisSN_16 = SN_SSP4D_16.loc[dict(year=thisYear, scenario=thisSSP)]
            pop_SN_over_16 = thisPop.where(thisSN_16>thisThreshold)
            pop_exposed_16.loc[dict(year=thisYear, scenario=thisSSP, 
                                  threshold=thisThreshold)]=int(np.sum(pop_SN_over_16))
            
            thisSN_84 = SN_SSP4D_84.loc[dict(year=thisYear, scenario=thisSSP)]
            pop_SN_over_84 = thisPop.where(thisSN_84>thisThreshold)
            pop_exposed_84.loc[dict(year=thisYear, scenario=thisSSP, 
                                  threshold=thisThreshold)]=int(np.sum(pop_SN_over_84))
 
# Save the output as netCDFs for future use. 
pop_exposed.to_netcdf(path=outPath+'dea2022_ECSfilter_pop_exposed_SN_SSP.nc')
pop_exposed_16.to_netcdf(path=outPath+'dea2022_ECSfilter_pop_exposed_SN_SSP_16.nc')
pop_exposed_84.to_netcdf(path=outPath+'dea2022_ECSfilter_pop_exposed_SN_SSP_84.nc')

##-----Same again, but for Area----##

areaGrid = xr.open_dataarray(areaPath)

# Set up empty DataArrays for storing the signal-to-noise for all scenarios 
SN_SSP4D = xr.DataArray(data=0.0, dims=['lat','lon','year','scenario'], 
                      coords=[areaGrid.lat,areaGrid.lon,years,scen_short], 
                      name='SN') 
SN_SSP4D_16 = SN_SSP4D.copy(deep=True)
SN_SSP4D_84 = SN_SSP4D.copy(deep=True)

i = 0
for thisSSP in scen_short:
    print(thisSSP)
    SN_models = xr.open_dataarray(SNpath+'MM_dea2022_ECSfilter_SN_'+thisSSP+'_allYears.nc')
    
    # Multi-model averaging
    if avgType == 'median':
        SN_Avg = SN_models.median('model', keep_attrs=True, skipna=True)
        SN_16 = SN_models.quantile(0.16, 'model', keep_attrs=True, skipna=True)
        SN_84 = SN_models.quantile(0.84, 'model', keep_attrs=True, skipna=True)
    elif avgType == 'mean':
        SN_Avg = SN_models.mean('model', keep_attrs=True, skipna=True)
    else:
        print('Specify mean or median for avgType')
        exit
    
    # Regrid SN data to the area grid with xESMF   
    
    ds_in = SN_Avg.to_dataset(promote_attrs=True)
    
    thisarea = areaGrid
    lat = thisarea.lat.values
    lon = thisarea.lon.values
    ds_out = xr.Dataset({'lat': (['lat'], lat), 'lon': (['lon'], lon)})
    
    regridder = xe.Regridder(ds_in, ds_out, interpMethod, periodic=True) 
    regridder  # print basic regridder information.
    
    SN_Avg = SN_Avg.transpose('lat','lon','year') # These need to be in the right order
    SN_Out = regridder(SN_Avg)
    SN_Out16 = regridder(SN_16)
    SN_Out84 = regridder(SN_84)
    
    #Reorder the dimensions as necessary:
    SN_Out = SN_Out.transpose('lat','lon','year')
    SN_SSP4D.loc[dict(scenario=thisSSP)]=SN_Out.values
    
    if avgType == 'median': 
        SN_Out16 = SN_Out16.transpose('lat','lon','year')
        SN_Out84 = SN_Out84.transpose('lat','lon','year')       
        SN_SSP4D_16.loc[dict(scenario=thisSSP)]=SN_Out16.values
        SN_SSP4D_84.loc[dict(scenario=thisSSP)]=SN_Out84.values
    
    i+=1

# Now we have dataarrays for the area and SN data with matching dimensions :)

#-----TOE data-----#
# We want to produce cumulative distribution plots showing when different proportions
# of the world's land area crosses different signal-to-noise thresholds. 

# Create an empty dataarray with dimensions years, scenarios, and thresholds. 
# This will store the global land area with signal-to-noise above a range of thresholds
# across the years, under the different scenarios.
 
area_exposed = xr.DataArray(data=0.0, dims=['year','scenario','threshold'], 
                           coords=[years,scen_short,thresholds], name='km2')
area_exposed_16 = xr.DataArray(data=0.0, dims=['year','scenario','threshold'], 
                           coords=[years,scen_short,thresholds], name='km2')
area_exposed_84 = xr.DataArray(data=0.0, dims=['year','scenario','threshold'], 
                           coords=[years,scen_short,thresholds], name='km2')

# Loop across scenarios, thresholds, and years. Use the .where command to filter 
# the land area data based on where the corresponding SN data exceed the thresholds. 

for i in range(len(scen_short)):
    thisSSP = scen_short[i]
    print(thisSSP)
    for thisThreshold in thresholds:
        print(thisThreshold)
        for thisYear in years:
            thisarea = areaGrid
            thisSN = SN_SSP4D.loc[dict(year=thisYear, scenario=thisSSP)]
            area_SN_over = thisarea.where(thisSN>thisThreshold)
            area_exposed.loc[dict(year=thisYear, scenario=thisSSP, 
                                 threshold=thisThreshold)]=int(np.sum(area_SN_over))
            
            thisSN_16 = SN_SSP4D_16.loc[dict(year=thisYear, scenario=thisSSP)]
            area_SN_over_16 = thisarea.where(thisSN_16>thisThreshold)
            area_exposed_16.loc[dict(year=thisYear, scenario=thisSSP, 
                                 threshold=thisThreshold)]=int(np.sum(area_SN_over_16))
            
            thisSN_84 = SN_SSP4D_84.loc[dict(year=thisYear, scenario=thisSSP)]
            area_SN_over_84 = thisarea.where(thisSN_84>thisThreshold)
            area_exposed_84.loc[dict(year=thisYear, scenario=thisSSP, 
                                 threshold=thisThreshold)]=int(np.sum(area_SN_over_84))
 
# Save the output as netCDFs for future use. 
area_exposed.to_netcdf(path=outPath+'dea2022_ECSfilter_area_exposed_SN_SSP.nc')
area_exposed_16.to_netcdf(path=outPath+'dea2022_ECSfilter_area_exposed_SN_SSP_16.nc')
area_exposed_84.to_netcdf(path=outPath+'dea2022_ECSfilter_area_exposed_SN_SSP_84.nc')
