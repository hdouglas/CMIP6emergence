#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 12:43:24 2021

@author: 1001883
"""

# Douglas et al. 2022 Script 1 (population data processing)
# Combine the SEDAC gridded population datasets into a dataarray with dimensions
# lat, lon, year, scenario. Interpolate projections to get more frequent data.

# Inputs: SSP-aligned population projections from NASA's Socioeconomic Data and 
# Applications Center (SEDAC) available from 
# https://cmr.earthdata.nasa.gov/search/concepts/C2022275525-SEDAC.html

# Outputs: A rather large netcdf file with the gridded population data organised by scenario

import os
import xarray as xr
import numpy as np

###############################
####----USER-CHANGEABLE----####

# The SEDAC zip files should be extracted to this folder:
basePath = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/SEDAC/'

# Start and end years of the projections
y0 = 2010
y1 = 2100

# Only change if you downloaded other than the 1default 0-yearly projections
years = np.arange(y0, y1+1, 10) 

# Desired temporal resolution of output from interpolation 
outputRes = 1 # (years)
years2 = np.arange(y0, y1+1, outputRes) 

# Desired spatial resolution (native is 0.125) - suggest downscaling for managable file sizes
outResSpat = 0.25 # degrees (please set in multiples of 0.125)

# Path for where outputs should be saved:
outPath = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/'

####----END USER-CHANGEABLE----####
###################################

scenarios = ['ssp1','ssp2','ssp3','ssp4','ssp5']
scenariosCaps = ['SSP1','SSP2','SSP3','SSP4','SSP5']
os.chdir(basePath)

# Create an empty xarray in which to store the results
# Load a dummy file to get the lat/lon arrays
dumFile = 'popdynamics-1-8th-pop-base-year-projection-ssp-2000-2100-rev01-proj-ssp1-netcdf/SSP1/Total/NetCDF/ssp1_2010.nc'
dumArr = xr.open_dataarray(dumFile)
lat = dumArr.lat
lon = dumArr.lon
scenArr = xr.DataArray(data=np.zeros(len(scenarios)),dims='scenario',coords=[scenarios])
zeroArr = np.zeros([len(lat),len(lon),len(years)])
popArr0 = xr.DataArray(data=zeroArr,dims=['lat','lon','year'],coords=[lat,lon,years])
popArr_dum, dummy = xr.broadcast(popArr0,scenArr)
popArr = popArr_dum.copy(deep=True)

# Loop over all scenarios and add the results to the popArr
for i in np.arange(len(scenarios)):
    thisScen = scenarios[i]
    print(thisScen)
    thisScenCaps = scenariosCaps[i]
    os.chdir('popdynamics-1-8th-pop-base-year-projection-ssp-2000-2100-rev01-proj-'+thisScen+'-netcdf')
    os.chdir(thisScenCaps+'/Total/NetCDF/')
    
    fileList = np.sort(os.listdir())
    for thisFile in fileList:
        #thisFile = fileList[0]
        thisArr = xr.open_dataarray(thisFile)
        thisYear = int(thisArr.long_name[-4:])
        print(thisYear)
        #Maybe add a year dimension and then concatenate along it
        thisArr = thisArr.assign_coords({'year':thisYear})
        if thisFile == fileList[0]:
            thisScenData = thisArr
        else:
            thisScenData = xr.concat((thisScenData, thisArr), dim='year')  
    thisScenData = thisScenData.transpose('lat','lon','year')    
    popArr.loc[dict(scenario=thisScen)]=thisScenData
    os.chdir(basePath)
    
# Interpolate the results to the desired temporal resolution
popArr2 = popArr.fillna(0)
popArr2 = popArr2.interp(year=years2,method='quadratic')
popArr2 = popArr2.transpose('lat','lon','year','scenario') 

print('Pop array shape: '+str(popArr2.shape))

# Native resolution is 0.125 deg, which makes for ridiculously large files down
# the line. Coarsen to the user-set resolution.
if outResSpat > 0.125:
    print('Coarsening spatial resolution')
    Fscale = outResSpat/0.125
    if np.mod(Fscale,1) != 0:
        print('WARNING: scale factor not a multiple of 0.125. Rounding down to nearest multiple.')
    Fscale = int(Fscale)
    
    popArr2 = popArr2.coarsen(lat=Fscale,boundary='trim').sum()
    popArr2 = popArr2.coarsen(lon=Fscale,boundary='exact').sum()
    print('Pop array shape: '+str(popArr2.shape))

popArr2.to_netcdf(path=outPath+'popGridSSP_SEDAC.nc')

# Optional: Test out the results by plotting
import matplotlib.pyplot as plt
globPop = popArr2.sum('lat').sum('lon')
scenarios = globPop.scenario.values
years = globPop.year.values

fig = plt.figure(1)
ax = fig.add_subplot(111)
plt.plot(years,globPop.loc[dict(scenario=scenarios[0])],color='purple')
plt.plot(years,globPop.loc[dict(scenario=scenarios[1])],color='royalblue')
plt.plot(years,globPop.loc[dict(scenario=scenarios[2])],color='limegreen')
plt.plot(years,globPop.loc[dict(scenario=scenarios[3])],color='gold')
plt.plot(years,globPop.loc[dict(scenario=scenarios[4])],color='orangered')
plt.legend(scenarios)
plt.title('World population')

# Looks exactly the same as my previous work! Good :)