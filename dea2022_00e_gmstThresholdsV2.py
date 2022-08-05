#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:55:26 2022

@author: 1001883
"""

# Douglas et al. 2022 optional script
# Compute S/N at different warming thresholds (e.g. 1, 1.5, 2.0, etc.) 
# and compare between generations.

# Inputs: population and area exposed netcdfs from Script 3, gridded population data,
# multi-model S, N, S/N netcdfs from script 2, as well as outputs from CMIP5 series
# of scripts and the population processing series of scripts. 
# Outputs: Figure S4

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

###############################
####----USER-CHANGEABLE----####

# Path to the multi-model S/N etc. data
inPath = '/Volumes/GeoCCRI_01/users-data/douglahu/CMIP6_data/tas/Amon/01_MM_output/'
inPath_rcp = '/Volumes/GeoCCRI_01/users-data/douglahu/CMIP5_data/tas/Amon/01_MM_output/'

scenarios = ['ssp119','ssp126','ssp245','ssp370','ssp585']
scenarios_rcp = ['rcp26','rcp45','rcp85']

avgType = 'median'

# Path to the 'N_RCP_eoc' and 'N_SSP_eoc' files
noisePath = '/Volumes/GeoCCRI_01/users-data/douglahu/Scripts/FEA2017/'

# GWL thresholds
tHold = [1.5,2.0,2.5,3.0]

# Figure 1 colours 
paletteF1 = ['#FFFFFF','#FFFEC7', '#FCD287', '#E67474', '#B13E3E']

####----END USER-CHANGEABLE----####
###################################

print('loading files')
# S/N ratios
SN_ssp119_orig = xr.open_dataarray(inPath+'MM_dea2022_SN_ssp119_allYears.nc')
SN_ssp126 = xr.open_dataarray(inPath+'MM_dea2022_SN_ssp126_allYears.nc')
SN_ssp245 = xr.open_dataarray(inPath+'MM_dea2022_SN_ssp245_allYears.nc')
SN_ssp370 = xr.open_dataarray(inPath+'MM_dea2022_SN_ssp370_allYears.nc')
SN_ssp585 = xr.open_dataarray(inPath+'MM_dea2022_SN_ssp585_allYears.nc')

SN_rcp26 = xr.open_dataarray(inPath_rcp+'MM_dea2022_SN_rcp26_allYears.nc')
SN_rcp45 = xr.open_dataarray(inPath_rcp+'MM_dea2022_SN_rcp45_allYears.nc')
SN_rcp85 = xr.open_dataarray(inPath_rcp+'MM_dea2022_SN_rcp85_allYears.nc')

# signal
signal_ssp119_orig = xr.open_dataarray(inPath+'MM_dea2022_signal_ssp119_allYears.nc')
signal_ssp126 = xr.open_dataarray(inPath+'MM_dea2022_signal_ssp126_allYears.nc')
signal_ssp245 = xr.open_dataarray(inPath+'MM_dea2022_signal_ssp245_allYears.nc')
signal_ssp370 = xr.open_dataarray(inPath+'MM_dea2022_signal_ssp370_allYears.nc')
signal_ssp585 = xr.open_dataarray(inPath+'MM_dea2022_signal_ssp585_allYears.nc')

signal_rcp26 = xr.open_dataarray(inPath_rcp+'MM_dea2022_signal_rcp26_allYears.nc')
signal_rcp45 = xr.open_dataarray(inPath_rcp+'MM_dea2022_signal_rcp45_allYears.nc')
signal_rcp85 = xr.open_dataarray(inPath_rcp+'MM_dea2022_signal_rcp85_allYears.nc')

# GMST
gmst_ssp119_orig = xr.open_dataarray(inPath+'MM_dea2022_gmst_ssp119.nc')
gmst_ssp126 = xr.open_dataarray(inPath+'MM_dea2022_gmst_ssp126.nc')
gmst_ssp245 = xr.open_dataarray(inPath+'MM_dea2022_gmst_ssp245.nc')
gmst_ssp370 = xr.open_dataarray(inPath+'MM_dea2022_gmst_ssp370.nc')
gmst_ssp585 = xr.open_dataarray(inPath+'MM_dea2022_gmst_ssp585.nc')

gmst_rcp26 = xr.open_dataarray(inPath_rcp+'MM_dea2022_gmst_rcp26.nc')
gmst_rcp45 = xr.open_dataarray(inPath_rcp+'MM_dea2022_gmst_rcp45.nc')
gmst_rcp85 = xr.open_dataarray(inPath_rcp+'MM_dea2022_gmst_rcp85.nc')

# Expand the ssp119 data to fit the 37 models used by the other scenarios by filling with nan
lat = SN_ssp245.lat.values
lon = SN_ssp245.lon.values
models = SN_ssp245.model.values
models_gmst = gmst_ssp245.model.values
years = SN_ssp245.year.values

# Set up some empty dataarrays
SN_ssp119 = xr.DataArray(data=np.nan, dims=['lat','lon','model','year'], 
                         coords=[lat,lon,models,years])
signal_ssp119 = xr.DataArray(data=np.nan, dims=['lat','lon','model','year'], 
                             coords=[lat,lon,models,years])
gmst_ssp119 = xr.DataArray(data=np.nan, dims=['model','year'], 
                           coords=[models_gmst,years])

# Loop through models and fill the array with available data
for thisMod in models:
    try:
        SN_ssp119.loc[dict(model=thisMod)] = SN_ssp119_orig.loc[dict(model=thisMod)].values
        #signal_ssp119.loc[dict(model=thisMod)] = signal_ssp119_orig.loc[dict(model=thisMod)].values
        gmst_ssp119.loc[dict(model=thisMod)] = gmst_ssp119_orig.loc[dict(model=thisMod)].values
    except:
        print('skipping '+str(thisMod))


# Multi-model averaging
print('multi-model averaging')
if avgType == 'median':
    # SN_ssp119_MM = SN_ssp119.median('model')
    # SN_ssp126_MM = SN_ssp126.median('model')
    # SN_ssp245_MM = SN_ssp245.median('model')
    # SN_ssp370_MM = SN_ssp370.median('model')
    # SN_ssp585_MM = SN_ssp585.median('model') 
    
    # signal_ssp119_MM = signal_ssp119.median('model')
    # signal_ssp126_MM = signal_ssp126.median('model')
    # signal_ssp245_MM = signal_ssp245.median('model')
    # signal_ssp370_MM = signal_ssp370.median('model')
    # signal_ssp585_MM = signal_ssp585.median('model') 
    
    gmst_ssp119_MM = gmst_ssp119.median('model',skipna=True)
    gmst_ssp126_MM = gmst_ssp126.median('model')
    gmst_ssp245_MM = gmst_ssp245.median('model')
    gmst_ssp370_MM = gmst_ssp370.median('model')
    gmst_ssp585_MM = gmst_ssp585.median('model')
    
    # SN_rcp26_MM = SN_rcp26.median('model')
    # SN_rcp45_MM = SN_rcp45.median('model')
    # SN_rcp85_MM = SN_rcp85.median('model')
    
    # signal_rcp26_MM = signal_rcp26.median('model')
    # signal_rcp45_MM = signal_rcp45.median('model')
    # signal_rcp85_MM = signal_rcp85.median('model')
    
    gmst_rcp26_MM = gmst_rcp26.median('model')
    gmst_rcp45_MM = gmst_rcp45.median('model')
    gmst_rcp85_MM = gmst_rcp85.median('model')
    
elif avgType == 'mean':
    # SN_ssp119_MM = SN_ssp119.mean('model')
    # SN_ssp126_MM = SN_ssp126.mean('model')
    # SN_ssp245_MM = SN_ssp245.mean('model')
    # SN_ssp370_MM = SN_ssp370.mean('model')
    # SN_ssp585_MM = SN_ssp585.mean('model')
    
    # signal_ssp119_MM = signal_ssp119.mean('model')
    # signal_ssp126_MM = signal_ssp126.mean('model')
    # signal_ssp245_MM = signal_ssp245.mean('model')
    # signal_ssp370_MM = signal_ssp370.mean('model')
    # signal_ssp585_MM = signal_ssp585.mean('model') 
    
    gmst_ssp119_MM = gmst_ssp119.mean('model',skipna=True)
    gmst_ssp126_MM = gmst_ssp126.mean('model')
    gmst_ssp245_MM = gmst_ssp245.mean('model')
    gmst_ssp370_MM = gmst_ssp370.mean('model')
    gmst_ssp585_MM = gmst_ssp585.mean('model')
    
    # SN_rcp26_MM = SN_rcp26.mean('model')
    # SN_rcp45_MM = SN_rcp45.mean('model')
    # SN_rcp85_MM = SN_rcp85.mean('model')
    
    # signal_rcp26_MM = signal_rcp26.mean('model')
    # signal_rcp45_MM = signal_rcp45.mean('model')
    # signal_rcp85_MM = signal_rcp85.mean('model')
    
    gmst_rcp26_MM = gmst_rcp26.mean('model')
    gmst_rcp45_MM = gmst_rcp45.mean('model')
    gmst_rcp85_MM = gmst_rcp85.mean('model')
    
# Collate all gmst into dataarrays
print('collating gmst into single array')
# One for all the data
gmst_all = xr.DataArray(data=np.nan,dims=['model','year','scenario'],
                           coords=[gmst_ssp245.model.values,gmst_ssp245.year.values,scenarios])
gmst_all.loc[dict(scenario='ssp119')] = gmst_ssp119
gmst_all.loc[dict(scenario='ssp126')] = gmst_ssp126
gmst_all.loc[dict(scenario='ssp245')] = gmst_ssp245
gmst_all.loc[dict(scenario='ssp370')] = gmst_ssp370
gmst_all.loc[dict(scenario='ssp585')] = gmst_ssp585

# One for the multi-model averages
gmst_all_MM = xr.DataArray(data=np.nan,dims=['year','scenario'],
                           coords=[gmst_ssp119_MM.year.values,scenarios])
gmst_all_MM.loc[dict(scenario='ssp119')] = gmst_ssp119_MM
gmst_all_MM.loc[dict(scenario='ssp126')] = gmst_ssp126_MM
gmst_all_MM.loc[dict(scenario='ssp245')] = gmst_ssp245_MM
gmst_all_MM.loc[dict(scenario='ssp370')] = gmst_ssp370_MM
gmst_all_MM.loc[dict(scenario='ssp585')] = gmst_ssp585_MM

# Repeat for RCPs 
gmst_all_rcp = xr.DataArray(data=np.nan,dims=['model','year','scenario'],
                               coords=[gmst_rcp26.model.values,gmst_rcp26.year.values,scenarios_rcp])
gmst_all_rcp.loc[dict(scenario='rcp26')] = gmst_rcp26
gmst_all_rcp.loc[dict(scenario='rcp45')] = gmst_rcp45
gmst_all_rcp.loc[dict(scenario='rcp85')] = gmst_rcp85

gmst_all_MM_rcp = xr.DataArray(data=np.nan,dims=['year','scenario'],
                               coords=[gmst_rcp26_MM.year.values,scenarios_rcp])
gmst_all_MM_rcp.loc[dict(scenario='rcp26')] = gmst_rcp26_MM
gmst_all_MM_rcp.loc[dict(scenario='rcp45')] = gmst_rcp45_MM
gmst_all_MM_rcp.loc[dict(scenario='rcp85')] = gmst_rcp85_MM

# Compute 20-year rolling gmst averages
gmst_all_ra = gmst_all.rolling(year=20).mean(skipna=True)#.dropna('year')
gmst_all_rcp_ra = gmst_all_rcp.rolling(year=20).mean()#.dropna('year')

gmst_all_MM_ra = gmst_all_MM.rolling(year=20).mean()#.dropna('year')
gmst_all_MM_rcp_ra = gmst_all_MM_rcp.rolling(year=20).mean()#.dropna('year')
  
# Create single dataarrays with all scenarios
print('collating SSPs into single array')

# For the full dataset
SN_all = xr.DataArray(data=np.nan,dims=['lat','lon','model','year','scenario'],coords=[lat,lon,models,years,scenarios])
SN_all.loc[dict(scenario='ssp119')]=SN_ssp119
SN_all.loc[dict(scenario='ssp126')]=SN_ssp126
SN_all.loc[dict(scenario='ssp245')]=SN_ssp245
SN_all.loc[dict(scenario='ssp370')]=SN_ssp370
SN_all.loc[dict(scenario='ssp585')]=SN_ssp585

signal_all = xr.DataArray(data=np.nan,dims=['lat','lon','model','year','scenario'],coords=[lat,lon,models,years,scenarios])
signal_all.loc[dict(scenario='ssp119')]=signal_ssp119
signal_all.loc[dict(scenario='ssp126')]=signal_ssp126
signal_all.loc[dict(scenario='ssp245')]=signal_ssp245
signal_all.loc[dict(scenario='ssp370')]=signal_ssp370
signal_all.loc[dict(scenario='ssp585')]=signal_ssp585

SN_all.to_netcdf(noisePath+'SN_SSP_all.nc')

# # And for the multi-model averages 
# SN_all_MM = xr.DataArray(data=np.nan,dims=['lat','lon','year','scenario'],coords=[lat,lon,years,scenarios])
# SN_all_MM.loc[dict(scenario='ssp119')]=SN_ssp119_MM
# SN_all_MM.loc[dict(scenario='ssp126')]=SN_ssp126_MM
# SN_all_MM.loc[dict(scenario='ssp245')]=SN_ssp245_MM
# SN_all_MM.loc[dict(scenario='ssp370')]=SN_ssp370_MM
# SN_all_MM.loc[dict(scenario='ssp585')]=SN_ssp585_MM

# signal_all_MM = xr.DataArray(data=np.nan,dims=['lat','lon','year','scenario'],coords=[lat,lon,years,scenarios])
# signal_all_MM.loc[dict(scenario='ssp119')]=signal_ssp119_MM
# signal_all_MM.loc[dict(scenario='ssp126')]=signal_ssp126_MM
# signal_all_MM.loc[dict(scenario='ssp245')]=signal_ssp245_MM
# signal_all_MM.loc[dict(scenario='ssp370')]=signal_ssp370_MM
# signal_all_MM.loc[dict(scenario='ssp585')]=signal_ssp585_MM

# Repeat for RCPs
print('collating RCPs into single array')
lat_rcp = SN_rcp26.lat.values
lon_rcp = SN_rcp26.lon.values
models_rcp = SN_rcp26.model.values
years_rcp = SN_rcp26.year.values

# For the full dataset
SN_all_rcp = xr.DataArray(data=np.nan,dims=['lat','lon','model','year','scenario'],
                          coords=[lat_rcp,lon_rcp,models_rcp,years_rcp,scenarios_rcp])
SN_all_rcp.loc[dict(scenario='rcp26')]=SN_rcp26
SN_all_rcp.loc[dict(scenario='rcp45')]=SN_rcp45
SN_all_rcp.loc[dict(scenario='rcp85')]=SN_rcp85

signal_all_rcp = xr.DataArray(data=np.nan,dims=['lat','lon','model','year','scenario'],
                              coords=[lat_rcp,lon_rcp,models_rcp,years_rcp,scenarios_rcp])
signal_all_rcp.loc[dict(scenario='rcp26')]=signal_rcp26
signal_all_rcp.loc[dict(scenario='rcp45')]=signal_rcp45
signal_all_rcp.loc[dict(scenario='rcp85')]=signal_rcp85

SN_all_rcp.to_netcdf(noisePath+'SN_RCP_all.nc')

# # And for the multi-model averages 
# SN_all_MM_rcp = xr.DataArray(data=np.nan,dims=['lat','lon','year','scenario'],
#                               coords=[lat_rcp,lon_rcp,years_rcp,scenarios_rcp])
# SN_all_MM_rcp.loc[dict(scenario='rcp26')]=SN_rcp26_MM
# SN_all_MM_rcp.loc[dict(scenario='rcp45')]=SN_rcp45_MM
# SN_all_MM_rcp.loc[dict(scenario='rcp85')]=SN_rcp85_MM

# signal_all_MM_rcp = xr.DataArray(data=np.nan,dims=['lat','lon','year','scenario'],
#                               coords=[lat_rcp,lon_rcp,years_rcp,scenarios_rcp])
# signal_all_MM_rcp.loc[dict(scenario='rcp26')]=signal_rcp26_MM
# signal_all_MM_rcp.loc[dict(scenario='rcp45')]=signal_rcp45_MM
# signal_all_MM_rcp.loc[dict(scenario='rcp85')]=signal_rcp85_MM


# Plot GMST by scenario
print('plotting gmst')
fig = plt.figure(0, figsize=(9,6))
fig.tight_layout()

yearsAll = gmst_all_MM.year.values

ax = fig.add_subplot(111)
ax.plot([2000,2110],[1.5,1.5],'k-',alpha=0.2)
ax.plot([2000,2110],[2.0,2.0],'k-',alpha=0.2)
ax.plot([2000,2110],[2.5,2.5],'k-',alpha=0.2)
ax.plot([2000,2110],[3.0,3.0],'k-',alpha=0.2)
ax.plot(yearsAll,gmst_all_MM.loc[dict(scenario='ssp585')],color='darkred')
ax.plot(yearsAll,gmst_all_MM_rcp.loc[dict(scenario='rcp85')],color='darkred',linestyle='dashed')
ax.plot(yearsAll,gmst_all_MM.loc[dict(scenario='ssp370')],color='darkorange')
ax.plot(yearsAll,gmst_all_MM.loc[dict(scenario='ssp245')],color='limegreen')
ax.plot(yearsAll,gmst_all_MM_rcp.loc[dict(scenario='rcp45')],color='limegreen',linestyle='dashed')
ax.plot(yearsAll,gmst_all_MM.loc[dict(scenario='ssp126')],color='deepskyblue')
ax.plot(yearsAll,gmst_all_MM_rcp.loc[dict(scenario='rcp26')],color='deepskyblue',linestyle='dashed')
ax.plot(yearsAll,gmst_all_MM.loc[dict(scenario='ssp119')],color='mediumblue')
ax.axis([2010,2100,0.5,5.5])
ax.legend(['_','_','_','_','ssp585','rcp85','ssp370','ssp245','rcp45','ssp126','rcp26','ssp119'],loc='upper left')
ax.set_ylabel('K')
ax.set_title('GMST above piControl')

fig.savefig(noisePath+'Plots/gmst.png', dpi=150, bbox_inches='tight')

# Repeat with rolling average
fig = plt.figure(1, figsize=(9,6))
fig.tight_layout()

yearsAll = gmst_all_MM_ra.year.values

ax = fig.add_subplot(111)
ax.plot([2000,2110],[1.5,1.5],'k-',alpha=0.2)
ax.plot([2000,2110],[2.0,2.0],'k-',alpha=0.2)
ax.plot([2000,2110],[2.5,2.5],'k-',alpha=0.2)
ax.plot([2000,2110],[3.0,3.0],'k-',alpha=0.2)
ax.plot(yearsAll,gmst_all_MM_ra.loc[dict(scenario='ssp585')],color='darkred')
ax.plot(yearsAll,gmst_all_MM_rcp_ra.loc[dict(scenario='rcp85')],color='darkred',linestyle='dashed')
ax.plot(yearsAll,gmst_all_MM_ra.loc[dict(scenario='ssp370')],color='darkorange')
ax.plot(yearsAll,gmst_all_MM_ra.loc[dict(scenario='ssp245')],color='limegreen')
ax.plot(yearsAll,gmst_all_MM_rcp_ra.loc[dict(scenario='rcp45')],color='limegreen',linestyle='dashed')
ax.plot(yearsAll,gmst_all_MM_ra.loc[dict(scenario='ssp126')],color='deepskyblue')
ax.plot(yearsAll,gmst_all_MM_rcp_ra.loc[dict(scenario='rcp26')],color='deepskyblue',linestyle='dashed')
ax.plot(yearsAll,gmst_all_MM_ra.loc[dict(scenario='ssp119')],color='mediumblue')
ax.axis([2010,2100,0.5,5.5])
ax.legend(['_','_','_','_','ssp585','rcp85','ssp370','ssp245','rcp45','ssp126','rcp26','ssp119'],loc='upper left')
ax.set_ylabel('K')
ax.set_title('GMST above piControl (20-year roll avg)')

fig.savefig(noisePath+'Plots/gmst_ra.png', dpi=150, bbox_inches='tight')

# When are thresholds crossed?
print('calculating threshold crossings for all models')
tHoldDatesAll = xr.DataArray(data=np.nan, dims=['model','scenario','threshold'],
                             coords=[models,scenarios,tHold])
for thisMod in models:
    for thisScen in scenarios:
        for this_tHold in tHold:
            try:
                yn = int(next(i for i in gmst_all_ra.loc[dict(model=thisMod, scenario=thisScen)] if i > this_tHold).year.values)
                print(yn)
                tHoldDatesAll.loc[dict(model=thisMod, scenario=thisScen, threshold=this_tHold)] = yn
            except:
                print('not crossed')
print(tHoldDatesAll)

tHoldDatesAll_rcp = xr.DataArray(data=np.nan, dims=['model','scenario','threshold'],
                                 coords=[models_rcp,scenarios_rcp,tHold])
for thisMod in models_rcp:
    for thisScen in scenarios_rcp:
        for this_tHold in tHold:
            try:
                yn = int(next(i for i in gmst_all_rcp_ra.loc[dict(model=thisMod, scenario=thisScen)] if i > this_tHold).year.values)
                print(yn)
                tHoldDatesAll_rcp.loc[dict(model=thisMod, scenario=thisScen, threshold=this_tHold)] = yn
            except:
                print('not crossed')
print(tHoldDatesAll_rcp)

# Save the threshold crossing dates as netcdfs
tHoldDatesAll.to_netcdf(path=noisePath+'GWLcrossDates.nc')
tHoldDatesAll_rcp.to_netcdf(path=noisePath+'GWLcrossDates_rcp.nc')

# print('calculating threshold crossings')
# tHold = [1.5,2.0,2.5,3.0]
# tHoldDates = pd.DataFrame(data=np.nan, index=tHold, columns=scenarios)
# for thisScen in scenarios:

#     for this_tHold in tHold:
#         try:
#             yn = int(next(i for i in gmst_all_MM_ra.loc[dict(scenario=thisScen)] if i > this_tHold).year.values)
#             tHoldDates[thisScen][this_tHold] = yn
#         except:
#             print('not crossed')
# tHoldDates['Count']=0
# print(tHoldDates)

# tHoldDates_rcp = pd.DataFrame(data=np.nan, index=tHold, columns=scenarios_rcp)
# for thisScen in scenarios_rcp:
#     for this_tHold in tHold:
#         try:
#             yn = int(next(i for i in gmst_all_MM_rcp_ra.loc[dict(scenario=thisScen)] if i > this_tHold).year.values)
#             tHoldDates_rcp[thisScen][this_tHold] = yn
#         except:
#             print('not crossed')
# tHoldDates_rcp['Count']=0
# print(tHoldDates_rcp)





# Next: get the average S/N and signal from the 20 years prior to the threshold 
# crossings and average across scenarios, then compare CMIP6 and CMIP5 by doing 
# another version of Fig 2. 

# I think everything up until here is working okay.

# Count the number of instances (model/scenario) that cross each threshold
counts = xr.DataArray(data=0, dims=['threshold'], coords=[tHold])
counts_rcp = xr.DataArray(data=0, dims=['threshold'], coords=[tHold])

SN_tHold_all = xr.DataArray(data=0.0, dims=['lat','lon','model','threshold'], coords=[lat,lon,models,tHold])
for this_tHold in tHold:
    print(this_tHold)
    for thisMod in models:
        print(thisMod)
        numCount = 0
        for thisScen in scenarios:
            print(thisScen)
            thisYr = tHoldDatesAll.loc[dict(model=thisMod, scenario=thisScen, threshold=this_tHold)].values
            if np.isnan(thisYr):
                print('not crossed')
            else:
                thisYr = int(thisYr)
                print(thisYr)
                theseSNvals = SN_all.loc[dict(scenario=thisScen, model=thisMod, year=slice(thisYr-19,thisYr))]
                # Get the 20-yr average
                thisSNavg = theseSNvals.mean('year')
                # Convert any nan values to 0 to avoid voiding out data
                thisSNavg = thisSNavg.where(~np.isnan(thisSNavg), other=0.0)
                
                SN_tHold_all.loc[dict(threshold=this_tHold, model=thisMod)] += thisSNavg.values
                numCount += 1
            
        if numCount != 0:
            # Average across the scenarios
            SN_tHold_all.loc[dict(threshold=this_tHold, model=thisMod)] = SN_tHold_all.loc[dict(threshold=this_tHold, model=thisMod)]/numCount   
        counts.loc[dict(threshold = this_tHold)] += numCount
       
signal_tHold_all = xr.DataArray(data=0.0, dims=['lat','lon','model','threshold'], coords=[lat,lon,models,tHold])
for this_tHold in tHold:
    print(this_tHold)
    for thisMod in models:
        print(thisMod)
        numCount = 0
        for thisScen in scenarios:
            print(thisScen)
            thisYr = tHoldDatesAll.loc[dict(model=thisMod, scenario=thisScen, threshold=this_tHold)].values
            if np.isnan(thisYr):
                print('not crossed')
            else:
                thisYr = int(thisYr)
                print(thisYr)
                thesesignalvals = signal_all.loc[dict(scenario=thisScen, model=thisMod, year=slice(thisYr-19,thisYr))]
                # Get the 20-yr average
                thissignalavg = thesesignalvals.mean('year')
                # Convert any nan values to 0 to avoid voiding out data
                thissignalavg = thissignalavg.where(~np.isnan(thissignalavg), other=0.0)
                
                signal_tHold_all.loc[dict(threshold=this_tHold, model=thisMod)] += thissignalavg.values
                numCount += 1
            
        if numCount != 0:
            # Average across the scenarios
            signal_tHold_all.loc[dict(threshold=this_tHold, model=thisMod)] = signal_tHold_all.loc[dict(threshold=this_tHold, model=thisMod)]/numCount


# SN_tHold = xr.DataArray(data=0.0, dims=['lat','lon','threshold'], coords=[lat,lon,tHold])
# for this_tHold in tHold:
#     print(this_tHold)
#     numCount = 0
#     for thisScen in scenarios:
#         print(thisScen)
#         thisYr = tHoldDates[thisScen][this_tHold]
#         if np.isnan(thisYr):
#             print('not crossed')
#         else:
#             print(thisYr)
#             theseSNvals = SN_all_MM.loc[dict(scenario=thisScen, year=slice(thisYr-19,thisYr))]
#             # Get the 20-yr average
#             thisSNavg = theseSNvals.mean('year')
#             SN_tHold.loc[dict(threshold=this_tHold)] += thisSNavg.values
#             numCount += 1
            
#     if numCount != 0:
#         # Average across the scenarios
#         SN_tHold.loc[dict(threshold=this_tHold)] = SN_tHold.loc[dict(threshold=this_tHold)]/numCount   
#     tHoldDates['Count'][this_tHold] = numCount
        
# signal_tHold = xr.DataArray(data=0.0, dims=['lat','lon','threshold'], coords=[lat,lon,tHold])
# for this_tHold in tHold:
#     print(this_tHold)
#     numCount = 0
#     for thisScen in scenarios:
#         print(thisScen)
#         thisYr = tHoldDates[thisScen][this_tHold]
#         if np.isnan(thisYr):
#             print('not crossed')
#         else:
#             print(thisYr)
#             thesesignalvals = signal_all_MM.loc[dict(scenario=thisScen, year=slice(thisYr-19,thisYr))]
#             # Get the 20-yr average
#             thissignalavg = thesesignalvals.mean('year')
#             signal_tHold.loc[dict(threshold=this_tHold)] += thissignalavg.values
#             numCount += 1
            
#     if numCount != 0:
#         # Average across the scenarios
#         signal_tHold.loc[dict(threshold=this_tHold)] = signal_tHold.loc[dict(threshold=this_tHold)]/numCount


# Repeat for the RCPs

SN_tHold_all_rcp = xr.DataArray(data=0.0, dims=['lat','lon','model','threshold'], coords=[lat,lon,models_rcp,tHold])
for this_tHold in tHold:
    print(this_tHold)
    for thisMod in models_rcp:
        print(thisMod)
        numCount = 0
        for thisScen in scenarios_rcp:
            print(thisScen)
            thisYr = tHoldDatesAll_rcp.loc[dict(model=thisMod, scenario=thisScen, threshold=this_tHold)].values
            if np.isnan(thisYr):
                print('not crossed')
            else:
                thisYr = int(thisYr)
                print(thisYr)
                theseSNvals = SN_all_rcp.loc[dict(scenario=thisScen, model=thisMod, year=slice(thisYr-19,thisYr))]
                # Get the 20-yr average
                thisSNavg = theseSNvals.mean('year')
                # Convert any nan values to 0 to avoid voiding out data
                thisSNavg = thisSNavg.where(~np.isnan(thisSNavg), other=0.0)
                
                SN_tHold_all_rcp.loc[dict(threshold=this_tHold, model=thisMod)] += thisSNavg.values
                numCount += 1
            
        if numCount != 0:
            # Average across the scenarios
            SN_tHold_all_rcp.loc[dict(threshold=this_tHold, model=thisMod)] = SN_tHold_all_rcp.loc[dict(threshold=this_tHold, model=thisMod)]/numCount
        counts_rcp.loc[dict(threshold = this_tHold)] += numCount
        
signal_tHold_all_rcp = xr.DataArray(data=0.0, dims=['lat','lon','model','threshold'], coords=[lat,lon,models_rcp,tHold])
for this_tHold in tHold:
    print(this_tHold)
    for thisMod in models_rcp:
        print(thisMod)
        numCount = 0
        for thisScen in scenarios_rcp:
            print(thisScen)
            thisYr = tHoldDatesAll_rcp.loc[dict(model=thisMod, scenario=thisScen, threshold=this_tHold)].values
            if np.isnan(thisYr):
                print('not crossed')
            else:
                thisYr = int(thisYr)
                print(thisYr)
                thesesignalvals = signal_all_rcp.loc[dict(scenario=thisScen, model=thisMod, year=slice(thisYr-19,thisYr))]
                # Get the 20-yr average
                thissignalavg = thesesignalvals.mean('year')
                # Convert any nan values to 0 to avoid voiding out data
                thissignalavg = thissignalavg.where(~np.isnan(thissignalavg), other=0.0)
                
                signal_tHold_all_rcp.loc[dict(threshold=this_tHold, model=thisMod)] += thissignalavg.values
                numCount += 1
            
        if numCount != 0:
            # Average across the scenarios
            signal_tHold_all_rcp.loc[dict(threshold=this_tHold, model=thisMod)] = signal_tHold_all_rcp.loc[dict(threshold=this_tHold, model=thisMod)]/numCount

# A test - save before doing the conversion
SN_tHold_all.to_netcdf(path=noisePath+'SN_tHold_all_test.nc')
SN_tHold_all_rcp.to_netcdf(path=noisePath+'SN_tHold_all_test_rcp.nc')

print('Checking if there are any zero values in the arrays.')
print('SN_tHold_all')
print(np.min(np.abs(SN_tHold_all.values)))
print('signal_tHold_all')
print(np.min(np.abs(signal_tHold_all.values)))
print('SN_tHold_all_rcp')
print(np.min(np.abs(SN_tHold_all_rcp.values)))
print('signal_tHold_all_rcp')
print(np.min(np.abs(signal_tHold_all_rcp.values)))

# Convert zero values to nan to prevent skewing averages
SN_tHold_all = SN_tHold_all.where(SN_tHold_all!=0.0, other=np.nan)
signal_tHold_all = signal_tHold_all.where(signal_tHold_all!=0.0, other=np.nan)

SN_tHold_all_rcp = SN_tHold_all_rcp.where(SN_tHold_all_rcp!=0.0, other=np.nan)
signal_tHold_all_rcp = signal_tHold_all_rcp.where(signal_tHold_all_rcp!=0.0, other=np.nan)

print('Checking again if there are any zero values in the arrays.')
print('SN_tHold_all')
print(np.min(np.abs(SN_tHold_all.values)))
print('signal_tHold_all')
print(np.min(np.abs(signal_tHold_all.values)))
print('SN_tHold_all_rcp')
print(np.min(np.abs(SN_tHold_all_rcp.values)))
print('signal_tHold_all_rcp')
print(np.min(np.abs(signal_tHold_all_rcp.values)))

# Save these arrays for future reference
SN_tHold_all.to_netcdf(path=noisePath+'SN_tHold_all.nc')
SN_tHold_all_rcp.to_netcdf(path=noisePath+'SN_tHold_all_rcp.nc')

signal_tHold_all.to_netcdf(path=noisePath+'signal_tHold_all.nc')
signal_tHold_all_rcp.to_netcdf(path=noisePath+'signal_tHold_all_rcp.nc')


# SN_tHold_rcp = xr.DataArray(data=0.0, dims=['lat','lon','threshold'], coords=[lat,lon,tHold])
# for this_tHold in tHold:
#     print(this_tHold)
#     numCount = 0
#     for thisScen in scenarios_rcp:
#         print(thisScen)
#         thisYr = tHoldDates_rcp[thisScen][this_tHold]
#         if np.isnan(thisYr):
#             print('not crossed')
#         else:
#             print(thisYr)
#             theseSNvals = SN_all_MM_rcp.loc[dict(scenario=thisScen, year=slice(thisYr-19,thisYr))]
#             # Get the 20-yr average
#             thisSNavg = theseSNvals.mean('year')
#             SN_tHold_rcp.loc[dict(threshold=this_tHold)] += thisSNavg.values
#             numCount += 1
            
#     if numCount != 0:
#         # Average across the scenarios
#         SN_tHold_rcp.loc[dict(threshold=this_tHold)] = SN_tHold_rcp.loc[dict(threshold=this_tHold)]/numCount
#     tHoldDates_rcp['Count'][this_tHold] = numCount
        
# signal_tHold_rcp = xr.DataArray(data=0.0, dims=['lat','lon','threshold'], coords=[lat,lon,tHold])
# for this_tHold in tHold:
#     print(this_tHold)
#     numCount = 0
#     for thisScen in scenarios_rcp:
#         print(thisScen)
#         thisYr = tHoldDates_rcp[thisScen][this_tHold]
#         if np.isnan(thisYr):
#             print('not crossed')
#         else:
#             print(thisYr)
#             thesesignalvals = signal_all_MM_rcp.loc[dict(scenario=thisScen, year=slice(thisYr-19,thisYr))]
#             # Get the 20-yr average
#             thissignalavg = thesesignalvals.mean('year')
#             signal_tHold_rcp.loc[dict(threshold=this_tHold)] += thissignalavg.values
#             numCount += 1
            
#     if numCount != 0:
#         # Average across the scenarios
#         signal_tHold_rcp.loc[dict(threshold=this_tHold)] = signal_tHold_rcp.loc[dict(threshold=this_tHold)]/numCount


# Now, as a last step, perform the multi-model averaging

print('multi-model averaging')
if avgType == 'median':
    SN_tHold = SN_tHold_all.median('model')
    SN_tHold_rcp = SN_tHold_all_rcp.median('model')
    signal_tHold = signal_tHold_all.median('model')
    signal_tHold_rcp = signal_tHold_all_rcp.median('model')
    
elif avgType == 'mean':
    SN_tHold = SN_tHold_all.mean('model')
    SN_tHold_rcp = SN_tHold_all_rcp.mean('model')
    signal_tHold = signal_tHold_all.mean('model')
    signal_tHold_rcp = signal_tHold_all_rcp.mean('model')

###---- Calculate the differences and make a new version of Fig 2 ----###

print('calculating differences')

N_RCP_eoc = xr.open_dataarray(noisePath+'N_RCP_eoc.nc')
N_SSP_eoc = xr.open_dataarray(noisePath+'N_SSP_eoc.nc')

# pcts = SN_RCP_pct.percentile
valLabs = ['$\Delta$ Noise', '$\Delta$ Signal', '$\Delta$ S/N', '$\Delta$ S/N (%)']

N_SSP_cut = N_SSP_eoc.loc[dict(scenario=['ssp126','ssp245','ssp585'])]

# In order to subtract RCPs from the SSPs, need to give them the same dimensions
N_RCP_eoc = N_RCP_eoc.assign_coords(scenario=['ssp126','ssp245','ssp585'])

diffArr_N = N_SSP_cut - N_RCP_eoc
diffArr_S = signal_tHold - signal_tHold_rcp
diffArr_SN = SN_tHold - SN_tHold_rcp
diffArr_SN_pc = (SN_tHold - SN_tHold_rcp)/SN_tHold_rcp*100 # Might need to convert to datasets
print('Max: ',np.max(diffArr_SN))
print('Min: ',np.min(diffArr_SN))
print('95th %ile: ',np.percentile(diffArr_SN,95))
print('5th %ile: ',np.percentile(diffArr_SN,5))

# Compute area-average changes
weights = np.cos(np.deg2rad(diffArr_N['lat']))
N_diff_globAvg = (diffArr_N*weights).sum('lat').mean('lon')/np.sum(weights)
print(N_diff_globAvg)
S_diff_globAvg = (diffArr_S*weights).sum('lat').mean('lon')/np.sum(weights)
print(S_diff_globAvg)
SN_diff_globAvg = (diffArr_SN*weights).sum('lat').mean('lon')/np.sum(weights)
print(SN_diff_globAvg)
SN_pc_diff_globAvg = (diffArr_SN_pc*weights).sum('lat').mean('lon')/np.sum(weights)
print(SN_pc_diff_globAvg)


# Make some plots
print('plotting big grid')

labs = ['1.5 K', '2.0 K', '2.5 K', '3.0 K']
cbLabs = ['K','K','$\sigma$','%']

def multi_plotter(thisArray_N, thisArray_S, thisArray_SN, thisArray_SN_pc):
    
    import cartopy.crs as ccrs
    import matplotlib as mpl
    
    # Packages for subplots
    from cartopy.mpl.geoaxes import GeoAxes
    from mpl_toolkits.axes_grid1 import AxesGrid
    
    # Packages to make it wrap
    from cartopy.util import add_cyclic_point
    import matplotlib.ticker as mticker
    from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
    
    title = 'Signal-to-noise at varying warming levels (20-yr avg): Change from CMIP5 to CMIP6'
    #levels = [-2.0,-1.5,-1.0,-0.5,0,0.5,1.0,1.5,2.0]
    #cbTicks = np.arange(0,30.1,(30.0/9))
    #cbLabels = ['0.0','1.6','2.2','2.8','3.6','4.4','5.4','6.7','9.0','30.0']
    
    #cmap = mpl.cm.RdBu_r
    #norm = mpl.colors.BoundaryNorm(levels, cmap.N)
    
    proj = ccrs.Robinson(central_longitude=0)
    axes_class = (GeoAxes,dict(map_projection=proj))
    
    fig=plt.figure(figsize=(20,12))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(len(tHold), 4),
                    axes_pad=0.4,
                    cbar_location='bottom',
                    cbar_mode='edge',#'single',
                    cbar_pad=0.2,
                    cbar_size='8%',
                    label_mode='')  # note the empty label_mode
    
    for i, ax in enumerate(axgr):
        print(i)
        thisTH = thisArray_SN.threshold.values[int(np.floor(i/4))] # thisArray_SN.threshold.values[i]#
        thisPct = 50 #pcts[np.mod(i,3)]
        
        # Select noise, signal, or S/N
        if np.mod(i,4)==0:
            theseData = thisArray_N.loc[dict(scenario='ssp126')]
            vmin_i = -0.2
            vmax_i = 0.2
            cmap = mpl.cm.PiYG_r
        elif np.mod(i,4)==1:
            theseData = thisArray_S.loc[dict(threshold=thisTH)]
            vmin_i = -2
            vmax_i = 2
            cmap = mpl.cm.RdBu_r
        elif np.mod(i,4)==2:
            theseData = thisArray_SN.loc[dict(threshold=thisTH)]
            vmin_i = -2
            vmax_i = 2
            cmap = mpl.cm.RdBu_r
        elif np.mod(i,4)==3:
            theseData = thisArray_SN_pc.loc[dict(threshold=thisTH)]
            vmin_i = -50
            vmax_i = 50
            cmap = mpl.cm.RdBu_r
        
        lon = theseData.coords['lon']
        print("Original shape: ", theseData.shape)
        
        lon_idx = theseData.dims.index('lon')
        wrap_data, wrap_lon = add_cyclic_point(theseData.values, coord=lon, axis=lon_idx)
        print('New shape: ', wrap_data.shape)
            
        cf = ax.pcolormesh(wrap_lon, theseData.lat.values, wrap_data, 
                            transform=ccrs.PlateCarree(), cmap = cmap,
                            rasterized = True,
                            vmin=vmin_i,
                            vmax=vmax_i)
               
        ax.coastlines()
        
        # Label the columns
        if i < 4 :
            ax.set_title(valLabs[i])
            axgr.cbar_axes[i].colorbar(cf) 
        
        # Label the rows
        if np.mod(i,4) == 0:
            thisSSPn = int(counts.loc[dict(threshold=thisTH)].values) #tHoldDates['Count'][thisTH]
            thisRCPn = int(counts_rcp.loc[dict(threshold=thisTH)].values) #tHoldDates_rcp['Count'][thisTH]
            thisLab = labs[int(np.floor(i/4))]+' ('+str(thisSSPn)+' SSPs, '+str(thisRCPn)+' RCPs)'
            ax.text(-0.04, 0.55, thisLab, va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=ax.transAxes)
        
        # Gridline customisation
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=2, color='lightgrey', alpha=0.5, linestyle=':')
        gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        gl.ylocator = mticker.FixedLocator([-90, -45, 0, 45, 90])
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {'size': 15, 'color': 'gray'}
        gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
    
    # Colourbar parameters
    j=0
    for cax in axgr.cbar_axes:
        axis = cax.axis[cax.orientation]
        if j<4:
            axis.label.set_text(cbLabs[j])
        j+=1
    
    #fig.suptitle(title, fontsize=20)
    #fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    #plt.show()
    
    return fig

gridPlot = multi_plotter(diffArr_N, diffArr_S, diffArr_SN, diffArr_SN_pc)
gridPlot.savefig(noisePath+'Plots/N-S-SN_SSP-RCP_tHold_v2.pdf', dpi=300, bbox_inches='tight')

##################################

###---- Now, make a new version of Fig 1 (probs should have done so first) ----###
print('Figure 1 redux in progress')

pcts = [16,50,84]
pctLabs = ['16th', '50th', '84th']
# Set up a dataarray for multi-model SN at each GWL (16th, 50th, 84th percentile)
SN_TH_pct = xr.DataArray(data=0.0, coords=[lat,lon,tHold,pcts], 
                          dims=['lat','lon','threshold','percentile'], name='SN')

# Calculate the multi-model 16th, 50th, and 84th percentiles
print('SSPs')
for thisTH in tHold:
    print(thisTH)
    try:
        # S/N
        SN_models = SN_tHold_all.loc[dict(threshold=thisTH)]
        print('calculating median')
        SN_Avg = SN_models.median('model', keep_attrs=True, skipna=True)
        print('calculating 16th percentile')
        SN_16 = SN_models.quantile(0.16, 'model', keep_attrs=True, skipna=True)
        print('calculating 84th percentile')
        SN_84 = SN_models.quantile(0.84, 'model', keep_attrs=True, skipna=True)
        SN_TH_pct.loc[dict(threshold=thisTH, percentile=16)]= SN_16.values
        SN_TH_pct.loc[dict(threshold=thisTH, percentile=50)]= SN_Avg.values
        SN_TH_pct.loc[dict(threshold=thisTH, percentile=84)]= SN_84.values
    except:
        print('Multi-model averaging failed.')
        pass

# Print the maximum of the EOC SN, for the colourbar
print(SN_TH_pct)
print('Maximum:')
print(np.max(SN_TH_pct))

#-----Same again, but for the RCPs----# 

SN_TH_pct_rcp = xr.DataArray(data=0.0, coords=[lat,lon,tHold,pcts], 
                             dims=['lat','lon','threshold','percentile'], name='SN')

# Calculate the multi-model 16th, 50th, and 84th percentiles
print('RCPs')
for thisTH in tHold:
    print(thisTH)
    try:
        # S/N
        SN_models = SN_tHold_all_rcp.loc[dict(threshold=thisTH)]
        print('calculating median')
        SN_Avg = SN_models.median('model', keep_attrs=True, skipna=True)
        print('calculating 16th percentile')
        SN_16 = SN_models.quantile(0.16, 'model', keep_attrs=True, skipna=True)
        print('calculating 84th percentile')
        SN_84 = SN_models.quantile(0.84, 'model', keep_attrs=True, skipna=True)
        SN_TH_pct_rcp.loc[dict(threshold=thisTH, percentile=16)]= SN_16.values
        SN_TH_pct_rcp.loc[dict(threshold=thisTH, percentile=50)]= SN_Avg.values
        SN_TH_pct_rcp.loc[dict(threshold=thisTH, percentile=84)]= SN_84.values
    except:
        print('Multi-model averaging failed.')
        pass


# Make a big grid plot of all GWLs and the 16th, 50th, and 84th percentiles

def multi_plotter(thisArray,THnum):
    plt.rc('font', size=10)
    
    import cartopy.crs as ccrs
    import matplotlib as mpl
    
    # Packages for subplots
    from cartopy.mpl.geoaxes import GeoAxes
    from mpl_toolkits.axes_grid1 import AxesGrid
    
    # Packages to make it wrap
    from cartopy.util import add_cyclic_point
    import matplotlib.ticker as mticker
    from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
    
    title = 'Modelled annual mean S/N values for GWLs'#+str(Yn1)+'-'+str(Yn2-1)#'2071–2100'
    cbar_label = '$\sigma$'
    #levels = [0.0,1.6,2.2,2.8,3.6,4.4,5.4,6.7,9.0,15.3]
    levels = [0.0,1.0,2.0,3.0,5.0,16.9]
    #cbTicks = np.arange(0,30.1,(30.0/9))
    #cbLabels = ['0.0','1.6','2.2','2.8','3.6','4.4','5.4','6.7','9.0','15.3']
    cbLabels = ['0.0','1.0','2.0','3.0','5.0','16.9']
    
    #cmap = mpl.cm.YlOrRd
    cmap = ListedColormap([paletteF1[0],paletteF1[1],paletteF1[2],paletteF1[3],paletteF1[4]])
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
    
    proj = ccrs.Robinson(central_longitude=0)
    axes_class = (GeoAxes,dict(map_projection=proj))
    
    fig=plt.figure(figsize=(10,2*THnum))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(THnum, 3),
                    axes_pad=0.2,
                    cbar_location='bottom',
                    cbar_mode='single',
                    cbar_pad=0,
                    cbar_size='4%',
                    label_mode='')  # note the empty label_mode
    
    for i, ax in enumerate(axgr):
        print(i)
        thisTH = thisArray.threshold.values[int(np.floor(i/3))]
        thisPct = pcts[np.mod(i,3)]
        
        theseData = thisArray.loc[dict(threshold=thisTH, percentile=thisPct)]
        
        lon = theseData.coords['lon']
        print("Original shape: ", theseData.shape)
        
        lon_idx = theseData.dims.index('lon')
        wrap_data, wrap_lon = add_cyclic_point(theseData.values, coord=lon, axis=lon_idx)
        print('New shape: ', wrap_data.shape)
            
        cf = ax.pcolormesh(wrap_lon, theseData.lat.values, wrap_data, 
                            #vmin=np.min(levels), vmax=np.max(levels),
                            rasterized = True, # if False, every gridcell becomes a vector file. File workability is bad.
                            transform=ccrs.PlateCarree(), cmap = cmap, norm = norm)
        # cf = ax.contourf(wrap_lon, theseData.lat.values, wrap_data, 
        #                    #vmin=np.min(levels), vmax=np.max(levels),
        #                    rasterized = True, # if False, every gridcell becomes a vector file. File workability is bad.
        #                    transform=ccrs.PlateCarree(), cmap = cmap, norm = norm)
               
        ax.coastlines(linewidth=0.5)
        
        # Label the columns
        if i < 3 :
            ax.set_title(pctLabs[i])
        
        # Label the rows
        if np.mod(i,3) == 0:
            ax.text(-0.04, 0.55, thisTH, va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=ax.transAxes)
        
        # Gridline customisation
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=1, color='lightgrey', alpha=0.5, linestyle=':')
        gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
        gl.ylocator = mticker.FixedLocator([-90, -45, 0, 45, 90])
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {'size': 15, 'color': 'gray'}
        gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
    
    # Colourbar parameters
    axgr.cbar_axes[0].colorbar(cf)#, ticks=cbTicks)
    cax = axgr.cbar_axes[0]
    axis = cax.axis[cax.orientation]
    axis.label.set_text(cbar_label)
    cax.set_xticklabels(cbLabels)
    
    # Optional supertitle 
    #fig.suptitle(title, fontsize='xx-large')
    
    #fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    #plt.show()
    
    return fig

gridPlot = multi_plotter(SN_TH_pct,4)
gridPlot.savefig(noisePath+'Plots/dea2022_SSP_SN-grid-all_GWLs.svg', bbox_inches='tight', dpi=300)
# .svg for editing in Inkscape to add the cartograms

# Repeat for the RCPs (only 3 of these that we're looking at)
gridPlot = multi_plotter(SN_TH_pct_rcp,4)
gridPlot.savefig(noisePath+'Plots/dea202_RCP_SN-grid-all_GWLs.pdf', bbox_inches='tight', dpi=300)