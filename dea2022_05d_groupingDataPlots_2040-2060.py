#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:58:19 2021

@author: 1001883
"""

# Douglas et al. 2022 Script 5
# Country grouping plots

# Inputs: population and area exposed netcdfs from Script 3, gridded population data,
# country grouping csv file, multi-model S, N, S/N netcdfs from script 2, 
# as well as outputs from CMIP5 series of scripts and the population processing 
# series of scripts. 
# Outputs: The country grouping plots

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
#import xesmf as xe

###############################
####----USER-CHANGEABLE----####

# Output path
outPath = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/'
# Paths for where the time-of-emergence netcdfs are saved
toePath = '/Volumes/GeoCCRI_01/users-data/douglahu/Scripts/FEA2017/'
toePath_rcp = '/Volumes/GeoCCRI_01/users-data/douglahu/Scripts/CMIP5/'

# Load the gridded population data
popPath = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/popGridSSP_SEDAC.nc'
popGridSSP4D = xr.open_dataarray(popPath)
scenarios = popGridSSP4D.scenario.values
scen_long = [scenarios[0], scenarios[0], scenarios[1], scenarios[2], scenarios[4]]

# Load the groupings (mostly as per Frame et al. 2017, with some educated guesses for GEM and LDC)
groupPath='/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/groupings_codes.csv'
groupList = pd.read_csv(groupPath)
groupings = np.unique(groupList.Grouping)
print(groupings)

# Load the mask
maskPath = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/mask_3D_ext_SEDAC.nc'
mask_3D_ext = xr.open_dataarray(maskPath)

# Parameters for multi-model averaging and regridding
#avgType = 'median' # 'mean'
#interpMethod = 'bilinear' # 'patch' # 'conservative' #  Set the interpolation method for regridding

# Path to locate the multi-model signal-to-noise files
SNpath = '/Volumes/GeoCCRI_01/users-data/douglahu/CMIP6_data/tas/Amon/01_MM_output/'
scen_short = ['ssp119','ssp126','ssp245','ssp370','ssp585']

# Years in the end-of century period
# POI = np.arange(2070,2101,5)# (if 5-yearly resolution)
POIvals = [2040,2060]
POI = np.arange(POIvals[0], POIvals[1]+1, 1)

# Colours for plotting
colours = ['mediumblue', 'deepskyblue', 'limegreen', 'darkred', 'black'] #'darkorange', 
# A list of strings for aligning the RCPs with the SSPs
rcpinp = ['skip','rcp26','rcp45','skip','rcp85']

# Background colours
#palette = ['#FFFEC7', '#FEED9B', '#FBB246', '#FB4E2B'] #>1, >2, >3, >5
palette = ['#FFFEC7', '#FCD287', '#E67474', '#B13E3E']

####----END USER-CHANGEABLE----####
###################################

# # Filter the mask by the country codes in the groupings to get grouping masks

# Note that some countries (e.g. Singapore) may be too small to be included in the 
# mask, depending on the resolution, so searching for them will throw an error. Need to 
# filter for only those present.
# Add a new column that says if the country is included in the mask dataarray
groupList.insert(4,'InMask','0')
exceptions = []
for i in range(len(groupList)):
    try:
        mask_3D_ext.loc[dict(region=groupList.iloc[i].Code)]
        groupList.iat[i,4] = 1
    except:
        exceptions.append(groupList.iloc[i].Country)
print(exceptions)
groupListMask = groupList[groupList.InMask==1]        

# Create new masks for each of the country groupings

codesAOSIS = groupListMask[groupListMask.Grouping=='AOSIS'].Code.values
codesASEAN = groupListMask[groupListMask.Grouping=='ASEAN'].Code.values
codesGEM = groupListMask[groupListMask.Grouping=='GEM'].Code.values
codesLDC = groupListMask[groupListMask.Grouping=='LDC'].Code.values
codesOECD90 = groupListMask[groupListMask.Grouping=='OECD90'].Code.values

mask_AOSIS = mask_3D_ext.loc[dict(region=codesAOSIS)]
mask_ASEAN = mask_3D_ext.loc[dict(region=codesASEAN)]
mask_GEM = mask_3D_ext.loc[dict(region=codesGEM)]
mask_LDC = mask_3D_ext.loc[dict(region=codesLDC)]
mask_OECD90 = mask_3D_ext.loc[dict(region=codesOECD90)]

# Sum over the regions to create 2D masks
mask_AOSIS = mask_AOSIS.sum('region')
mask_ASEAN = mask_ASEAN.sum('region')
mask_GEM = mask_GEM.sum('region')
mask_LDC = mask_LDC.sum('region')
mask_OECD90 = mask_OECD90.sum('region')

# ####----START COMMENTED SECTION IF LOADING FROM PRIOR----####
# # Make a 5D dataArray with the groupings as the new dimension
# popGridSSP5D = popGridSSP4D.copy(deep=True)
# groupingsArr = xr.DataArray(data=np.zeros(len(groupings)),coords=[groupings],dims='grouping')
# popGrid_dum, dummy = xr.broadcast(popGridSSP5D, groupingsArr)
# popGridSSP5D = popGrid_dum.copy(deep=True)

# # Filter the population data to only include the gridcells within the groupings
# popGridSSP5D.loc[dict(grouping='ASEAN')]=popGridSSP5D.loc[dict(grouping='ASEAN')].where(mask_ASEAN)
# popGridSSP5D.loc[dict(grouping='AOSIS')]=popGridSSP5D.loc[dict(grouping='AOSIS')].where(mask_AOSIS)
# popGridSSP5D.loc[dict(grouping='GEM')]=popGridSSP5D.loc[dict(grouping='GEM')].where(mask_GEM)
# popGridSSP5D.loc[dict(grouping='LDC')]=popGridSSP5D.loc[dict(grouping='LDC')].where(mask_LDC)
# popGridSSP5D.loc[dict(grouping='OECD90')]=popGridSSP5D.loc[dict(grouping='OECD90')].where(mask_OECD90)

# # Quick check that the nan values haven't saved weirdly
# c=popGridSSP5D.loc[dict(year=2010,scenario=popGridSSP5D.scenario[0],grouping='ASEAN')]
# print('NAN check')
# print(np.min(c))
# print(np.max(c))

# # Save the dataArray for future use
# popGridSSP5D.to_netcdf(path=outPath+'popGridSSP_Grouping.nc')
# ####----END COMMENTED SECTION IF LOADING FROM PRIOR----####

# Optional: load the array instead of calculating them if re-running this code to save time
popGridSSP5D = xr.open_dataarray(outPath+'popGridSSP_Grouping.nc')

# Load the SN netcdfs that've been regridded to match the population dataset from Script 3
SN_SSP4D = xr.open_dataarray(toePath+'dea2022_SN_SSP4D.nc')
SN_RCP4D = xr.open_dataarray(toePath_rcp+'dea2022_SN_RCP4D.nc')

# To produce plots matching FEA2017 Fig 4, we need end-of-century S/N for each of  
# the scenarios across all the gridpoints within a region. Flatten lat & lon.

# First, get POI averages.
popGridSSP_POI = popGridSSP5D.loc[dict(year=POI)]
popGridSSP_POI = popGridSSP_POI.mean('year')
SN_SSP4D_POI = SN_SSP4D.loc[dict(year=POI)]
SN_SSP4D_POI = SN_SSP4D_POI.mean('year')
SN_RCP4D_POI = SN_RCP4D.loc[dict(year=POI)]
SN_RCP4D_POI = SN_RCP4D_POI.mean('year')


# Convert population from persons to a proportion of the total
propGridSSP_POI = popGridSSP_POI.copy(deep=True)
for thisScen in propGridSSP_POI.scenario:
    for thisGroup in propGridSSP_POI.grouping:
        thisPop = int(np.sum(popGridSSP_POI.loc[dict(scenario=thisScen,grouping=thisGroup)]))
        propGridSSP_POI.loc[dict(scenario=thisScen,grouping=thisGroup)]=propGridSSP_POI.loc[dict(scenario=thisScen,grouping=thisGroup)]/thisPop
       
# To produce the CDFs, I'm going to arrange the data into a Pandas dataframe, 
# sort by the S/N values, and calculate a cumulative proportion of the population
# to add as a new column. 

groupcount = 0 
# Legend labels
leg_lab = [scen_short[0],scen_short[1],rcpinp[1],scen_short[2],rcpinp[2],
            scen_short[3],scen_short[4],rcpinp[4]]

# # Testing out the plot
# for i in range(len(scen_short)):
#     # Plot the CDFs for SSPs and RCPs together
#     fig = plt.figure(1)
#     #ax = fig.add_subplot(111)
#     plt.plot([0+i,8+i],[0,1],color=colours[i])
#     plt.axis([0,20,0,1])
#     plt.title('test')
#     plt.ylabel('Cumulative fraction of population exposed')
#     plt.xlabel('S/N$_{2071-2100}$')
#     plt.legend(['a','b','c','d','e'],loc='lower right')
# plt.axvspan(1, 2, color=palette[0], alpha=1)
# plt.axvspan(2, 3, color=palette[1], alpha=1)
# plt.axvspan(3, 5, color=palette[2], alpha=0.6)
# plt.axvspan(5, 99, color=palette[3], alpha=0.5)


# Loop across the country groupings
for thisGroup in propGridSSP_POI.grouping:
    # Loop across the scenarios
    for i in range(len(scen_short)):
        thisSN = SN_SSP4D_POI.loc[dict(scenario=scen_short[i])] # Locate this scenario's data
        SN1D = thisSN.stack(x=('lat','lon')) # stack into 1 dimension
        thisProp = propGridSSP_POI.loc[dict(scenario=scen_long[i],grouping=thisGroup)] # This proportion
        prop1D = thisProp.stack(y=('lat','lon')) 
        prop1Dvals = prop1D.values
        prop1Dvals[np.isnan(prop1Dvals)] = 0 # Convert nan to 0
        # Make a pandas dataframe
        thisDF = pd.DataFrame({'S/N_2070-2100':SN1D.values,'Proportion':prop1Dvals})
        # Sort by S/N
        thisDF = thisDF.sort_values('S/N_2070-2100')
        # Make a new column with cumulative proportion of the population (0-1)
        thisDF['Cumulative']=thisDF['Proportion'].cumsum()
        
        # Add the RCPs
        if rcpinp[i] != 'skip':
            thisSN_rcp = SN_RCP4D_POI.loc[dict(scenario=rcpinp[i])]
            SN1D_rcp = thisSN_rcp.stack(x=('lat','lon'))
            thisDF_rcp = pd.DataFrame({'S/N_2070-2100':SN1D_rcp.values,'Proportion':prop1Dvals})
            thisDF_rcp = thisDF_rcp.sort_values('S/N_2070-2100')
            thisDF_rcp['Cumulative']=thisDF_rcp['Proportion'].cumsum()
        
        # Plot the CDFs for SSPs and RCPs together
        fig = plt.figure(groupcount)
        plt.plot(thisDF['S/N_2070-2100'],thisDF['Cumulative'],color=colours[i])
        if rcpinp[i] != 'skip':
            plt.plot(thisDF_rcp['S/N_2070-2100'],thisDF_rcp['Cumulative'],color=colours[i],linestyle='dashed')
        plt.axis([0,10,0,1])
        plt.title(str(thisGroup.values))
        if thisGroup == 'OECD90':
            plt.legend(leg_lab,loc='lower right')
        plt.ylabel('Cumulative fraction of population exposed')
        plt.xlabel('S/N$_{'+str(POIvals[0])+'-'+str(POIvals[1])+'}$')
        
    plt.axvspan(1, 2, color=palette[0], alpha=1, lw=None)
    plt.axvspan(2, 3, color=palette[1], alpha=1, lw=None)
    plt.axvspan(3, 5, color=palette[2], alpha=0.6, lw=None)
    plt.axvspan(5, 99, color=palette[3], alpha=0.5, lw=None)
    
    fig.savefig('./Plots/dea2022_Group_cdf_'+str(POIvals[0])+'-'+str(POIvals[1])+'_'+str(thisGroup.values)+'.svg')
    groupcount+=1

# Make a map showing the countries' groupings
import regionmask
mapColours = [plt.cm.Accent(4),plt.cm.Accent(1),plt.cm.Accent(3),plt.cm.Accent(0),plt.cm.Accent(2)]
groupix = [0,1,2,3,4]
zeroArr = np.zeros([len(mask_3D_ext.lat),len(mask_3D_ext.lon),len(groupix)])
groupMap = xr.DataArray(data=zeroArr,dims=['lat','lon','region'],coords=[mask_3D_ext.lat,mask_3D_ext.lon,groupix])
groupMap.loc[dict(region=groupix[0])] = mask_AOSIS
groupMap.loc[dict(region=groupix[1])] = mask_ASEAN
groupMap.loc[dict(region=groupix[2])] = mask_GEM
groupMap.loc[dict(region=groupix[3])] = mask_LDC
groupMap.loc[dict(region=groupix[4])] = mask_OECD90
groupMap.loc[dict(region=groupix[0])]=groupMap.loc[dict(region=groupix[0])].where(mask_AOSIS)
groupMap.loc[dict(region=groupix[1])]=groupMap.loc[dict(region=groupix[1])].where(mask_ASEAN)
groupMap.loc[dict(region=groupix[2])]=groupMap.loc[dict(region=groupix[2])].where(mask_GEM)
groupMap.loc[dict(region=groupix[3])]=groupMap.loc[dict(region=groupix[3])].where(mask_LDC)
groupMap.loc[dict(region=groupix[4])]=groupMap.loc[dict(region=groupix[4])].where(mask_OECD90)

groupMapB = groupMap.where(np.isnan(groupMap),other=True)
groupMapB = groupMapB.where(groupMapB==True,other=False)

regionmask.plot_3D_mask(groupMapB, add_colorbar=True, cmap='viridis')

mask_2D = (groupMapB * groupMapB.region).sum("region")

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from cartopy.util import add_cyclic_point
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
import cartopy.feature as cfeat
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

proj = ccrs.Robinson(central_longitude=0)
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj, aspect='auto')
for i in np.flip(groupix): #Reverse order just for a nicer z ordering (prioritise ASEAN & AOSIS)
    thisArray = groupMapB.loc[dict(region=i)]
    
    lon = thisArray.coords['lon']
    print("Original shape: ", thisArray.shape)
    lon_idx = thisArray.dims.index('lon')
    wrap_data, wrap_lon = add_cyclic_point(thisArray.values, coord=lon, axis=lon_idx)
    print('New shape: ', wrap_data.shape)
    
    # Plotting parameters
    cbar_label = 'Grouping'
    levels = [0,1,2]
    thisHue = mapColours[i]
    myMap = ListedColormap([(1,1,1,0.0),(1,1,1,0.0),(thisHue)])#(1,0,0,1)])
    
    cmap = myMap#mpl.cm.Accent
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
    
    ax.pcolormesh(wrap_lon, thisArray.lat.values, wrap_data, vmin=np.min(levels), 
                  vmax=np.max(levels), transform=ccrs.PlateCarree(), cmap = cmap, norm = norm,
                  rasterized=True)
    ax.text(-135,0-i*10,groupings[i],backgroundcolor=thisHue,transform=ccrs.PlateCarree())

ax.coastlines()
ax.add_feature(cfeat.BORDERS)

# Gridline customisation
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=2, color='lightgrey', alpha=0.5, linestyle=':')
gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
gl.ylocator = mticker.FixedLocator([-90, -45, 0, 45, 90])
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
gl.xlabel_style = {'size': 15, 'color': 'gray'}
gl.xlabel_style = {'color': 'red', 'weight': 'bold'}

plt.show()

fig.savefig(toePath+'Plots/groupings.svg',dpi=300)


# Add a supplementary plot for comparing groupings under SSP119 and SSP126

scens1 = ['ssp119','ssp126']
#colours = ['darkorchid', 'deepskyblue', 'lime', 'darkorange', 'darkred']
mapColours2 = [plt.cm.Accent(4),plt.cm.Accent(1),plt.cm.Set3(11),plt.cm.Accent(0),plt.cm.Set2(1)]
lineStyles = ['solid','dashed']
legLabs = [groupings[0],'_',groupings[1],'_',groupings[2],'_',
           groupings[3],'_',groupings[4],'_']
legLabs2 = ['ssp119','ssp126','_','_','_','_','_','_','_','_']
groupCount = 0 
for thisGroup in groupings:
    for i in [0,1]:
        thisScen = scens1[i]
        thisSN = SN_SSP4D_POI.loc[dict(scenario=thisScen)]
        SN1D = thisSN.stack(x=('lat','lon'))
        thisProp = propGridSSP_POI.loc[dict(scenario=scen_long[i],grouping=thisGroup)]
        prop1D = thisProp.stack(y=('lat','lon'))
        prop1Dvals = prop1D.values
        prop1Dvals[np.isnan(prop1Dvals)] = 0 
        thisDF = pd.DataFrame({'S/N_2070-2100':SN1D.values,'Proportion':prop1Dvals})
        thisDF = thisDF.sort_values('S/N_2070-2100')
        thisDF['Cumulative']=thisDF['Proportion'].cumsum()

        fig1 = plt.figure(10)
        plt.plot(thisDF['S/N_2070-2100'],thisDF['Cumulative'],color=mapColours2[groupCount],linestyle=lineStyles[i])
        plt.axis([0,10,0,1])
        if groupCount == 0:
            plt.legend(scens1,loc='upper right')
        plt.ylabel('Cumulative fraction of population exposed')
        plt.xlabel('S/N$_{'+str(POIvals[0])+'-'+str(POIvals[1])+'}$')
        
        # sec_legend = plt.legend(scens1, loc='lower center')
        # plt.gca().add_artist(sec_legend)

    groupCount+=1

plt.axvspan(1, 2, color=palette[0], alpha=1)
plt.axvspan(2, 3, color=palette[1], alpha=1)
plt.axvspan(3, 5, color=palette[2], alpha=0.6)
plt.axvspan(5, 99, color=palette[3], alpha=0.5)

first_legend = plt.legend(legLabs, loc='lower right')
plt.gca().add_artist(first_legend)
sec_legend = plt.legend(legLabs2, loc='center right')
plt.gca().add_artist(sec_legend)

fig1.savefig(toePath+'Plots/Group_cdf_all_ssp1_'+str(POIvals[0])+'-'+str(POIvals[1])+'.pdf')