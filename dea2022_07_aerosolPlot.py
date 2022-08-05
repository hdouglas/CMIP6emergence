#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 22:09:03 2022

@author: 1001883
"""

# Douglas et al. 2022 Script 7
# Load and compare the aerosol emissions specified for CMIP5 and CMIP6
# Plots the changes and optionally add a plot of the total emissions in 
# CMIP6 with the same colour scale.

# Inputs: multi-ensemble netcdf files (S, N, S/N) from script 01 for all models
# Outputs: multi-model netcdf files with all models, on a common grid

import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
from cartopy.util import add_cyclic_point
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
import cartopy.feature as cfeat
import xesmf as xe 
from matplotlib.colors import LogNorm
import cftime

###############################
####----USER-CHANGEABLE----####
# RCP netcdf files of emissions available from 
# https://tntcat.iiasa.ac.at/RcpDb/dsd?Action=htmlpage&page=spatial
C5dir = '/Volumes/GeoCCRI_01/users-data/douglahu/CMIP5_data/SpatialAerEms/'
specList = ['SO2','BCE','OCE'] # sulfur dioxide, black carbon, organic carbon
scens = ['R26','R45','R85']

# Netcdf files of SSP emissions are available from 
# https://esgf-node.llnl.gov/search/input4mips/ 
C6dir = '/Volumes/GeoCCRI_01/users-data/douglahu/CMIP6_data/SpatialAerEms/'
scenList = ['ssp126','ssp245','ssp585']
specListC6 = ['SO2','BC','OC'] # slightly different naming convention
specLabsC6 = ['SO$_2$','BC','OC'] # labels for plotting

outDir = '/Volumes/GeoCCRI_01/users-data/douglahu/Scripts/FEA2017/Plots/'

# Set common gridsize
dlat = 2.5 # Size (in degrees) of latitude of each grid cell
dlon = 2.5 # Size (in degrees) of longitude of each grid cell
interpMethod = 'bilinear' # 'patch' # 'conservative' #  Set the interpolation method for regridding

# Labels for the combined plots
rowLabs = ['SSP126-RCP26', 'SSP245-RCP45', 'SSP585-RCP85']
panels = ['a','b','c','d','e','f','g','h','i']

POI = [2040,2060] # Period of interest over which emissions are averaged

####----END USER-CHANGEABLE----####
###################################

#---------------------------
# Part 1: CMIP5-era emissions

# Just a quick test to see what we're working with
testDS = xr.open_dataset(C5dir+'R26_SO2.nc')
print(testDS)
sources = ['ENE', 'IND', 'TRA', 'DOM', 'SLV', 'AGR', 'AWB', 'WST', 'LCF', 'SAV',
           'SHP', 'AIR']
sourceslc = ['ene', 'ind', 'tra', 'dom', 'slv', 'agr', 'awb', 'wst', 'lcf', 
             'sav', 'shp', 'air']
tot = 0
for thisSource in sources:
    thisArr = testDS['EMISS_'+thisSource]
    print(thisArr.coords)
    thisSum = np.sum(thisArr)
    print(thisSource+': '+str(thisSum))
    tot+=thisSum
    
print(tot)

# Create an empty dataarray to store the total emissions
aerosols_C5 = xr.DataArray(data=np.nan, dims=['species','lat','lon','time','scenario'],
                     coords=[specList,thisArr.LAT,thisArr.LON,thisArr.TIME,scens],
                     attrs=dict(units='kg m-2 sec-1'))

# Loop through files to get the data and store in the dataarrays
for thisFile in os.listdir(C5dir):
    if thisFile != '.DS_Store':
        # Get the file from the species and scenario in the filename
        print(thisFile)
        thisSpec = thisFile[4:7]
        print(thisSpec)
        thisScen = thisFile[0:3]
        print(thisScen)
        thisDS = xr.open_dataset(C5dir+thisFile)
        
        # Calculate the total emissions
        # Complication, some datasets use uppercase, some lowercase
        try:
            emiss_tot = thisDS['EMISS_ENE'].copy(deep=True)
            lc = False
        except:
            emiss_tot = thisDS['emiss_ene'].copy(deep=True)
            lc = True
        
        # Set the total values everywhere to 0
        emiss_tot = emiss_tot.where(emiss_tot == 999999, other=0.0)
        if lc:
            sourceList = sourceslc
            emtag = 'emiss'
        else:
            sourceList = sources
            emtag = 'EMISS'
        for thisSource in sourceList:
            thisArr = thisDS[emtag+'_'+thisSource]
            # Set any nan values to zero
            thisArr = thisArr.fillna(0) #.where(emiss_tot != np.nan, other=0.0)
            try:
                thisArr = thisArr.transpose('lat','lon','time')
            except:
                thisArr = thisArr.transpose('LAT','LON','TIME')
            emiss_tot += thisArr#.values
            try:
                emiss_tot = emiss_tot.transpose('lat','lon','time')
            except:
                emiss_tot = emiss_tot.transpose('LAT','LON','TIME')
        
        # Save the total emissions in the dataarray
        aerosols_C5.loc[dict(species=thisSpec,scenario=thisScen)] = emiss_tot

# Calculate and then plot some of the end-of-century emissions (average)
aerosols_C5_poi = aerosols_C5.loc[dict(time=slice(np.datetime64('2069-12-31'),
                                                  np.datetime64('2101-01-01')))]
aerosols_C5_poi = aerosols_C5_poi.mean('time')
# Zero values are unhelpful for log-scale plotting 
aerosols_C5_poi = aerosols_C5_poi.where(aerosols_C5_poi != 0.0)

# Define a plotting function
def plotter_global(thisArray,ttl,cBarLbl,lvls=[0],cMap=mpl.cm.YlOrRd):
    
    lon = thisArray.coords['lon']
    print("Original shape: ", thisArray.shape)
    lon_idx = thisArray.dims.index('lon')
    wrap_data, wrap_lon = add_cyclic_point(thisArray.values, coord=lon, axis=lon_idx)
    print('New shape: ', wrap_data.shape)
    
    proj = ccrs.Robinson(central_longitude=0)
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection=proj, aspect='auto')
    
    # Plotting parameters
    title = ttl
    cmap = cMap # mpl.cm.viridis #RdYlBu_r#PiYG_r    
    
    cf = ax.pcolormesh(wrap_lon, thisArray.lat.values, wrap_data, #vmin=np.min(levels), vmax=np.max(levels),
                                          transform=ccrs.PlateCarree(),
                                          cmap = cmap, 
                                          rasterized=True,
                                          #norm = LogNorm(vmin=float(np.min(thisArray)), vmax=float(np.max(thisArray))))
                                          norm = LogNorm(vmin=1e-18, vmax=1e-8))
    
    cb = plt.colorbar(cf, ax=ax, orientation='horizontal', label=cBarLbl)
    ax.coastlines()
    ax.add_feature(cfeat.BORDERS)
    plt.title(title)
    
    # Gridline customisation
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=2, color='lightgrey', alpha=0.5, linestyle=':')
    gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl.ylocator = mticker.FixedLocator([-90, -45, 0, 45, 90])
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.xlabel_style = {'color': 'red', 'weight': 'bold'}


# Plot the fields (optional)  
# plotter_global(aerosols_C5_eoc.loc[dict(species='OCE',scenario='R26')], '2070-2100 avg OC emissions, RCP26', 'kg m-2 s-1')
# plotter_global(aerosols_C5_eoc.loc[dict(species='OCE',scenario='R45')], '2070-2100 avg OC emissions, RCP45', 'kg m-2 s-1')
# plotter_global(aerosols_C5_eoc.loc[dict(species='OCE',scenario='R85')], '2070-2100 avg OC emissions, RCP85', 'kg m-2 s-1')
            
#---------------------------

# Part 2: Now repeat for CMIP6 emissions

fileList = os.listdir(C6dir)

# Load one to see what we're dealing with
testDS = xr.open_dataset(C6dir+'BC-em-AIR-anthro_input4MIPs_emissions_ScenarioMIP_IAMC-IMAGE-ssp126-1-1_gn_201501-210012.nc')

# Through this process, I worked out that the openburning-share files relate to 
# some fraction; they're not reporting the actual flux values. I've removed them
# from the database. 
# Only use the XX-em-AIR-anthro_... and XX-em-anthro_... files.

# Create an empty dataarray for storing the values
aerosols_C6 = xr.DataArray(data=0.0, dims=['species','lat','lon','time','scenario'],
                     coords=[specListC6,testDS.lat,testDS.lon,testDS.time,scenList],
                     attrs=dict(units='kg m-2 sec-1'))

# Loop through and add data to the arrays
for thisFile in fileList:
    if thisFile.find('.nc') > 0:
        thisSource = thisFile.split('_')[0]
        thisSpec = thisSource.split('-')[0]
        thisMod = thisFile.split('_')[4]
        thisSpecName = thisSource.replace('-','_')
        isShare = thisSource.find('share')
        thisDS = xr.open_dataset(C6dir+thisFile)
        
        if len(thisMod.split('-')) == 6:
            thisScen = thisMod.split('-')[3]
        elif len(thisMod.split('-')) == 5:
            thisScen = thisMod.split('-')[2]
        else:
            thisScen = 'ERROR'
        
        thisArr = thisDS[thisSpecName]
        
        # Set any nan values to zero
        thisArr = thisArr.fillna(0)
        
        # Some datasets have levels or sectors associated with them
        if len(thisArr.coords) > 3:
            try:
                thisArr = thisArr.sum('level')
            except:
                thisArr = thisArr.sum('sector')
        print(len(thisArr.coords))
        
        aerosols_C6.loc[dict(species=thisSpec,scenario=thisScen)] += thisArr
        
# Calculate and then plot some of the end-of-century emissions (average)
POIlen = POI[1]-POI[0] + 1
timeBnds = cftime.num2date([0,365*POIlen],'days since '+str(POI[0])+'-01-01 00:00:00 UTC',calendar='noleap')
aerosols_C6_poi = aerosols_C6.loc[dict(time=slice(timeBnds[0],timeBnds[1]))]
aerosols_C6_poi = aerosols_C6_poi.mean('time')
# Zero values are unhelpful for log-scale plotting 
aerosols_C6_poi = aerosols_C6_poi.where(aerosols_C6_poi != 0.0)

# Plot the fields (optional)  
# plotter_global(aerosols_C6_eoc.loc[dict(species='OC',scenario='ssp126')], 
#                '2070-2100 avg OC emissions, SSP126', 'kg m-2 s-1')
# plotter_global(aerosols_C6_eoc.loc[dict(species='OC',scenario='ssp245')], 
#                '2070-2100 avg OC emissions, SSP245', 'kg m-2 s-1')
# plotter_global(aerosols_C6_eoc.loc[dict(species='OC',scenario='ssp585')], 
#                '2070-2100 avg OC emissions, SSP585', 'kg m-2 s-1')

#---------------------------

# Part 3: Combine and compare generations

# Regrid to a courser gridsize
# Because the emissions data is defined in terms of emissions per unit area, we
# can average across gridcells, we don't need to sum up the results. 
lat = np.arange(-90+dlat/2, 90, dlat)
lon = np.arange(0+dlon/2, 360, dlon)
ds_out = xr.Dataset({'lat': (['lat'], lat),
                     'lon': (['lon'], lon),
                    }
                   )
regridder = xe.Regridder(aerosols_C5_poi.to_dataset(name='Emissions'), ds_out, 
                         interpMethod, periodic=True) 
regridder  # print basic regridder information.

# The bilinear interpolation doesn't play nice with nan, so let's convert to zero, 
# then convert back
aerosols_C5_poi = aerosols_C5_poi.fillna(0.0)
aerosols_C6_poi = aerosols_C6_poi.fillna(0.0)

aerosols_C5_poi_course = regridder(aerosols_C5_poi)
aerosols_C6_poi_course = regridder(aerosols_C6_poi)

plotter_global(aerosols_C5_poi_course.loc[dict(species='OCE',scenario='R26')], 
               str(POI[0])+'-'+str(POI[1])+' avg OC emissions, RCP26', 'kg m-2 s-1')
plotter_global(aerosols_C6_poi_course.loc[dict(species='OC',scenario='ssp126')], 
               str(POI[0])+'-'+str(POI[1])+' avg OC emissions, SSP126', 'kg m-2 s-1')

# Compute the difference
# In order to subtract RCPs from the SSPs, need to give them the same dimensions
aerosols_C5_poi_course = aerosols_C5_poi_course.assign_coords(scenario=scenList)
aerosols_C5_poi_course = aerosols_C5_poi_course.assign_coords(species=specListC6)

aerosols_C6C5diff = aerosols_C6_poi_course - aerosols_C5_poi_course

# Define a plotting function for the difference
def plotter_global_diff(thisArray,ttl,cBarLbl,lvls=[0],cMap=mpl.cm.BrBG_r):
    
    lon = thisArray.coords['lon']
    print("Original shape: ", thisArray.shape)
    lon_idx = thisArray.dims.index('lon')
    wrap_data, wrap_lon = add_cyclic_point(thisArray.values, coord=lon, axis=lon_idx)
    print('New shape: ', wrap_data.shape)
    
    proj = ccrs.Robinson(central_longitude=0)
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection=proj, aspect='auto')
    
    # Plotting parameters
    title = ttl
    cmap = cMap # mpl.cm.viridis #RdYlBu_r#PiYG_r
    
    if len(lvls)>1:
        levels = lvls
    else:
        levels = np.arange(-1e-11,1e-11,1e-13) #lvls #np.arange(0,6,6/100)#[0.0,0.2,0.4,0.6,0.8,1.0,2,3,4,5,6,7]    
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
    
    cf = ax.pcolormesh(wrap_lon, thisArray.lat.values, wrap_data, #vmin=np.min(levels), vmax=np.max(levels),
                                          transform=ccrs.PlateCarree(),
                                          rasterized=True,
                                          cmap = cmap, 
                                          norm = norm)
                                          #norm = LogNorm(vmin=-1e-11, vmax=1e-11))
                                          #norm = LogNorm(vmin=1e-18, vmax=1e-8))
            
    
    cb = plt.colorbar(cf, ax=ax, orientation='horizontal', label=cBarLbl)
    ax.coastlines()
    ax.add_feature(cfeat.BORDERS)
    plt.title(title)
    
    # Gridline customisation
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=2, color='lightgrey', alpha=0.5, linestyle=':')
    gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl.ylocator = mticker.FixedLocator([-90, -45, 0, 45, 90])
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.xlabel_style = {'color': 'red', 'weight': 'bold'}

# Optional: single plots of each species/scenario
#plotter_global_diff(aerosols_C6C5diff.loc[dict(species='SO2',scenario='ssp126')], 'Change in 2070-2100 avg SO2 emissions, RCP26 to SSP126', 'kg m-2 s-1',np.arange(-1e-11,1e-11,1e-13))
#plotter_global_diff(aerosols_C6C5diff.loc[dict(species='BC',scenario='ssp126')], 'Change in 2070-2100 avg BC emissions, RCP26 to SSP126', 'kg m-2 s-1',np.arange(-1e-11,1e-11,1e-13))
#plotter_global_diff(aerosols_C6C5diff.loc[dict(species='OC',scenario='ssp126')], 'Change in 2070-2100 avg OC emissions, RCP26 to SSP126', 'kg m-2 s-1',np.arange(-1e-11,1e-11,1e-13))

# Plot it all together on one big plot
def multi_plotter(thisArray, scenNo, levels, cMap, myRowLabs):
    
    import cartopy.crs as ccrs
    import matplotlib as mpl
    
    # Packages for subplots
    from cartopy.mpl.geoaxes import GeoAxes
    from mpl_toolkits.axes_grid1 import AxesGrid
    
    # Packages to make it wrap
    from cartopy.util import add_cyclic_point
    import matplotlib.ticker as mticker
    from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
    
    title = 'Change in '+str(POI[0])+'-'+str(POI[1])+' avg aerosol emissions'
    cbar_label = 'kg m-2 s-1'#'Fraction of global 99th percentile emissions'

    levels = np.array(levels)*1e-11
    
    cmap = cMap # mpl.cm.BrBG_r
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
    
    proj = ccrs.Robinson(central_longitude=0)
    axes_class = (GeoAxes,dict(map_projection=proj))
    
    fig=plt.figure(figsize=(10,2*scenNo))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(scenNo, 3),
                    axes_pad=0.2,
                    cbar_location='bottom',
                    cbar_mode='single',
                    cbar_pad=0,
                    cbar_size='4%',
                    label_mode='')  # note the empty label_mode
    
    for i, ax in enumerate(axgr):
        print(i)
        thisScen = thisArray.scenario.values[int(np.floor(i/3))]
        thisSpec = specListC6[np.mod(i,3)]
        thisLab = myRowLabs[int(np.floor(i/3))]
        
        theseData = thisArray.loc[dict(scenario=thisScen, species=thisSpec)]
        
        lon = theseData.coords['lon']
        print("Original shape: ", theseData.shape)
        
        lon_idx = theseData.dims.index('lon')
        wrap_data, wrap_lon = add_cyclic_point(theseData.values, coord=lon, axis=lon_idx)
        print('New shape: ', wrap_data.shape)
            
        cf = ax.pcolormesh(wrap_lon, theseData.lat.values, wrap_data,
                           rasterized=True,
                           transform=ccrs.PlateCarree(), cmap = cmap, norm = norm)
               
        ax.coastlines(linewidth=0.5)
        
        # Label the columns
        if i < 3 :
            ax.set_title(specLabsC6[i])
        
        # Label the rows
        if np.mod(i,3) == 0:
            ax.text(-0.04, 0.55, thisLab, va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=ax.transAxes)
            
        # Label each panel
        ax.text(0.02,0.9,panels[i], va='bottom', ha='center',
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
    
    fig.subplots_adjust(top=0.95)
    
    return fig

# Set the colourbar levels
myLevs = [-1.0,-0.81,-0.64,-0.49,-0.36,-0.25,-0.16,-0.09,-0.04,-0.01,0.0,
          0.01,0.04,0.09,0.16,0.25,0.36,0.49,0.64,0.81,1.0]
allPlot = multi_plotter(aerosols_C6C5diff,3,myLevs,mpl.cm.BrBG_r,rowLabs)
allPlot.savefig(outDir+'aerosolChanges_'+str(POI[0])+'-'+str(POI[1])+'.pdf', bbox_inches='tight', dpi=300)

#---------------------------

# Part 4: Get some summary stats to quantify the changes

#Make a gridded dataset with the value being the area within each gridcell
dx = dlon
dy = dlat
# Make a new, empty dataarray for the area data
lat = np.arange(-90+dy/2, 90-dy/2+0.1, dy)
lon = np.arange(0+dx/2, 360-dx/2+0.1, dx)
zeroArr = np.zeros([len(lat), len(lon)])
areaArr = xr.DataArray(data=zeroArr,coords=[lat,lon],dims=['lat','lon'],name='area')
# Calculate the area of each gridcell
Rearth = 6371000 # radius of earth in m
for i in lat:
    print(i)
    for j in lon:
        Agrid = abs(2*np.pi*Rearth/(360/dy)*np.cos(np.deg2rad(i))) * 2*np.pi*Rearth/(360/dx)
        areaArr.loc[dict(lat=i, lon=j)]=Agrid
gridsTot = np.sum(areaArr)
sphereTot = 4*np.pi*Rearth**2
print(gridsTot/sphereTot)

nSecs = 60*60*24*365.25 # Seconds per year

# Convert kg/m2/s to kg/year
aerosols_C6_poi_course_tot = aerosols_C6_poi_course*areaArr*nSecs
aerosols_C5_poi_course_tot = aerosols_C5_poi_course*areaArr*nSecs
aerosols_C6C5diff_tot = aerosols_C6C5diff*areaArr*nSecs

# Compare some stats
def statPrinter(thisScen,thisSpec):
    absArr = aerosols_C6_poi_course_tot.loc[dict(scenario=thisScen,species=thisSpec)]
    absArrC5 = aerosols_C5_poi_course_tot.loc[dict(scenario=thisScen,species=thisSpec)]
    diffArr = aerosols_C6C5diff_tot.loc[dict(scenario=thisScen,species=thisSpec)]
    incArr = diffArr.where(diffArr>0)
    decArr = diffArr.where(diffArr<0)
    print(thisScen+', '+thisSpec)
    print('        CMIP5          CMIP6          CMIP5-CMIP6 change')
    print('Tots = '+format(float(np.sum(absArrC5)),'.2E')+'   '+format(float(np.sum(absArr)),'.2E')+
          '   '+format(float(np.sum(diffArr)),'.2E')+' ('+str(format(float(np.sum(diffArr)/np.sum(absArrC5))*100,'.1f'))+'% of CMIP5 total)')
    print('AbsSum: '+format(float(np.sum(abs(absArrC5))),'.2E')+'   '+format(float(np.sum(abs(absArr))),'.2E')+'   '+format(float(np.sum(abs(diffArr))),'.2E'))
    print('5th pc: '+format(float(np.percentile(absArrC5,5)),'.2E')+'   '+format(float(np.percentile(absArr,5)),'.2E')+'   '+format(float(np.percentile(diffArr,5)),'.2E'))
    print('25th pc: '+format(float(np.percentile(absArrC5,25)),'.2E')+'   '+format(float(np.percentile(absArr,25)),'.2E')+'   '+format(float(np.percentile(diffArr,25)),'.2E'))
    print('50th pc: '+format(float(np.percentile(absArrC5,50)),'.2E')+'   '+format(float(np.percentile(absArr,50)),'.2E')+'   '+format(float(np.percentile(diffArr,50)),'.2E'))
    print('75th pc: '+format(float(np.percentile(absArrC5,75)),'.2E')+'   '+format(float(np.percentile(absArr,75)),'.2E')+'   '+format(float(np.percentile(diffArr,75)),'.2E'))
    print('95th pc: '+format(float(np.percentile(absArrC5,95)),'.2E')+'   '+format(float(np.percentile(absArr,95)),'.2E')+'   '+format(float(np.percentile(diffArr,95)),'.2E'))
    print('Increases: '+format(float(np.sum(abs(incArr))),'.2E')+' ('+str(format(float(np.sum(abs(incArr))/np.sum(absArrC5))*100,'.1f'))+'% of CMIP5 total)')
    print('Decreases: '+format(float(np.sum(abs(decArr))),'.2E')+' ('+str(format(float(np.sum(abs(decArr))/np.sum(absArrC5))*100,'.1f'))+'% of CMIP5 total)')
    
statPrinter('ssp126','SO2')
statPrinter('ssp245','SO2')
statPrinter('ssp585','SO2')

statPrinter('ssp126','BC')
statPrinter('ssp245','BC')
statPrinter('ssp585','BC')

statPrinter('ssp126','OC')
statPrinter('ssp245','OC')
statPrinter('ssp585','OC')

#---------------------------

# Part 5: Make a similar plot of CMIP6 emissions for comparison

# It will need a custom colourmap
myLevs = [0.0,0.01,0.04,0.09,0.16,0.25,0.36,0.49,0.64,0.81,1.0]
myCols = ['#F5F2E8','#F6EBCF','#EFDCAD','#E3C989','#D4AC62','#C38936','#AB6E1F','#8F540C','#734208','#543005']
myCmap = mpl.colors.ListedColormap(myCols)

allPlot = multi_plotter(aerosols_C6_poi_course,3,myLevs,myCmap,scenList)
allPlot.savefig(outDir+'aerosolsCMIP6_'+str(POI[0])+'-'+str(POI[1])+'.pdf', bbox_inches='tight', dpi=300)


    
