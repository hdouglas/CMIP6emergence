#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:46:03 2021

@author: 1001883
"""

# Douglas et al. 2022 optional script for diagnostic plotting.
# Plots spatial figures of multi-model avg signal, noise, and signal-to-noise for 
# each year and scenario. 

# Inputs: The multi-model netcdfs from dea2022_02_combineModels.py
# Outputs: Individual plots of multi-model average noise, signal, and 
# signal-to-noise for each year.

import os
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt

# Code to make plots wrap (not break at longitude 180)
from cartopy.util import add_cyclic_point
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
import cartopy.feature as cfeat

###############################
####----USER-CHANGEABLE----####

avgType = 'median'
variable_id = 'tas'
table_id = 'Amon'

homedir = '/Volumes/GeoCCRI_01/users-data/douglahu/CMIP6_data'
outdir = '/Volumes/GeoCCRI_01/users-data/douglahu/Scripts/FEA2017/Plots/'

####----END USER-CHANGEABLE----####
###################################

os.chdir(homedir+'/'+variable_id+'/'+table_id+'/01_MM_output')

def plotter_fn(thisArray,ssp,var,type_avg,year):
    
    lon = thisArray.coords['lon']
    print("Original shape: ", thisArray.shape)
    
    lon_idx = thisArray.dims.index('lon')
    wrap_data, wrap_lon = add_cyclic_point(thisArray.values, coord=lon, axis=lon_idx)
    print('New shape: ', wrap_data.shape)
    
    proj = ccrs.Robinson(central_longitude=0)
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection=proj, aspect='auto')
    
    # Plotting parameters
    title = 'Signal-to-noise ('+ssp+')'
    cbar_label = 'N'
    
    # Set contour levels and title etc. based on variable type
    if var == 'noise':
        title = 'Noise - $\sigma$, piControl ('+ssp+'), '+year
        cbar_label = 'K'
        levels = [0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8]
    elif var == 'signal':
        title = 'Signal - warming 2071-2100 ('+ssp+'), '+year
        cbar_label = 'K'
        levels = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
    elif var == 'SN':
        title = 'Signal-to-noise ('+ssp+') '+type_avg+', '+year
        cbar_label = 'N'
        levels = [0.0,1.6,2.2,2.8,3.6,4.4,5.4,6.7,9.0,25.5]
    else:
        title = 'wrong var argument given'
        cbar_label = ''
        levels = [0.0,1.6,2.2,2.8,3.6,4.4,5.4,6.7,9.0,25.5]
    
    cmap = mpl.cm.YlOrRd
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
    
    cf = ax.pcolormesh(wrap_lon, thisArray.lat.values, wrap_data, vmin=np.min(levels), vmax=np.max(levels),
                                          transform=ccrs.PlateCarree(),
                                          cmap = cmap, 
                                          norm = norm)
    
    cb = plt.colorbar(cf, ax=ax, orientation='horizontal')
    cb.set_label(cbar_label)
    
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
    
    return fig


#-------Annual averages------
scen_short = ['ssp119','ssp126','ssp245','ssp370','ssp585']
for thisSSP in scen_short:
    print(thisSSP)
    noise_models=xr.open_dataarray('MM_dea2022_noise_'+thisSSP+'.nc')
    signal_models=xr.open_dataarray('MM_dea2022_signal_'+thisSSP+'_allYears.nc')
    SN_models=xr.open_dataarray('MM_dea2022_SN_'+thisSSP+'_allYears.nc')
    
    print(SN_models.model.values)
    
    # Multi-model averaging
    if avgType == 'median':
        noiseAvg = noise_models.median('model', keep_attrs=True, skipna=True)
        signalAvg = signal_models.median('model', keep_attrs=True, skipna=True)
        SN_Avg = SN_models.median('model', keep_attrs=True, skipna=True)
        SN_16 = SN_models.quantile(0.16, 'model', keep_attrs=True, skipna=True)
        SN_84 = SN_models.quantile(0.84, 'model', keep_attrs=True, skipna=True)
    elif avgType == 'mean':
        noiseAvg = noise_models.mean('model', keep_attrs=True, skipna=True)
        signalAvg = signal_models.mean('model', keep_attrs=True, skipna=True)
        SN_Avg = SN_models.mean('model', keep_attrs=True, skipna=True)
    else:
        print('Specify mean or median for avgType')
        exit
    
    a = plotter_fn(noiseAvg,thisSSP,'noise',avgType,'')
    a.savefig(outdir+thisSSP+'/01_noise_mmAvg_'+thisSSP+'.png')
    
    for thisYear in SN_Avg.year:
        print(thisYear)
        
        b = plotter_fn(signalAvg.loc[dict(year=thisYear)],thisSSP,'signal',avgType,str(int(thisYear)))
        c = plotter_fn(SN_Avg.loc[dict(year=thisYear)],thisSSP,'SN',avgType,str(int(thisYear)))
        
        b.savefig(outdir+thisSSP+'/02_signal_mmAvg_'+thisSSP+'_'+str(int(thisYear))+'.png')
        c.savefig(outdir+thisSSP+'/03_S-N_mmAvg_'+thisSSP+'_'+str(int(thisYear))+'.png')
        
        if avgType == 'median':
            d = plotter_fn(SN_16.loc[dict(year=thisYear)],thisSSP,'SN','16th %ile',str(int(thisYear)))
            e = plotter_fn(SN_84.loc[dict(year=thisYear)],thisSSP,'SN','84th %ile',str(int(thisYear)))
            
            d.savefig(outdir+thisSSP+'/03_S-N_mm16th_'+thisSSP+'_'+str(int(thisYear))+'.png')
            e.savefig(outdir+thisSSP+'/03_S-N_mm84th_'+thisSSP+'_'+str(int(thisYear))+'.png')
        
