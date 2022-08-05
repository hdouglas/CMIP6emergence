#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 21:42:22 2021

@author: 1001883
"""

# Douglas et al. 2022 Script 3 (population data processing)
# Generate a land area gridded dataset

# Inputs: Natural Earth shapefiles (land area)downloaded from https://www.naturalearthdata.com/downloads/

# Outputs: A netcdf mask file with gridded land area

import pandas as pd
import geopandas as gp
import numpy as np
import xarray as xr
import regionmask
from scipy import stats

regionmask.__version__

###############################
####----USER-CHANGEABLE----####

# Path to the land area shapefile:
ne_50m_path = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/Geographies/ne_50m_land'
ne_50m = gp.read_file(ne_50m_path)

# map projections
epsgs = ['8857','3410','6933'] 
# 6933 = equal-area, supersedes 3410, but only 86N to 86S. https://epsg.io/6933
# 8857 = WGS 84 / Equal Earth Greenwich

# Spatial resolution (degrees)
dx = 0.25
dy = 0.25

# Path for where outputs should be saved:
outPath = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/'

####----END USER-CHANGEABLE----####
###################################

#-----Calculate the area of land on the earth-----#

proj1=[]
proj2=[]
proj3=[]

ne_50m_p1 = ne_50m.to_crs(epsg=epsgs[0])
ne_50m_p2 = ne_50m.to_crs(epsg=epsgs[1])
ne_50m_p3 = ne_50m.to_crs(epsg=epsgs[2])

for i in range(0,len(ne_50m)):
    
    print(i)
    
    thisPoly = ne_50m_p1.iloc[i]
    area = int(pd.to_numeric(thisPoly['geometry'].area)/10**6)
    proj1.append(area)
    
    thisPoly = ne_50m_p2.iloc[i]
    area = int(pd.to_numeric(thisPoly['geometry'].area)/10**6)
    proj2.append(area)
    
    thisPoly = ne_50m_p3.iloc[i]
    area = int(pd.to_numeric(thisPoly['geometry'].area)/10**6)
    proj3.append(area)

print(np.sum(proj1))
print(np.sum(proj2))
print(np.sum(proj3))

# I'm going to go with epsg 8857.

Aland = np.sum(proj1)

#-----Make a gridded dataset with the value being the land area within each gridcell----#

# The trick here is that regionmask only tells you if the centre of the grid cell 
# is within the shapefile. There will be a decent amount of coastal area left out 
# by this. Try the first way with no corrections, then see how far off it is. 

# Make a new, empty dataarray for the area data
lat = np.arange(-90+dy/2, 90-dy/2+0.1, dy)
lon = np.arange(-180+dx/2, 180-dx/2+0.1, dx)
zeroArr = np.zeros([len(lat), len(lon)])
areaArr = xr.DataArray(data=zeroArr,coords=[lat,lon],dims=['lat','lon'],name='area')

# Calculate the area of each gridcell
Rearth = 6371 # radius of earth in km

for i in lat:
    print(i)
    for j in lon:
        Agrid = abs(2*np.pi*Rearth/(360/dy)*np.cos(np.deg2rad(i))) * 2*np.pi*Rearth/(360/dx)
        areaArr.loc[dict(lat=i, lon=j)]=Agrid
        
gridsTot = np.sum(areaArr)
sphereTot = 4*np.pi*Rearth**2

print(gridsTot/sphereTot)

# RegionMask code adapted from https://regionmask.readthedocs.io/en/stable/notebooks/geopandas.html

mask = regionmask.mask_geopandas(ne_50m, lon, lat)
#mask_3D = regionmask.mask_3D_geopandas(ne_50m, lon, lat)

# What's the area within the mask, i.e. the land area?

# Convert mask to boolean
maskB = mask.where(np.isnan(mask),other=1)
maskB = maskB.where(maskB==1,other=0)

landArr = maskB * areaArr
print(int(np.sum(landArr)))
print(Aland)
print(int(np.sum(landArr))/Aland)

# Turns out the coastal area errors mostly cancel out. Grid captures 99.98% of 
# the vector land area at 0.5 deg resolution.

# Save the land area array as a netCDF
landArr.to_netcdf(path=outPath+'landAreas_'+str(dx)+'deg.nc')

# Optional: Show off the handiwork

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
from cartopy.util import add_cyclic_point

import matplotlib.ticker as mticker
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
import cartopy.feature as cfeat

lon = landArr.coords['lon']

lon_idx = landArr.dims.index('lon')
wrap_data, wrap_lon = add_cyclic_point(landArr.values, coord=lon, axis=lon_idx)

proj = ccrs.Robinson(central_longitude=0)
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection=proj, aspect='auto')

cmap = mpl.cm.YlOrRd

cf = ax.pcolormesh(wrap_lon, landArr.lat.values, wrap_data, vmin=0, vmax=float(np.max(landArr)),
                                      transform=ccrs.PlateCarree(),
                                      cmap = cmap)

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


