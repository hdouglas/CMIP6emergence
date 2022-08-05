#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 21:42:22 2021

@author: 1001883
"""

# Douglas et al. 2022 Script 2 (population data processing)
# Mask the SEDAC population dataarray by country.

# Inputs: The netcdf from Script 1, Natural Earth shapefiles (countries and map units)
# Shapefiles downloaded from https://www.naturalearthdata.com/downloads/

# Outputs: A netcdf mask file that assigns gridpoints to countries

import geopandas as gp
import numpy as np
import xarray as xr
import regionmask
from scipy import stats

regionmask.__version__

###############################
####----USER-CHANGEABLE----####

# Path to the countries shapefile:
ne_50m_path = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/Geographies/ne_50m_admin_0_countries'
# Path to the map units shapefile:
ne_50mMU_path = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/Geographies/ne_50m_admin_0_map_units'
# Note that both are required because the population projections treat some territories
# as countries, like Reunion. 

# Path for where outputs should be saved:
outPath = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/'

####----END USER-CHANGEABLE----####
###################################


# Load and manipulate the Natural Earth datasets to get what we need for the SSPs
ne_50m = gp.read_file(ne_50m_path)
ne_50mMU = gp.read_file(ne_50mMU_path)
# The unique three-letter codes are denoted ADM0_A3m but some tweaking is required 
# to align with the population projections

missers = ['GLP','GUF','MTQ','MYT','REU'] # Dependencies that are treated as countries in SSP
for miss in missers:
    ne_50mMU.loc[ne_50mMU['ADM0_A3_IS']== miss, ['ADM0_A3']] = miss
    ne_50m = ne_50m.append(ne_50mMU.loc[ne_50mMU['ADM0_A3']== miss])

ne_50m.loc[ne_50m['ADM0_A3']=='PSX', ['ADM0_A3']] = 'PSE' # Align the code for Palestine with SSP   

ne_50m = ne_50m.sort_values('ADM0_A3') # Sort by the codes we'll be using
ne_50m = ne_50m.reset_index(drop=True) # Re-index the array for neatness


# Load the population data
popPath = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/popGridSSP_SEDAC.nc'
popGrid = xr.open_dataarray(popPath)

# If using SEDAC grid, select out just one instance
popGrid = popGrid.loc[dict(scenario='ssp2',year=2010)]

# RegionMask code adapted from https://regionmask.readthedocs.io/en/stable/notebooks/geopandas.html
lon = popGrid.lon.values
lat = popGrid.lat.values

mask = regionmask.mask_geopandas(ne_50m, lon, lat)
mask_3D = regionmask.mask_3D_geopandas(ne_50m, lon, lat)

# What's the world population within the mask?

# Convert mask to boolean
maskB = mask.where(np.isnan(mask),other=1)
maskB = maskB.where(maskB==1,other=0)

WrldPop = popGrid.where(maskB) # Population within the mask
print(np.sum(WrldPop).values)
print(np.sum(popGrid).values)
pct = np.sum(WrldPop).values/np.sum(popGrid).values*100
print("Captures %.2f percent of the population" % pct)

#-----Implementing a coastal neighbour inclusion algorithm for the world-----#

# The naive approach checks each gridpoint to see if its centre is within a country
# boundary. This excludes coastal gridcells where the centre is in the ocean but
# some land (possibly with some people) is present. 
# Here, I take an approach where any unassigned gridcell that is adjacent to one
# or more assigned gridcells is then assigned to its modal (most common) neighbour.

lats = mask.lat.values
lons = mask.lon.values

# Create an equal-sized mask of zeroes for adding the neighbour values to
neighbMask = mask.where(np.isnan(mask),other=0)
neighbMask = neighbMask.where(neighbMask==0,other=0)

counter = 0
icount = 0
jcount = 0
for i in lats:
    if i != min(lats) and i !=max(lats): #skip the edge cases 
        for j in lons:
            if j != min(lons) and j !=max(lons):
                thiscell = mask.loc[dict(lat=i, lon=j)]
                if np.isnan(thiscell.values):
                    # Get the values of the eight neighbouring cells
                    N  = float(mask[icount-1][jcount])
                    NE = float(mask[icount-1][jcount+1])
                    E  = float(mask[icount][jcount+1])
                    SE = float(mask[icount+1][jcount+1])
                    S  = float(mask[icount+1][jcount])
                    SW = float(mask[icount+1][jcount-1])
                    W  = float(mask[icount][jcount-1])
                    NW = float(mask[icount-1][jcount-1])
                    neighbs = np.array([N, NE, E, SE, S, SW, W, NW])
                    
                    if np.isfinite(neighbs).any(): # Ignore if all are nan
                        counter+=1
                        # Fill the empty cell with its most common neighbour value
                        neighbMask[icount][jcount] = stats.mode(neighbs[np.isfinite(neighbs)])[0][0]
            jcount+=1
    icount+=1
    print(icount)
    jcount=0
print(counter)

# Create a new mask that joins the original mask and the coastal neighbours mask
jointMask = mask.where(~np.isnan(mask), other=0) # Same as original mask, but with 0 instead of nan
jointMask = jointMask + neighbMask # Add the neighbour values
jointMask = jointMask.where(jointMask>0, other=np.nan) # Convert the zeros back to nans

# Optional: Save this for easier reference
#jointMask.to_netcdf(path='mask_ext_SEDAC.nc') 

# What's the world population within the extended mask?

# Convert mask to boolean
maskB = jointMask.where(np.isnan(jointMask),other=1)
maskB = maskB.where(maskB==1,other=0)

WrldPop = popGrid.where(maskB) # Population within the extended mask
print(np.sum(WrldPop).values)
print(np.sum(popGrid).values)
pct = np.sum(WrldPop).values/np.sum(popGrid).values*100
print("Captures %.2f percent of the population" % pct)


# Use the extended 2D mask to create an extended 3D mask
mask_3D_ext = mask_3D.copy(deep=True) # Make a deep copy so the original 3D mask is retained

for thisReg in mask_3D_ext.region:
    thisMask = jointMask.where(jointMask==thisReg, other=False) # Filter the 2D mask
    thisMask = thisMask/np.max(thisMask) # Now it's all 0 or 1
    mask_3D_ext.loc[dict(region = thisReg)] = thisMask # Assign the values to the right layer in the 3D mask

# Rename the regions with the ADM0_A3 code instead of the index
regList = ne_50m['ADM0_A3'].values
regListShort = regList[mask_3D_ext.region.values]
mask_3D_ext = mask_3D_ext.assign_coords(region=regListShort)

# Save the mask for use in other scripts
mask_3D_ext.to_netcdf(path=outPath+'mask_3D_ext_SEDAC.nc')

# Optional: Show off the handiwork
#regionmask.plot_3D_mask(mask_3D, add_colorbar=False, cmap="plasma")
#regionmask.plot_3D_mask(mask_3D_ext, add_colorbar=False, cmap="plasma")
