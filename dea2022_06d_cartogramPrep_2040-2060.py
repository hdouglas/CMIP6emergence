#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:58:19 2021

@author: 1001883
"""

# Douglas et al. 2022 Script 6
# Script to calculate the average S/N for each country (2071-2100, 16/50/84th percentile)
# for use in the R script for generating the population-weighted cartograms

# Inputs: multi-ensemble netcdf files (S, N, S/N) from script 01 for all models, 
# country mask, 
# Outputs: 1) netcdf of population by country and scenario
# 2) gridded netcdf of S/N by country. Dimensions: region, percentile, scenario.
# 3) CSVs of the above for use in R (one set of population, one set of SN)

import numpy as np
import xarray as xr
import pandas as pd
import xesmf as xe  

###############################
####----USER-CHANGEABLE----####

variable_id = 'tas'
table_id = 'Amon'
scen_short = ['ssp119','ssp126','ssp245','ssp370','ssp585']
interpMethod = 'bilinear' # 'patch' # 'conservative' #  Set the interpolation method for regridding

# Path where the SSP population data are stored in a csv (available from 
# https://tntcat.iiasa.ac.at/SspDb/dsd?Action=htmlpage&page=20 - login required)
homedir = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/'

# Chose the model type
thisMod = 'IIASA-WiC POP'  # or 'NCAR' # 

# Paths for outputs
outPath = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/'
csvOutPath = outPath+'CSVs/'

# Load the mask
maskPath = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/mask_3D_ext_SEDAC.nc'
mask_3D_ext = xr.open_dataarray(maskPath)

modPath = '/Volumes/GeoCCRI_01/users-data/douglahu/CMIP6_data/'+variable_id+'/'+table_id+'/01_MM_output/'

scen_short = ['ssp119','ssp126','ssp245','ssp370','ssp585']

POI = [2040,2060] # Period of interest over which signal and SN are averaged

####----END USER-CHANGEABLE----####
###################################

####----Step 1 - Save the population by country/scenario to a netcdf----####

## Load the SSP country population data and store in a more usable format
csv = pd.read_csv(homedir+'SspDb_country_data_2013-06-12.csv')

# Filter to only use the total population
pops = csv.loc[csv['VARIABLE'] == 'Population'] 

# Filter to only use the model we want
pops = pops.loc[pops['MODEL'] == thisMod]

countries = np.sort(pd.unique(pops['REGION']))
scenarios = np.sort(pd.unique(pops['SCENARIO']))

if thisMod=='IIASA-WiC POP':
    years = ['2010','2015','2020','2025','2030','2035','2040','2045','2050','2055',
             '2060','2065','2070','2075','2080','2085','2090','2095','2100']
    yearsnum = np.arange(2010,2101,5)
elif thisMod=='NCAR':
    years = ['2010','2020','2030','2040','2050','2060','2070','2080','2090','2100']
    yearsnum = np.arange(2010,2101,10)

pops = pops.sort_values('REGION')

# Set an empty xarray for storing the values
popArr = np.zeros([len(countries),len(years),len(scenarios)])
popData = xr.DataArray(data=popArr, dims=['country','year','scenario'], 
                       coords=[countries,yearsnum,scenarios], name="Population")

for i in scenarios:
    thesepops = pops.loc[pops['SCENARIO'] == i]
    thesepops = thesepops.filter(items=years, axis=1)
    popData.loc[dict(scenario=i)]=thesepops

popData.attrs['units'] = "millions"
    
popData.to_netcdf(path=outPath+'countryPopulation_'+thisMod+'.nc')

####----Step 2 - Signal-to-noise by country and scenario----####

# Set up an xarray for the S/N results. Dimensions: region, percentile, scenario.
regions = mask_3D_ext.region 
zeroArr = np.zeros([len(regions),3,len(scen_short)])
SN_countries = xr.DataArray(data=zeroArr,dims=['region','percentile','scenario'],
                       coords=[regions,[16,50,84],scen_short])

i=0
for thisSSP in scen_short:
    print(thisSSP)
    SN_models = xr.open_dataarray(modPath+'MM_dea2022_SN_'+str(POI[0])+'-'+str(POI[1])+'_'+thisSSP+'.nc')
    SN_Avg = SN_models.median('model', keep_attrs=True, skipna=True)
    SN_16 = SN_models.quantile(0.16, 'model', keep_attrs=True, skipna=True)
    SN_84 = SN_models.quantile(0.84, 'model', keep_attrs=True, skipna=True)
    
    # Regrid SN array to match the mask
    ds_in = SN_Avg.to_dataset(promote_attrs=True)
    
    lat = mask_3D_ext.lat.values
    lon = mask_3D_ext.lon.values
    ds_out = xr.Dataset({'lat': (['lat'], lat), 'lon': (['lon'], lon)})
    
    regridder = xe.Regridder(ds_in, ds_out, interpMethod, periodic=True) 
    regridder  # print basic regridder information.
    
    SN_Avg = SN_Avg.transpose('lat','lon') # These need to be in the right order
    SN_Out50 = regridder(SN_Avg)
    SN_Out16 = regridder(SN_16)
    SN_Out84 = regridder(SN_84)
    
    #Reorder the dimensions as necessary:
    SN_Out50 = SN_Out50.transpose('lat','lon')
    SN_Out16 = SN_Out16.transpose('lat','lon')
    SN_Out84 = SN_Out84.transpose('lat','lon')    
    
    for thisRegion in mask_3D_ext.region.values:
        print(thisRegion)
        thisMask = mask_3D_ext.loc[dict(region=thisRegion)].drop('region')
        thisSN50 = SN_Out50.where(thisMask>0)
        thisSN16 = SN_Out16.where(thisMask>0)
        thisSN84 = SN_Out84.where(thisMask>0)
        
        # Average across the country and store in the dataarray
        # Make sure average is properly area-weighted
        weights = np.cos(np.deg2rad(thisSN50['lat'])) 
        # Count number of country gridcells at each lat band (needed for proper weighting)
        counts = thisMask.sum('lon') 
        thisSN50avg = (thisSN50*weights).sum('lon').sum('lat')/(weights*counts).sum('lat')
        #thisSN50avg = (thisSN50*weights).mean('lat').mean('lon')
        SN_countries.loc[dict(region=thisRegion, percentile=50, 
                              scenario=thisSSP)]=float(thisSN50avg.values)
        thisSN16avg = (thisSN16*weights).sum('lon').sum('lat')/(weights*counts).sum('lat')
        #thisSN16avg = (thisSN16*weights).mean('lat').mean('lon')
        SN_countries.loc[dict(region=thisRegion, percentile=16, 
                              scenario=thisSSP)]=float(thisSN16avg.values)
        thisSN84avg = (thisSN50*weights).sum('lon').sum('lat')/(weights*counts).sum('lat')
        #thisSN84avg = (thisSN84*weights).mean('lat').mean('lon')
        SN_countries.loc[dict(region=thisRegion, percentile=84, 
                              scenario=thisSSP)]=float(thisSN84avg.values)
    
    i+=1
  
SN_countries.to_netcdf(path=outPath+'SN_countries_SEDAC.nc')      

# Optional verification 
for thisRegion in mask_3D_ext.region.values:
    theseData = SN_countries.loc[dict(region=thisRegion)]
    print(thisRegion)
    print(float(np.sum(theseData)))
    if np.sum(theseData) == 0:
        print(thisRegion)


####----Step 3 - Save country-level population and S/N as CSVs for R to use----####
        
# Load SSP population projections
#SSPproj = xr.open_dataarray(outPath+'countryPopulation_'+thisMod+'.nc')
SSPproj = popData

# Load the L21C country-average S/N values
#SN_countries = xr.open_dataarray(outPath+'SN_countries_SEDAC.nc')

# Get the average population for POI:
l21c = np.arange(POI[0],POI[1]+1,5)
SSPpop_l21c = SSPproj.loc[dict(year=l21c)]
SSPpop_l21c = SSPpop_l21c.mean('year')

scen_long = [SSPpop_l21c.scenario.values[0],SSPpop_l21c.scenario.values[0],SSPpop_l21c.scenario.values[1],
             SSPpop_l21c.scenario.values[2],SSPpop_l21c.scenario.values[4]]

# Save the country populations as csv files
i = 0
for thisSSP in scen_long:
    pops = SSPpop_l21c.loc[dict(scenario=thisSSP)].values
    countries = SSPpop_l21c.country.values
    countryPops = np.vstack((countries,pops))
    countryPops = np.transpose(countryPops)
    np.savetxt(csvOutPath+'countryPop_'+scen_short[i]+'_'+str(POI[0])+'-'+str(POI[1])+'.csv',
               countryPops,delimiter=',', fmt=['%s','%f'],
               header='Country,Population (millions)_'+scen_short[i],comments='')
    i+=1 

# Save the country S/N values as csv files
for thisSSP in scen_short:
    for thispct in SN_countries.percentile:
        SN = SN_countries.loc[dict(scenario=thisSSP,percentile=thispct)].values
        countries = SN_countries.region.values
        countrySN = np.vstack((countries,SN))
        countrySN = np.transpose(countrySN)
        np.savetxt(csvOutPath+'countrySN_'+str(thispct.values)+'_'+thisSSP+'_'+str(POI[0])+'-'+str(POI[1])+'.csv',
                   countrySN,delimiter=',',fmt=['%s','%f'],
                   header='Country,S/N_'+str(thispct.values)+'_'+thisSSP,comments='')
