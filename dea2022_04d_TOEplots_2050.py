#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:58:19 2021

@author: 1001883
"""

# Douglas et al. 2022 Script 4
# Time-of-emergence and gridded signal-to-noise plots 

# Inputs: population and area exposed netcdfs from Script 3, gridded population data,
# multi-model S, N, S/N netcdfs from script 2, as well as outputs from CMIP5 series
# of scripts and the population processing series of scripts. 
# Outputs: Most of the main figures

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

###############################
####----USER-CHANGEABLE----####

variable_id = 'tas'
table_id = 'Amon'

# Load the gridded population data
popPath = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/popGridSSP_SEDAC.nc'
popGridSSP4D = xr.open_dataarray(popPath)
scenarios = popGridSSP4D.scenario.values
print(scenarios)
# Note: now includes ssp534-over
scen_long = [scenarios[0], scenarios[0], scenarios[1], scenarios[2], scenarios[4], scenarios[4]]

# Load the gridded area data
areaPath = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/landAreas_0.5deg.nc'
areaGrid = xr.open_dataarray(areaPath)

# Sum across latitude and longitude to get global totals
popTots = popGridSSP4D.sum('lat')
popTots = popTots.sum('lon')

# Paths to the population/area exposure data
toePath = '/Volumes/GeoCCRI_01/users-data/douglahu/Scripts/FEA2017/'
toePath_rcp = '/Volumes/GeoCCRI_01/users-data/douglahu/Scripts/CMIP5/'

# Load the model data, just to get the total number of models per scenario
modPath = '/Volumes/GeoCCRI_01/users-data/douglahu/CMIP6_data/'+variable_id+'/'+table_id+'/01_MM_output/'
scen_short = ['ssp119','ssp126','ssp245','ssp370','ssp585']

modPath_rcp = '/Volumes/GeoCCRI_01/users-data/douglahu/CMIP5_data/'+variable_id+'/'+table_id+'/01_MM_output/'
scen_short_rcp = ['rcp26','rcp45','rcp85']

# A list of strings for aligning the SSPs and RCPs 
rcp_inp = ['skip','rcp26','rcp45','skip','rcp85']

# Colour palette for plotting
palette = ['#f598ae', '#d24799', '#73297e', '#242952']

# Percentiles for Figure 1 to show model uncertainty
pcts = [16,50,84]
pctLabs = ['16th', '50th', '84th']

# Define Period of interest (end of century) Note that if a rolling average is used,
# the last year may be earlier than 2100
Yn1 = 2040 #2071 # (inclusive)
Yn2 = 2061 #2101 # (non-inclusive)

# Set years of interest to show as separate points in area-population plot
yoi = [2022,2030,2050] 

# Note there are also colourbar parameters etc. in the plotting fuctions that need
# to be edited there instead of in this block (e.g. "levels").

# Figure 1 colours 
# paletteF1 = ['#FFFFFF','#FFFEC7', '#FEED9B', '#FBB246', '#FB4E2B']
# paletteF1 = ['#FFFFFF','#FFEE97', '#FDB535', '#E02612', '#BB1323']
paletteF1 = ['#FFFFFF','#FFFEC7', '#FCD287', '#E67474', '#B13E3E']

####----END USER-CHANGEABLE----####
###################################

# Load the time-of-emergence data (population exposed to S-N over threshold x)
pop_exp = xr.open_dataarray(toePath+'dea2022_pop_exposed_SN_SSP.nc')
pop_exp_16 = xr.open_dataarray(toePath+'dea2022_pop_exposed_SN_SSP_16.nc')
pop_exp_84 = xr.open_dataarray(toePath+'dea2022_pop_exposed_SN_SSP_84.nc')

# Same for the RCPs
pop_exp_rcp = xr.open_dataarray(toePath_rcp+'dea2022_pop_exposed_SN_RCP.nc')

# Normalise by total population to get a proportion exposed
prop_exp = pop_exp.copy(deep=True)
prop_exp.name='proportion'
prop_exp_16 = pop_exp_16.copy(deep=True)
prop_exp_16.name='proportion'
prop_exp_84 = pop_exp_84.copy(deep=True)
prop_exp_84.name='proportion'

prop_exp_rcp = pop_exp_rcp.copy(deep=True)
prop_exp_rcp.name='proportion'

i = 0
for thisScen in prop_exp.scenario:
    print(thisScen)
    prop_exp.loc[dict(scenario=thisScen)]=prop_exp.loc[dict(scenario=thisScen)]/popTots.loc[dict(scenario=scen_long[i])]
    prop_exp_16.loc[dict(scenario=thisScen)]=prop_exp_16.loc[dict(scenario=thisScen)]/popTots.loc[dict(scenario=scen_long[i])]
    prop_exp_84.loc[dict(scenario=thisScen)]=prop_exp_84.loc[dict(scenario=thisScen)]/popTots.loc[dict(scenario=scen_long[i])]
    i+=1
    
scen_inp = [scenarios[0], scenarios[1], scenarios[4]] #aligns with SSPs 1, 2, 5
i = 0
for thisScen in prop_exp_rcp.scenario:
    prop_exp_rcp.loc[dict(scenario=thisScen)]=prop_exp_rcp.loc[dict(scenario=thisScen)]/popTots.loc[dict(scenario=scen_inp[i])]
    i+=1

# Count the number of models per scenario
modNums = []
for thisSSP in scen_short:
    thisMod = xr.open_dataarray(modPath+'MM_dea2022_SN_'+thisSSP+'_allYears.nc')
    modNums.append(len(thisMod.model))
    print(np.sort(thisMod.model))
    
modNums_rcp = []
for thisRCP in scen_short_rcp:
    thisMod = xr.open_dataarray(modPath_rcp+'MM_dea2022_SN_'+thisRCP+'_allYears.nc')
    modNums_rcp.append(len(thisMod.model))
    print(np.sort(thisMod.model))
    
# Print some outputs for 2022 from the projections
print('2022 projections')
print(prop_exp.loc[dict(year=2022,threshold=1)])
print(prop_exp.loc[dict(year=2022,threshold=2)])

# Truncate the exposure data after 2060
prop_exp = prop_exp.loc[dict(year=slice(np.min(pop_exp.year),2060))]
prop_exp_16 = prop_exp_16.loc[dict(year=slice(np.min(pop_exp_16.year),2060))]
prop_exp_84 = prop_exp_84.loc[dict(year=slice(np.min(pop_exp_84.year),2060))]
prop_exp_rcp = prop_exp_rcp.loc[dict(year=slice(np.min(pop_exp_rcp.year),2060))]

####################
#-----Plotting-----#
####################

################################################################
##---TOE plot with the population added on a secondary axis---##
def scenPopPlotter(array, array_16, array_84, scen, thisScen_long, modNum, figi, 
                   rcp, array_rcp, modNum_rcp):
    plt.rc('font', size=12)
    fig, ax1 = plt.subplots()
   
    ax1.plot(array.year, array.loc[dict(scenario=scen, threshold=1)]*100, color=palette[0])
    ax1.fill_between(array_16.year, array_16.loc[dict(scenario=scen, threshold=1)]*100, 
                    array_84.loc[dict(scenario=scen, threshold=1)]*100, color=palette[0], alpha=0.2)
    
    ax1.plot(array.year, array.loc[dict(scenario=scen, threshold=2)]*100, color=palette[1])
    ax1.fill_between(array_16.year, array_16.loc[dict(scenario=scen, threshold=2)]*100, 
                    array_84.loc[dict(scenario=scen, threshold=2)]*100, color=palette[1], alpha=0.2)    
    
    ax1.plot(array.year, array.loc[dict(scenario=scen, threshold=3)]*100, color=palette[2])
    ax1.fill_between(array_16.year, array_16.loc[dict(scenario=scen, threshold=3)]*100, 
                    array_84.loc[dict(scenario=scen, threshold=3)]*100, color=palette[2], alpha=0.2)
    
    ax1.plot(array.year, array.loc[dict(scenario=scen, threshold=5)]*100, color=palette[3])
    ax1.fill_between(array_16.year, array_16.loc[dict(scenario=scen, threshold=5)]*100, 
                    array_84.loc[dict(scenario=scen, threshold=5)]*100, color=palette[3], alpha=0.2)
    
    #add the RCPs
    if rcp != 'skip':
        ax1.plot(array.year, array_rcp.loc[dict(scenario=rcp, threshold=1)]*100, color=palette[0], linestyle='dashed')
        ax1.plot(array.year, array_rcp.loc[dict(scenario=rcp, threshold=2)]*100, color=palette[1], linestyle='dashed')
        ax1.plot(array.year, array_rcp.loc[dict(scenario=rcp, threshold=3)]*100, color=palette[2], linestyle='dashed')
        ax1.plot(array.year, array_rcp.loc[dict(scenario=rcp, threshold=5)]*100, color=palette[3], linestyle='dashed')
        plt.title(scen+' (%.0f models)' % modNum+' + '+rcp+' (%.0f models)' % modNum_rcp)
        if scen == 'ssp119':
            ax1.legend(['S/N1 (unusual)', 'S/N2 (unfamiliar)', 'S/N3 (unknown)', 'S/N5 (inconceivable)', 'RCP'],loc='upper left')
    else:
        plt.title(scen+' (%.0f models)' % modNum)
        if scen == 'ssp119':
            ax1.legend(['S/N1 (unusual)', 'S/N2 (unfamiliar)', 'S/N3 (unknown)', 'S/N5 (inconceivable)'],loc='upper left')
       
    ax1.axis([2010,2060,0,100])
    ax1.set_ylabel('Population exposed (%)')
    
    # twin object for two different y-axis on the sample plot
    ax2=ax1.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(popTots.year, popTots.loc[dict(scenario=thisScen_long)],color='grey', linestyle='dotted')
    ax2.set_ylabel("Total global population, billions")
    ax2.axis([2010,2060,0,12e9])
    ax2.set_yticks([0, 2e9, 4e9, 6e9, 8e9, 1e10, 1.2e10])
    ax2.set_yticklabels([0, 2, 4, 6, 8, 10, 12])
    if scen == 'ssp119':
        ax2.legend(['Population'],loc='center left')

    return fig

modNumsIn = [0,modNums_rcp[0],modNums_rcp[1],0,modNums_rcp[2]]
for i in np.arange(0,len(scen_short)):
    if scen_short[i][0:4] != scen_long[i]:
        print('Error! Misalignment between scenario lists scen_short and scen_long.')
    else:
        print(scen_short[i])
        thisPlot = scenPopPlotter(prop_exp,prop_exp_16,prop_exp_84,scen_short[i],scen_long[i],
                                  modNums[i],i,rcp_inp[i],prop_exp_rcp,modNumsIn[i])
        thisPlot.savefig('./Plots/dea2022_pop-SN_TOE_'+str(Yn1)+'-'+str(Yn2-1)+'_'+scen_short[i]+'.svg')


#################################
##---As above, but with area--##

# Sum across latitude and longitude to get global totals
areaTot = float(np.sum(areaGrid))

# Load the time-of-emergence data (population exposed to S-N over threshold x)
area_exp = xr.open_dataarray(toePath+'dea2022_area_exposed_SN_SSP.nc')
area_exp_16 = xr.open_dataarray(toePath+'dea2022_area_exposed_SN_SSP_16.nc')
area_exp_84 = xr.open_dataarray(toePath+'dea2022_area_exposed_SN_SSP_84.nc')

#toePath = '/Users/hunterdouglas/Documents/VUW/Exercises/CMIP5/'
area_exp_rcp = xr.open_dataarray(toePath_rcp+'dea2022_area_exposed_SN_RCP.nc')

# Normalise by total population to get a proportion exposed
prop_a_exp = area_exp.copy(deep=True)
prop_a_exp.name='proportion'
prop_a_exp_16 = area_exp_16.copy(deep=True)
prop_a_exp_16.name='proportion'
prop_a_exp_84 = area_exp_84.copy(deep=True)
prop_a_exp_84.name='proportion'

prop_a_exp_rcp = area_exp_rcp.copy(deep=True)
prop_a_exp_rcp.name='proportion'

for thisScen in prop_a_exp.scenario:
    prop_a_exp.loc[dict(scenario=thisScen)]=prop_a_exp.loc[dict(scenario=thisScen)]/areaTot
    prop_a_exp_16.loc[dict(scenario=thisScen)]=prop_a_exp_16.loc[dict(scenario=thisScen)]/areaTot
    prop_a_exp_84.loc[dict(scenario=thisScen)]=prop_a_exp_84.loc[dict(scenario=thisScen)]/areaTot

for thisScen in prop_a_exp_rcp.scenario:
    prop_a_exp_rcp.loc[dict(scenario=thisScen)]=prop_a_exp_rcp.loc[dict(scenario=thisScen)]/areaTot

# Truncate the exposure data after 2060
prop_a_exp = prop_a_exp.loc[dict(year=slice(np.min(prop_a_exp.year),2060))]
prop_a_exp_16 = prop_a_exp_16.loc[dict(year=slice(np.min(prop_a_exp_16.year),2060))]
prop_a_exp_84 = prop_a_exp_84.loc[dict(year=slice(np.min(prop_a_exp_84.year),2060))]
prop_a_exp_rcp = prop_a_exp_rcp.loc[dict(year=slice(np.min(prop_a_exp_rcp.year),2060))]

# Time of emergence of all thresholds, one plot per scenario
def scenPlotter(array, array_16, array_84, scen, modNum, figi, rcp, array_rcp,modNum_rcp):
    plt.rc('font', size=12)
    fig = plt.figure(figi)
    ax = fig.add_subplot(111)
    ax.plot(array.year, array.loc[dict(scenario=scen, threshold=1)]*100, color=palette[0])
    ax.fill_between(array_16.year, array_16.loc[dict(scenario=scen, threshold=1)]*100, 
                    array_84.loc[dict(scenario=scen, threshold=1)]*100, color=palette[0], alpha=0.2)
    
    ax.plot(array.year, array.loc[dict(scenario=scen, threshold=2)]*100, color=palette[1])
    ax.fill_between(array_16.year, array_16.loc[dict(scenario=scen, threshold=2)]*100, 
                    array_84.loc[dict(scenario=scen, threshold=2)]*100, color=palette[1], alpha=0.2)    
    
    ax.plot(array.year, array.loc[dict(scenario=scen, threshold=3)]*100, color=palette[2])
    ax.fill_between(array_16.year, array_16.loc[dict(scenario=scen, threshold=3)]*100, 
                    array_84.loc[dict(scenario=scen, threshold=3)]*100, color=palette[2], alpha=0.2) 
    
    ax.plot(array.year, array.loc[dict(scenario=scen, threshold=5)]*100, color=palette[3])
    ax.fill_between(array_16.year, array_16.loc[dict(scenario=scen, threshold=5)]*100, 
                    array_84.loc[dict(scenario=scen, threshold=5)]*100, color=palette[3], alpha=0.2)
    
    #add the RCPs
    if rcp != 'skip':
        ax.plot(array.year, array_rcp.loc[dict(scenario=rcp, threshold=1)]*100, color=palette[0], linestyle='dashed')
        ax.plot(array.year, array_rcp.loc[dict(scenario=rcp, threshold=2)]*100, color=palette[1], linestyle='dashed')
        ax.plot(array.year, array_rcp.loc[dict(scenario=rcp, threshold=3)]*100, color=palette[2], linestyle='dashed')
        ax.plot(array.year, array_rcp.loc[dict(scenario=rcp, threshold=5)]*100, color=palette[3], linestyle='dashed')
        plt.title(scen+' (%.0f models)' % modNum+' + '+rcp+' (%.0f models)' % modNum_rcp)
        if scen == 'ssp119':
            plt.legend(['S/N1 (unusual)', 'S/N2 (unfamiliar)', 'S/N3 (unknown)', 'S/N5 (inconceivable)', 'RCP'],loc='upper left')
    else:
        plt.title(scen+' (%.0f models)' % modNum)
        if scen == 'ssp119':
            plt.legend(['S/N1 (unusual)', 'S/N2 (unfamiliar)', 'S/N3 (unknown)', 'S/N5 (inconceivable)'],loc='upper left')
    
    plt.axis([2010,2060,0,100])
    plt.ylabel('Land area exposed (%)')
 
    return fig

for i in np.arange(0,len(scen_short)):
    thisPlot = scenPlotter(prop_a_exp,prop_a_exp_16,prop_a_exp_84,scen_short[i],modNums[i],
                           i+10,rcp_inp[i],prop_a_exp_rcp,modNumsIn[i])
    thisPlot.savefig('./Plots/dea2022_area-SN_TOE_'+str(Yn1)+'-'+str(Yn2-1)+'_'+scen_short[i]+'.svg')


#####################################
##---Area-population median plot---##

def medianCompPlot(areaArr, popArr, scen, figi, yoi_in):
    plt.rc('font', size=12)
    fig = plt.figure(figi+20)
    ax = fig.add_subplot(111)
    
    # Lines
    ax.plot(popArr.loc[dict(scenario=scen, threshold=1)]*100, 
            areaArr.loc[dict(scenario=scen, threshold=1)]*100, color=palette[0])
    ax.plot(popArr.loc[dict(scenario=scen, threshold=2)]*100, 
            areaArr.loc[dict(scenario=scen, threshold=2)]*100, color=palette[1])
    ax.plot(popArr.loc[dict(scenario=scen, threshold=3)]*100, 
            areaArr.loc[dict(scenario=scen, threshold=3)]*100, color=palette[2])
    
    # Markers
    ax.scatter(popArr.loc[dict(scenario=scen, threshold=1, year=yoi)]*100,
               areaArr.loc[dict(scenario=scen, threshold=1, year=yoi)]*100,
               marker='o', color=palette[0], s=60)
    ax.scatter(popArr.loc[dict(scenario=scen, threshold=2, year=yoi)]*100,
               areaArr.loc[dict(scenario=scen, threshold=2, year=yoi)]*100,
               marker='o', color=palette[1], s=60)
    ax.scatter(popArr.loc[dict(scenario=scen, threshold=3, year=yoi)]*100,
               areaArr.loc[dict(scenario=scen, threshold=3, year=yoi)]*100,
               marker='o', color=palette[2], s=60)
            
    for thisYear in yoi_in: 
        lab = str(thisYear)
        if plotOn.loc[dict(scenario=scen,year=thisYear,threshold=1)]:
            xpt = popArr.loc[dict(scenario=scen, threshold=1, year=thisYear)]*100
            ypt = areaArr.loc[dict(scenario=scen, threshold=1, year=thisYear)]*100
            ax.annotate(lab, (xpt, ypt), textcoords="offset points", xytext=(5,-5), 
                        ha='left', color = palette[0]) 
        if plotOn.loc[dict(scenario=scen,year=thisYear,threshold=2)]:
            xpt = popArr.loc[dict(scenario=scen, threshold=2, year=thisYear)]*100
            ypt = areaArr.loc[dict(scenario=scen, threshold=2, year=thisYear)]*100
            ax.annotate(lab, (xpt, ypt), textcoords="offset points", xytext=(5,-5), 
                        ha='left', color = palette[1]) 
        if plotOn.loc[dict(scenario=scen,year=thisYear,threshold=3)]:
            xpt = popArr.loc[dict(scenario=scen, threshold=3, year=thisYear)]*100
            ypt = areaArr.loc[dict(scenario=scen, threshold=3, year=thisYear)]*100
            ax.annotate(lab, (xpt, ypt), textcoords="offset points", xytext=(5,-5), 
                        ha='left', color = palette[2]) 
    
    ax.plot([0,100],[0,100],'k-',linewidth=0.7)
    
    plt.axis([0,100,0,100])
    plt.ylabel('Land area exposed (%)')
    plt.xlabel('Population exposed (%)')
    if scen == 'ssp119':
        plt.legend(['S/N1', 'S/N2', 'S/N3'],loc='upper left')
    plt.title(scen)
    
    return fig

# Create an xarray to instruct whether or not to print the labels.
plotOn = xr.DataArray(data = np.array(np.ones([5,4,len(yoi)]),dtype='bool'),
                     coords=dict(scenario = scen_short, threshold = [1,2,3,5],
                                 year = yoi))
plotOn.loc[dict(scenario='ssp119',threshold=2,year=2030)]=False
plotOn.loc[dict(scenario='ssp119',threshold=3,year=2022)]=False
plotOn.loc[dict(scenario='ssp119',threshold=3,year=2030)]=False
plotOn.loc[dict(scenario='ssp126',threshold=2,year=2022)]=False
plotOn.loc[dict(scenario='ssp126',threshold=3,year=2022)]=False
plotOn.loc[dict(scenario='ssp245',threshold=3,year=2022)]=False
plotOn.loc[dict(scenario='ssp585',threshold=1,year=2030)]=False
plotOn.loc[dict(scenario='ssp585',threshold=1,year=2050)]=False
    
for i in np.arange(0,len(scen_short)):
    thisPlot = medianCompPlot(prop_a_exp,prop_exp,scen_short[i],i+20,yoi)
    thisPlot.savefig('./Plots/dea2022_area-pop-exposed_'+str(Yn1)+'-'+str(Yn2-1)+'_'+scen_short[i]+'.svg')

##################################
##-----Signal-to-noise plots----##

# First, load the model output and calculate multi-model 16th, 50th, and 84th percentiles
# We're focussing on a particular period of interest (POI), following Frame et al. 2017, 
# this will be 2071 to 2100.

# Create an empty xarray with dimensions lat, lon, scenario, percentile
# Get the required lat & lon arrays from a representative file. 
dummy = xr.open_dataarray(modPath+'MM_dea2022_SN_'+str(Yn1)+'-'+str(Yn2-1)+'_ssp126.nc')
lat = dummy.lat.values
lon = dummy.lon.values

N_SSP = xr.DataArray(data=0.0, coords=[lat,lon,scen_short], 
                     dims=['lat','lon','scenario'], name='noise')
S_SSP = xr.DataArray(data=0.0, coords=[lat,lon,scen_short], 
                     dims=['lat','lon','scenario'], name='signal')
SN_SSP_pct = xr.DataArray(data=0.0, coords=[lat,lon,scen_short,pcts], 
                          dims=['lat','lon','scenario','percentile'], name='SN')

# Calculate the multi-model median for all scenarios
for thisSSP in scen_short:
    print(thisSSP)
    try:
        # Noise
        N_models = xr.open_dataarray(modPath+'MM_dea2022_noise_'+thisSSP+'.nc')
        noiseAvg = N_models.median('model', keep_attrs=True, skipna=True)
        N_SSP.loc[dict(scenario=thisSSP)]= noiseAvg.values
        
        # Signal
        S_models = xr.open_dataarray(modPath+'MM_dea2022_signal_'+str(Yn1)+'-'+str(Yn2-1)+'_'+thisSSP+'.nc')
        signalAvg = S_models.median('model', keep_attrs=True, skipna=True)
        S_SSP.loc[dict(scenario=thisSSP)]= signalAvg.values
        
        # S/N
        SN_models = xr.open_dataarray(modPath+'MM_dea2022_SN_'+str(Yn1)+'-'+str(Yn2-1)+'_'+thisSSP+'.nc')
        SN_Avg = SN_models.median('model', keep_attrs=True, skipna=True)
        SN_16 = SN_models.quantile(0.16, 'model', keep_attrs=True, skipna=True)
        SN_84 = SN_models.quantile(0.84, 'model', keep_attrs=True, skipna=True)
        SN_SSP_pct.loc[dict(scenario=thisSSP, percentile=16)]= SN_16.values
        SN_SSP_pct.loc[dict(scenario=thisSSP, percentile=50)]= SN_Avg.values
        SN_SSP_pct.loc[dict(scenario=thisSSP, percentile=84)]= SN_84.values
    except:
        print('Multi-model averaging failed.')
        pass

# Output these for future use to speed up time
N_SSP.to_netcdf(path=toePath+'N_SSP_eoc.nc')
S_SSP.to_netcdf(path=toePath+'S_SSP_eoc_'+str(Yn1)+'-'+str(Yn2-1)+'.nc')
SN_SSP_pct.to_netcdf(path=toePath+'SN_SSP_pct_'+str(Yn1)+'-'+str(Yn2-1)+'.nc')

# Print the maximum of the EOC SN, for the colourbar
print(SN_SSP_pct)
print('Maximum:')
print(np.max(SN_SSP_pct))

#-----Same again, but for the RCPs----# 

# Create an empty xarray with dimensions lat, lon, scenario, percentile
# Get the required lat & lon arrays from a representative file. 
dummy_rcp = xr.open_dataarray(modPath_rcp+'MM_dea2022_SN_'+str(Yn1)+'-'+str(Yn2-1)+'_rcp26.nc')
lat_rcp = dummy_rcp.lat.values
lon_rcp = dummy_rcp.lon.values

N_RCP = xr.DataArray(data=0.0, coords=[lat_rcp,lon_rcp,scen_short_rcp], 
                     dims=['lat','lon','scenario'], name='noise')
S_RCP = xr.DataArray(data=0.0, coords=[lat_rcp,lon_rcp,scen_short_rcp], 
                     dims=['lat','lon','scenario'], name='signal')
SN_RCP_pct = xr.DataArray(data=0.0, coords=[lat_rcp,lon_rcp,scen_short_rcp,pcts], 
                      dims=['lat','lon','scenario','percentile'], name='SN')

# Print checkpoints for troubleshooting
print('CP0')
for thisRCP in scen_short_rcp:
    print(thisRCP)
    try:
        # Noise
        N_models = xr.open_dataarray(modPath_rcp+'MM_dea2022_noise_'+thisRCP+'.nc')
        print('CP1')
        noiseAvg = N_models.median('model', keep_attrs=True, skipna=True)
        print('CP2')
        N_RCP.loc[dict(scenario=thisRCP)]= noiseAvg.values
        print('CP3')
        
        # Signal
        S_models = xr.open_dataarray(modPath_rcp+'MM_dea2022_signal_'+str(Yn1)+'-'+str(Yn2-1)+'_'+thisRCP+'.nc')
        signalAvg = S_models.median('model', keep_attrs=True, skipna=True)
        S_RCP.loc[dict(scenario=thisRCP)]= signalAvg.values
        print('CP4')
        
        # S/N
        SN_models = xr.open_dataarray(modPath_rcp+'MM_dea2022_SN_'+str(Yn1)+'-'+str(Yn2-1)+'_'+thisRCP+'.nc')
        SN_Avg = SN_models.median('model', keep_attrs=True, skipna=True)
        SN_16 = SN_models.quantile(0.16, 'model', keep_attrs=True, skipna=True)
        SN_84 = SN_models.quantile(0.84, 'model', keep_attrs=True, skipna=True)
        SN_RCP_pct.loc[dict(scenario=thisRCP, percentile=16)]= SN_16.values
        SN_RCP_pct.loc[dict(scenario=thisRCP, percentile=50)]= SN_Avg.values
        SN_RCP_pct.loc[dict(scenario=thisRCP, percentile=84)]= SN_84.values
        print('CP5')
    except:
        print('Multi-model averaging failed.')
        pass

# Save for future speed-up
N_RCP.to_netcdf(path=toePath+'N_RCP_eoc_'+str(Yn1)+'-'+str(Yn2-1)+'.nc')
S_RCP.to_netcdf(path=toePath+'S_RCP_eoc_'+str(Yn1)+'-'+str(Yn2-1)+'.nc')
SN_RCP_pct.to_netcdf(path=toePath+'SN_RCP_pct_'+str(Yn1)+'-'+str(Yn2-1)+'.nc')

# Make a big grid plot of all scenarios and the 16th, 50th, and 84th percentiles

def multi_plotter(thisArray,scenNo):
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
    
    title = 'Modelled annual mean S/N values for '+str(Yn1)+'-'+str(Yn2-1)#'2071–2100'
    cbar_label = '$\sigma$'
    #levels = [0.0,1.6,2.2,2.8,3.6,4.4,5.4,6.7,9.0,15.3]
    levels = [0.0,1.0,2.0,3.0,5.0,15.3]
    #cbTicks = np.arange(0,30.1,(30.0/9))
    #cbLabels = ['0.0','1.6','2.2','2.8','3.6','4.4','5.4','6.7','9.0','15.3']
    cbLabels = ['0.0','1.0','2.0','3.0','5.0','15.3']
    
    #cmap = mpl.cm.YlOrRd
    cmap = ListedColormap([paletteF1[0],paletteF1[1],paletteF1[2],paletteF1[3],paletteF1[4]])
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
        thisPct = pcts[np.mod(i,3)]
        
        theseData = thisArray.loc[dict(scenario=thisScen, percentile=thisPct)]
        
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
            ax.text(-0.04, 0.55, thisScen, va='bottom', ha='center',
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

gridPlot = multi_plotter(SN_SSP_pct,5)
gridPlot.savefig('./Plots/dea2022_SSP_SN-grid-all_'+str(Yn1)+'-'+str(Yn2-1)+'.svg', bbox_inches='tight', dpi=300)
# .svg for editing in Inkscape to add the cartograms

# Repeat for the RCPs (only 3 of these that we're looking at)
gridPlot = multi_plotter(SN_RCP_pct,3)
gridPlot.savefig('./Plots/dea202_RCP_SN-grid-all_'+str(Yn1)+'-'+str(Yn2-1)+'.pdf', bbox_inches='tight', dpi=300)


#####################################
##---CMIP5-CMIP6 difference plot---##

# Make copies of the earlier-computed files to avoid overwriting things
N_RCP_eoc = N_RCP.copy(deep=True) # EOC noise for RCPs
N_SSP_eoc = N_SSP.copy(deep=True) # EOC noise for SSPs
S_RCP_eoc = S_RCP.copy(deep=True) # EOC signal for RCPs
S_SSP_eoc = S_SSP.copy(deep=True) # EOC signal for SSPs
SN_RCP_plot = SN_RCP_pct.copy(deep=True) # EOC signal-to-noise with percentiles
SN_SSP_plot = SN_SSP_pct.copy(deep=True) # EOC signal-to-noise with percentiles

pcts = SN_RCP_plot.percentile

# Trim to just include scenarios with RCP counterparts
N_SSP_cut = N_SSP_eoc.loc[dict(scenario=['ssp126','ssp245','ssp585'])]
S_SSP_cut = S_SSP_eoc.loc[dict(scenario=['ssp126','ssp245','ssp585'])]
SN_SSP_cut = SN_SSP_plot.loc[dict(scenario=['ssp126','ssp245','ssp585'])]

# In order to subtract RCPs from the SSPs, need to give them the same dimensions
N_RCP_eoc = N_RCP_eoc.assign_coords(scenario=['ssp126','ssp245','ssp585'])
S_RCP_eoc = S_RCP_eoc.assign_coords(scenario=['ssp126','ssp245','ssp585'])
SN_RCP_plot = SN_RCP_plot.assign_coords(scenario=['ssp126','ssp245','ssp585'])

# Calculate the differences
diffArr_N = N_SSP_cut - N_RCP_eoc
diffArr_S = S_SSP_cut - S_RCP_eoc
diffArr_SN = SN_SSP_cut - SN_RCP_plot
diffArr_SN_pc = (SN_SSP_cut - SN_RCP_plot)/SN_RCP_plot*100 

# A few checks
print('Max: ',np.max(diffArr_SN))
print('Min: ',np.min(diffArr_SN))
print('95th %ile: ',np.percentile(diffArr_SN,95))
print('5th %ile: ',np.percentile(diffArr_SN,5))

# Compute area-average changes and print out for use in the table
weights = np.cos(np.deg2rad(diffArr_N['lat']))
N_diff_globAvg = (diffArr_N*weights).sum('lat').mean('lon')/np.sum(weights)
print('Global average noise:')
print(N_diff_globAvg)
S_diff_globAvg = (diffArr_S*weights).sum('lat').mean('lon')/np.sum(weights)
print('Global average signal:')
print(S_diff_globAvg)
SN_diff_globAvg = (diffArr_SN.loc[dict(percentile=50)]*weights).sum('lat').mean('lon')/np.sum(weights)
print('Global average signal-to-noise:')
print(SN_diff_globAvg)
SN_pc_diff_globAvg = (diffArr_SN_pc.loc[dict(percentile=50)]*weights).sum('lat').mean('lon')/np.sum(weights)
print('Global average signal-to-noise (%):')
print(SN_pc_diff_globAvg)

# Labels for the columns of the plot
valLabs = ['$\Delta$ Noise', '$\Delta$ Signal', '$\Delta$ S/N', '$\Delta$ S/N (%)']
# Row labels
labs = ['ssp126-rcp26','ssp245-rcp45','ssp585-rcp85']
# Colourbar labels
cbLabs = ['K','K','$\sigma$','%']



def multi_plotter_diff(thisArray_N, thisArray_S, thisArray_SN, thisArray_SN_pc):
    
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
    
    title = 'Signal-to-noise at '+str(Yn1)+'-'+str(Yn2-1)+': Median SSP - corresponding RCP'
    
    proj = ccrs.Robinson(central_longitude=0)
    axes_class = (GeoAxes,dict(map_projection=proj))
    
    fig=plt.figure(figsize=(20,10))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(3, 4),
                    axes_pad=0.4,
                    cbar_location='bottom',
                    cbar_mode='edge',#'single',
                    cbar_pad=0.2,
                    cbar_size='8%',
                    label_mode='')  # note the empty label_mode
    
    for i, ax in enumerate(axgr):
        print(i)
        thisScen = thisArray_SN.scenario.values[int(np.floor(i/4))] # thisArray_SN.scenario.values[i]#
        thisPct = 50 # Only use the median plot
        
        # Select noise, signal, or S/N
        if np.mod(i,4)==0:
            theseData = thisArray_N.loc[dict(scenario=thisScen)]
            vmin_i = -0.15
            vmax_i = 0.15
            cmap = mpl.cm.PiYG_r
        elif np.mod(i,4)==1:
            theseData = thisArray_S.loc[dict(scenario=thisScen)]
            vmin_i = -1.5
            vmax_i = 1.5
            cmap = mpl.cm.RdBu_r
        elif np.mod(i,4)==2:
            theseData = thisArray_SN.loc[dict(scenario=thisScen, percentile=thisPct)]
            vmin_i = -1.5
            vmax_i = 1.5
            cmap = mpl.cm.RdBu_r
        elif np.mod(i,4)==3:
            theseData = thisArray_SN_pc.loc[dict(scenario=thisScen, percentile=thisPct)]
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
                            rasterized = True, # if False, every gridcell becomes a vector file. File workability is bad.
                            vmin=vmin_i,
                            vmax=vmax_i)        
        ax.coastlines()
        
        # Label the columns
        if i < 4 :
            ax.set_title(valLabs[i])
            axgr.cbar_axes[i].colorbar(cf) 
        
        # Label the rows
        if np.mod(i,4) == 0:
            ax.text(-0.04, 0.55, labs[int(np.floor(i/4))], va='bottom', ha='center',
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
    
    # Optional supertitle:
    #fig.suptitle(title, fontsize=20)
    
    #fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    #plt.show()
    
    return fig

gridPlot = multi_plotter_diff(diffArr_N, diffArr_S, diffArr_SN, diffArr_SN_pc)
gridPlot.savefig('./Plots/dea2022_N-S-SN_SSP-RCP_'+str(Yn1)+'-'+str(Yn2-1)+'.pdf', bbox_inches='tight', dpi=300)