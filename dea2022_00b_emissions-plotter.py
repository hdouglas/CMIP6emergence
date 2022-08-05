#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 10:00:20 2021

@author: 1001883
"""
# Douglas et al. 2022 optional script
# Produces Supplementary Figure S1

# Inputs: SSP & RCP country-level emission data
# Outputs: A nice plot

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# SSP emissions data available from https://tntcat.iiasa.ac.at/SspDb/dsd?Action=htmlpage&page=20
# Look for 'SSP_CMIP6_201811.csv.zip'

# RCP emissions data available from https://tntcat.iiasa.ac.at/RcpDb/dsd?Action=htmlpage&page=download
# Download 'R26_bulk.xls', 'R45_bulk.xls', and 'R85_bulk.xls'

homeDir = '/Volumes/GeoCCRI_01/users-data/douglahu/SSP_data/'
outDir = '/Volumes/GeoCCRI_01/users-data/douglahu/Scripts/FEA2017/Plots/'

os.chdir(homeDir+'SSP_CMIP6_201811/')

fileList = np.sort(os.listdir())

# Chose the emissions of interest
thisEm = 'CMIP6 Emissions|CO2' # 'CMIP6 Emissions|OC' # 'CMIP6 Emissions|Sulfur' # 'CMIP6 Emissions|BC' 
# 'CMIP6 Emissions|CH4' # 'CMIP6 Emissions|N2O' # 
# 'CMIP6 Emissions|CF4' # 'CMIP6 Emissions|CO' # 'CMIP6 Emissions|NH3' 
# 'CMIP6 Emissions|NOx' # 'CMIP6 Emissions|VOC'

emOptions = ['CMIP6 Emissions|OC', 'CMIP6 Emissions|Sulfur', 'CMIP6 Emissions|BC',
             'CMIP6 Emissions|CH4', 'CMIP6 Emissions|CO2', 'CMIP6 Emissions|N2O',
             'CMIP6 Emissions|CF4', 'CMIP6 Emissions|CO', 'CMIP6 Emissions|NH3',
             'CMIP6 Emissions|NOx', 'CMIP6 Emissions|VOC']
    
emData_co2 = xr.open_dataarray('countryEmissions_CMIP6 Emissions|CO2.nc')
emData_ch4 = xr.open_dataarray('countryEmissions_CMIP6 Emissions|CH4.nc')
emData_n2o = xr.open_dataarray('countryEmissions_CMIP6 Emissions|N2O.nc')

# Get the metadata for the whole set
csv = pd.read_csv('SSP_CMIP6_201811.csv')
emissions = csv.loc[csv['VARIABLE'] == thisEm] 
countries = np.sort(pd.unique(emissions['REGION']))
scenarios = np.sort(pd.unique(emissions['SCENARIO']))
years = ['2015','2020','2030','2040','2050','2060','2070','2080','2090','2100']
yearsnum = np.zeros(len(years))
for i in range(0,len(years)): 
    yearsnum[i] = int(years[i])
emissions_co2 = csv.loc[csv['VARIABLE'] == 'CMIP6 Emissions|CO2'] 
co2Unit = emissions_co2['UNIT'].iloc[0] 
emissions_ch4 = csv.loc[csv['VARIABLE'] == 'CMIP6 Emissions|CH4'] 
ch4Unit = emissions_ch4['UNIT'].iloc[0] 
emissions_n2o = csv.loc[csv['VARIABLE'] == 'CMIP6 Emissions|N2O'] 
n2oUnit = emissions_n2o['UNIT'].iloc[0]    



####-----Repeat for RCPs----#####

# Load the excel spreadsheets
rcp26data = pd.read_excel(homeDir+'R26_bulk.xls',header=0)
rcp45data = pd.read_excel(homeDir+'R45_bulk.xls',header=0)
rcp85data = pd.read_excel(homeDir+'R85_bulk.xls',header=0)

# Filter to just the global emissions
rcp26data = rcp26data[rcp26data.Region=='World']
rcp45data = rcp45data[rcp45data.Region=='World']
rcp85data = rcp85data[rcp85data.Region=='World']

# Drop the notes
rcp26data = rcp26data.drop(labels=['Notes'],axis=1)
rcp45data = rcp45data.drop(labels=['Notes'],axis=1)
rcp85data = rcp85data.drop(labels=['Notes'],axis=1)

# Save in xarrays
rcp_years = rcp26data.columns[4:]
rcp_yearsnum = np.zeros(len(rcp_years))
for i in range(0,len(rcp_years)): 
    rcp_yearsnum[i] = int(rcp_years[i])
rcp_tags = ['rcp26','rcp45','rcp85']

rcp_emData_co2 = xr.DataArray(data=0.0, dims=['year','scenario'], coords=[rcp_yearsnum,rcp_tags], name="CO2 Emissions")
rcp_emData_ch4 = xr.DataArray(data=0.0, dims=['year','scenario'], coords=[rcp_yearsnum,rcp_tags], name="CH4 Emissions")
rcp_emData_n2o = xr.DataArray(data=0.0, dims=['year','scenario'], coords=[rcp_yearsnum,rcp_tags], name="N2O Emissions")

rcp_emData_co2.loc[dict(scenario='rcp26')] = rcp26data.loc[rcp26data['Variable'] == 'CO2 emissions - Total'].iloc[:,4:].values[0]
rcp_emData_co2.loc[dict(scenario='rcp45')] = rcp45data.loc[rcp26data['Variable'] == 'CO2 emissions - Total'].iloc[:,4:].values[0]
rcp_emData_co2.loc[dict(scenario='rcp85')] = rcp85data.loc[rcp26data['Variable'] == 'CO2 emissions - Total'].iloc[:,4:].values[0]

rcp_emData_ch4.loc[dict(scenario='rcp26')] = rcp26data.loc[rcp26data['Variable'] == 'CH4 emissions -Total'].iloc[:,4:].values[0]
rcp_emData_ch4.loc[dict(scenario='rcp45')] = rcp45data.loc[rcp26data['Variable'] == 'CH4 emissions -Total'].iloc[:,4:].values[0]
rcp_emData_ch4.loc[dict(scenario='rcp85')] = rcp85data.loc[rcp26data['Variable'] == 'CH4 emissions -Total'].iloc[:,4:].values[0]

rcp_emData_n2o.loc[dict(scenario='rcp26')] = rcp26data.loc[rcp26data['Variable'] == 'N2O emissions - Total'].iloc[:,4:].values[0]
rcp_emData_n2o.loc[dict(scenario='rcp45')] = rcp45data.loc[rcp26data['Variable'] == 'N2O emissions - Total'].iloc[:,4:].values[0]
rcp_emData_n2o.loc[dict(scenario='rcp85')] = rcp85data.loc[rcp26data['Variable'] == 'N2O emissions - Total'].iloc[:,4:].values[0]

# Convert Pg C to Mt CO2
rcp_emData_co2 = rcp_emData_co2 * (44/12 * 1e3)

# 1 Tg CH4 = Mt CH4 
 
# Convert Tg N2O to kt N2O
rcp_emData_n2o = rcp_emData_n2o * 1e3
    


####----Plot global emissions----####

globEm_co2 = emData_co2.loc[dict(country='World')]
globEm_ch4 = emData_ch4.loc[dict(country='World')]
globEm_n2o = emData_n2o.loc[dict(country='World')]
#globEm.attrs['units'] = "billions"

fig = plt.figure(0, figsize=(9,14))
fig.tight_layout()

ax = fig.add_subplot(311)
ax.plot([1900,2100],[0,0],color='grey',linestyle='solid',linewidth=0.5)
ax.plot(yearsnum,globEm_co2.loc[dict(scenario=scenarios[8])],color='darkred')
ax.plot(rcp_yearsnum,rcp_emData_co2.loc[dict(scenario=rcp_tags[2])],color='darkred',linestyle='dashed')
ax.plot(yearsnum,globEm_co2.loc[dict(scenario=scenarios[3])],color='darkorange')
ax.plot(yearsnum,globEm_co2.loc[dict(scenario=scenarios[2])],color='limegreen')
ax.plot(rcp_yearsnum,rcp_emData_co2.loc[dict(scenario=rcp_tags[1])],color='limegreen',linestyle='dashed')
ax.plot(yearsnum,globEm_co2.loc[dict(scenario=scenarios[1])],color='deepskyblue')
ax.plot(rcp_yearsnum,rcp_emData_co2.loc[dict(scenario=rcp_tags[0])],color='deepskyblue',linestyle='dashed')
ax.plot(yearsnum,globEm_co2.loc[dict(scenario=scenarios[0])],color='mediumblue')
ax.axis([2000,2100,-20000,140000])
ax.legend(['_','ssp585','rcp85','ssp370','ssp245','rcp45','ssp126','rcp26','ssp119'],loc='upper left')
ax.set_ylabel(co2Unit)
ax.set_title('Projected Annual Emissions, CO$_2$')

ax = fig.add_subplot(312)
ax.plot([1900,2100],[0,0],color='grey',linestyle='solid',linewidth=0.5)
ax.plot(yearsnum,globEm_ch4.loc[dict(scenario=scenarios[8])],color='darkred')
ax.plot(rcp_yearsnum,rcp_emData_ch4.loc[dict(scenario=rcp_tags[2])],color='darkred',linestyle='dashed')
ax.plot(yearsnum,globEm_ch4.loc[dict(scenario=scenarios[3])],color='darkorange')
ax.plot(yearsnum,globEm_ch4.loc[dict(scenario=scenarios[2])],color='limegreen')
ax.plot(rcp_yearsnum,rcp_emData_ch4.loc[dict(scenario=rcp_tags[1])],color='limegreen',linestyle='dashed')
ax.plot(yearsnum,globEm_ch4.loc[dict(scenario=scenarios[1])],color='deepskyblue')
ax.plot(rcp_yearsnum,rcp_emData_ch4.loc[dict(scenario=rcp_tags[0])],color='deepskyblue',linestyle='dashed')
ax.plot(yearsnum,globEm_ch4.loc[dict(scenario=scenarios[0])],color='mediumblue')
ax.axis([2000,2100,0,1000])
#ax.legend(['_','ssp585','rcp85','ssp370','ssp245','rcp45','ssp126','rcp26','ssp119'],loc='upper left')
ax.set_ylabel(ch4Unit)
ax.set_title('CH$_4$')

ax = fig.add_subplot(313)
ax.plot([1900,2100],[0,0],color='grey',linestyle='solid',linewidth=0.5)
ax.plot(yearsnum,globEm_n2o.loc[dict(scenario=scenarios[8])],color='darkred')
ax.plot(rcp_yearsnum,rcp_emData_n2o.loc[dict(scenario=rcp_tags[2])],color='darkred',linestyle='dashed')
ax.plot(yearsnum,globEm_n2o.loc[dict(scenario=scenarios[3])],color='darkorange')
ax.plot(yearsnum,globEm_n2o.loc[dict(scenario=scenarios[2])],color='limegreen')
ax.plot(rcp_yearsnum,rcp_emData_n2o.loc[dict(scenario=rcp_tags[1])],color='limegreen',linestyle='dashed')
ax.plot(yearsnum,globEm_n2o.loc[dict(scenario=scenarios[1])],color='deepskyblue')
ax.plot(rcp_yearsnum,rcp_emData_n2o.loc[dict(scenario=rcp_tags[0])],color='deepskyblue',linestyle='dashed')
ax.plot(yearsnum,globEm_n2o.loc[dict(scenario=scenarios[0])],color='mediumblue')
ax.axis([2000,2100,0,25000])
#ax.legend(['_','ssp585','rcp85','ssp370','ssp245','rcp45','ssp126','rcp26','ssp119'],loc='upper left')
ax.set_ylabel(n2oUnit)
ax.set_title('N$_2$O')


fig.savefig(outDir+'worldEmissions_ssp-rcp_all.pdf', bbox_inches='tight')

