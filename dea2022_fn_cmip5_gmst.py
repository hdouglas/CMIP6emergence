# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:12:01 2020

@author: ler45
"""

# Function for calculating global average temperature (not smoothed), relative to 
# the pre-industrial control period.
# This version for CMIP5 data

def gmst_calc(model, RCP, Yend, rcpEns):

    import numpy as np  
    from matplotlib import pyplot as plt
    import os
    import xarray as xr   
    from statsmodels.nonparametric.smoothers_lowess import lowess
    import cftime
    
    #This block of code creates a list that contains all of the files we need to analyse 
    #a single simulation run. `historical` simulations cover the time period 1850-2015 
    #and scenario (e.g. `ssp245`) simulations cover the time period 2015-2100. So 
    #a single simulation from 1850-2100 needs both the `historical` and the predictive 
    #`ssp245` to get full temporal coverage. 
    #
    #The `DATAPATH` is the path to the directory on your computer that contains 
    #the cmip6 data.
    
    ###############################
    ####----USER-CHANGEABLE----####
    
    # Set the parameters of interest:
    variable_id = 'tas'
    table_id = 'Amon'
    
    scenarios = ['historical',RCP]
    
    DATAPATH = '/Volumes/GeoCCRI_01/users-data/douglahu/CMIP5_data/'+variable_id+'/'+table_id+'/'+model+'/'
    
    # Set whether or not to use the use_cftime option in xr.open_dataset
    # (this avoids issues if the dates go past 2262, but might slow performance. Not sure.
    usingCFtime = True
    
    ####----END USER-CHANGEABLE----####
    ###################################
    
    os.chdir(DATAPATH)
    
    ensembles = []
    fileList = []
    
    print('Checkpoint1')
    
    for scenario in scenarios:
        
        if scenario == 'historical':
            # Get the ensemble member for the historical run 
            ensembleList = np.sort(os.listdir(DATAPATH+scenario))
            print(ensembleList)
            if ensembleList[0] == '.DS_Store':
                ensembleList = ensembleList[1:]
            if len(ensembleList)  > 1:
                print('WARNING: Multiple ensembles exist for this instance. Only using first.')
            ensembles.append(ensembleList[0])
        else:
            ensembles.append(rcpEns) #Use the specified variant
            # This assumes that ssp585 has all the same variants as ssp534-over
            
        # Pick the ensemble relating to this scenario (this can vary between scenarios)
        ensemble = ensembles[scenarios.index(scenario)]
        
        # Create a file list of all raw data to collate
        try:
            files = [f for f in os.listdir(DATAPATH+scenario+'/'+ensemble) if ensemble in f]
        except:
            print('Error in reading specified filename. Might be misalignment between ssp585 and ssp534-over.')    
        for f in files:
            fileList.append(DATAPATH+scenario+'/'+ensemble+'/'+f)
    
    fileList.sort()
    #print(fileList)
     
    #Uncomment lines below if data are across multiple pressure levels.
    
    def preprocess(ds):
            #data = ds.isel(plev=slice(None,7))
            #data = data.bfill('plev')
            #data = data.isel(plev=0)
            
            data = ds.groupby('time.year').mean('time').load()
            try:
                test=len(data.lat)
            except:
                print('Renaming latitude and longitude')
                data=data.rename({'latitude':'lat','longitude':'lon'})
    
            return data
    
    print('Checkpoint2')
    
    # Caution: some of the historical and SSP files overlap in time, so open_mfdataset fails
    # So, I set up a less efficient, but still functional alternative
    try:
        with xr.open_mfdataset(fileList, preprocess=preprocess, parallel=True, use_cftime=usingCFtime) as file:
            data = file
        # EC-Earth3 has a single digit off in the 13th decimal place for two 
        # of the latitude values. This ruins the combined dataset. Need to 
        # check this and throw an error to prompt the exception code to run.
        print('Checking lengths')
        init_file = fileList[0]
        initData = xr.open_dataset(init_file, use_cftime=True)
        try:
            test=len(initData.lat)
        except:
            print('Renaming latitude and longitude')
            initData=initData.rename({'latitude':'lat','longitude':'lon'})
        if len(data.lat) != len(initData.lat):
            fileList.index('boguscheck') # Intentially throw an error
        # Also throw an error if there are any repeated years or gaps in the data
        if int(max(data.year).values)-int(min(data.year).values)+1 != len(data.year):
            print('Something is off with the years. Applying workaround.')
            fileList.index('boguscheck') # Intentially throw an error
    except:
        print('xr.open_mfdataset() failed. Applying workaround.')
        init_file = fileList[0]
        
        # Initiate the xarray that will have all the data
        allData = xr.open_dataset(init_file, use_cftime=True)
        try:
            test=len(allData.lat)
        except:
            print('Renaming latitude and longitude')
            allData=allData.rename({'latitude':'lat','longitude':'lon'})
        realLats = allData.lat
    
        for i in fileList:
            # Skip over the initial file
            if i != init_file: 
                print(i)
                # Open and process this file
                dset = xr.open_dataset(i, use_cftime=True)
                try:
                    test=len(dset.lat)
                except:
                    print('Renaming latitude and longitude')
                    dset=dset.rename({'latitude':'lat','longitude':'lon'})
                # Check for and correct latitude mismatch
                if np.sum(abs(dset.lat.values-realLats.values)) != 0.0:
                    print('Latitude mismatch = ',str(np.sum(abs(dset.lat.values-realLats.values))))
                    dset = dset.assign_coords(lat=realLats)
                    
                                        
                maxTime = np.max(allData.time.values)
                minTime = np.min(dset.time.values)
                
                if maxTime > minTime:
                    print('Overlap between historical & simulation. Truncating sim data.')
                    if maxTime.month == 12:
                        startRCP = str(maxTime.year+1)+'-01'
                    else:
                        startRCP = str(maxTime.year)+'-'+str(maxTime.month+1)
                    endRCP = str(np.max(dset.time.values).year)+'-'+str(np.max(dset.time.values).month)
                    dset = dset.loc[dict(time=slice(startRCP, endRCP))]
                    
                # Concatenate along time axis
                allData = xr.concat((allData, dset), dim="time")  
                
        
        print('Checkpoint2.2')        
        
        try:
            data = allData.groupby('time.year').mean('time') 
        except:
            print('Sorting failed. Calendar mismatch. Engaging workaround.')
            realDates = []
            for j in range(len(allData.time)):
                thisDate = allData.time.values[j]
                realDates.append(cftime.datetime(thisDate.year,thisDate.month,thisDate.day))
            allData = allData.assign_coords(time=realDates)
            print('Checkpoint 2.3')
            data = allData.groupby('time.year').mean('time') 
        
    print(data.lat.shape)
    
    # Before fitting the polynomial, it's beneficial to truncate the data after
    # 2100. Inspecting some of the fits showed that having extended datasets really
    # skews the fit over important periods. 
    data = data.loc[dict(year=slice(1850,Yend))]
    
    # calculate tasAverage (note that GMST has been used to refer to the smoothed
    # curve in other scripts.)
        
    weights = np.cos(np.deg2rad(data['lat']))
    
    data['tasAverage'] = (data[variable_id]*weights).sum('lat').mean('lon')/np.sum(weights)

    # Now load the piControl data to calculate the baseline to subtract away, 
    # leaving tasAverage as degrees K above pre-industrial.
    
    # Get the ensemble and grid label
    ensembleList = np.sort(os.listdir(DATAPATH+'piControl/'))
    print(ensembleList)
    if ensembleList[0] == '.DS_Store':
        ensembleList = ensembleList[1:]
    if len(ensembleList)  > 1:
        print('WARNING: Multiple ensembles exist for this instance. Only using first.')
    ensembles.append(ensembleList[0])
    # Pick the ensemble relating to piControl
    ensemble = ensembles[-1] 
    
    print('Checkpoint3')
    
    fileList = []
    files = [f for f in os.listdir(DATAPATH+'piControl/'+ensemble) if ensemble in f]
    for f in files:
        fileList.append(DATAPATH+'piControl/'+ensemble+'/'+f)

    fileList.sort()
    #print(fileList)
    
    print('Checkpoint3.1')
    
    # HadGEM2-ES has this annoying thing where each file ends in November and 
    # the next one starts in December, so the pre-processing doesn't work.
    try:
        with xr.open_mfdataset(fileList, preprocess=preprocess, parallel=True, use_cftime=usingCFtime) as file:
            piCdata = file
    except:
        print('xr.open_mfdataset() failed. Applying workaround.')
        init_file = fileList[0]
        
        # Initiate the xarray that will have all the data
        allData = xr.open_dataset(init_file, use_cftime=True)
        try:
            test=len(allData.lat)
        except:
            print('Renaming latitude and longitude')
            allData=allData.rename({'latitude':'lat','longitude':'lon'})
        realLats = allData.lat
    
        for i in fileList:
            # Skip over the initial file
            if i != init_file: 
                print(i)
                # Open and process this file
                dset = xr.open_dataset(i, use_cftime=True)
                try:
                    test=len(dset.lat)
                except:
                    print('Renaming latitude and longitude')
                    dset=dset.rename({'latitude':'lat','longitude':'lon'})
                # Check for and correct latitude mismatch
                if np.sum(abs(dset.lat.values-realLats.values)) != 0.0:
                    print('Latitude mismatch = ',str(np.sum(abs(dset.lat.values-realLats.values))))
                    dset = dset.assign_coords(lat=realLats)
                    
                # Concatenate along time axis
                allData = xr.concat((allData, dset), dim="time")  
            
        print('Checkpoint3.1b')        
        
        try:
            piCdata = allData.groupby('time.year').mean('time') 
        except:
            print('Sorting failed. Calendar mismatch. Engaging workaround.')
            realDates = []
            for j in range(len(allData.time)):
                thisDate = allData.time.values[j]
                realDates.append(cftime.datetime(thisDate.year,thisDate.month,thisDate.day))
            allData = allData.assign_coords(time=realDates)
            print('Checkpoint 3.1c')
            piCdata = allData.groupby('time.year').mean('time') 
    
    # Only use the last 200 years of the piControl data, to allow for ramp-up
    piCend = np.max(piCdata.year.values)
    print('Checkpoint3.2')
    piCdata = piCdata.loc[dict(year=slice(piCend-199,piCend))]
    print('Checkpoint3.3')
    # Area-average gmst over the piControl period:
    weights = np.cos(np.deg2rad(piCdata['lat']))
    print('Checkpoint3.4')
    piCgmstAvg = (piCdata*weights).sum('lat').mean('lon')/np.sum(weights)
    print('Checkpoint3.5')
    piCgmstAvg = piCgmstAvg[variable_id].mean('year') 
    print(piCgmstAvg)
    print('Checkpoint3.6')
    
    # Subtract this from tasAverage to get GMST relative to piC baseline
    
    # But MIROC5 has a weird issue where the piC data are way too cold. 
    # In that case only, I'll calculate GMST relative to 1850-1900 avg temps.
    if model=='MIROC5':
        histGmstAvg = data['tasAverage'].loc[dict(year=slice(1850,1900))].mean('year')
        data['tasAverageRel'] = data['tasAverage'] - histGmstAvg
    else:
        data['tasAverageRel'] = data['tasAverage'] - piCgmstAvg
    
    return data
