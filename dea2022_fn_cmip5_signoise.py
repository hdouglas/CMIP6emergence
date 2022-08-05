# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:12:01 2020

@author: ler45
"""

# Function for calculating signal and noise using a range of methods. The CMIP5
# datasets have a slightly different naming convention (no gridlabels).

def signoise_calc(model, RCP, framework, Ystart, Yend, rcpEns):

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
    #the cmip5 data.
    
    ###############################
    ####----USER-CHANGEABLE----####
    
    # Set the parameters of interest:
    variable_id = 'tas'
    table_id = 'Amon'
    
    scenarios = ['historical',RCP]
    
    # Set the baseline period for defining the signal
    Y0 = 1986 # Start year for baseline period (inclusive)
    Yn = 2006 # End year for baseline period (non-inclusive)
    
    # Set the period for comparison (average signal over what years?)
    Yn1 = Ystart #Yend-29 # 2071 # (inclusive)
    Yn2 = Yend+1 # 2101 # (non-inclusive)
    
    # Set the rolling average bin width (if used)
    rab = 20 
    
    # Set a path for where the netcdf files are stored
    DATAPATH = '/Volumes/GeoCCRI_01/users-data/douglahu/CMIP5_data/'+variable_id+'/'+table_id+'/'+model+'/'
    # Set a path for saving plots to inspect the smoothing algorithm fit
    plotOutPath = '/Volumes/GeoCCRI_01/users-data/douglahu/CMIP5_data/'+variable_id+'/'+table_id+'/02_InspectionPlots/'
    
    # Set whether or not to use the use_cftime option in xr.open_dataset
    # (this avoids issues if the dates go past 2262, but may slow performance. 
    usingCFtime = True
    
    ####----END USER-CHANGEABLE----####
    ###################################
    
    os.chdir(DATAPATH)
    
    ensembles = []
    fileList = []
    
    #print('Checkpoint1')
    
    for scenario in scenarios:
        
        if scenario == 'historical':
            # Get the ensemble member for the historical run 
            ensembleList = np.sort(os.listdir(DATAPATH+scenario))
            #print(ensembleList)
            if ensembleList[0] == '.DS_Store':
                ensembleList = ensembleList[1:]
            if len(ensembleList)  > 1:
                print('WARNING: Multiple ensembles exist for this instance. Only using first.')
            ensembles.append(ensembleList[0])
        else:
            ensembles.append(rcpEns) #Use the specified variant

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
            
    #print('Checkpoint2')
    
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
                
        #print('Checkpoint2.2')        
        
        try:
            data = allData.groupby('time.year').mean('time') 
        except:
            print('Sorting failed. Calendar mismatch. Engaging workaround.')
            realDates = []
            for j in range(len(allData.time)):
                thisDate = allData.time.values[j]
                realDates.append(cftime.datetime(thisDate.year,thisDate.month,thisDate.day))
            allData = allData.assign_coords(time=realDates)
            #print('Checkpoint 2.3')
            data = allData.groupby('time.year').mean('time') 
        
    #print(data.lat.shape)
    
    # Before fitting the polynomial, it's beneficial to truncate the data after
    # 2100. Inspecting some of the fits showed that having extended datasets really
    # skews the fit over important periods. 
    data = data.loc[dict(year=slice(1850,2100))]
    
    #Following the work of Frame et. al. 2017, we calculate a smooth representation 
    #of the temperature signal by fitting a fourth order polynomial to the global mean 
    #(area weighted) surface air temperature. The following code block determines the 
    #global mean surface air temperature (`GMST`). The mean temperature between 1985 
    #and 2006 is subtracted so that the signal now represents the temperature increase 
    #since that time period.
        
    weights = np.cos(np.deg2rad(data['lat']))
    
    data['tasAverage'] = (data[variable_id]*weights).sum('lat').mean('lon')/np.sum(weights)
    offsetMask = (data['year'] > Y0)*( data['year'] < Yn)
    
    GMST_poi = np.mean(data['tasAverage'][offsetMask])
    
    data['tasAverage'] = data['tasAverage'] - GMST_poi
    
    if framework == 'Hawkins2020':
        data['GMST'] = data['tasAverage']*0.0 + lowess(data['tasAverage'],data['year'],41.0/len(data['year']))[:,1]
        
    elif framework == 'Frame2017':
        polynomial = np.polyfit(data['year'],data['tasAverage'],4)
        data['GMST'] = data['tasAverage']*0.0 + np.polyval(polynomial,data['year'])
        
    elif framework == 'rollavg':
        # Rolling average for each gridpoint (not recommended)
        polynomial = np.polyfit(data['year'],data['tasAverage'],4)
        data['GMST'] = data['tasAverage']*0.0 + np.polyval(polynomial,data['year'])
        
    elif framework == 'globrollavg':
        # Gets a rolling average of GSMT and regresses local temps against this. 
        # Recommended over 'rollavg' approach.
        data['GMST'] = data.tasAverage.rolling(year=rab).mean().dropna('year')
    
    #print('Checkpoint3')    
    
    # The signal for each grid point is then found by performing a linear regression 
    #between the `GMST` and the air temperature in the grid point. We do this using 
    #a linear algebra least-squares method. This could be done iteratively using for 
    #loops, but it is much faster using linear algebra.
    
    A = np.zeros((2,2))
    b = np.zeros(2)
    
    A[0,0] = np.sum(data['GMST']**2)
    A[0,1] = np.sum(data['GMST'])
    A[1,0] = A[0,1]
    A[1,1] = len(data['GMST'])
    
    Ainv = np.linalg.inv(A)
    
    a = (data[variable_id]*data['GMST']).sum('year')
    b = (data[variable_id]).sum('year')
    
    p0 = Ainv[0,0]*a + Ainv[0,1]*b
    p1 = Ainv[1,0]*a + Ainv[1,1]*b
    
    if framework == 'rollavg':
        signal = data.tas.rolling(year=rab).mean().dropna('year')
        #Cut off the earlier data 
        data = data.loc[dict(year=slice(1850+rab-1,Yend))]
        offsetMask = offsetMask.loc[dict(year=slice(1850+rab-1,Yend))]
        
    else:
        signal = p0*data['GMST'] + p1
    
    #print('Checkpoint4')
    
    #----Optional plotting code to inspect the polynomial fit----#
    # Some extraneous code to plot the unadjusted GMST alongside the signal

    testlatlon = [[-41.290599, 174.768083],[0,0],[1.341838, 103.963735],
                  [-34.791384, -58.702692],[-77.835155, 166.691438],
                  [40.748760, -73.977961],[46.868368, 104.922882],
                  [78.887013, 16.476489]]
    testlabels = ['Wellington','0,0','Singapore','Buenos Aires','Antarctica',
                  'New York, NY','Mongolia','Svalbard']
    
    test0 = np.zeros(p0.shape) + 1
    test1 = xr.DataArray(p0)
    test1.values = test0
    test2 = test1*data['GMST'] + GMST_poi  # p1 # #use GMST_poi if looking at actual GSMT, not anomaly translated to same baseline
    
    fig2 = plt.figure(figsize=(12,18))
    ax = fig2.add_subplot(5,2,1)
    ax.plot(data['year'],data['tasAverage'], 'k.')
    ax.plot(data['year'],data['GMST'])
    plt.ylabel('Temperature Change Since '+str(Y0)+'-'+str(Yn)+' (K)')
    plt.title(model+': GMST and fitted trend, '+RCP)
    plt.legend(['GMST data','GMST trend'])
    for i in range(len(testlatlon)):
        ax = fig2.add_subplot(5,2,i+3)
        ax.plot(data['year'],data[variable_id].sel(lat=testlatlon[i][0],lon=testlatlon[i][1],method='nearest'),'y.')
        ax.plot(data['year'],signal.sel(lat=testlatlon[i][0],lon=testlatlon[i][1],method='nearest'),'b-')
        ax.plot(data['year'],test2.sel(lat=testlatlon[i][0],lon=testlatlon[i][1],method='nearest'),'r-')
        plt.title(testlabels[i])
        plt.ylabel('Surface Air Temperature (K)')
        plt.legend(['data','signal','GMST'])
    #plt.show()

    if framework == 'rollavg':
        fig2.savefig(plotOutPath+'GMST_rollavg_inspection_'+RCP+'_'+model+'.png',dpi=150)
    elif framework == 'globrollavg':
        fig2.savefig(plotOutPath+'GMST_globrollavg_inspection_'+RCP+'_'+model+'.png',dpi=150)
    else:
        fig2.savefig(plotOutPath+'GMST_inspection_'+RCP+'_'+model+'.png',dpi=150)
    #----End optional plotting code----#
    
    #print('Checkpoint5')
    
    # The noise is estimated as the root mean square error (RMSE) of the smoothed signal compared to the data.
    
    if framework=='Hawkins2020':
        data['signal'] = signal.transpose('year','lat','lon')*1.0
        data['noise'] = np.sqrt((((data[variable_id] - data['signal'])**2).rolling(year=41,center=True).mean('year')))
        data['noise'] = data['noise'].ffill('year')
        data['noise'] = data['noise'].bfill('year')
        data['signal'] -= np.mean((data['signal'][offsetMask]),axis=0)
        data['signaltonoise'] = data['signal']/data['noise'] 
        
    else: #if framework=='Frame2017':
        
        #print('Checkpoint6')
        # From Frame et al. 2017: "The N term is the standard deviation of annual 
        # mean temperatures in the pre-industrial control simulations at each grid point."
        
        # Get the ensemble list
        ensembleList = np.sort(os.listdir(DATAPATH+'piControl/'))
        #print(ensembleList)
        if ensembleList[0] == '.DS_Store':
            ensembleList = ensembleList[1:]
        if len(ensembleList)  > 1:
            print('WARNING: Multiple ensembles exist for this instance. Only using first.')
        ensembles.append(ensembleList[0])
        # Pick the ensemble relating to piControl
        ensemble = ensembles[-1] 
        
        #print('Checkpoint7')
        
        fileList = []
        files = [f for f in os.listdir(DATAPATH+'piControl/'+ensemble) if ensemble in f]
        for f in files:
            fileList.append(DATAPATH+'piControl/'+ensemble+'/'+f)
    
        fileList.sort()
        #print(fileList)
        
        #print('Checkpoint7.1')
        
        #The simulation output we are looking at contain monthly means of the air temperature 
        #at set pressure levels. We are only interested in the annual mean surface air 
        #temperature, so we use some `xarray` functions to only load the data we are interested in.
        
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
                
            #print('Checkpoint7.1b')        
            
            try:
                piCdata = allData.groupby('time.year').mean('time') 
            except:
                print('Sorting failed. Calendar mismatch. Engaging workaround.')
                realDates = []
                for j in range(len(allData.time)):
                    thisDate = allData.time.values[j]
                    realDates.append(cftime.datetime(thisDate.year,thisDate.month,thisDate.day))
                allData = allData.assign_coords(time=realDates)
                #print('Checkpoint 2.3')
                piCdata = allData.groupby('time.year').mean('time') 
        
        # Only use the last 200 years of the piControl data, to allow for ramp-up
        piCend = np.max(piCdata.year.values)
        #print('Checkpoint7.2')
        #print(piCend)
        
        data['noise'] = piCdata.loc[dict(year=slice(piCend-199,piCend))][variable_id].std('year')
                    
        #------------------------------------#  
        
        #print('Checkpoint8')
        
        data['signal'] = signal.transpose('year','lat','lon')*1.0
        data['signal'] -= (data['signal'][offsetMask]).mean('year')
        
        # Uncomment if you wish to use RMSE as the noise metric:
        #data['noise0'] = np.sqrt((((data[variable_id] - data['signal'])**2)[offsetMask]).mean('year')) # Dylan's original code
    
        data['signaltonoise'] = data['signal']/data['noise'] 
        #data['signaltonoise0'] = data['signal']/data['noise0'] 
        
        #print('Checkpoint9')
        
        # Calculate average values for the Period Of Interest
        offsetMask2 = (data['year'] > Yn1)*( data['year'] < Yn2)
        data['POIsignal'] = data['signal'][offsetMask2].mean('year')
        data['POIsignaltonoise'] = data['signaltonoise'][offsetMask2].mean('year')
        #data['POIsignaltonoise0'] = data['signaltonoise0'][offsetMask2].mean('year')

    return data
