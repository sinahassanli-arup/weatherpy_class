import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
# from ..analysis.processing import _generate_windspeed_ranges, _generate_label, spdLocal_to_10tc2
from ..analysis import processing 
from ..analysis.query import query_range 

NUMBER_OF_WINDDIRS = 16
DEFAULT_WIND_SPEED_RANGES = [0, 2, 4, 6, 8, 10, 15]
TEMPERATURE_RANGE = list(np.r_[0:51])
DEFAULT_TEMP_RANGES = list(np.r_[0:51])
WIND_RANGE = list(np.r_[0:12:2, 15])
WIND_ANGLES = list(range(10, 361, 10)) # [10, 20, 30, ..., 340, 350, 360]

DATA_DESCRIPTION = {
    'WindDirection': ('Wind Direction', '°', 'Wind Direction (°)'),
    'WindSpeed': ('Mean Wind Speed', 'm/s', 'Wind Speed (m/s)'),
    'WindGust': ('Wind Gust', 'm/s', 'Wind Gust (m/s)'),
    'SeaLevelPressure': ('Sea Level Pressure', 'hPa', 'Sea Level Pressure (hPa)'),
    'DryBulbTemperature': ('Dry Bulb Temperature', '°C', 'Dry Bulb Temperature (°C)'),
    'WetBulbTemperature': ('Wet Bulb Temperature', '°C', 'Wet Bulb Temperature (°C)'),
    'DewPointTemperature': ('Dew Point Temperature', '°C', 'Dew Point Temperature (°C)'),
    'RelativeHumidity': ('Relative Humidity', '%', 'Relative Humidity (%)'),
    'Rain': ('Rain', 'mm', 'Rain (mm)'),
    'RainIntensity': ('Rain Intensity', 'mm/min', 'Rain Intensity (mm/min)'),
    'RainCumulative': ('Rain Cumulative', 'mm', 'Rain Cumulative (mm)'),
    'CloudHeight': ('Cloud Height', 'm', 'Cloud Height (m)'),
    'Visibility': ('Visibility (m)'),
    'WindType': ('Wind Type', '', 'Wind Type'),
    'CloudOktas': ('Cloud Oktas', '', 'Cloud Oktas'),
    'MW1_0': ('MW1_0', '', 'MW1_0'),
    'MW1_1': ('MW1_1', '', 'MW1_1'),
    'AJ1_0': ('AJ1_0', '', 'AJ1_0'),
    'ReportType': ('Report Type', '', 'Report Type'),
    'QCName': ('QC Name', '', 'QC Name'),
    'QCWindSpeed': ('QC Wind Speed', '', 'QC Wind Speed'),
    'QCWindDir': ('QC Wind Dir', '', 'QC Wind Dir')}

# %% CPDF LOCAL

def cpdf_local(cpdf10tc2_station, spdMaxRatio, Mzcat_ref_over_10tc2, numDirs, wind_spd_ranges):
    """
    Calculate cumpulative pdf for a single point

    Parameters
    ----------
    cpdf10tc2_station : pandas.dataframe
        Cumulative pdf for the weather station.
    
    spdMaxRatio : numpy.array [1D array]
        maximum of mean and gem wind speed ratio for a single point
        
    Mzcat_ref_over_10tc2 : int or array
        Mzcat of reference to 10m, TC2. Length should corresponds to number of wind directions.
        If integer is used, the same value will be used for all directions.
    
    numDirs : int
        Number of tested wind directions. Should be the same as number of columns in cpdf10tc2_station
    
    wind_spd_ranges : list
        list of local wind speeds to consider. 
        It is best to usually set this to cover all wind speeds (e.g. list(np.r_[0:45]))

    Returns
    -------
    cpdfLocal : pandas.dataframe
        Cumpulative pdf for a single point.
            rows: wind speeds
            columns: wind directions
    """
    
    # wind speed at 10m TC2 per direction which causes measured local wind speed in wind tunnel.
    spd10tc2 = processing.spdLocal_to_10tc2(spdMaxRatio, Mzcat_ref_over_10tc2, numDirs, wind_spd_ranges)

    cpdfLocal = pd.DataFrame(index=spd10tc2.index, columns=spd10tc2.columns)
    
    cpdfLocal.index.name = 'local windSpd'
    
    for s in range(len(wind_spd_ranges)):
        for d in range(numDirs):
            cpdfLocal.iloc[s,d] = np.interp(spd10tc2.iloc[s,d], wind_spd_ranges, cpdf10tc2_station.iloc[:,d])
            
    return  cpdfLocal  

# %% CPDF LOCAL ALL

def cpdf_local_all(configNames, cpdf10tc2_station, spdMaxRatio_all, Mzcat_ref_over_10tc2, numDirs, wind_spd_ranges):
    """
    Calculate cumpulative pdf for all points and configurations

    Parameters
    ----------
    configNames : list of str
        List of configuration names which corresponds to xlsx sheet_names.
    
    cpdf10tc2_station : pandas.dataframe
        Cumulative pdf for the weather station.
    
    spdMaxRatio_all : numpy.array [1D array]
        maximum of mean and gem wind speed ratio for a single point
        
    Mzcat_ref_over_10tc2 : int or array
        Mzcat of reference to 10m, TC2. Length should corresponds to number of wind directions.
        If integer is used, the same value will be used for all directions.
    
    numDirs : int
        Number of tested wind directions. Should be the same as number of columns in cpdf10tc2_station
    
    wind_spd_ranges : list
        list of local wind speeds to considers. 
        It is best to usually set this to cover all wind speeds (e.g. list(np.r_[0:45]))
        

    Returns
    -------
    cpdf10tc2MeasuredAll : numpy.ndarray [4D array]
        Cumpulative pdf for all testing configurations points 
            1-D: testing configurations
            2-D: testing points
            3-D: wind speeds
            4-D: wind directions
    """
    print('\tCalculating local CPDF for all points')
    
    cpdfLocalALL = np.zeros((len(configNames), spdMaxRatio_all.shape[1] , len(wind_spd_ranges), numDirs));
    
    cpdf_df = pd.DataFrame()
    
    for p in range(spdMaxRatio_all.shape[1]):
        for c in range(spdMaxRatio_all.shape[0]):
            spdMaxRatio = spdMaxRatio_all[c,p,:]
            cpdfLocalALL[c,p,:,:] = cpdf_local(cpdf10tc2_station, spdMaxRatio, Mzcat_ref_over_10tc2, numDirs, wind_spd_ranges)
    

    return  cpdfLocalALL

# %% Legacy cpdf

def cpdf_all(cpdf, spdMaxRatio, Mzcat_inlet2ref, dodgeFactor, wind_spd_ranges=DEFAULT_WIND_SPEED_RANGES):
    """
    Calculate cumulative probability distribution, cpdf, fucntion for all points and all directions 
    Return normalized max(mean, GEM) and where which one is greater.

    Parameters
    ----------
    cpdf : pandas.DataFrame
        Cumulative probability distribution function.
    spdMaxRatio : 3d numpy array (configuration x number of points x wind directions)
        Maximum of normalized mean and GEM (gust equivalent mean) wind speed ratios.
    Mzcat_inlet2ref : Dict
        modelling (wind tunnel or CFD) parameters.
    dodgeFactor : Dict
        input parameters.
    wind_spd_ranges : list, optional
        Wind speed intervals. Default is: [0, 2, 4, 6, 8, 10, 15]

    Returns
    -------
    cpdfInterp : TYPE
        DESCRIPTION.

    """

    numConfig = spdMaxRatio.shape[0]
    
    # check if limited or all points are selected for processing 
    # if modellingParams['numPts']==-1:
    numPts = spdMaxRatio.shape[1]
    # else:
    #     numPts = modellingParams['numPts']
    
    numDirs = spdMaxRatio.shape[2]
    
    SF = dodgeFactor
    
    # try:
    CF = Mzcat_inlet2ref # conversion factor: V_inlet/V_10,TC2
    # except:
    #     TC_local = modellingParams['TerrainHeight']['TC_local']
    #     height_local = modellingParams['TerrainHeight']['height_local']
    #     TC_ref = modellingParams['TerrainHeight']['TC_reference']
    #     height_ref = modellingParams['TerrainHeight']['height_reference']
    #     conversionFactor = TerrainHeightMultiplier(TC_local, height_local, TC_ref, height_ref) # conversion factor: V_inlet/V_10,TC2  It could be a number or a list of correction factors for all directions   
    #     CF=conversionFactor*np.ones(numDirs) # need uodate. work in progress. should allow for 16 Conversion Factor
    print('\nCorrection factors (Mzca reference to local) are {}\n'.format(CF))
    
    # try:
    #     wind_spd_ranges = inputParams['customRange']
    # except:
    #     wind_spd_ranges = list(np.r_[0:46])
    
    
    Spd10mTC2=np.zeros((numConfig , numPts , numDirs ,len(wind_spd_ranges))); # configs X Pts X dirs X wind_spd_ranges
    cpdfAll=np.zeros((numConfig , numPts , numDirs ,len(wind_spd_ranges)));
    
    
    for c in range(numConfig): # testing configuration
       for p in range(numPts): # testing points
           for s in range(len(wind_spd_ranges)): # local wind speed 
                for d in range(numDirs): # tetsing wind directions
                    '''
                    Evaluate wind speed at TC2,10m, V_2,10m, based on local wind, V_3,1.5m
                    V_2,10m = V_3,1.5m/(V_3,1.5m/V_3,10m)/(V_3,10/V_2,10)
                    first division, (V_3,1.5m/V_3,10m) , is from CFD
                    second division, CF = (V_3,10/V_2,10), from Standard
                    '''
                    
                    
                    Spd10mTC2[c,p,d,s] = wind_spd_ranges[s]/(SF*spdMaxRatio[c,p,d]*CF[d])
                    
                    cpdfAll[c,p,d,s] = np.interp(Spd10mTC2[c,p,d,s], wind_spd_ranges, cpdf.iloc[0:,d])
                    
                    
    
    return cpdfAll

# %% CPDF PARALLEL

def cpdf_all_parallel(cpdf, spdMaxRatio, Mzcat_inlet2ref, dodgeFactor, wind_spd_ranges=DEFAULT_WIND_SPEED_RANGES):
    """
    Calculate cumulative probability distribution, cpdf, fucntion for all points and all directions 
    Return normalized max(mean, GEM) and where which one is greater.

    Parameters
    ----------
    cpdf : pandas.DataFrame
        Cumulative probability distribution function.
    spdMaxRatio : 3d numpy array (configuration x number of points x wind directions)
        Maximum of normalized mean and GEM (gust equivalent mean) wind speed ratios.
    Mzcat_inlet2ref : Dict
        modelling (wind tunnel or CFD) parameters.
    dodgeFactor : Dict
        input parameters.
    wind_spd_ranges : list, optional
        Wind speed intervals. Default is: [0, 2, 4, 6, 8, 10, 15]

    Returns
    -------
    cpdfInterp : TYPE
        DESCRIPTION.

    """

    numConfig = spdMaxRatio.shape[0]
    
    # check if limited or all points are selected for processing 
    # if modellingParams['numPts']==-1:
    numPts = spdMaxRatio.shape[1]
    # else:
    #     numPts = modellingParams['numPts']
    
    numDirs = spdMaxRatio.shape[2]
    
    SF = dodgeFactor
    
    # try:
    CF = Mzcat_inlet2ref # conversion factor: V_inlet/V_10,TC2
    # except:
    #     TC_local = modellingParams['TerrainHeight']['TC_local']
    #     height_local = modellingParams['TerrainHeight']['height_local']
    #     TC_ref = modellingParams['TerrainHeight']['TC_reference']
    #     height_ref = modellingParams['TerrainHeight']['height_reference']
    #     conversionFactor = TerrainHeightMultiplier(TC_local, height_local, TC_ref, height_ref) # conversion factor: V_inlet/V_10,TC2  It could be a number or a list of correction factors for all directions   
    #     CF=conversionFactor*np.ones(numDirs) # need uodate. work in progress. should allow for 16 Conversion Factor
    print('\nCorrection factors (Mzca reference to local) are {}\n'.format(CF))
    
    # try:
    #     wind_spd_ranges = inputParams['customRange']
    # except:
    #     wind_spd_ranges = list(np.r_[0:46])
    
    def TC2(c,p,s,d, wind_spd_ranges,SF,spdMaxRatio,CF):
        
        Spd10mTC2[c,p,d,s] = wind_spd_ranges[s]/(SF*spdMaxRatio[c,p,d]*CF[d])
        cpdfAll[c,p,d,s] = np.interp(Spd10mTC2[c,p,d,s], wind_spd_ranges, cpdf.iloc[0:,d])
        
        return cpdfAll[c,p,d,s]
    
    Spd10mTC2=np.zeros((numConfig , numPts , numDirs ,len(wind_spd_ranges))); # configs X Pts X dirs X wind_spd_ranges
    cpdfAll=np.zeros((numConfig , numPts , numDirs ,len(wind_spd_ranges)));
    
    
    for c in range(numConfig): # testing configuration
       for p in range(numPts): # testing points
           for s in range(len(wind_spd_ranges)): # local wind speed 
                for d in range(numDirs): # tetsing wind directions
                    '''
                    Evaluate wind speed at TC2,10m, V_2,10m, based on local wind, V_3,1.5m
                    V_2,10m = V_3,1.5m/(V_3,1.5m/V_3,10m)/(V_3,10/V_2,10)
                    first division, (V_3,1.5m/V_3,10m) , is from CFD
                    second division, CF = (V_3,10/V_2,10), from Standard
                    '''
                    
                    # c = 1, p = 25903, s = 45, d = 16
                    
                    # p = multiprocessing.Pool(processes=6)
                    cpdfAll.map(TC2(c, p, s, d, wind_spd_ranges, SF, spdMaxRatio, CF), p)
                    
                    
                    
    
    return cpdfAll

# %% OPTIMIZED CPDF ALL
# from numba import jit
# @jit
def cpdf_local_optimization(cpdf10tc2_station,spdMaxRatio,Mzcat_ref_over_10tc2,configNames,numDirs,wind_spd_ranges):
    
    import warnings

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    
    cpdf10tc2_station_index = cpdf10tc2_station.reset_index()
    cpdf10tc2_station_index = cpdf10tc2_station_index.drop('index',axis=1)

    # cpdf_local_all = pd.DataFrame(columns=cpdf10tc2_station_index.index)
    
    cpdf_all = np.zeros((len(configNames),spdMaxRatio.shape[1],len(wind_spd_ranges),numDirs))
    
    # pbar = tqdm(total=len(spdMaxRatio.shape[1]))
    
    for c in range(len(configNames)):
        for p in range(spdMaxRatio.shape[1]):
            
            # pts = spdMaxRatio.shape[1]
            # p_checkpts = [1000,round(pts/4),round(pts/2),round(pts*3/4),(pts*0.9)]
            
            # if p in p_checkpts:
            #     print(f'Progressed through {p} of {pts} points')
            
            # Update progress bar
            # pbar.update(len(p))
            
            spdMaxRatio_ = spdMaxRatio[c,p,:]
            
            spd_local_over_10tc2 = spdMaxRatio_*Mzcat_ref_over_10tc2
            
            windDirs = np.linspace(0, 360, numDirs+1)[0:-1]
            spd10tc2 = pd.DataFrame(index=wind_spd_ranges, columns=windDirs)
            spd10tc2.index.name = 'local windSpd'
            for s in range(len(wind_spd_ranges)):
                spd10tc2.iloc[s,:] =  wind_spd_ranges[s]/spd_local_over_10tc2
        
            cpdfLocal = pd.DataFrame(index=spd10tc2.index, columns=spd10tc2.columns)
            
            cpdfLocal.index.name = 'local windSpd'
            
            for d in range(numDirs):
                ind = d*22.5
                cpdfLocal[ind] = np.interp(spd10tc2[ind], 
                                            wind_spd_ranges, 
                                            cpdf10tc2_station_index[ind].values)
            
            cpdfLocal = cpdfLocal.to_numpy()
            
            cpdf_all[c,p,:,:] = cpdfLocal
            
    return cpdf_all
    
# %% COMPLETENESS

def completeness(data, columns='all', site_description='', timeframe='daily', save=False, save_dirpath=None, use_pcolormesh=True):
    """
    Plots seaborn heatmap of data completeness based on timestamps / per hour.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to be resampled and analyzed.
        
    columns : string or list, optional
        Columns to be selected for the completeness map. Default is 'all'.
    
    site_description : string
        Title block for the plot.
        
    timeframe : string
        Default is 'daily'. Accepts either 'daily' or 'monthly'.
    
    save : bool, optional
        Indicates if the produced chart should be saved. Default is False.
    
    save_dirpath : str, optional
        Save location if save is True.

    use_pcolormesh : bool, optional
        Use pcolormesh for continuous plot. Default is False.

    Returns
    -------
    df_cross_ph : pandas.DataFrame
        DataFrame showing the completeness of the data.
    """
    
    # Selecting columns
    if columns == 'all':
        df = data.copy()
    elif isinstance(columns, list):
        df = data[columns].copy()
    else:
        raise ValueError('Choose columns for completeness map as a list of columns or use string "all" to select all')
    
    # Adding year, day of year, and month to the DataFrame
    df['Year'] = df.index.year
    df['DOY'] = df.index.dayofyear
    df['Month'] = df.index.month
    
    year_start = df['Year'].min()
    year_end = df['Year'].max()
    
    # Creating crosstab for daily or monthly timeframe
    if timeframe == 'daily':
        df_cross = pd.crosstab(df.Year, df.DOY)
        hours = 24
    else:
        df_cross = pd.crosstab(df.Year, df.Month)
        hours_per_month = {1: 744, 2: 672, 3: 744, 4: 720, 5: 744, 6: 720, 7: 744, 8: 744, 9: 720, 10: 744, 11: 720, 12: 744}
        leap_years = [year for year in range(year_start, year_end + 1) if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)]
        for year in leap_years:
            hours_per_month[2] = 696
        df_cross = df_cross.divide([hours_per_month[m] for m in df_cross.columns], axis=1)
        hours = 1
    
    # Reversing the DataFrame and filling missing values
    df_cross = df_cross[::-1].fillna(0) / hours
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot using pcolormesh if use_pcolormesh is True
    if use_pcolormesh:
        c = ax.pcolormesh(np.arange(df_cross.shape[1] + 1), np.arange(df_cross.shape[0] + 1), df_cross.values, cmap='inferno', shading='auto')
        fig.colorbar(c, ax=ax)
        if timeframe == 'daily':
            ax.set_xticks(np.arange(0, 366, 30) + 0.5)
            ax.set_xticklabels(np.arange(0, 366, 30))
        else:
            ax.set_xticks(np.arange(1, 13) + 0.5)
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_yticks(np.arange(len(df_cross.index)) + 0.5)
        ax.set_yticklabels(df_cross.index)
        ax.invert_yaxis()  # Invert y-axis to have minimum year at bottom and maximum year at top
    else:
        # Plot using imshow if use_pcolormesh is False
        df_cross_ph = df_cross.copy()
        ct_arr = np.array(df_cross_ph).flatten()
        dfmax = ct_arr.max()
        dfmin = ct_arr.min()
        all_ticks = np.array([0, 0.3, 1, 2, 6, 60])
        select_ticks = all_ticks[all_ticks > dfmax]
        colours = plt.cm.inferno
        ct_arr = ct_arr[ct_arr != 0]
        ticks = [dfmin]
        for i in all_ticks:
            if dfmin <= i <= dfmax:
                ticks.append(i)
        ticks.extend([dfmax, select_ticks[0]])
        list_ticks = [[0, 0], [0, 0.3], [0.3, 1], [1, 2], [2, 6], [6, 60], [60, np.inf]]
        colours_all = sns.mpl_palette('inferno', len(list_ticks))
        colour_dict = {colour: bound for colour, bound in zip(colours_all, list_ticks)}
        norm = matplotlib.colors.BoundaryNorm(ticks[:-1], colours.N, extend='both')
        for colour, bound in colour_dict.items():
            cut_df = df_cross_ph[(df_cross_ph >= bound[0]) & (df_cross_ph < bound[1])]
            try:
                if not cut_df.isnull().all().all():
                    ax.imshow(cut_df, cmap=colours, norm=norm, interpolation='none', aspect=len(df_cross_ph.columns) / len(df_cross_ph.index))
            except KeyError:
                pass
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colours), ax=ax)
    
    # Set title and axis labels
    ax.set_title(f'Completeness recordings/hour\n{site_description}')
    
    if timeframe == 'daily':
        ax.set_xlabel('Day of Year')
        ax.set_ylabel('Year')
        ax.set_xticks(np.arange(0, 366, 30))
        ax.set_xticklabels(np.arange(0, 366, 30))
    else:
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        ax.set_xticks(np.arange(0.5, 12.5, 1))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Centering y-axis labels
    y_labels = df_cross.index
    ax.set_yticks(np.arange(len(y_labels)) + 0.5)
    ax.set_yticklabels(y_labels)
    
    # Save the figure if save is True
    if save and save_dirpath:
        import os
        site_dir = os.path.join(save_dirpath, site_description)
        if not os.path.exists(site_dir):
            os.makedirs(site_dir)
        filename = f'completeness_{site_description}_{year_start}-{year_end}.png'
        fig.savefig(os.path.join(site_dir, filename), bbox_inches='tight')
    
    return df_cross


# %% ANNUAL MEAN CALL

def plot_annual_mean(
        input_data,
        site_description,
        column = 'WindSpeed',
        figsize=(14,6),
        whis=[5,95],
        show_min=False,
        show_max=False,
        save=False,
        save_dirpath=None
        ):
    
    '''
    Plots annual box plots, number of recordings, maximum, and mean values of column data.
    
    Parameters
    ----------
    input_data : pandas dataframe
        Data to be resampled and analysed
        
    site_description : string
        Title block
    
    column : string
        Default 'WindSpeed'. 
        Determines what column of dataframe is used for plotting.
        Must be a named column of dataframe containing numerical values.
        
    figsize : tuple of ints
        Default is (14,6)
        Controls the size of the plot.
    
    maxes : bool
        Default True
        Controls whether maximum values (not all outliers) are plotted above 
        box and whisker plots
        
    save : bool
        Default False
        Controls whether the image is saved.
    
    save_filepath : str
        Default None
        Controls where the image is saved.
        
    Returns
    ----------
    data_desc : pandas dataframe
        Plotted data
    
    '''
    
    data = input_data.copy()
    if column == 'Rain':
        data = data[data['Rain']>0]
        
    data_year_dict = {}
    data_year_desc = {}
    data_year = data[column].resample('YE')
    for i in data_year:
        data_annual = i[1]
        desc = data_annual.describe()
        data_year_dict.update({i[0].year:data_annual})
        data_year_desc.update({i[0].year:desc})
        
    data_desc = pd.DataFrame(data_year_desc)
    fig, ax = plt.subplots(1,1,figsize=(figsize))
    ax2 = ax.twinx()
    ax.boxplot(data_year_dict.values(),0,whis=whis,showfliers=False,showmeans=False,positions=data_desc.columns)

    # calculate calms
    if column=='WindSpeed':
        data_year_all = data.resample('YE')
        calmPercent = []
        for i in data_year_all:
            if i[1].empty:  # Check if the year has no data
                calmPercent.append('N/A')  # Append 'N/A' or a placeholder for years with no data
            else:
                mask = (i[1]['WindSpeed'] == 0) & (i[1]['WindDirection'] == 0)
                calmPercent.append(round(sum(mask) / len(i[1]['WindSpeed']) * 100, 2))
     
    # Update x-axis labels with the calm percentages
    ax.set_xticklabels([f'{i}\n{j}%' if j != 'N/A' else f'{i}\nN/A' for i, j in zip(data_desc.columns, calmPercent)])
    ax.set_xlabel('Year (calm %)')

    ax.plot(data_desc.columns,data_desc.loc['mean'],'-x',label='Annual Mean')
    if show_max:
        ax.plot(data_desc.columns,data_desc.loc['max'],'^',c='r',label='Annual Maximum')
    if show_min:
        ax.plot(data_desc.columns,data_desc.loc['min'],'v',c='b',label='Annual Minimum')
    ax2.plot(data_desc.columns,data_desc.loc['count'],'--',c='g',label='Recordings in year')
    
    arr_pts = np.array([whis[0],25,50,75,whis[1]])
    ax3 = fig.add_axes([0.7,0.925,0.125,0.06])
    ax3.boxplot(arr_pts,widths=0.6,vert=False)
    ax3.set_yticks([])
    arr_ticks=[]
    for i in arr_pts:
        arr_ticks.append(f'{i}%')
    ax3.set_xticks(arr_pts)
    ax3.set_xticklabels(arr_ticks)
    
    y_ax_label = DATA_DESCRIPTION[column][0] + ' ({})'.format(DATA_DESCRIPTION[column][1])

    ax.set_ylabel(y_ax_label)
    # ax.set_xlabel('Year')
    ax.tick_params(axis='x',labelrotation = 45)
    ax2.set_ylabel('Recordings in year')
    fig.suptitle(f'{site_description}')
    fig.legend(fontsize='small')
    
    if save and save_dirpath:

        site_dir = os.path.join(save_dirpath,site_description)
        if not os.path.exists(site_dir):
            os.makedirs(site_dir)
        write_name = f'annual_mean_plot_{site_description}'
        fig.savefig(os.path.join(site_dir, write_name+'.png'),bbox_inches='tight',dpi=300)
        data_desc = data_desc.T
        data_desc.to_csv(os.path.join(site_dir, write_name+'.csv'))
    
    return data_desc

# %% PERCENTILE SUMMARY

def percentile_summary(
        data,
        column='WindSpeed',
        percentiles = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99, 1],
        ):
    
    """
    
    Generates a statistical overview of a selected column, and returns timestamps exceeding a chosen value.
    
    Parameters
    ----------
    
    data : dataframe
        Cleaned BOM weather data

    column : str, optional
        String of which column of the summary should be applied. Default is ''.

    percentiles : list of floats
        Percentiles to include in statistic summary of column within input data
        Default is [0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99, 1]

    Returns
    ----------
    
    summary_df : dataframe
        Simple dataframe showing number of occurences over each percentile value 
        of the selected column.
    
    dates_of_interest : dataframe
        A dataframe of timestamps with a column value exceeding the threshold value.
    
    """
        
    df = data[f'{column}']
    
    occurences={}
    quantiles={}
    for p in percentiles:
        k = '{:.1%}'.format(p)
        thres = data[f'{column}'].quantile(p)
        if  p <= 0.5:
            v = len(data[data[f'{column}'] <= thres])
        else:
            v = len(data[data[f'{column}'] >= thres])
        occurences.update({f'{k}':v})
        quantiles.update({f'{k}': thres})
    
    summary = pd.DataFrame(data={ f'{column} Values': quantiles.values(), 'Occurences, #' : occurences.values()}, index = occurences.keys())
    summary['Occurences, %'] = summary['Occurences, #']/len(data)*100
     
    return summary


def data_quantile(data, quantile = 0.99, column = 'WindSpeed'):
      
    thres = data[f'{column}'].quantile(quantile)
    
    if  quantile <= 0.5:
        dates_of_interest = data[data[f'{column}'] <= thres]
    else:
        dates_of_interest = data[data[f'{column}'] >= thres]

    return thres, dates_of_interest

    
# %% PLOT INTERVAL STATISTICS

def _plot_interval_statistics(data,
                             site_description,
                             dataType='BOM',
                             interval='monthly',
                             obType='WindSpeed',
                             save=False,
                             save_dirpath=''):
    """
    ! WORK IN PROGRESS !
    calculates and plots data averaged each day, month, or year for a given data type

    Parameters
    ----------
    data : pandas.dataframe
        Imported and unified weather data dataframe.
    site_description : str
        Site description used for legend and file name. Typically in this
        format: <StationNumber or SiteName> (<Station ID>).
    datatype : str, optional
        The data source of he data, either 'BOM', or 'NOAA'. Default is 'BOM'
    interval : str, optional
        The period of time to be considered for finding the mean/max. The options
        are "yearly", "monthly", "daily", or "hourly". Default is "monthly".
    obType : str, optional
        Desired data type to be averaged. can be any of the 15 data types in a
        unified dataframe: WindDirection, WindSpeed, WindGust,
        SeaLevelPressure, DryBulbTemperature, WetBulbTemperature,
        DewPointTemperature, RelativeHumidity, Rain, RainIntensity,
        RainCumulative, CloudHight, Visibility, WindType, CloudOktas
    save : bool, optional
        True or False for whether the user wishes to save the data. Default is False.
    save_dirpath : str, optional
        String with the address of the disired save location. The default is "".

    Returns
    -------
    df : pandas.DataFrame
        The dataframe with the statistical analyses is output

    """
    firstYear = data.index.year.min()
    lastYear = data.index.year.max()
    range_dict = {'YEARLY':('Y','year'),'MONTHLY':('M','month'),'DAILY':('D','day'),'HOURLY':('H','hour')}
    dataType_dict  = {'WindDirection': 'Wind Direction', 'WindSpeed': 'Wind Speed (m/s)', 'WindGust':'Wind Gust (m/s)',
    'SeaLevelPressure':'Sea Level Pressure (hPa)', 'DryBulbTemperature':'Dry Bulb Temperature (°C)',
    'WetBulbTemperature':'Wet Bulb Temperature (°C)','DewPointTemperature':'Dew Point Temperature (°C)',
    'RelativeHumidity':'Relative Humidity (%)', 'Rain':'Rain (mm)', 'RainIntensity':'Rain Intensity',
    'RainCumulative':'Rain Cumulative (mm)', 'CloudHight':'Cloud Hight', 'Visibility':'Visibility',
    'WindType':'Wind Type', 'CloudOktas':'Cloud Oktas'}
    
    data_noRecording = data[obType].resample(range_dict[interval.upper()][0]).count()
    
    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_

    df = pd.DataFrame([])
    percents = ('min', percentile(25), percentile(50), percentile(75), 'max','mean')
    col_names = ('Minimum', '25th Percentile', '50th Percentile', '75th Percentile', 'Maximum', 'Mean')
    
    for i, n in zip(percents, col_names):
        stats = processing.interval_converter(data,timeframe = interval,aggregate_function = i)
        df[n] = stats[obType]

    fig, ax1 = plt.subplots()
    fig.set_figheight(5)
    fig.set_figwidth(10)
    ax1.grid()
    ax2 = ax1.twinx()
    ax1.plot(df)
    ax2.plot(data_noRecording, 'r:')
    ax1.set_xlabel(f'\nYears: {firstYear} - {lastYear}')
    ax1.set_ylabel(dataType_dict[obType])
    ax2.set_ylabel(f'No. of observation per {range_dict[interval.upper()][1]}')
    fig.suptitle(f'\n{site_description}')
    
    lgnd_lst = list(df.columns)
    lgnd_lst.append('Measurement count')
    
    fig.legend(lgnd_lst,  bbox_to_anchor=(0.3, 1.25))

    if save == True:
        save_path = os.path.join(save_dirpath,f'{site_description}_{obType}_{interval.capitalize()}_statistics')
        df.to_csv(save_path + '.csv')
        fig.savefig(save_path + '.png', bbox_inches='tight')
        
    return df


# %% PDF TEMPERATURE PLOT

def pdf_temperature(data, site_description, tempType='DB', wind_speed_ranges=WIND_RANGE, temp_range=TEMPERATURE_RANGE, orientation='horizontal', save=False, save_dirpath=None):
    """
    Plot probability distribution functionof temperature for 8 cardinal wind direction.

    Parameters
    ----------
    data : pandas.dataframe
        Historic weather data in unified format with correct labling      
    site_description : str
        Site description used for legend and fille name. Typically in this format: <StationNumber or SiteName> (<Station ID>)
    tempType : str, optional
        DB (Dry-bulb) or WB (Wet-bulb), default is DB
    wind_speed_ranges : array, optional
        list of wind speeds for plotting. If not specified the following default value will be used:
        e.g. [0, 2, 4, 6, 8, 10, 15]
    temp_range : array, optional
        list of temperature intervals. Default is [0,1,2,...,50]
    orientation : str, optional
        'vertical', or 'horizontal'. The default is horizontal.
    save : boolean, optional
        Check to save the windrose. The default is False.
    save_dirpath : str, optional
        Path to directory to save the figure. The default is None.

    Returns
    -------
    None.
    
    """    

    firstYear = data.index.year.min()
    lastYear = data.index.year.max()

    numDirs = 8  # pdf for 8 cardinal wind directions
    WIND_DIRECTIONS = ['North', 'North-East', 'East',
                       'South-East', 'South', 'South-West', 'West', 'North-West']

    if orientation=='vertical':
        fig, axs = plt.subplots(int(numDirs/2), 2, sharey=True,
                            tight_layout=True, figsize=(15, 24))
        table_x_loc = 0.85
        percent_y_loc = 0.90
        
    elif orientation=='horizontal':
        fig, axs = plt.subplots(2, int(numDirs/2), sharey=True,
                    tight_layout=True, figsize=(24, 15))
        
        table_x_loc = 0.80
        percent_y_loc = 0.95
        
    axs = axs.flatten()  # making axes from 2d to 1d array so can loop though
    
    # Calculating pdf, cpdf and temperature exceedances for DB or WB
    if tempType=='DB':
        temp_col = 'DryBulbTemperature'
        # temperature PDF for all wind directions
        Tpdf, Tcpdf, Texceed = temperature_statistics(
            data["WindDirection"], data[temp_col], site_description, numDirs=numDirs, temp_ranges=temp_range)
    elif tempType=='WB':
        temp_col = 'WetBulbTemperature'
        # temperature PDF for all wind directions
        Tpdf, Tcpdf, Texceed = temperature_statistics(
            data["WindDirection"], data[temp_col], site_description, numDirs=numDirs, temp_ranges=temp_range)
    else:
        print('tempType should be set as DB or WB')
        
    # Finding the range of temperature
    nonzeroBool = Tpdf.sum(axis=1).ne(0)
    Tmin_lbl = nonzeroBool.idxmax()
    Tmax_lbl = nonzeroBool.iloc[::-1].idxmax()
    Tmin = nonzeroBool.index.get_loc(Tmin_lbl)
    Tmax = nonzeroBool.index.get_loc(Tmax_lbl)
    
    # finding maximum probability to set for y axis
    probMax = round((Tpdf.max()/Tpdf.sum()).max()*1.2,2)
    
    def generate_label(spd_range):

        lower_bound, upper_bound = spd_range
        if np.isinf(upper_bound):
            return ">"+str(lower_bound)+" m/s"
        else:
            return ">"+str(lower_bound)+"-"+str(upper_bound)+" m/s"

    def generate_windspeed_ranges(spd_range):
        return list(zip(spd_range, spd_range[1:]+[np.inf]))

    def wind_range(spd):
        for s in range(len(wind_speed_ranges)-1):
            if spd < wind_speed_ranges[s+1]:
                return wind_range_label[s]
                break
        if spd > wind_speed_ranges[6]:
            return wind_range_label[6]

    wind_range_label = [generate_label(
        spd_range) for spd_range in generate_windspeed_ranges(wind_speed_ranges)]

    for d in range(numDirs):
        
        data_filtered=pd.DataFrame()
        dir_interval = 360/numDirs
        
        # filter the appropriate start and end wind directions
        if d == 0:
            windDirection_range = [360-dir_interval/2, dir_interval/2]
            # print(windDirection_range)
            data_filtered, _, _ = query_range(data,{'WindDirection':windDirection_range})
        else:
            windDirection_range = [
                dir_interval/2+(d-1)*dir_interval, dir_interval/2+(d)*dir_interval]
            # print(windDirection_range)
            data_filtered, _, _ = query_range(data,{'WindDirection':windDirection_range})

        # percentage of occurance of the specified wind direction
        ofTotal = 0
        for dd in Tcpdf.columns.values:
            if d == 0:
                # TODO:check below if need windDirection_range_0 or windDirection_range
                if (int(dd) >= windDirection_range[0]) | (int(dd) < windDirection_range[1]):
                    ofTotal = ofTotal+Tcpdf.loc['>0 C', dd]
            else:
                if (int(dd) >= windDirection_range[0]) & (int(dd) < windDirection_range[1]):
                    ofTotal = ofTotal+Tcpdf.loc['>0 C', dd]

    
        data_filtered.insert(0,'wind range label',data_filtered["WindSpeed"].apply(wind_range).values)

        lst = []
        for i in range(numDirs-1):
            x = data_filtered[data_filtered['wind range label']
                              == wind_range_label[i]]
            lst.append(x[temp_col].values)

        def percentile_t(p):
            return data_filtered[temp_col].quantile(p)

        # Plotting
        # change math.ceil to np.ceil
        noBins = np.ceil(percentile_t(1)-percentile_t(0))*2
        n, bins, patches = axs[d].hist(lst, bins=int(
            noBins), stacked=True, density=True, label=wind_range_label, color=plt.cm.jet(np.linspace(0, 1, 7)))

        # on plot test of temperature for different percentile
        txt = 'Temperature       \nStatistics:\n\n' + \
            r'$T_{min}$ = '+str(round(percentile_t(0), 1))+'$^{\\circ}$C\n' + \
            r'$T_{max}$ = '+str(round(percentile_t(1), 1))+'$^{\\circ}$C\n' + \
            r'$T_{mean}$ = '+str(round(percentile_t(0.50), 1))+'$^{\\circ}$C\n' + \
            r'$T_{99}$ = '+str(round(percentile_t(0.01), 1))+'$^{\\circ}$C\n' + \
            r'$T_{95}$ = '+str(round(percentile_t(0.05), 1))+'$^{\\circ}$C\n' + \
            r'$T_{10}$ = '+str(round(percentile_t(0.9), 1))+'$^{\\circ}$C\n' + \
            r'$T_{5}$ = '+str(round(percentile_t(0.95), 1))+'$^{\\circ}$C\n' + \
            r'$T_{2}$ = '+str(round(percentile_t(0.98), 1))+'$^{\\circ}$C\n' + \
            r'$T_{1}$ = '+str(round(percentile_t(0.99), 1))+'$^{\\circ}$C\n' + \
            r'$T_{0.4}$ = '+str(round(percentile_t(0.996), 1))+'$^{\\circ}$C'

        axs[d].legend(wind_range_label, fontsize=8)


        axs[d].text(table_x_loc, 0.3, s=txt, fontsize=8, transform=axs[d].transAxes, bbox=dict(
            facecolor='none', edgecolor='black'))
        axs[d].text(0.05, percent_y_loc, s=str(round(ofTotal*100, 1))+'% of total', fontsize=10,
                    transform=axs[d].transAxes, bbox=dict(facecolor='none', edgecolor='black'))
        
        axs[d].set_xlabel('{} Temperature'.format(tempType)+', $^{\\circ}$C')
        axs[d].set_ylabel('Probability distribution, %')
        axs[d].set_title('winds from '+WIND_DIRECTIONS[d])
        axs[d].set_xlim((Tmin, Tmax))
        axs[d].set_ylim((0, probMax))

        data_filtered = []

    print('\nPDF for {} temperature is plotted'.format(tempType))

    if save and save_dirpath:

        site_dir = os.path.join(save_dirpath,site_description)
        if not os.path.exists(site_dir):
            os.makedirs(site_dir)
        fileName = f"pdf_{tempType}temperature_{numDirs}Dirs_{site_description}_{firstYear}-{lastYear}.png"
        fig.savefig(os.path.join(site_dir,fileName), dpi=150)

    return Tpdf, Tcpdf, Texceed

# %% WINDSPEED STATISTICS

def windspeed_pdf(windDirs, windSpds, wind_spd_ranges=DEFAULT_WIND_SPEED_RANGES, flag_nomapping=False):
    """
    PDF of wind speeds for a given direction and wind speed band.

    Parameters
    ----------------------------------
    windDirs : numpy.array or pandas.Series
        Vector of wind directions (in degrees, 0 is calm, 360 is north)
    windSpds : numpy.array or pandas.Series
        Vector of wind speeds (in m/s)
    wind_spd_ranges : list, optional
        List of wind speeds that bound ranges.Default: [0, 2, 4, 6, 8, 10, 15]
    
    Returns
    ----------------------------------
    pdf : pandas.Dataframe
    """
    windDir_diff = WIND_ANGLES[1] - WIND_ANGLES[0]

    def _count(windDirsection, wind_speed_range):
        lower_bound, upper_bound = wind_speed_range
        mask = (windDirs==direction) & (lower_bound<windSpds) & (windSpds<=upper_bound)
        return np.sum(mask)

    # modified counting to only bin within the wind directions specified 
    # (rather than 10 degree buckets)
    def _count_nomapping(windDirsection, wind_speed_range):
        lower_bound, upper_bound = wind_speed_range
        if direction  == 360:
            upper_bin = 0 + windDir_diff/2
            lower_bin = 360 - windDir_diff/2
            mask = ((windDirs>=lower_bin) & (lower_bound<windSpds) & (windSpds<=upper_bound) | \
                    (windDirs<upper_bin) & (lower_bound<windSpds) & (windSpds<=upper_bound))
        else:
            upper_bin = direction+windDir_diff/2
            lower_bin = direction-windDir_diff/2
            mask = (windDirs>=lower_bin) & (windDirs<upper_bin) & \
                (lower_bound<windSpds) & (windSpds<=upper_bound)
        
        return np.sum(mask)
            
    def _probability(windDirsection, wind_speed_range, flag_nomapping):
        if flag_nomapping:
            return _count_nomapping(windDirsection, wind_speed_range)/windSpds.size            
        else:
            return _count(windDirsection, wind_speed_range)/windSpds.size

    wind_speed_ranges = processing._generate_windspeed_ranges(wind_spd_ranges) # creating ranges (2-4 m/s, etc) for the final pdf dataframe

    wind_speed_pdf = pd.DataFrame()

    for direction in WIND_ANGLES: # if wind angles are not rounded to near 10 degree it does nto work
        wind_speed_pdf[direction] = pd.Series(
            [_probability(direction, spd_range, flag_nomapping) for spd_range in wind_speed_ranges],
            index = [processing._generate_label(spd_range) for spd_range in wind_speed_ranges])
        
    return wind_speed_pdf

def _sector_windDirections(no_sectors):
    return np.linspace(360/no_sectors, 360, no_sectors, endpoint=True)

def _map_to_nsectors(N):
    """Function to produce a map to reduce number of wind sectors from 36.

    Parameters
    -----------------
    N : int
        Number of sectors to reduce to

    Returns
    ----------------------------------
    defaultdict(<class 'list'>,
        Map providing the fraction of frequencies to be distributed from
        each of the 36 wind directions.

        Example output for 16 wind directions:

            {22.5: [[0.4444444444444444, 10],
                    [0.8888888888888888, 20],
                    [0.6666666666666667, 30],
                    [0.2222222222222222, 40]],
             45.0: [[0.3333333333333333, 30],
                    [0.7777777777777778, 40],
                    [0.7777777777777778, 50],
                    [0.33333333333333337, 60]],
             67.5: [[0.2222222222222222, 50],
                    [0.6666666666666666, 60],
                    .
                    .
                    .
    """
    wind_angles_reduced = _sector_windDirections(N)

    map_to_sector = defaultdict(list)

    for wind_angle in WIND_ANGLES:
        upper_index = np.searchsorted(wind_angles_reduced, wind_angle)
        upper_wind_angle = wind_angles_reduced[upper_index]
        if upper_index == 0:
            lower_wind_angle = WIND_ANGLES[-1]
            fraction_to_upper = wind_angle/upper_wind_angle
        else:
            lower_wind_angle = wind_angles_reduced[upper_index - 1]
            fraction_to_upper = (wind_angle - lower_wind_angle)/(upper_wind_angle - lower_wind_angle)
        map_to_sector[upper_wind_angle].append([fraction_to_upper, wind_angle])
        map_to_sector[lower_wind_angle].append([1-fraction_to_upper, wind_angle])
    
    return map_to_sector

# Function to map degree to one of the 16 cardinal directions
def map_wind_direction(deg, num_bins):
    """
    Maps a wind direction in degrees to the numeric midpoint of its bin,
    where the circle is divided into num_bins equal parts.
    """
    bin_size = 360 / num_bins       # Each bin's angular width.
    half_bin = bin_size / 2         # Half the bin size for centering.
    # Shift the degree by half a bin and wrap around 360°
    shifted_deg = (deg + half_bin) % 360
    bin_index = int(shifted_deg / bin_size)
    # The midpoint is the bin index times the bin size.
    return bin_index * bin_size

def speed_exceedance_total(dfs_dict, pecentile):
    
    total_size = sum(v.shape[0] for i,v in dfs_dict.items())
    # Build a dictionary where keys are wind direction labels and values are arrays
    # of wind speed thresholds for each percentile.
    spdExceed_dict = {}
    for key, value in dfs_dict.items():
        # Calculate the relative frequency (probability) of this wind direction
        dir_prob = value.shape[0] / total_size
        thresholds = []
        for p in pecentile:
            q = 1 - p / dir_prob  # Global quantile level for this direction and percentile p
            # Only compute if q is within [0,1]; otherwise assign NaN
            if 0 <= q <= 1:
                thresholds.append(value['WindSpeed'].quantile(q))
            else:
                thresholds.append(np.nan)
        spdExceed_dict[key] = thresholds
    
    # Create a DataFrame: index is the percentile labels and columns are wind directions.
    spdExceed_df = pd.DataFrame(spdExceed_dict, index=[f'{i*100:g}%' for i in pecentile])
    
    return spdExceed_df

def speed_exceedance_directional(dfs_dict, pecentile):
    
    qile = [1-i for i in pecentile]

    spdExceed_dir = pd.DataFrame()
    
    for key,value in dfs_dict.items():
        spdExceed_dir[key] = value['WindSpeed'].quantile(qile).values
    
    spdExceed_dir.index = [f'{i*100:g}%' for i in pecentile]
    return spdExceed_dir
        
def windspeed_statistics(windDirs, windSpds, numDirs=NUMBER_OF_WINDDIRS, wind_spd_ranges=DEFAULT_WIND_SPEED_RANGES, flag_mapping=True, exceed_from_pdf=False):
    """
    Produce a wind speed pdf for specified number of wind directions.
    
    Parameters
    ----------------------------------
    windDirs : numpy.array or pandas.Series
        Vector of wind directions (in degrees, 0 is calm, 360 is north)
    windSpds : numpy.array or pandas.Series
        Vector of wind speeds (in m/s)
    numDirs : int, optional (default 16)
        Number of sectors of the wind rose
    wind_spd_ranges : list of int, optional (default [0,2,4,6,8,10,15])
        List of wind speeds that bound wind speed ranges
    
    Returns
    ----------------------------------
    pdf : pandas.DataFrame
    cpdf : pandas.DataFrame
    spdExceed_perDirection : pandas.DataFrame
    spdExceed_total : pandas.DataFrame
    """
    
    wind_spd_ranges = list(wind_spd_ranges)
    
    if flag_mapping == False:
        # choice to remove the smoothing of wind rose. Does NOT bin in 10 degree
        # buckets and does NOT apply any weighing factor to any of the bins
        global WIND_ANGLES
        WIND_ANGLES = np.arange(360/numDirs,360+360/numDirs,360/numDirs)
        map_to_sector = {direction: [[1, direction]] for direction in WIND_ANGLES}        
    else:
        map_to_sector = _map_to_nsectors(numDirs)
        
    pdfAll = windspeed_pdf(windDirs, windSpds, wind_spd_ranges)
    _windspeed_statistics = pd.DataFrame()
    for sector_windDirsection, probability_share in map_to_sector.items():
        _windspeed_statistics[sector_windDirsection] = sum(fraction*pdfAll[wd] for fraction, wd in probability_share)

    # sort columns before returning
    wind_angles_reduced = list(_sector_windDirections(numDirs)) # order of coloums

    pdf=_windspeed_statistics[wind_angles_reduced]
    
    # bring 360.0 Column to the first column and rename to 0.0
    _zeroDir=pdf[360.0]; pdf=pdf.drop(labels=[360.0], axis=1); pdf.insert(0, 0.0, _zeroDir)

    # cumulative pdf
    _cpdf=(pdf.iloc[::-1]).cumsum();
    cpdf=_cpdf.iloc[::-1];
    cpdf_index=[">"+str (i)+" m/s" for i in wind_spd_ranges];
    cpdf.set_index([cpdf_index],inplace=True, drop=True)
    
    
    probExceed=[0.000114,0.00022,0.001,0.01,0.015,0.05,0.1,0.2,0.5,0.8,0.95,0.99]
    probExceedLabel=['0.0114%','0.022%','0.1%','1%','1.5%','5%','10%','20%','50%','80%','95%','99%']
    
    # calculating exceedaces from wind directly
    # calculating from probabilities is depreciated.

    if exceed_from_pdf:
        # probability of exceedance

        spdExceed_perDirection=pd.DataFrame(0, index=probExceedLabel, columns=pdf.columns)
        for i in range(len(probExceed)):
            for j in range(len(cpdf.columns)):
                d=probExceed[i]*cpdf.iloc[0,j] # finding the probability exceedance for the speciifc direction
                spdExceed_perDirection.iloc[i,j]=np.interp(d,cpdf.iloc[:,j],wind_spd_ranges,period=100) #fidning the corresponding speed
                
        spdExceed_total=pd.DataFrame(0, index=probExceedLabel, columns=pdf.columns)

        for i in range(len(probExceed)):
            for j in range(len(cpdf.columns)):
                d=probExceed[i]
                spdExceed_total.iloc[i,j]=np.interp(d,cpdf.iloc[:,j],wind_spd_ranges,period=100) #fidning the corresponding speed
    else:

        df = pd.DataFrame(data={'WindSpeed': windSpds, 'WindDirection': windDirs})
        
        # removing calms:
        df = df[(df['WindSpeed']!=0)&(df['WindDirection']!=0)]
        
        # Apply the label
        df['windDirLabel'] = df['WindDirection'].apply(lambda d: map_wind_direction(d, numDirs))

        dfs_dict = dict(tuple(df.groupby('windDirLabel')))

        probExceed_total = np.r_[0.0001, 0.001, np.arange(0.01, 0.21, 0.01)]
        spdExceed_total = speed_exceedance_total(dfs_dict, probExceed_total)
        
        spdExceed_dir = speed_exceedance_directional(dfs_dict, probExceed)
        qile = [1-i for i in probExceed]
        spdExceed_dir['total'] = df['WindSpeed'].quantile(qile).values


    return pdf, cpdf, spdExceed_dir, spdExceed_total

# %% TEMPERATURE STATISTICS

def temperature_statistics(windDirs, temp, siteLabel, numDirs=NUMBER_OF_WINDDIRS, temp_ranges=DEFAULT_TEMP_RANGES):
    """
    Produce a temperature pdf for specified number of wind directions.
    
    Parameters
    ----------------------------------
    windDirs : numpy.array or pandas.Series
        Vector of wind directions (in degrees, 0 is calm, 360 is north)
    temp : numpy.array or pandas.Series
        Vector of temperatures
    numDirs : int, optional
        Number of sectors of the wind rose. Default is 16.
    temp_ranges : list, optional
        List of temperature intevals. Default is [0,1,2,...,50].
    
    Returns
    ----------------------------------
    pdf : pandas.DataFrame
    cpdf : pandas.DataFrame
    spdExceed_perDirection : pandas.DataFrame
    spdExceed_total : pandas.DataFrame
    """
    
    pdfAll = windspeed_pdf(windDirs, temp, temp_ranges)
    map_to_sector = _map_to_nsectors(numDirs)
    _temperature_statistics = pd.DataFrame()
    for sector_windDirsection, probability_share in map_to_sector.items():
        _temperature_statistics[sector_windDirsection] = sum(fraction*pdfAll[wd] for fraction, wd in probability_share)

    # sort columns before returning
    wind_angles_reduced = list(_sector_windDirections(numDirs)) # order of coloums
    Tpdf=_temperature_statistics[wind_angles_reduced]
    # bring 360.0 Column to the first column and rename to 0.0
    _zeroDir=Tpdf[360.0]; Tpdf=Tpdf.drop(labels=[360.0], axis=1); Tpdf.insert(0, 0.0, _zeroDir)

    # cumulative pdf
    _Tcpdf=(Tpdf.iloc[::-1]).cumsum();
    Tcpdf=_Tcpdf.iloc[::-1];
    Tcpdf_index=[">"+str (i)+" C" for i in temp_ranges];
    Tcpdf.set_index([Tcpdf_index],inplace=True, drop=True)
    Tpdf.set_index([Tcpdf_index],inplace=True, drop=True)

    # probability of exceedance
    probExceed=[0.01,0.05,0.1,0.2,0.5,0.8,0.9,0.95,0.99]
    probExceedLabel=['1%','5%','10%','20%','50%','80%','90%','95%','99%']
    Texceed=pd.DataFrame(0, index=probExceedLabel, columns=Tpdf.columns, dtype=np.float64)
    for i in range(len(probExceed)):
        for j in range(len(Tpdf.columns)):
            d=probExceed[i]*Tcpdf.iloc[0,j] # finding the probability exceedance for the speciifc direction
            Texceed.iloc[i,j]=np.interp(d,Tcpdf.iloc[:,j],temp_ranges,period=100) #fidning the corresponding speed
    
    return Tpdf, Tcpdf, Texceed



#%% Functions added to work with NOAA and match MATLAB

"""
Global variable changes
"""
def global_turnoff_sector_mapping(switch):
    global flag_nomapping
    if switch.lower() == 'off':
        flag_nomapping = True
    elif switch.lower() == 'on':
        flag_nomapping = False
        print("No smoothing applied to PDF/CDF for each wind direction.\nWill affect wind roses and exceedance analysis")
    else:
        raise TypeError("Please specify whether or not you want the directional sector mapping: 'on' or 'off'")



"""
HISTOGRAM
"""
def plot_histogram(
        input_data,
        site_description,
        column='WindSpeed',
        figsize=(10,8),
        save=False,
        save_filepath=''
        ):

    data = input_data.copy()
    if column == 'Rain':
        data = data[data['Rain']>0]
        
    nbins = round(max(data[column]) - min(data[column]))
    
    fig, ax = plt.subplots(1,1,figsize=(figsize))
    out = ax.hist(data[column],nbins,color='#E61E28',edgecolor='black',linewidth=1)
    n = out[0]
    bins = out[1]
    ax.grid(linestyle=':',alpha=0.7,linewidth=1)
    ax.set_ylabel('Number of occurrences')
    ax.set_xlabel('Wind Velocity (m/s)')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    if save and save_filepath :
        filename = f'histogram_{site_description}'
        fig.savefig(os.path.join(save_filePath, filename+'.png'),bbox_inches='tight')
            
    return n, bins, fig

def plot_monthly_mean(
        input_data,
        site_description,
        column = 'WindSpeed',
        figsize=(16,6),
        maxes=True,
        save=False,
        save_filepath='',
        dpi=300,
        ):
    
    '''
    Plots monthly box plots, number of recordings, maximum, and mean values of column data.
    
    Parameters
    ----------
    input_data : pandas dataframe
        Data to be resampled and analysed
        
    site_description : string
        Title block
    
    column : string
        Default 'WindSpeed'. 
        Determines what column of dataframe is used for plotting.
        Must be a named column of dataframe containing numerical values.
        
    figsize : tuple of ints
        Default is (14,6)
        Controls the size of the plot.
    
    maxes : bool
        Default True
        Controls whether maximum values (not all outliers) are plotted above 
        box and whisker plots
        
    save : bool
        Default False
        Controls whether the image is saved.
    
    save_filepath : str
        Default None
        Controls where the image is saved.
        
    Returns
    ----------
    data_desc : pandas dataframe
        Plotted data
    
    '''
    
    data = input_data.copy()
    if column == 'Rain':
        data = data[data['Rain']>0]
        
    data_month_dict = {}
    data_month_desc = {}
    data_month = data[column].resample('M')
    for i in data_month:
        data_monthly = i[1]
        desc = data_monthly.describe()
        data_month_dict.update({f"{i[0].year}-{i[0].month:02d}":data_monthly})
        data_month_desc.update({f"{i[0].year}-{i[0].month:02d}":desc})
        
    data_desc = pd.DataFrame(data_month_desc)
    fig, ax = plt.subplots(1,1,figsize=(figsize))
    ax2 = ax.twinx()
    plotx = np.arange(0,len(data_desc.columns))
    ax.boxplot(data_month_dict.values(),0,'',showmeans=False,positions=plotx)
    ax.plot(plotx,data_desc.loc['mean'],'-x',label='Monthly Mean')
    if maxes:
        ax.plot(plotx,data_desc.loc['max'],'+',c='g',label='Monthly Maximum')
    ax2.plot(plotx,data_desc.loc['count'],'o-',c='r',label='Recordings in month')
    
    arr_pts = np.array([5,25,50,75,95])
    ax3 = fig.add_axes([0.7,0.925,0.125,0.06])
    ax3.boxplot(arr_pts,widths=0.6,vert=False)
    ax3.set_yticks([])
    arr_ticks=[]
    for i in arr_pts:
        arr_ticks.append(f'{i}%')
    ax3.set_xticks(arr_pts)
    ax3.set_xticklabels(arr_ticks)
    
    # add divider for month
    divider_loc,divider_label = \
        zip(*[(list(data_desc.columns).index(x), x) for x in list(data_desc.columns) if '-01' in x])
    ax.vlines(divider_loc,ax.get_ylim()[0], ax.get_ylim()[1],colors='b',linewidth=0.5)
    
    
    ax.set_ylabel(f'{column}')
    ax.set_xlabel('Month')
    # ax.tick_params(axis='x',labelrotation = 90)
    ax.set_xticklabels(data_desc.columns, fontsize=8, rotation=90)
    ax2.set_ylabel('Recordings in month')
    fig.suptitle(f'{site_description}')
    fig.legend()
    
    # VL
    write_name = f'monthly_mean_plot_{site_description}'
    fig_savepath = os.path.join(save_filepath,"Plots")
    if not os.path.exists(fig_savepath):
        os.makedirs(fig_savepath)
    if save:
        if save_filepath:
            fig.savefig(os.path.join(fig_savepath,write_name),bbox_inches='tight',dpi=300)
            
        data_desc = data_desc.T
        data_desc.to_csv(os.path.join(save_filepath,'stats_month.csv'))
    
    return data_desc
