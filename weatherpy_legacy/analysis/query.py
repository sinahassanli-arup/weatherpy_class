from .processing import generate_ranges, generate_ranges, generate_ranges_wind_direction
from collections import OrderedDict
import numpy as np
import pandas as pd
import os

BOUNDS_INCLUDED_SINGS = {'both':    ['>=','<='],
                         'upper':   ['>','<='],
                         'lower':   ['>=','<']}

# %% QUERY RANGE

def query_range(data, queries, closed='upper', loud=False,site_description='', save=False, savePath=None):
    """
    Query data based on query ranges. 
    Returned data will be inclusive of lower and upper boundary

    Parameters
    ----------
    data : pandas.DataFrame
        Unified weather data to be queried.
    queries : dictionary
        dictionary with keys as data fields and values as list of respective range thresholds.
    closed: str, optional 
        Select one of the following options:
            - both:    range inclusive of both bounds
            - upper:   range inclusive of upper bound only
            - lower:   range inclusive of lower bound only
        Default is upper.
    loud : bool, optional
        Set as True to have information printed to console. Default is False.
    
    Returns
    -------
    data_query : pandas.DataFrame
        data filtered based on query.
    probability : float
        probability of time query range happens.
    mask : pandas.Series
        A boolean series with a datetime index where True indicates the timestamp
        is included within the query and False indicates that it is not included
    """
    
    # check which bounded to be included in comparison
    sign = BOUNDS_INCLUDED_SINGS[closed]
    
    # index expended to new columns of year, month, day, and hour to be able to filter based on date and time
    data_datetimeExpanded = data.assign(Year=data.index.year, Month=data.index.month, Day=data.index.day, Hour=data.index.hour, Minute=data.index.minute)

    #creating the conditions list
    query_range_joined = []
    for k, v in queries.items():
        if isinstance(v, list) or isinstance(v, tuple):
            if v[1]>v[0]:
                query_range_joined.append('(({0}{2[0]}{1[0]})&({0}{2[1]}{1[1]}))'.format(k, v,sign))
            if v[0]>v[1]:
                query_range_joined.append('(({0}{2[0]}{1[0]})|({0}{2[1]}{1[1]}))'.format(k, v,sign))
                if (k=='WindDirection') and (k[0]!=0 or k[1]!=0):
                    query_range_joined.append('({}!=0&WindSpeed!=0)'.format(k))
    
    #creating conditions string from list
    query_range_joined_all = ' & '.join(query_range_joined)

    #creating mask based on conditions
    mask = data_datetimeExpanded.eval(query_range_joined_all)
    
    #filter the data and calculate probability
    data_filtered = data[mask]
    probability = data_filtered.shape[0]/data.shape[0]
    
    if loud:
        print('Querying Data')
        for k, v in queries.items():
            print('\t{}s ({} bounds) between {} - {}'.format(k, closed, v[0], v[1])) 
        
        print('\t--Query Complete--\n')
    
    if save:
        # Save query data to excel sheet in local directory
        filename = '{}_{}.xlsx'.format('query_range', site_description)
        filePath = os.path.join(savePath, filename)
        writer = pd.ExcelWriter(filePath, engine='xlsxwriter')
        
        def convert_to_timezone_unaware(df):
            
            print(df.dtypes)
            # Convert index to timezone unaware
            df.index = df.index.tz_localize(None)
        
            # Convert all datetime columns to timezone unaware
            df = df.apply(lambda col: col.dt.tz_localize(None) if pd.api.types.is_datetime64_any_dtype(col) else col)

            print(df.dtypes)
            return df

        for k, v in data_filtered.items():
            v = convert_to_timezone_unaware(v)
            v.to_excel(writer, sheet_name = k)
            
        writer.close()
        
    return data_filtered, probability, mask

# %% QUERY SELECTION
def query_selection(data, selections):
    """
    Filter data based on selection query
    for example to select January and February and December and only half hourly data 
    you can use {'Month': [12,1,2], 'Minute':[0,30]} as selections

    Parameters
    ----------
    data : pandas.DataFrame
        Weather data from BOM or NOAA
    selections : dict of lists
        A dictionary with keys as attribute types or Year, Month, Day, Hour, Minute and values as a list to include
        it can have multiple selections

    Returns
    -------
    data_filtered : pandas.DataFrame
        Filtered data.
    probability : float
        A float representing the percent of the whole data set that is included in the query  
    mask: pandas.Series
        A boolean series with a datetime index where True indicates the timestamp
        is included within the query and False indicates that it is not included
    """

    data_datetimeExpanded = data.assign(Year=data.index.year , Month=data.index.month, Day=data.index.day, Hour=data.index.hour, Minute=data.index.minute)

    conditions = []
    for attrType, values in selections.items():
        conditions.append(data_datetimeExpanded[attrType].isin(values))

    mask = conditions[0]
    for c in conditions[1:]:
        mask = mask & c
    
    data_filtered = data[mask]
    probability = len(data_filtered)/len(data_datetimeExpanded)

    return data_filtered, probability, mask

# %% DATA SPLITTER

def data_binsplit(data, splitter, closed='upper'):
    """
    Split data into list of data based on splitter.
    If splitter is Wind Direction, only number of direction as value need specified and
    calm events will be added as a additional column.
    For others splitter is a list of lower and upper bounds for a data field specified as the key.
    NOTE: BOM wind direction interval is 10 degree, hence combining into 16 or 12 would skew the results slightly.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Unified weather data to be splitted.
    splitter : dict
        For Wind Direction, key as 'WindDirection' and value as int for number of Directions
        For other data fields, value should contain a list of range thresholds 
    closed: str, optional
        Select one of the following options:
            - both:    range inclusive of both bounds
            - upper:   range inclusive of upper bound only
            - lower:   range inclusive of lower bound only
        Default is upper.

    Returns
    -------
    data_splitted_dict :list
        data splitted into list of data based on splitter.

    """

    # check which bounded to be included in comparison
    sign = BOUNDS_INCLUDED_SINGS[closed]

    splitter_key = next(iter(splitter))
    splitter_value = splitter[splitter_key]
    
    # splitter for wind direction
    if splitter_key=='WindDirection':
        
        range_list, range_label = generate_ranges_wind_direction(splitter_value)
        data_splitted=[]
        
        for i in range(len(range_list)):
            # Calm condition
            if i==0:
                data_splitted.append(data[data['WindDirection']==0])
            # Winds from north
            elif i==1: 
                filt = '(({0}{2[0]}{1[0]})|({0}{2[1]}{1[1]})) & ({0}!=0&WindSpeed!=0)'.format(splitter_key,range_list[i], sign) # 

                mask = data.eval(filt)
                data_splitted.append(data[mask])
            # Other wind directions
            else:
                # list for other wind directions
                filt = '({0}{2[0]}{1[0]}) & ({0}{2[1]}{1[1]})'.format(splitter_key,range_list[i], sign)
                mask = data.eval(filt)
                data_splitted.append(data[mask])
        
        range_label_with_calm = range_label
        data_splitted_dict = OrderedDict(zip(range_label_with_calm,data_splitted))
    
    # splitter for other fields
    else:
        range_list, range_label = generate_ranges(splitter_value, closed)
        data_splitted=[]
        for i in range(len(range_list)):
            filt = '({0}{2[0]}{1[0]}) & ({0}{2[1]}{1[1]})'.format(splitter_key,range_list[i], sign)
            mask = data.eval(filt)
            data_splitted.append(data[mask])
            
        data_splitted_dict = OrderedDict(zip(range_label,data_splitted))
    
    return data_splitted_dict

# %% Data multi splitter
def data_multisplit(data, splitters):
    """
    This function split data based on splitters by finding all combinations between 
    different ranges specified in the splitter.
    - If two values are given it uses range: (max>= attr >=min)
    - If three or more values are given for example 1, 2, 3 then it uses 
      equal or: (attr==1 or attr=2 or attr=3)

    Parameters
    ----------
    data : pd.dataframe
        input weather data.
    splitters : dict of dicts
        dict of dicts of attibutes with ranges [min,max] or selected values [1,2,3]
        main keys are attribute types (Standard weather attributes plus time/date types
        main values are dict of ranges or selected values for that attirbutes.
        if two values are given it uses range (max>=attr>=min)
        if three or more values are given for example 1, 2, 3 then it uses 
        (attr==1 or attr=2 or attr=3)

    Returns
    -------
    data_split : dict of dataframes
        A dict with keys as all combination of subkeys and values as combinations
        of the ranges.

    """
    
    # index expended to new columns of year, month, day, and hour to be able to filter based on date and time
    data_datetimeExpanded = data.assign(Year=data.index.year , Month=data.index.month, Day=data.index.day, Hour=data.index.hour, Minute=data.index.minute)
    
    combinations = [[]]
    
    # generate all possible combinations
    for key in splitters.keys():
        new_combinations = []
        for combination in combinations:
            for subkey, value_range in splitters[key].items():
                new_combinations.append(combination + [(key, subkey, value_range)])
        combinations = new_combinations
    
    # create all possible conditions based on combinations
    conditions = {}
    for combination in combinations:
        condition_parts = []
        subkeys=[]
        for key, subkey, value_range in combination:
            
            # using range if it is 2 values (min, max)
            if len(value_range)==2: 
                if value_range[0]<value_range[1]:
                    cond_str = f'({value_range[0]} <= {key} <= {value_range[1]})'
                else:
                    cond_str = f'({value_range[0]} <= {key} | {value_range[1]} >= {key})'
            
            # using selected valeus instead of range
            elif len(value_range)>2:
                cond_str_lst=[]
                for val in value_range:
                    cond_str_lst.append(f'({key}=={val})')
                cond_str = ' | '.join(cond_str_lst)
                
            condition_parts.append(cond_str)
            subkeys.append(subkey)
        name = '_'.join(subkeys)
        condition = ' & '.join(condition_parts)
        conditions[name]= condition
    
    data_split={}
    for key, value in conditions.items():
        df = data_datetimeExpanded.query(value)
        columns_to_remove = ['Year', 'Month', 'Day', 'Hour', 'Minute']
        df = df.drop(columns_to_remove, axis=1)
        data_split[key]=df
        
    return data_split


# %% TODO: Convert this to query_range for grids
def probability_grid(data, threshold, threshold_label=None):
    
    data_split = {}
    probability = {}
    hours = data.shape[0]
    if isinstance(threshold,dict):
        for key, value in threshold.items():
            if value[0]<value[1]:
                probability[key] = ((data>=value[0]) & (data<=value[1])).sum(axis=0).astype(int)/hours
            else:
                probability[key] = ((data<=value[0]) & (data>=value[1])).sum(axis=0).astype(int)/hours
    elif isinstance(threshold,list):
        for i, value in enumerate(threshold):
            if i==0:
                lbl = threshold_label[i] if threshold_label else f"<=value"
                probability[lbl] = (data<=value).sum(axis=0).astype(int)/hours
            elif i==len(threshold)-1:
                lbl = threshold_label[i+1] if threshold_label else f">=value"
                probability[lbl] = (data>=value).sum(axis=0).astype(int)/hours
            else:
                lbl = threshold_label[i] if threshold_label else f"{threshold[i]}-{threshold[i+1]}"
                probability[lbl] = ((data<=threshold[i+1]) & (data>=threshold[i])).sum(axis=0).astype(int)/hours
                
    return probability


def resample_data(data, frequency, columns=['WindSpeed'], aggregations=['mean', 'std'], col_aggr=None, time_filter=None):
    """
    Resample the data based on input arguments and apply optional filtering.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with datetime index.

    frequency : str
        String indicating the frequency for resampling.
        Options include: 
        'H' hourly frequency,
        'D' daily frequency
        'W' weekly frequency, 
        'M' month end frequency,
        'Q' quarter end frequency, 
        'A' year end frequency, 

    columns : list, optional
        List of column names to be evaluated (default: ['WindSpeed']).

    aggregations : list, optional
        List of aggregation functions for each column.
        Options include: 'sum', 'mean', 'std', 'var', 'min', 'max', 'count', 'first', 'last', 'median', 'mad', 'prod', 'sem', 'skew', 'kurt'.
        Options also include percentile in form of '95%','50%' (computaitonally more expensive)

    col_aggr : dict, optional
        Dictionary specifying custom aggregation functions for specific columns (default: None).

    time_filter : dict, optional
        Dictionary containing filtering conditions based on hour and month (default: {'Hour': [1, 6], 'Month': [2, 4]}).

    Returns
    -------
    pandas.DataFrame
        Resampled and aggregated DataFrame considering the provided filters.
        When using percentile some of the cells might return np.nan especially if the number of samples is not enough for percentile
    """

    if time_filter:
        if 'Hour' in time_filter:
            start_hour, end_hour = time_filter['Hour']
            data = data[(data.index.hour >= start_hour) & (data.index.hour <= end_hour)]

        if 'Month' in time_filter:
            start_month, end_month = time_filter['Month']
            data = data[(data.index.month >= start_month) & (data.index.month <= end_month)]

    if not col_aggr:
        col_aggr = {col: aggregations for col in columns}
    
    # working with percentiles
    agg_dict = {}
    for keys, vals in col_aggr.items():
        agg_dict[keys]=[]
        for v in vals:
            if v.endswith('%'):
                p = int(v.replace('%', ''))
                def percentile_n(x, percentile=p):
                    return np.percentile(x, percentile) if x.size else np.nan
                percentile_n.__name__ = str(p)+'%'
                vv = percentile_n
            else:
                vv = v
            agg_dict[keys].append(vv)
    
    data_resampled = data.resample(frequency).agg(agg_dict)
    
    return data_resampled