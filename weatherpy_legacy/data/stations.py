# Import required packages
import os
import numpy as np
import pandas as pd

#%% haversine
def haversine(lat1, lat2, lon1, lon2):
    """
    The haversine function is used to calculate distance between two sets of
    latitude and longitude coordinates, as well as the bearing from point one
    to point two.
    
    Parameters
    ----------
    lat1 : float
        The latitude of the first point
    lat2 : float
        The latitude of the second point
    lon1 : float
        The longitude of the first point
    lon2 : float
        The longitude of the second point
    
    Returns
    ----------
    distance : float
         The calculated distance between the two input points 
    bearing : 
        Bearing from point one to point two in decimal degrees where 0° is
        north and 180° is south.
    """
    
    # Convert Decimal degrees to radians
    lat1, lat2, lon1, lon2 = map(np.radians, [lat1, lat2, lon1, lon2])
    
    # Haversine conversion
    lat_d = lat2 - lat1
    lon_d = lon2 - lon1
    x = np.sin(lat_d / 2)**2
    y = np.cos(lat1)*np.cos(lat2)*np.sin(lon_d / 2)**2
    a = x + y
    c = 2 * np.arcsin(np.sqrt(a))
    distance = c * 6371000
    
    # Bearing Calculation
    m = np.cos(lat2)*np.sin(lon_d)
    n = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(lon_d)
    bearing = (np.arctan2(m,n)) * 180/np.pi
    
    # Distance and bearing ouput as a tuple
    return distance, bearing


#%% find_stations
def find_stations(stationID=None,
                  coord=None,
                  city=None,
                  nearest=None,
                  radius=None,
                  dataType='BOM',
                  printOutput=True,
                  save = False,
                  outputType='csv',
                  saveDirpath=''):
    """
    Search for nearest weather stations from BOM and NOAA station databases.
    
    The origin point is either defined by the stationID argument, which will
    consider the location of that station as the origin, the coord argument, or
    the city argument.
    
    The function can return all stations within a specified km radius (radius
    argument), or the nearest "n" number of stations around the origin.
    
    The function will only perform one search per instance, so if the origin
    is specified by BOTH station ID and coord, the search will only be run
    around the stationID.
    
    Similarly, if the search criteria are defined by BOTH nearest and radius,
    the function will only search for the nearest number.
    
    Parameters
    ----------
    stationID : str, optional
        For searching around a given station. Default is None.
    coord : list or tuple, optional
        A list or tuple of the latitude and longitude. Default is None.
    city : str, optional
        A name of a city to search in/around. Default is None.
    nearest : int, optional
        To return the nearest "n" number of stations. Default is None
    radius : int, optional
        To return all stations within a radius of the origin. Default is None.
    dataType : str, optional
        Which data type to use: "BOM" or "NOAA". Default is "BOM".
    printOutput : bool, optional
        If the user wishes to have the matching data points printed to console.
        Default is True.
    save : bool, optional
        If the user wishes to have the found stations data frame output to the
        desired location. Default is False.
    outputType : str, optional
        If the user wishes to have data frame saved, the type of output can be
        specified as either 'csv', or 'kml'. Only necessary if save = True.
        Default is 'csv'.
    saveDirpath : str, optional
        If the user wishes to have data frame saved, the save location can be
        specified. Only necessary if save = True. Default is ''.
    
    Returns
    ----------
    stationsOut = pandas.Dataframe
        The function will return a data frame with the data points that matched
        the specified criteria from the arguments. If the function is missing 
        valid inputs, a NoneType object will be returned
    """

    # Imports data
    if dataType == 'NOAA':
        absolute_path = os.path.dirname(__file__)
        relative_path = os.path.join('src','NOAA_stations_full.csv')
        full_path = os.path.join(absolute_path, relative_path)
    elif dataType == 'BOM':
        absolute_path = os.path.dirname(__file__)
        relative_path = os.path.join('src','BOM_stations_clean.csv')
        full_path = os.path.join(absolute_path, relative_path)
    else:
        raise NameError('Please enter a valid dataSource argument (\"BOM\" or \"NOAA\")')
    
    stations = pd.read_csv(full_path,converters={'Station Code':str,
        'Station Name':str,'Country':str,'State':str,'Latitude':float,
        'Longitude':float,'Elevation':str,'Start':str,'End':str,
        'Timezone':str,'Source':str,'Wind Direction':str,
        'Wind Speed':str,'Wind Gust':str,'Sea Level Pressure':str,
        'Dry Bulb Temperature':str,'Wet Bulb Temperature':str,
        'Relative Humidity':str,'Rain':str,'Rain Intensity':str,
        'Cloud Oktas':str},index_col=False)
    
    # Initialise output
    stationsOut = None

    # Check for valid origin input
    if stationID != None:
        origin_type = 's'
    elif coord != None:
        origin_type = 'c'
    elif city != None:
        origin_type = 'v'
    else:
        raise TypeError('No valid origin argument entered')
        
    # Check for valid criteria input
    if nearest != None:
        criteria_type = 'n'
        nearest_b = nearest
    elif radius != None:
        criteria_type = 'r'
    else:
        raise TypeError('No valid criteria argument entered')
    
    # Initialises titles
    title_origin = ''
    title_criteria = ''
    
    # Checks the stationID origin point input (s = stationID)
    if origin_type == 's':
        inpt_id = str(stationID)
        station_row = stations.loc[stations['Station Code'] == inpt_id]
        lat1 = station_row['Latitude']
        lon1 = station_row['Longitude']
        
        # If lat1 and lon1 are = 1, then the stationID has been found
        if len(lat1) == 1 and len(lon1) == 1:
            lat1 = float(lat1)
            lon1 = float(lon1)
            station_name = station_row['Station Name']
            station_name = station_name.reset_index()['Station Name'][0]
            title_origin = station_name + ' (ID: ' +inpt_id + ')'
        else:
            raise IndexError('Station ID could not be found in database')
    
    # Checks the lat/lon origin input (c = coordinate)
    elif origin_type == 'c':
        try:
            lat1 = float(coord[0])
            lon1 = float(coord[1])
            title_origin = '(' + str(lat1) + ', ' + str(lon1) + ')'
        except:
            raise TypeError('Please enter a float or integer for \"lon\"')
    
    # Checks the city input (v = city)
    elif origin_type == 'v':
        absolute_path = os.path.dirname(__file__)
        relative_path = os.path.join('src','Cities_database.csv')
        full_path = os.path.join(absolute_path, relative_path)
        cities = pd.read_csv(full_path)
        city_row = cities.loc[cities['city'] == city].reset_index(drop=True)
        
        if len(city_row) > 1:
            for i, r in city_row.iterrows():
                print(f'City Number: {i+1}')
                print('  City:       {}'.format(r['city']))
                print('  State:      {}'.format(r['admin_name']))                
                print('  Country:    {}'.format(r['country']))
                print('  Population: {}'.format(round(r['population'])))
                print()   
            print(f'There are multiple cities which have the name: {city}')
            city_numb = int(input('Which of the above cities is correct? (Type the city number): '))
                
            try:
                print('\nOrigin specified as: {}, {}, {}'.format(city_row['city'][city_numb-1],city_row['admin_name'][city_numb-1],city_row['country'][city_numb-1]))
                lat1 = float(city_row['lat'][city_numb-1])
                lon1 = float(city_row['lng'][city_numb-1])
                title_origin = '{} | {} | {}'.format(city_row['city'][city_numb-1],city_row['admin_name'][city_numb-1],city_row['country'][city_numb-1])
            except:
                raise IndexError('Invalid city number chosen')
        
        elif len(city_row) < 1:
            raise IndexError(f'City name: \"{city}\" not found in database. Please ensure spelling is correct.')
        
        else:
            lat1 = float(city_row['lat'])
            lon1 = float(city_row['lng'])
            title_origin = '{} | {} | {}'.format(city_row['city'][0],city_row['admin_name'][0],city_row['country'][0])
    
    # Checks the nearest criteria input (n = nearest number)
    if criteria_type == 'n':
        try:
            nearest = int(nearest)
            if nearest < 1:
                raise IndexError
                
            elif nearest > len(stations):
                raise IndexError
                
            elif nearest== 1:
                title_criteria = 'The closest station to: '
            else:
                title_criteria = ('The closest ' + str(nearest) +
                    ' stations to: ')
        
        except IndexError:
            raise ValueError('nearest argument must be between 1 and the total ' +
                'station count (' + str(len(stations)) + ' stations)')
        
        except:
            raise TypeError('\"nearest\" argument must be an integer ')
    
    # Checks the radius criteria input (r = radius)
    elif criteria_type == 'r':
        try:
            radius = int(radius)
            radius = radius * 1000
            if radius < 1 or radius > 10000000:
                raise IndexError

            else:
                title_criteria = ('The stations within ' + str(radius/1000) +
                    'km of: ')
        
        except IndexError:
            raise ValueError('\"radius\" argument must be between 1 and 10000')
        
        except:
            raise TypeError('\"radius\" argument must be an integer')
    
    # Calculate distances and bearings using vectorisation
    stations['Distance (km)'] = round(((haversine(
        lat1,stations['Latitude'],lon1,stations['Longitude'])[0])/1000),3)

    stations['Distamce (mi)'] = round(
        (stations['Distance (km)'] * 0.62137119),3)

    stations['Bearing'] = round((haversine(
        lat1,stations['Latitude'],lon1,stations['Longitude'])[1]),3)
    
    stations.loc[stations['Bearing'] < 0, 'Bearing'] = \
        stations['Bearing'] + 360
    
    # The data frame is sorted by the distance column in ascending order
    stations = stations.sort_values(by='Distance (km)',ascending=True
        ).reset_index(drop=True)
    
    # New data frame is trimmed to only include criteria matching rows
    if origin_type == 's' and criteria_type == 'n':
        nearest += 1

    if criteria_type == 'n':
        stationsOut = stations.head(nearest)
    else:
        counter = 0
        radius_check = True
        while radius_check == True:
            if stations['Distance (km)'][counter] < radius/1000:
                counter += 1
                if counter == len(stations):
                    radius_check = False  
            else:
                radius_check = False
        stationsOut = stations.head(counter)

    if origin_type == 's':
        stationsOut = stationsOut.iloc[1: , :]
    
    stationsOut = stationsOut.reset_index(drop=True)
    
    # The data points matching the criteria are printed 
    if printOutput:
        print('\n\n=========================================================')
        print(title_criteria + title_origin)
        print('=========================================================\n')  
        
        print_stations(stationsOut,head=10)
           
        if len(stationsOut) == 0:
            print('No stations found in search')
        elif len(stationsOut) == 1:
            print('1 station found')
        elif len(stationsOut) <= 10:
            print(str(len(stationsOut)) + ' stations found')
        else:
            print('Displaying first 10 of ' + str(len(stationsOut)) + \
                  ' station found')

    if save == True and saveDirpath is not None:
        # Initialises filename output
        if dataType.upper() == 'BOM':
            filename = os.path.join(saveDirpath,'Stations_BOM_')
        elif dataType == 'NOAA':
            filename = os.path.join(saveDirpath,'Stations_NOAA_')
        
        if criteria_type == 'r':
            filename = filename + f'within{round(radius/1000)}km_'
        else:
            filename = filename + f'nearest{nearest_b}_'
        
        if origin_type == 'c':
            filename = filename + f'{lat1}_{lon1}'
        elif origin_type == 'v':
            filename = filename + f'{city}'
        else:
            filename = filename + f'{str(stationID)}'
        
        output_stations(stationsOut, filename, outputType=outputType)

    # The data frame of matching data points is returned
    return stationsOut
    

#%% filter_stations
def filter_stations(stationData,
                    country=None,
                    state=None,
                    measurementType=None,
                    printOutput=True,
                    save=False,
                    outputType='csv',
                    saveDirpath=''):
    """
    filters a weatherstation dataframe by a series of parameters. The default
    for each filter parameter is none, meaning entries will not be filtered.
    
    for filtering by available measurement types (wind speed, wind direction, 
    etc...), a string or a list of strings argument should be passed into the 
    masurementType argument. See parameters of type codes.
    
    Parameters
    ----------
    stationData : pandas.DataFrame
        Takes in a dataframe of weather stations.
    country : str or list, optional
        Keep stations only from the specified country or countries.
    state : str or list, optional
        Keep stations only from the specified state or states: NSW, VIC, QLD, 
        WA, SA, NT, TAS, ANT(Antarctica), ISL(Remote Island)
    measurementType : str or list, optional
        One or multiple data types that will be filtered by. Only stations that
        have data for these measurement types will pass the filter. The valid
        arguments are are two-letter codes as listed below:
        - WD : windDirection
        - WS : windSpeed
        - WG : windGust
        - SP : sealevelPressure
        - DB : dryBulbTemp
        - WB : wetBulbTemp
        - RH : relativeHumidity
        - RA : rain
        - RI : rainIntensity
        - CO : cloudOktas
        Additionally, the user can enter "ALL" if they wish only to see
        stations that have observation for all data types.
    printOutput : bool, optional
        if the User wishes for the matching stations ot be output to the
        console. Default is True.
    save : bool, optional
        If the user wishes to have the found stations data frame output to the
        desired location. Default is False.
    outputType : str, optional
        If the user wishes to have data frame saved, the type of output can be
        specified as either 'csv', or 'kml'. Only necessary if save = True.
        Default is 'csv'.
    saveDirpath : str, optional
        If the user wishes to have data frame saved, the save location can be
        specified. Only necessary if save = True. Default is ''.
        
    Returns
    ----------
    stationData = pandas.Dataframe
        The function will return a dataframe with the data points that matched
        the specified filter arguments, if the function is missing valid
        inputs, a NoneType object will be returned
    """
    
    stationsOut = stationData.copy()
    types_dict = {'WD':'Wind Direction','WS':'Wind Speed','WG':'Wind Gust',
        'SP':'Sea Level Pressure','DB':'Dry Bulb Temperature',
        'WB':'Wet Bulb Temperature','RH':'Relative Humidity','RA':'Rain',
        'RI':'Rain Intensity','CO':'Cloud Oktas'}

    # Check function inputs
    if type(stationData) != type(pd.DataFrame([1,2],[3,4])):
        raise TypeError('The first argument must be a pandas dataframe object')
        
    elif len(stationData) == 0:
        raise ValueError('Input dataframe object is empty')
    
    # Filter by country
    if type(country) == type('string'):
        stationsOut = stationsOut[stationsOut['Country'] == country]
        saveDirpath = os.path.join(saveDirpath,'StationsFilteredBy_country')
    elif type(country) == type([1,2]):
        country_set = set(country)
        stationsOut = stationsOut[stationsOut['Country'].isin(country_set)]
        saveDirpath = os.path.join(saveDirpath,'StationsFilteredBy_country')
    elif country != None:
        raise TypeError('\"country\" argument must be a string of a list of strings')
        
    # Filter by state(s)
    if type(state) == type('string'):
        stationsOut = stationsOut[stationsOut['State'] == state.upper()]
        saveDirpath = saveDirpath + '_state'
    elif type(state) == type([1,2]):
        state_set = set([x.upper() for x in state])
        stationsOut = stationsOut[stationsOut['State'].isin(state_set)]
        saveDirpath = saveDirpath + '_state'
    elif state != None:
        raise TypeError('\"state\" argument must be a string or a list of string')
    
    # Filter By data type
    if type(measurementType) == type('string'):
        if measurementType.upper() == 'ALL':
            for n in list(types_dict.keys()):
                stationsOut = stationsOut[stationsOut[types_dict[n]] == 'True']
            saveDirpath = saveDirpath + '_ALL_dataTypes'
        else:
            stationsOut = stationsOut[stationsOut[types_dict[measurementType.upper()]] == 'True']
            saveDirpath = saveDirpath + f'_{measurementType}_dataType'
    elif type(measurementType) == type([1,2]):
        for n in measurementType:
            for i, r in stationData.iterrows():
                stationsOut = stationsOut[stationsOut[types_dict[n]] == 'True']
            saveDirpath = saveDirpath + f'_{n}'
        saveDirpath = saveDirpath + '_dataTypes'
    elif measurementType != None:
        raise TypeError('\"measurementType\" argument must be a string or a list of string')
    
    # Print output if requested 
    if printOutput == True:
        print_stations(stationsOut.head(10))
        print(f'\n{len(stationsOut)} station(s) of {len(stationData)} total station(s) passed through the filters\n')

    # Export dataframe if requested
    if save == True and saveDirpath is not None:
        output_stations(stationsOut, saveDirpath, outputType=outputType)
    
    # Returns filtered dataframe
    return stationsOut.reset_index(drop=True)


#%% print_stations

def print_stations(stationDF, head=None, info=False):
    '''
    Neatly prints a station database to console
    
    Parameters
    ----------
    stationDF : pandas.DataFrame
        A stations dataframe
    head : int, optional
        The first few stations the user wishes to display. Default is None
        which will display all the stations
    info : bool, optional
        Mark as True if the database has columns for distance and bearing at
        the end (i.e. has come from find_stations). Default is True.
    
    Returns
    -------
    None
    '''

    if head !=None:
        try:
            stationDF = stationDF.head(head).copy()
        except:
            print('Please enter a positive integer for the head argument')
            return
    
    for i, r in stationDF.iterrows():
        if info == False:
            print('Station Code:     ' + str(r['Station Code']))
        
        print('Station Name:     ' + str(r['Station Name']))
        print('Country:          ' + str(r['Country']))
        print('State:            ' + str(r['State']))
        print('Timezone:         ' + str(r['Timezone Name']))
        print('Timezone Ofset:   ' + str(r['Timezone UTC']))
        print('Coordinate:       (' + str(r['Latitude']) + ', ' + str(r['Longitude']) + ')')
        print('Altitude:         ' + str(r['Elevation']) + ' m')
        print('Years Active:     ' + str(r['Start'])+' - '+str(r['End']))
        print('Source:           ' + str(r['Source']))
        
        if info == False:
            print('Distance          '+str(round(r['Distance (km)'], 3))+' km')
            print('Bearing:          ' + str(round(r['Bearing'])) + '°')
        
        print('M Correction Factors: ', end = '')
        if str(r['Mean Correction Factors']) == '':
            print('N/A')
        else:
            print(str(r['Mean Correction Factors']))
            
        if stationDF['Source'][i].upper() == 'BOM':
            if stationDF['Wind Speed'][i] == 'N/A' or stationDF['Wind Speed'][i] == '':
                print('Data Available?:  No')

            else:
                print('Data Available?:  Yes')
                print('  Wind Direction:     ' + \
                      stationDF['Wind Direction'][i])
                print('  Wind Speed:         ' + stationDF['Wind Speed'][i])
                print('  Wind Gust:          ' + stationDF['Wind Gust'][i])
                print('  Sea Level Pressure: ' + \
                      stationDF['Sea Level Pressure'][i])
                print('  Dry Bulb Temp:      ' + \
                      stationDF['Dry Bulb Temperature'][i])
                print('  Wet Bulb Temp:      ' + \
                      stationDF['Wet Bulb Temperature'][i])
                print('  Relative Humidity:  ' + \
                      stationDF['Relative Humidity'][i])
                print('  Rain:               ' + stationDF['Rain'][i])
                print('  Rain Intensity:     ' + \
                      stationDF['Rain Intensity'][i])
                print('  Cloud Oktas:        ' + stationDF['Cloud Oktas'][i])
            
        print()
        print('-----------------------------------------------------') 
        print()
    return


#%% output_stations
def output_stations(stationsOut, saveDirpath, outputType='csv'):
    '''
    Outputs a station database, as either a csv or a kml
    
    Parameters
    ----------
    stationsOut : pandas.DataFrame
        A stations dataframe
    saveDirpath : str
        The desired save location of the output file
    outputType : str, optional
        The desired output file type. "csv" or "kml". Default is "csv"
    
    Returns
    -------
    None
    '''
    

    # Outputs to csv
    if outputType == 'csv':
        stationsOut.to_csv(saveDirpath + '.csv', index=False)
    
    # Outputs to kml
    elif outputType == 'kml':
        kml_str = ['<?xml version=\"1.0\" encoding=\"UTF-8\"?>',
            '<kml xmlns=\"http://www.opengis.net/kml/2.2\" xmlns:gx=\"http://www.google.com/kml/ext/2.2\">',
            '\t<Document>']
        
        # Adds all the points found 
        for i, r in stationsOut.iterrows():
            lat_coord = r['Latitude']
            lon_coord = r['Longitude']
            
            kml_str.append('\t\t<Placemark>')
            kml_str.append('\t\t\t<name>{} ({})</name>'.format(r['Station Name'],r['Station Code']))
            kml_str.append('\t\t\t<description>' +\
                           '\nCountry:\t{}'.format(r['Country']) +\
                           '\nState:\t{}'.format(r['State']) +\
                           '\nTimezone:\t{} ({})'.format(r['Timezone Name'], r['Timezone UTC']) +\
                           '\nYears Active:\t{} - {}\n'.format(r['Start'],r['End']) +\
                           '\nWind Direction:\t{}'.format(r['Wind Direction']) +\
                           '\nWind Speed:\t{}'.format(r['Wind Speed']) +\
                           '\nWind Gust:\t{}'.format(r['Wind Gust']) +\
                           '\nSea Level Pres:\t{}'.format(r['Sea Level Pressure']) +\
                           '\nDry Bulb Temp:\t{}'.format(r['Dry Bulb Temperature']) +\
                           '\nWet Bulb Temp:\t{}'.format(r['Wet Bulb Temperature']) +\
                           '\nRelative Humid:\t{}'.format(r['Relative Humidity']) +\
                           '\nRain:\t{}'.format(r['Rain']) +\
                           '\nRain Intensity:\t{}'.format(r['Rain Intensity']) +\
                           '\nCloud Oktas:\t{}'.format(r['Cloud Oktas']) +\
                           '</description>')
            kml_str.append('\t\t\t<Point>')
            kml_str.append('\t\t\t\t<coordinates>{},{},0</coordinates>'.format(lon_coord,lat_coord))
            kml_str.append('\t\t\t</Point>')
            kml_str.append('\t\t</Placemark>')

        kml_str.append('\t</Document>')
        kml_str.append('</kml>')
        
        # Writes to a kml file in the desired location
        with open(saveDirpath + '.kml','w') as f:
            f.write('\n'.join(kml_str))
    
    print(f'Stations output to {outputType} file to:' + '\n' + f'{saveDirpath}' + f'.{outputType}')
    
    return


#%% station_info
def station_info(stationID, printed=True):
    '''
    Prints the infomration about a given station and returns a dictionary

    Parameters
    ----------
    stationID : str
        Either a BOM or NOAA station ID.
    printed : Boolean, optional
        indicates whether the data should be printed to console. Default is True.

    Returns
    -------
    station_dict : dict
        A dictionary of all the printed information about the station
    '''
    # bom data
    ID_str = str(stationID)
    if len(ID_str)<=6:
        
        relative_path = os.path.join('src','BOM_stations_clean.csv')
        absolute_path = os.path.dirname(__file__)
        filenameB = os.path.join(absolute_path, relative_path)
        stations_BOM = pd.read_csv(filenameB, sep=',',converters={
            'Station Code':str,'Station Name':str,'Country':str,'State':str,
            'Latitude':float,'Longitude':float,'Elevation':str,'Start':str,
            'End':str,'Timezone UTC':str, 'Timezone Name':str,'Source':str,'Wind Direction':str,
            'Wind Speed':str,'Wind Gust':str,'Sea Level Pressure':str,
            'Dry Bulb Temperature':str,'Wet Bulb Temperature':str,
            'Relative Humidity':str,'Rain':str,'Rain Intensity':str,
            'Cloud Oktas':str,'Mean Correction Factors':str})
        
        stations_BOM['Station Code'] = stations_BOM['Station Code'].apply(lambda x: str(x).zfill(6)) # pad only Station Code' with 6 zeros
    
        ID = ID_str.zfill(6)
        
        stations = stations_BOM
    
    else: 
        absolute_path = os.path.dirname(__file__)
        relative_path = os.path.join('src','NOAA_stations_full.csv')
        filenameN = os.path.join(absolute_path, relative_path)
        stations_NOAA = pd.read_csv(filenameN, sep=',',converters={
            'Station Code':str,'Station Name':str,'Country':str,'State':str,
            'Latitude':float,'Longitude':float,'Elevation':str,'Start':str,
            'End':str,'Timezone':str,'Source':str,'Wind Direction':str,
            'Wind Speed':str,'Wind Gust':str,'Sea Level Pressure':str,
            'Dry Bulb Temperature':str,'Wet Bulb Temperature':str,
            'Relative Humidity':str,'Rain':str,'Rain Intensity':str,
            'Cloud Oktas':str,'Mean Correction Factors':str})
    
        stations_NOAA['Station Code'] = stations_NOAA['Station Code'].apply(lambda x: str(x).zfill(11)) # pad only Station Code' with 6 zeros
        
        ID = ID_str.zfill(11)
        
        stations = stations_NOAA
        
    # stations_all = pd.concat([stations_BOM,stations_NOAA]).reset_index(drop=True)
    
    try:
        station_dict = stations.loc[stations['Station Code'] == ID].reset_index(drop=True).transpose().to_dict()[0]
    except:
        #raise KeyError('Station ID not in database')
        pass
    
    station_df = pd.DataFrame.from_dict(station_dict, orient='index').transpose()
    
    if printed:
        print('\n=========================================================')
        print(f'Station ID: {stationID}')
        print('=========================================================\n')
        print_stations(station_df,info=True)

    return station_dict




