"""
Module for managing weather station data.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union

class WeatherStation:
    """Class representing a single weather station."""
    
    def __init__(self, data: pd.Series):
        """Initialize station with data."""
        self.data = data
        
    @property
    def id(self) -> str:
        """Station ID."""
        if 'Station Code' in self.data:
            return str(self.data['Station Code'])
        elif 'Station ID' in self.data:
            return str(self.data['Station ID'])
        return ""
    
    @property
    def name(self) -> str:
        """Station name."""
        return str(self.data['Station Name'])
    
    @property
    def country(self) -> str:
        """Station country."""
        return str(self.data.get('Country', ''))
    
    @property
    def state(self) -> str:
        """Station state."""
        return str(self.data.get('State', ''))
    
    @property
    def latitude(self) -> float:
        """Station latitude."""
        return float(self.data['Latitude'])
    
    @property
    def longitude(self) -> float:
        """Station longitude."""
        return float(self.data['Longitude'])
    
    @property
    def elevation(self) -> float:
        """Station elevation."""
        return float(self.data['Elevation'])
    
    @property
    def start_year(self) -> str:
        """Start year."""
        return str(self.data['Start'])
    
    @property
    def end_year(self) -> str:
        """End year."""
        return str(self.data['End'])
    
    @property
    def timezone_name(self) -> str:
        """Timezone name."""
        return str(self.data.get('Timezone Name', ''))
    
    @property
    def timezone_utc(self) -> str:
        """Timezone UTC offset."""
        return str(self.data.get('Timezone UTC', ''))
    
    @property
    def source(self) -> str:
        """Data source."""
        return str(self.data.get('Source', ''))
    
    @property
    def available_measurements(self) -> Dict[str, str]:
        """Available measurements."""
        measurements = {}
        measurement_types = [
            'Wind Direction', 'Wind Speed', 'Wind Gust',
            'Sea Level Pressure', 'Dry Bulb Temperature',
            'Wet Bulb Temperature', 'Relative Humidity',
            'Rain', 'Rain Intensity', 'Cloud Oktas'
        ]
        
        for col in measurement_types:
            if col in self.data.index and str(self.data[col]).lower() == 'true':
                measurements[col] = 'True'
        return measurements
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert station to dictionary."""
        return {
            'Station ID': self.id,
            'Station Name': self.name,
            'Country': self.country,
            'State': self.state,
            'Timezone': self.timezone_name,
            'Timezone Offset': self.timezone_utc,
            'Coordinate': (self.latitude, self.longitude),
            'Altitude': self.elevation,
            'Years Active': f"{self.start_year} - {self.end_year}",
            'Source': self.source,
            'Data Available': self.available_measurements
        }

class WeatherStationDatabase:
    """Class for managing weather station databases."""
    
    def __init__(self, data_type: str = 'BOM'):
        """
        Initialize the database.
        
        Parameters
        ----------
        data_type : str, optional
            Type of database. The default is 'BOM'.
        """
        self.data_type = data_type
        self._data = self._load_database()
        
    def _load_database(self) -> pd.DataFrame:
        """
        Load the station database.
        
        Returns
        -------
        pandas.DataFrame
            Station database.
        """
        # Get the path to the stations database
        current_dir = Path(__file__).resolve().parent
        
        if self.data_type == 'BOM':
            db_file = current_dir / 'src' / 'BOM_stations_clean.csv'
        elif self.data_type == 'NOAA':
            db_file = current_dir / 'src' / 'NOAA_stations.csv'
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
        
        # Check if the file exists
        if not db_file.exists():
            raise FileNotFoundError(f"Station database file not found at {db_file}")
        
        # Load database with string dtypes to preserve leading zeros
        try:
            return pd.read_csv(db_file, dtype=str)
        except Exception as e:
            raise IOError(f"Error loading station database: {e}")
    
    def get_station(self, station_id: str) -> WeatherStation:
        """
        Get a station by ID.
        
        Parameters
        ----------
        station_id : str
            Station ID.
        
        Returns
        -------
        WeatherStation
            Station object.
        
        Raises
        ------
        ValueError
            If the station is not found.
        """
        if self.data_type == 'BOM':
            station_id = str(station_id).zfill(6)
            station_data = self._data[self._data['Station Code'] == station_id]
        elif self.data_type == 'NOAA':
            station_id = str(station_id)
            station_data = self._data[self._data['Station ID'] == station_id]
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
        
        if len(station_data) == 0:
            raise ValueError(f"Station not found: {station_id}")
            
        return WeatherStation(station_data.iloc[0])
    
    def get_station_info(self, station_id: str) -> Dict[str, Any]:
        """
        Get station information in a dictionary format.
        
        Parameters
        ----------
        station_id : str
            Station ID.
        
        Returns
        -------
        Dict[str, Any]
            Station information.
        
        Raises
        ------
        ValueError
            If the station is not found.
        """
        try:
            station = self.get_station(station_id)
            return station.to_dict()
        except Exception as e:
            raise ValueError(f"Error getting station info: {e}")
    
    def get_all_stations(self) -> List[WeatherStation]:
        """
        Get all stations in the database.
        
        Returns
        -------
        List[WeatherStation]
            List of all stations.
        """
        return [WeatherStation(row) for _, row in self._data.iterrows()]
    
    def get_stations_by_type(self, data_type: str) -> List[WeatherStation]:
        """
        Get stations by data type.
        
        Parameters
        ----------
        data_type : str
            Data type to filter by.
        
        Returns
        -------
        List[WeatherStation]
            List of stations of the specified type.
        """
        if data_type not in ['BOM', 'NOAA']:
            raise ValueError(f"Unsupported data type: {data_type}")
            
        filtered_data = self._data[self._data['Source'] == data_type]
        return [WeatherStation(row) for _, row in filtered_data.iterrows()]
    
    def _haversine_distance(self, lat1: float, lat2: np.ndarray, lon1: float, lon2: np.ndarray) -> np.ndarray:
        """
        Calculate distances using Haversine formula.
        
        Parameters
        ----------
        lat1 : float
            Latitude of first point.
        lat2 : numpy.ndarray
            Latitudes of second points.
        lon1 : float
            Longitude of first point.
        lon2 : numpy.ndarray
            Longitudes of second points.
        
        Returns
        -------
        numpy.ndarray
            Distances in kilometers.
        """
        R = 6371  # Earth radius in kilometers
        
        # Convert to radians
        lat1, lat2 = np.radians(lat1), np.radians(lat2)
        lon1, lon2 = np.radians(lon1), np.radians(lon2)
        
        # Differences
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Haversine formula
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def find_stations(self,
                     city: Optional[str] = None,
                     coordinates: Optional[Tuple[float, float]] = None,
                     nearest: Optional[int] = None,
                     radius: Optional[float] = None) -> List[WeatherStation]:
        """
        Find stations based on criteria.
        
        Parameters
        ----------
        city : str, optional
            City name to search near.
        coordinates : tuple, optional
            (latitude, longitude) to search near.
        nearest : int, optional
            Number of nearest stations to return.
        radius : float, optional
            Radius in kilometers to search within.
        
        Returns
        -------
        List[WeatherStation]
            List of stations matching the criteria.
        
        Raises
        ------
        ValueError
            If invalid parameters are provided.
        """
        if city is not None:
            # Load cities database
            cities_file = Path(__file__).resolve().parent / 'src' / 'Cities_database.csv'
            if not cities_file.exists():
                raise FileNotFoundError(f"Cities database not found at {cities_file}")
                
            cities = pd.read_csv(cities_file)
            
            # Find city
            city_data = cities[cities['city'] == city]
            if len(city_data) == 0:
                raise ValueError(f"City not found: {city}")
            
            # Get coordinates
            lat = float(city_data.iloc[0]['lat'])
            lon = float(city_data.iloc[0]['lng'])
            coordinates = (lat, lon)
        
        if coordinates is None:
            raise ValueError("Must provide either city or coordinates")
        
        if nearest is None and radius is None:
            raise ValueError("Must provide either nearest or radius")
        
        # Calculate distances
        lat1, lon1 = coordinates
        distances = self._haversine_distance(
            lat1, self._data['Latitude'].astype(float),
            lon1, self._data['Longitude'].astype(float)
        )
        
        # Add distances to DataFrame
        df_copy = self._data.copy()
        df_copy['Distance (km)'] = distances
        
        # Sort by distance
        df_copy = df_copy.sort_values('Distance (km)')
        
        # Filter results
        if nearest is not None:
            results = df_copy.head(nearest)
        elif radius is not None:
            results = df_copy[df_copy['Distance (km)'] <= radius]
        else:
            results = df_copy
        
        # Convert to station objects
        return [WeatherStation(row) for _, row in results.iterrows()]
    
    def filter_stations(self,
                       country: Optional[Union[str, List[str]]] = None,
                       state: Optional[Union[str, List[str]]] = None,
                       measurement_type: Optional[Union[str, List[str]]] = None) -> List[WeatherStation]:
        """
        Filter stations by criteria.
        
        Parameters
        ----------
        country : str or list, optional
            Country or countries to filter by.
        state : str or list, optional
            State or states to filter by.
        measurement_type : str or list, optional
            Measurement type(s) to filter by.
        
        Returns
        -------
        List[WeatherStation]
            List of stations matching the criteria.
        """
        df_copy = self._data.copy()
        
        # Filter by country
        if isinstance(country, str):
            df_copy = df_copy[df_copy['Country'] == country]
        elif isinstance(country, list):
            df_copy = df_copy[df_copy['Country'].isin(country)]
        
        # Filter by state
        if isinstance(state, str):
            df_copy = df_copy[df_copy['State'] == state.upper()]
        elif isinstance(state, list):
            state_set = set([x.upper() for x in state])
            df_copy = df_copy[df_copy['State'].isin(state_set)]
        
        # Filter by measurement type
        types_dict = {
            'WD': 'Wind Direction',
            'WS': 'Wind Speed',
            'WG': 'Wind Gust',
            'SP': 'Sea Level Pressure',
            'DB': 'Dry Bulb Temperature',
            'WB': 'Wet Bulb Temperature',
            'RH': 'Relative Humidity',
            'RA': 'Rain',
            'RI': 'Rain Intensity',
            'CO': 'Cloud Oktas'
        }
        
        if isinstance(measurement_type, str):
            if measurement_type.upper() == 'ALL':
                for col in types_dict.values():
                    df_copy = df_copy[df_copy[col] == 'True']
            elif measurement_type.upper() in types_dict:
                df_copy = df_copy[df_copy[types_dict[measurement_type.upper()]] == 'True']
        elif isinstance(measurement_type, list):
            for mt in measurement_type:
                if mt.upper() in types_dict:
                    df_copy = df_copy[df_copy[types_dict[mt.upper()]] == 'True']
        
        # Convert to station objects
        return [WeatherStation(row) for _, row in df_copy.iterrows()] 