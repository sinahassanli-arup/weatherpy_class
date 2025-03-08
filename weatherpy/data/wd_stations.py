"""
Module for managing weather station data.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict

class WeatherStation:
    """Class representing a single weather station."""
    
    def __init__(self, data: pd.Series):
        """Initialize station with data."""
        self.data = data
        
    @property
    def code(self) -> str:
        """Station code."""
        return str(self.data['Station Code'])
    
    @property
    def name(self) -> str:
        """Station name."""
        return str(self.data['Station Name'])
    
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
    def available_measurements(self) -> Dict[str, str]:
        """Available measurements."""
        measurements = {}
        for col in self.data.index:
            if col in ['Wind Direction', 'Wind Speed', 'Wind Gust',
                      'Sea Level Pressure', 'Dry Bulb Temperature',
                      'Wet Bulb Temperature', 'Relative Humidity',
                      'Rain', 'Rain Intensity', 'Cloud Oktas']:
                if str(self.data[col]).lower() == 'true':
                    measurements[col] = 'True'
        return measurements

class WeatherStationDatabase:
    """Class for managing weather station databases."""
    
    def __init__(self, data_type: str = 'BOM'):
        """
        Initialize database.
        
        Parameters
        ----------
        data_type : str
            'BOM' or 'NOAA'
        """
        self.data_type = data_type.upper()
        self._load_database()
    
    def _load_database(self):
        """Load the station database."""
        # Get the path to the data directory
        current_dir = Path(__file__).resolve().parent
        data_dir = current_dir / 'src'
        
        # Choose database file
        if self.data_type == 'BOM':
            db_file = data_dir / 'BOM_stations_clean.csv'
            self.id_length = 6
        elif self.data_type == 'NOAA':
            db_file = data_dir / 'NOAA_stations_full.csv'
            self.id_length = 11
        else:
            raise ValueError("data_type must be 'BOM' or 'NOAA'")
        
        # Load database
        self.stations = pd.read_csv(db_file)
        
        # Pad station codes
        self.stations['Station Code'] = self.stations['Station Code'].astype(str).str.zfill(self.id_length)
    
    def get_station(self, station_id: str) -> WeatherStation:
        """Get station by ID."""
        # Pad ID if needed
        station_id = str(station_id).zfill(self.id_length)
        
        # Find station
        station_data = self.stations[self.stations['Station Code'] == station_id]
        if len(station_data) == 0:
            raise ValueError(f"Station not found: {station_id}")
        
        return WeatherStation(station_data.iloc[0])
    
    def find_stations(self,
                     city: Optional[str] = None,
                     coordinates: Optional[Tuple[float, float]] = None,
                     nearest: Optional[int] = None,
                     radius: Optional[float] = None) -> List[WeatherStation]:
        """Find stations based on criteria."""
        if city is not None:
            # Load cities database
            cities_file = Path(__file__).resolve().parent / 'src' / 'Cities_database.csv'
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
        
        # Calculate distances
        lat1, lon1 = coordinates
        distances = self._haversine_distance(
            lat1, self.stations['Latitude'].astype(float),
            lon1, self.stations['Longitude'].astype(float)
        )
        
        # Add distances to DataFrame
        self.stations = self.stations.copy()
        self.stations['Distance (km)'] = distances
        
        # Sort by distance
        self.stations = self.stations.sort_values('Distance (km)')
        
        # Filter results
        if nearest is not None:
            results = self.stations.head(nearest)
        elif radius is not None:
            results = self.stations[self.stations['Distance (km)'] <= radius]
        else:
            results = self.stations
        
        # Convert to station objects
        return [WeatherStation(row) for _, row in results.iterrows()]
    
    def _haversine_distance(self, lat1: float, lat2: np.ndarray, lon1: float, lon2: np.ndarray) -> np.ndarray:
        """Calculate distances using Haversine formula."""
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