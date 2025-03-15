"""
Base class for weather data operations.
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from .wd_stations import WeatherStation

class WeatherData:
    """Base class for weather data operations."""
    
    def __init__(self, data: pd.DataFrame, station: Optional[WeatherStation] = None, 
                 data_type: Optional[str] = None, interval: Optional[int] = None):
        """
        Initialize with weather data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Weather data DataFrame
        station : Optional[WeatherStation], optional
            Weather station information, by default None
        data_type : Optional[str], optional
            Type of data source (e.g., 'BOM', 'NOAA'), by default None
        interval : Optional[int], optional
            Data recording interval in minutes (1, 10, 30, 60), by default None
        """
        self._data = data.copy()  # Store as protected attribute
        self._station = station
        self._data_type = data_type  # Data source type (BOM, NOAA, etc.)
        self._interval = interval  # Data interval in minutes
        
        # Initialize data-driven attributes
        self._start_date = None
        self._end_date = None
        self._shape = (0, 0)  # (rows, columns)
        self._columns = []
        self._summary = None
        
        # Initialize operation log
        self._operations_log = []
        self._log_operation("Initialize", {"data_type": data_type, "interval": interval})
        
        # Update all data-driven attributes
        self._update_data_attributes()
    
    def _update_data_attributes(self):
        """
        Update all data-driven attributes based on current data.
        
        This method updates all attributes that depend on the data's current state.
        When adding new data-driven attributes to the class, update them here.
        """
        # Basic data properties
        self._shape = self._data.shape
        self._columns = list(self._data.columns)
        
        # Date range (if applicable)
        if not self._data.empty and self._data.index.dtype.kind == 'M':  # Check if index is datetime
            self._start_date = self._data.index.min()
            self._end_date = self._data.index.max()
        else:
            self._start_date = None
            self._end_date = None
        
        # Data summary statistics (can be expensive, so only compute if needed)
        self._summary = None
    
    def _log_operation(self, operation_name: str, parameters: Dict[str, Any] = None):
        """
        Log an operation applied to the data.
        
        Parameters
        ----------
        operation_name : str
            Name of the operation
        parameters : Dict[str, Any], optional
            Parameters used in the operation, by default None
        """
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "operation": operation_name,
            "parameters": parameters or {}
        }
        self._operations_log.append(log_entry)
    
    @property
    def operations_log(self) -> List[Dict[str, Any]]:
        """
        Get the operations log.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of logged operations
        """
        return self._operations_log.copy()  # Return a copy to prevent modification
    
    @property
    def data(self) -> pd.DataFrame:
        """
        Get the weather data DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Weather data
        """
        return self._data
    
    @data.setter
    def data(self, new_data: pd.DataFrame):
        """
        Set the weather data DataFrame and update related attributes.
        
        Parameters
        ----------
        new_data : pd.DataFrame
            New weather data DataFrame
        """
        self._data = new_data.copy()
        self._log_operation("UpdateData", {"shape": new_data.shape})
        self._update_data_attributes()  # Update all data-driven attributes
    
    @property
    def data_type(self) -> Optional[str]:
        """
        Get the data source type.
        
        Returns
        -------
        Optional[str]
            Data source type (BOM, NOAA, etc.)
        """
        return self._data_type
    
    @data_type.setter
    def data_type(self, data_type_value: str):
        """
        Set the data source type.
        
        Parameters
        ----------
        data_type_value : str
            Data source type (BOM, NOAA, etc.)
        """
        self._data_type = data_type_value
        self._log_operation("UpdateDataType", {"data_type": data_type_value})
    
    @property
    def interval(self) -> Optional[int]:
        """
        Get the data recording interval in minutes.
        
        Returns
        -------
        Optional[int]
            Data interval in minutes (1, 10, 30, 60)
        """
        return self._interval
    
    @interval.setter
    def interval(self, interval_value: int):
        """
        Set the data recording interval.
        
        Parameters
        ----------
        interval_value : int
            Data interval in minutes (1, 10, 30, 60)
        """
        valid_intervals = [1, 10, 30, 60]
        if interval_value not in valid_intervals:
            logging.warning(f"Unusual interval value: {interval_value}. Expected one of {valid_intervals}")
        self._interval = interval_value
        self._log_operation("UpdateInterval", {"interval": interval_value})
        
    @property
    def station(self) -> Optional[WeatherStation]:
        """
        Get the weather station information.
        
        This property allows access to the station object while keeping it protected.
        Access station attributes like: weather_data.station.longitude
        
        Returns
        -------
        Optional[WeatherStation]
            Weather station information
        """
        return self._station
        
    @station.setter
    def station(self, station: WeatherStation):
        """
        Set the weather station information.
        
        Parameters
        ----------
        station : WeatherStation
            Weather station information
        """
        self._station = station
        self._log_operation("UpdateStation", {"station_id": station.id if station else None})
    
    @property
    def start_date(self) -> Optional[pd.Timestamp]:
        """
        Get the start date of the data (not the station).
        
        Returns
        -------
        Optional[pd.Timestamp]
            Start date of the data
        """
        return self._start_date
    
    @property
    def end_date(self) -> Optional[pd.Timestamp]:
        """
        Get the end date of the data (not the station).
        
        Returns
        -------
        Optional[pd.Timestamp]
            End date of the data
        """
        return self._end_date
    
    @property
    def shape(self) -> Tuple[int, int]:
        """
        Get the shape of the data as (rows, columns).
        
        Returns
        -------
        Tuple[int, int]
            Data shape as (rows, columns)
        """
        return self._shape
    
    @property
    def row_count(self) -> int:
        """
        Get the number of rows in the data.
        
        Returns
        -------
        int
            Number of rows
        """
        return self._shape[0]
    
    @property
    def column_count(self) -> int:
        """
        Get the number of columns in the data.
        
        Returns
        -------
        int
            Number of columns
        """
        return self._shape[1]
    
    @property
    def columns(self) -> List[str]:
        """
        Get the column names of the data.
        
        Returns
        -------
        List[str]
            Column names
        """
        return self._columns
    
    @property
    def summary(self) -> pd.DataFrame:
        """
        Get summary statistics for the data.
        
        This property computes the summary on demand to avoid unnecessary computation.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        if self._summary is None:
            self._summary = self._data.describe()
        return self._summary
        
    def get_station_info(self) -> Dict[str, Any]:
        """
        Get station information as a dictionary.
        
        Returns
        -------
        Dict[str, Any]
            Station information
        
        Raises
        ------
        ValueError
            If no station information is available
        """
        if self._station is None:
            raise ValueError("No station information is available")
        
        return self._station.to_dict()
    
    def save(self, filepath: str, format: str = 'csv', save_metadata: bool = True) -> str:
        """
        Save the weather data and metadata to files.
        
        Parameters
        ----------
        filepath : str
            Path to save the data file
        format : str, optional
            File format ('csv', 'excel', 'parquet', 'hdf', 'pickle'), by default 'csv'
        save_metadata : bool, optional
            Whether to save metadata in a separate file, by default True
            
        Returns
        -------
        str
            Path to the saved data file
            
        Raises
        ------
        ValueError
            If the format is not supported
        """
        # Create directory if it doesn't exist
        directory = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(directory, exist_ok=True)
        
        # Log the save operation
        self._log_operation("Save", {"filepath": filepath, "format": format})
        
        # Save data in the specified format
        format = format.lower()
        if format == 'csv':
            self._data.to_csv(filepath, index=True)
        elif format == 'excel':
            self._data.to_excel(filepath, index=True)
        elif format == 'parquet':
            self._data.to_parquet(filepath, index=True)
        elif format == 'hdf':
            self._data.to_hdf(filepath, key='weather_data', index=True)
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            # If we save the entire object as pickle, we don't need separate metadata
            save_metadata = False
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv', 'excel', 'parquet', 'hdf', or 'pickle'.")
        
        # Save metadata if requested (and if not saving as pickle)
        if save_metadata:
            metadata_filepath = self._get_metadata_filepath(filepath)
            self._save_metadata(metadata_filepath)
        
        logging.info(f"Weather data saved to {filepath}")
        if save_metadata:
            logging.info(f"Metadata saved to {self._get_metadata_filepath(filepath)}")
        
        return filepath
    
    def _get_metadata_filepath(self, data_filepath: str) -> str:
        """
        Get the filepath for metadata based on the data filepath.
        
        Parameters
        ----------
        data_filepath : str
            Path to the data file
            
        Returns
        -------
        str
            Path to the metadata file
        """
        base, ext = os.path.splitext(data_filepath)
        return f"{base}_metadata.json"
    
    def _save_metadata(self, metadata_filepath: str):
        """
        Save metadata to a JSON file.
        
        Parameters
        ----------
        metadata_filepath : str
            Path to save the metadata
        """
        metadata = {
            "data_type": self._data_type,
            "interval": self._interval,
            "start_date": self._start_date.isoformat() if self._start_date else None,
            "end_date": self._end_date.isoformat() if self._end_date else None,
            "shape": self._shape,
            "columns": self._columns,
            "operations_log": self._operations_log
        }
        
        # Add station information if available
        if self._station:
            metadata["station"] = self._station.to_dict()
        
        # Save metadata as JSON
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str, load_metadata: bool = True) -> 'WeatherData':
        """
        Load weather data and metadata from files.
        
        Parameters
        ----------
        filepath : str
            Path to the data file
        load_metadata : bool, optional
            Whether to load metadata from a separate file, by default True
            
        Returns
        -------
        WeatherData
            Loaded weather data object
            
        Raises
        ------
        FileNotFoundError
            If the file does not exist
        ValueError
            If the file format is not supported
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Check if it's a pickle file (contains the entire object)
        if filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
            if not isinstance(obj, cls):
                raise ValueError(f"Loaded object is not a {cls.__name__} instance")
            obj._log_operation("Load", {"filepath": filepath})
            return obj
        
        # Load data based on file extension
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.csv':
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif ext in ['.xls', '.xlsx']:
            data = pd.read_excel(filepath, index_col=0, parse_dates=True)
        elif ext == '.parquet':
            data = pd.read_parquet(filepath)
        elif ext in ['.h5', '.hdf', '.hdf5']:
            data = pd.read_hdf(filepath, key='weather_data')
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        # Create a new instance with just the data
        obj = cls(data)
        obj._log_operation("Load", {"filepath": filepath})
        
        # Load metadata if requested
        if load_metadata:
            metadata_filepath = obj._get_metadata_filepath(filepath)
            if os.path.exists(metadata_filepath):
                obj._load_metadata(metadata_filepath)
        
        return obj
    
    def _load_metadata(self, metadata_filepath: str):
        """
        Load metadata from a JSON file.
        
        Parameters
        ----------
        metadata_filepath : str
            Path to the metadata file
        """
        try:
            with open(metadata_filepath, 'r') as f:
                metadata = json.load(f)
            
            # Set basic attributes
            self._data_type = metadata.get("data_type")
            self._interval = metadata.get("interval")
            
            # Set operations log
            self._operations_log = metadata.get("operations_log", [])
            
            # Load station information if available
            if "station" in metadata and metadata["station"]:
                from .wd_stations import WeatherStation
                self._station = WeatherStation.from_dict(metadata["station"])
            
            # Log the metadata loading
            self._log_operation("LoadMetadata", {"filepath": metadata_filepath})
            
        except Exception as e:
            logging.warning(f"Error loading metadata: {e}")
    
    def unify(self, additional_columns: Optional[List[str]] = None) -> 'WeatherData':
        """
        Unify this weather data by selecting specific columns.
        
        This method uses WeatherDataUnifier to select columns from the data
        based on a standard list of weather data columns.
        
        Parameters
        ----------
        additional_columns : Optional[List[str]], optional
            Additional columns to include beyond the standard columns, by default None
            
        Returns
        -------
        WeatherData
            Weather data object with selected columns
            
        Raises
        ------
        ImportError
            If WeatherDataUnifier is not available
        """
        try:
            from .wd_unifier import WeatherDataUnifier
        except ImportError:
            raise ImportError("WeatherDataUnifier is not available. Make sure wd_unifier.py is in the same package.")
        
        # Store original columns for logging
        original_cols = list(self._data.columns)
        
        # Create unifier and apply unification
        unifier = WeatherDataUnifier()
        unifier.unify(self, additional_columns=additional_columns)
        
        # Log the operation
        self._log_operation("Unify", {
            "columns_before": original_cols,
            "columns_after": list(self._data.columns)
        })
        
        return self