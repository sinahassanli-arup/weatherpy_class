"""
Base class for weather data operations.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union

class WeatherData:
    """Base class for weather data operations."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with weather data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Weather data DataFrame
        """
        self.data = data.copy()
        
    def get_data(self) -> pd.DataFrame:
        """
        Get the data DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Weather data
        """
        return self.data
        
    def set_data(self, data: pd.DataFrame):
        """
        Set new data.
        
        Parameters
        ----------
        data : pd.DataFrame
            New weather data DataFrame
        """
        self.data = data.copy()
        
    def get_columns(self) -> List[str]:
        """
        Get column names.
        
        Returns
        -------
        List[str]
            List of column names
        """
        return list(self.data.columns)
        
    def get_shape(self) -> tuple:
        """
        Get data shape.
        
        Returns
        -------
        tuple
            (rows, columns)
        """
        return self.data.shape
        
    def get_info(self) -> str:
        """
        Get data info.
        
        Returns
        -------
        str
            DataFrame info
        """
        return str(self.data.info())
        
    def get_summary(self) -> pd.DataFrame:
        """
        Get data summary statistics.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics
        """
        return self.data.describe()
        
    def get_missing(self) -> pd.Series:
        """
        Get missing value counts.
        
        Returns
        -------
        pd.Series
            Missing value counts by column
        """
        return self.data.isnull().sum()
        
    def get_unique(self) -> Dict[str, int]:
        """
        Get unique value counts.
        
        Returns
        -------
        Dict[str, int]
            Unique value counts by column
        """
        return {col: self.data[col].nunique() for col in self.data.columns}
        
    def get_dtypes(self) -> pd.Series:
        """
        Get column data types.
        
        Returns
        -------
        pd.Series
            Data types by column
        """
        return self.data.dtypes
        
    def get_memory_usage(self) -> pd.Series:
        """
        Get memory usage.
        
        Returns
        -------
        pd.Series
            Memory usage by column
        """
        return self.data.memory_usage(deep=True)
        
    def get_sample(self, n: int = 5) -> pd.DataFrame:
        """
        Get sample rows.
        
        Parameters
        ----------
        n : int, optional
            Number of rows, by default 5
            
        Returns
        -------
        pd.DataFrame
            Sample rows
        """
        return self.data.sample(n=n)
        
    def get_head(self, n: int = 5) -> pd.DataFrame:
        """
        Get first n rows.
        
        Parameters
        ----------
        n : int, optional
            Number of rows, by default 5
            
        Returns
        -------
        pd.DataFrame
            First n rows
        """
        return self.data.head(n)
        
    def get_tail(self, n: int = 5) -> pd.DataFrame:
        """
        Get last n rows.
        
        Parameters
        ----------
        n : int, optional
            Number of rows, by default 5
            
        Returns
        -------
        pd.DataFrame
            Last n rows
        """
        return self.data.tail(n)
        
    def get_value_counts(self, column: str) -> pd.Series:
        """
        Get value counts for a column.
        
        Parameters
        ----------
        column : str
            Column name
            
        Returns
        -------
        pd.Series
            Value counts
        """
        return self.data[column].value_counts()
        
    def get_correlation(self) -> pd.DataFrame:
        """
        Get correlation matrix.
        
        Returns
        -------
        pd.DataFrame
            Correlation matrix
        """
        return self.data.corr()
        
    def get_column_stats(self, column: str) -> Dict[str, Any]:
        """
        Get detailed statistics for a column.
        
        Parameters
        ----------
        column : str
            Column name
            
        Returns
        -------
        Dict[str, Any]
            Column statistics
        """
        stats = {}
        stats['mean'] = self.data[column].mean()
        stats['median'] = self.data[column].median()
        stats['std'] = self.data[column].std()
        stats['min'] = self.data[column].min()
        stats['max'] = self.data[column].max()
        stats['missing'] = self.data[column].isnull().sum()
        stats['unique'] = self.data[column].nunique()
        return stats
        
    def filter_data(self, column: str, value: Any, operator: str = '==') -> pd.DataFrame:
        """
        Filter data based on column value.
        
        Parameters
        ----------
        column : str
            Column name
        value : Any
            Filter value
        operator : str, optional
            Comparison operator, by default '=='
            
        Returns
        -------
        pd.DataFrame
            Filtered data
        """
        if operator == '==':
            return self.data[self.data[column] == value]
        elif operator == '!=':
            return self.data[self.data[column] != value]
        elif operator == '>':
            return self.data[self.data[column] > value]
        elif operator == '<':
            return self.data[self.data[column] < value]
        elif operator == '>=':
            return self.data[self.data[column] >= value]
        elif operator == '<=':
            return self.data[self.data[column] <= value]
        else:
            raise ValueError(f"Unsupported operator: {operator}")
            
    def sort_data(self, column: str, ascending: bool = True) -> pd.DataFrame:
        """
        Sort data by column.
        
        Parameters
        ----------
        column : str
            Column name
        ascending : bool, optional
            Sort order, by default True
            
        Returns
        -------
        pd.DataFrame
            Sorted data
        """
        return self.data.sort_values(column, ascending=ascending)
        
    def group_by(self, column: str) -> pd.DataFrame:
        """
        Group data by column.
        
        Parameters
        ----------
        column : str
            Column name
            
        Returns
        -------
        pd.DataFrame
            Grouped data
        """
        return self.data.groupby(column).agg(['mean', 'std', 'count'])
        
    def resample_data(self, freq: str, agg_func: str = 'mean') -> pd.DataFrame:
        """
        Resample time series data.
        
        Parameters
        ----------
        freq : str
            Resampling frequency (e.g., 'H' for hourly)
        agg_func : str, optional
            Aggregation function, by default 'mean'
            
        Returns
        -------
        pd.DataFrame
            Resampled data
        """
        if 'UTC' not in self.data.columns:
            raise ValueError("UTC column not found for resampling")
            
        self.data['UTC'] = pd.to_datetime(self.data['UTC'])
        self.data.set_index('UTC', inplace=True)
        resampled = self.data.resample(freq).agg(agg_func)
        self.data.reset_index(inplace=True)
        return resampled
        
    def interpolate_missing(self, method: str = 'linear') -> pd.DataFrame:
        """
        Interpolate missing values.
        
        Parameters
        ----------
        method : str, optional
            Interpolation method, by default 'linear'
            
        Returns
        -------
        pd.DataFrame
            Data with interpolated values
        """
        return self.data.interpolate(method=method)
        
    def drop_missing(self, threshold: float = 0.5) -> pd.DataFrame:
        """
        Drop columns with missing values above threshold.
        
        Parameters
        ----------
        threshold : float, optional
            Missing value threshold, by default 0.5
            
        Returns
        -------
        pd.DataFrame
            Data with dropped columns
        """
        missing_ratio = self.data.isnull().sum() / len(self.data)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index
        return self.data.drop(columns=cols_to_drop)
        
    def fill_missing(self, value: Any = 0) -> pd.DataFrame:
        """
        Fill missing values.
        
        Parameters
        ----------
        value : Any, optional
            Fill value, by default 0
            
        Returns
        -------
        pd.DataFrame
            Data with filled values
        """
        return self.data.fillna(value)
        
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Returns
        -------
        pd.DataFrame
            Data without duplicates
        """
        return self.data.drop_duplicates()
        
    def scale_column(self, column: str, method: str = 'minmax') -> pd.DataFrame:
        """
        Scale column values.
        
        Parameters
        ----------
        column : str
            Column name
        method : str, optional
            Scaling method ('minmax' or 'standard'), by default 'minmax'
            
        Returns
        -------
        pd.DataFrame
            Data with scaled column
        """
        if method == 'minmax':
            min_val = self.data[column].min()
            max_val = self.data[column].max()
            self.data[f"{column}_scaled"] = (self.data[column] - min_val) / (max_val - min_val)
        elif method == 'standard':
            mean_val = self.data[column].mean()
            std_val = self.data[column].std()
            self.data[f"{column}_scaled"] = (self.data[column] - mean_val) / std_val
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
        return self.data
        
    def bin_column(self, column: str, bins: int = 10) -> pd.DataFrame:
        """
        Bin column values.
        
        Parameters
        ----------
        column : str
            Column name
        bins : int, optional
            Number of bins, by default 10
            
        Returns
        -------
        pd.DataFrame
            Data with binned column
        """
        self.data[f"{column}_binned"] = pd.qcut(self.data[column], bins, labels=False)
        return self.data
        
    def encode_categorical(self, column: str, method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical column.
        
        Parameters
        ----------
        column : str
            Column name
        method : str, optional
            Encoding method ('onehot' or 'label'), by default 'onehot'
            
        Returns
        -------
        pd.DataFrame
            Data with encoded column
        """
        if method == 'onehot':
            encoded = pd.get_dummies(self.data[column], prefix=column)
            self.data = pd.concat([self.data, encoded], axis=1)
        elif method == 'label':
            self.data[f"{column}_encoded"] = self.data[column].astype('category').cat.codes
        else:
            raise ValueError(f"Unsupported encoding method: {method}")
        return self.data
        
    def add_datetime_features(self, column: str = 'UTC') -> pd.DataFrame:
        """
        Add datetime features.
        
        Parameters
        ----------
        column : str, optional
            Datetime column name, by default 'UTC'
            
        Returns
        -------
        pd.DataFrame
            Data with datetime features
        """
        if column not in self.data.columns:
            raise ValueError(f"Column not found: {column}")
            
        self.data[column] = pd.to_datetime(self.data[column])
        self.data[f"{column}_year"] = self.data[column].dt.year
        self.data[f"{column}_month"] = self.data[column].dt.month
        self.data[f"{column}_day"] = self.data[column].dt.day
        self.data[f"{column}_hour"] = self.data[column].dt.hour
        self.data[f"{column}_minute"] = self.data[column].dt.minute
        self.data[f"{column}_dayofweek"] = self.data[column].dt.dayofweek
        self.data[f"{column}_quarter"] = self.data[column].dt.quarter
        self.data[f"{column}_is_month_start"] = self.data[column].dt.is_month_start
        self.data[f"{column}_is_month_end"] = self.data[column].dt.is_month_end
        return self.data
        
    def add_rolling_features(self, column: str, window: int = 3) -> pd.DataFrame:
        """
        Add rolling window features.
        
        Parameters
        ----------
        column : str
            Column name
        window : int, optional
            Window size, by default 3
            
        Returns
        -------
        pd.DataFrame
            Data with rolling features
        """
        self.data[f"{column}_rolling_mean"] = self.data[column].rolling(window=window).mean()
        self.data[f"{column}_rolling_std"] = self.data[column].rolling(window=window).std()
        self.data[f"{column}_rolling_min"] = self.data[column].rolling(window=window).min()
        self.data[f"{column}_rolling_max"] = self.data[column].rolling(window=window).max()
        return self.data
        
    def add_lag_features(self, column: str, lags: List[int]) -> pd.DataFrame:
        """
        Add lagged features.
        
        Parameters
        ----------
        column : str
            Column name
        lags : List[int]
            List of lag periods
            
        Returns
        -------
        pd.DataFrame
            Data with lagged features
        """
        for lag in lags:
            self.data[f"{column}_lag_{lag}"] = self.data[column].shift(lag)
        return self.data
        
    def add_diff_features(self, column: str, periods: List[int]) -> pd.DataFrame:
        """
        Add difference features.
        
        Parameters
        ----------
        column : str
            Column name
        periods : List[int]
            List of difference periods
            
        Returns
        -------
        pd.DataFrame
            Data with difference features
        """
        for period in periods:
            self.data[f"{column}_diff_{period}"] = self.data[column].diff(period)
        return self.data
        
    def add_pct_change_features(self, column: str, periods: List[int]) -> pd.DataFrame:
        """
        Add percentage change features.
        
        Parameters
        ----------
        column : str
            Column name
        periods : List[int]
            List of periods
            
        Returns
        -------
        pd.DataFrame
            Data with percentage change features
        """
        for period in periods:
            self.data[f"{column}_pct_change_{period}"] = self.data[column].pct_change(period)
        return self.data 