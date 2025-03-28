"""
Fix for WeatherDataCleaner error in the examples/compare.data.ipynb notebook.

This script demonstrates the correct way to use the cleaner classes.
"""

import sys
import logging
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import weatherpy classes
from weatherpy.data.wd_importer import BOMWeatherDataImporter, NOAAWeatherDataImporter
from weatherpy.data.wd_unifier import BOMWeatherDataUnifier, NOAAWeatherDataUnifier
from weatherpy.data.wd_cleaner import BOMDataCleaner, NOAADataCleaner
from weatherpy.data.wd_base import WeatherData


def process_bom_data():
    """
    Example of processing BOM weather data using explicit classes.
    """
    logger.info("=== Processing BOM Weather Data ===")
    
    # Step 1: Import data using BOMWeatherDataImporter
    station_id = '066037'  # Example BOM station ID (Wyndham Aero)
    
    logger.info(f"Importing BOM data for station {station_id}")
    importer = BOMWeatherDataImporter(
        station_id=station_id,
        year_start=2019,
        year_end=2020,
        time_zone='LocalTime',
        interval=60  # 60-minute data
    )
    
    # Import the data
    weather_data = importer.import_data()
    logger.info(f"Imported data shape: {weather_data.data.shape}")
    logger.info(f"Imported data columns: {weather_data.data.columns.tolist()}")
    
    # Step 2: Unify data using BOMWeatherDataUnifier
    logger.info("Unifying BOM data")
    unifier = BOMWeatherDataUnifier()
    unified_data = unifier.unify_data(weather_data)
    logger.info(f"Unified data shape: {unified_data.data.shape}")
    logger.info(f"Unified data columns: {unified_data.data.columns.tolist()}")
    
    # Step 3: Clean data using BOMDataCleaner
    # FIXED: Use BOMDataCleaner instead of WeatherDataCleaner, and pass unified_data to constructor
    logger.info("Cleaning BOM data")
    cleaner = BOMDataCleaner(unified_data)  # Pass the WeatherData object
    cleaned_data = cleaner.clean_data(inplace=False)  # Call clean_data not clean
    logger.info(f"Cleaned data shape: {cleaned_data.data.shape}")
    
    # Display operations log
    logger.info("Operations performed on BOM data:")
    for op in cleaned_data.operations_log:
        logger.info(f"  - {op}")
    
    return cleaned_data


def process_noaa_data():
    """
    Example of processing NOAA weather data using explicit classes.
    """
    logger.info("\n=== Processing NOAA Weather Data ===")
    
    # Step 1: Import data using NOAAWeatherDataImporter
    station_id = '72509014739'  # Example NOAA station ID
    
    logger.info(f"Importing NOAA data for station {station_id}")
    importer = NOAAWeatherDataImporter(
        station_id=station_id,
        year_start=2019,
        year_end=2020,
        time_zone='UTC'
    )
    
    # Import the data
    weather_data = importer.import_data()
    logger.info(f"Imported data shape: {weather_data.data.shape}")
    logger.info(f"Imported data columns: {weather_data.data.columns.tolist()}")
    
    # Step 2: Unify data using NOAAWeatherDataUnifier
    logger.info("Unifying NOAA data")
    unifier = NOAAWeatherDataUnifier()
    unified_data = unifier.unify_data(weather_data)
    logger.info(f"Unified data shape: {unified_data.data.shape}")
    logger.info(f"Unified data columns: {unified_data.data.columns.tolist()}")
    
    # Step 3: Clean data using NOAADataCleaner
    # FIXED: Use NOAADataCleaner instead of WeatherDataCleaner, and pass unified_data to constructor
    logger.info("Cleaning NOAA data")
    cleaner = NOAADataCleaner(unified_data)  # Pass the WeatherData object
    cleaned_data = cleaner.clean_data(inplace=False)  # Call clean_data not clean
    logger.info(f"Cleaned data shape: {cleaned_data.data.shape}")
    
    # Display operations log
    logger.info("Operations performed on NOAA data:")
    for op in cleaned_data.operations_log:
        logger.info(f"  - {op}")
    
    return cleaned_data


def process_data_with_base_class():
    """
    Example of processing weather data using the WeatherData base class.
    This demonstrates the simplified API that chains operations.
    """
    logger.info("\n=== Processing Weather Data with Base Class ===")
    
    # Initialize WeatherData class
    wd = WeatherData()
    
    # Import, unify, and clean BOM data in a chain
    logger.info("Processing BOM data with WeatherData class")
    bom_data = wd.import_data(
        station_id='066037',
        year_start=2019,
        year_end=2020
    ).unify(
        inplace=True
    ).clean(
        inplace=True
    )
    
    logger.info(f"Processed BOM data shape: {bom_data.data.shape}")
    
    # Import, unify, and clean NOAA data in a chain
    logger.info("Processing NOAA data with WeatherData class")
    noaa_data = wd.import_data(
        station_id='72509014739',
        data_type='NOAA',
        year_start=2019,
        year_end=2020,
        time_zone='UTC'
    ).unify(
        inplace=True
    ).clean(
        inplace=True
    )
    
    logger.info(f"Processed NOAA data shape: {noaa_data.data.shape}")
    
    # Display operations log
    logger.info("Operations performed on NOAA data:")
    for op in noaa_data.operations_log:
        logger.info(f"  - {op}")
    
    return bom_data, noaa_data


def compare_methods(bom_explicit, bom_base, noaa_explicit, noaa_base):
    """
    Compare results from explicit class usage vs. base class usage.
    """
    logger.info("\n=== Comparing Methods ===")
    
    # Compare BOM data
    logger.info("Comparing BOM data processing methods:")
    bom_explicit_cols = set(bom_explicit.data.columns)
    bom_base_cols = set(bom_base.data.columns)
    
    logger.info(f"Explicit method columns: {len(bom_explicit_cols)}")
    logger.info(f"Base class method columns: {len(bom_base_cols)}")
    logger.info(f"Column differences: {bom_explicit_cols.symmetric_difference(bom_base_cols)}")
    
    # Compare NOAA data
    logger.info("Comparing NOAA data processing methods:")
    noaa_explicit_cols = set(noaa_explicit.data.columns)
    noaa_base_cols = set(noaa_base.data.columns)
    
    logger.info(f"Explicit method columns: {len(noaa_explicit_cols)}")
    logger.info(f"Base class method columns: {len(noaa_base_cols)}")
    logger.info(f"Column differences: {noaa_explicit_cols.symmetric_difference(noaa_base_cols)}")


if __name__ == "__main__":
    try:
        # Process data using explicit classes
        bom_explicit = process_bom_data()
        noaa_explicit = process_noaa_data()
        
        # Process data using base class
        bom_base, noaa_base = process_data_with_base_class()
        
        # Compare methods
        compare_methods(bom_explicit, bom_base, noaa_explicit, noaa_base)
        
        logger.info("\nAll processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True) 