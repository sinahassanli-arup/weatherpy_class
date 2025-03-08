"""
Examples demonstrating basic usage of weather station search functionality.
Shows both legacy and OOP approaches for each search type.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import weatherpy
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from weatherpy.data.stations import find_stations, station_info
    from weatherpy.data.wd_stations import WeatherStationDatabase
except ImportError as e:
    print(f"Error importing weatherpy modules: {e}")
    print("Make sure you're running this script from the correct directory")
    sys.exit(1)

def example_city_search():
    """Example of searching stations near a city."""
    print("\n=== Finding Stations Near a City ===")
    
    try:
        # Legacy approach
        print("\nLegacy Method:")
        legacy_stations = find_stations(
            city='Sydney',
            nearest=3,
            printOutput=True
        )
        
        # OOP approach
        print("\nOOP Method:")
        db = WeatherStationDatabase('BOM')
        stations = db.find_stations(city='Sydney', nearest=3)
        
        print("\nOOP Method Results:")
        for station in stations:
            print(f"\nStation: {station.name}")
            print(f"Location: ({station.latitude}, {station.longitude})")
    except Exception as e:
        print(f"Error in city search example: {e}")

def example_coordinate_search():
    """Example of searching stations near coordinates."""
    print("\n=== Finding Stations Near Coordinates ===")
    
    try:
        sydney_coords = (-33.8688, 151.2093)
        
        # Legacy approach
        print("\nLegacy Method:")
        legacy_stations = find_stations(
            coord=sydney_coords,
            radius=50,
            printOutput=True
        )
        
        # OOP approach
        print("\nOOP Method:")
        db = WeatherStationDatabase('BOM')
        stations = db.find_stations(coordinates=sydney_coords, radius=50)
        
        print("\nOOP Method Results:")
        for station in stations[:5]:  # Show first 5 stations
            print(f"\nStation: {station.name}")
            print(f"Distance: {station.data.get('Distance (km)', 'N/A')} km")
    except Exception as e:
        print(f"Error in coordinate search example: {e}")

def example_station_info():
    """Example of getting detailed station information."""
    print("\n=== Getting Station Information ===")
    
    try:
        station_id = '066062'  # Sydney Observatory Hill
        
        # Legacy approach
        print("\nLegacy Method:")
        info = station_info(station_id, printed=True)
        
        # OOP approach
        print("\nOOP Method:")
        db = WeatherStationDatabase('BOM')
        station = db.get_station(station_id)
        print(station)
    except Exception as e:
        print(f"Error in station info example: {e}")

def example_noaa_search():
    """Example of searching NOAA stations."""
    print("\n=== Searching NOAA Stations ===")
    
    try:
        # Legacy approach
        print("\nLegacy Method:")
        legacy_stations = find_stations(
            city='New York',
            nearest=3,
            dataType='NOAA',
            printOutput=True
        )
        
        # OOP approach
        print("\nOOP Method:")
        db = WeatherStationDatabase('NOAA')
        stations = db.find_stations(city='New York', nearest=3)
        
        print("\nOOP Method Results:")
        for station in stations:
            print(f"\nStation: {station.name}")
            print(f"ID: {station.code}")
            print(f"Years: {station.start_year} - {station.end_year}")
    except Exception as e:
        print(f"Error in NOAA search example: {e}")

def example_filter_stations():
    """Example of filtering stations by measurements."""
    print("\n=== Filtering Stations by Measurements ===")
    
    try:
        # OOP approach with filtering
        db = WeatherStationDatabase('BOM')
        stations = db.find_stations(city='Melbourne', nearest=10)
        
        # Filter for stations with wind and temperature measurements
        wind_temp_stations = [
            station for station in stations
            if ('Wind Speed' in station.available_measurements and
                'Dry Bulb Temperature' in station.available_measurements)
        ]
        
        print(f"\nFound {len(wind_temp_stations)} stations with wind and temperature data:")
        for station in wind_temp_stations:
            print(f"\nStation: {station.name}")
            print("Available measurements:")
            for measurement in station.available_measurements:
                print(f"- {measurement}")
    except Exception as e:
        print(f"Error in filter stations example: {e}")

def main():
    """Run all examples."""
    print("Weather Station Search Examples")
    print("==============================")
    
    try:
        example_city_search()
        example_coordinate_search()
        example_station_info()
        example_noaa_search()
        example_filter_stations()
        
        print("\nAll examples completed successfully!")
    except Exception as e:
        print(f"\nError running examples: {e}")

if __name__ == '__main__':
    main() 