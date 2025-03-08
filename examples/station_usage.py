"""
Basic examples of using the WeatherStationDatabase class.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from weatherpy.data.wd_stations import WeatherStationDatabase

def main():
    try:
        # Initialize BOM database
        db = WeatherStationDatabase('BOM')
        
        # Example 1: Find BOM stations near Sydney
        print("\nExample 1: Finding BOM stations near Sydney")
        print("-" * 50)
        stations = db.find_stations(city='Sydney', nearest=3)
        for station in stations:
            print(f"Station: {station.name} ({station.code})")
            print(f"Location: ({station.latitude}, {station.longitude})")
            print(f"Years: {station.start_year} - {station.end_year}")
            print()
            
        # Example 2: Find stations within 50km of Melbourne coordinates
        print("\nExample 2: Finding stations within 50km of Melbourne")
        print("-" * 50)
        melbourne_coords = (-37.8136, 144.9631)  # Melbourne coordinates
        stations = db.find_stations(coordinates=melbourne_coords, radius=50)
        for station in stations:
            print(f"Station: {station.name} ({station.code})")
            print(f"Location: ({station.latitude}, {station.longitude})")
            print()
            
        # Example 3: Get details of a specific station
        print("\nExample 3: Getting details of Sydney Observatory Hill")
        print("-" * 50)
        station = db.get_station('066062')  # Sydney Observatory Hill
        print(f"Station: {station.name} ({station.code})")
        print(f"Location: ({station.latitude}, {station.longitude})")
        print(f"Elevation: {station.elevation}m")
        print("\nAvailable measurements:")
        for measurement, available in station.available_measurements.items():
            print(f"- {measurement}")
            
    except Exception as e:
        import traceback
        print(f"\nError: {str(e)}")
        print("\nStack trace:")
        print(traceback.format_exc())

if __name__ == '__main__':
    main() 