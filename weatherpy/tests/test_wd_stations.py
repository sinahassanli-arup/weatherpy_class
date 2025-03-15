import unittest
import pandas as pd
from weatherpy.data.wd_stations import WeatherStation, WeatherStationDatabase

class TestWeatherStation(unittest.TestCase):
    def setUp(self):
        # Sample data for a weather station
        self.sample_data = pd.Series({
            'Station ID': '000001',
            'Station Name': 'Test Station',
            'Country': 'Testland',
            'State': 'TS',
            'Latitude': '12.34',
            'Longitude': '56.78',
            'Elevation': '100',
            'Start': '2000',
            'End': '2020',
            'Timezone Name': 'Test/Timezone',
            'Timezone UTC': '+00:00',
            'Source': 'BOM',
            'Mean Correction Factor': {'Temperature': 1.0},
            'Wind Direction': 'True',
            'Wind Speed': 'True'
        })
        self.station = WeatherStation(self.sample_data)

    def test_station_attributes(self):
        self.assertEqual(self.station.id, '000001')
        self.assertEqual(self.station.name, 'Test Station')
        self.assertEqual(self.station.country, 'Testland')
        self.assertEqual(self.station.state, 'TS')
        self.assertEqual(self.station.latitude, 12.34)
        self.assertEqual(self.station.longitude, 56.78)
        self.assertEqual(self.station.elevation, 100)
        self.assertEqual(self.station.start_year, '2000')
        self.assertEqual(self.station.end_year, '2020')
        self.assertEqual(self.station.timezone_name, 'Test/Timezone')
        self.assertEqual(self.station.timezone_utc, '+00:00')
        self.assertEqual(self.station.source, 'BOM')
        self.assertEqual(self.station.mean_correction_factors, {'Temperature': 1.0})
        self.assertEqual(self.station.available_measurements, {'Wind Direction': 'True', 'Wind Speed': 'True'})

class TestWeatherStationDatabase(unittest.TestCase):
    def setUp(self):
        # Mock the _load_database method to return a DataFrame
        self.db = WeatherStationDatabase(data_type='BOM')
        self.db._data = pd.DataFrame([
            {
                'Station ID': '000001',
                'Station Name': 'Test Station',
                'Country': 'Testland',
                'State': 'TS',
                'Latitude': '12.34',
                'Longitude': '56.78',
                'Elevation': '100',
                'Start': '2000',
                'End': '2020',
                'Timezone Name': 'Test/Timezone',
                'Timezone UTC': '+00:00',
                'Source': 'BOM',
                'Mean Correction Factor': {'Temperature': 1.0},
                'Wind Direction': 'True',
                'Wind Speed': 'True'
            }
        ])

    def test_get_station(self):
        station = self.db.get_station('000001')
        self.assertEqual(station.id, '000001')
        self.assertEqual(station.name, 'Test Station')

if __name__ == '__main__':
    unittest.main() 