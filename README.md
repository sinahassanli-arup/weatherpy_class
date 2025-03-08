# WeatherPy

A Python package for importing, processing, and analyzing weather data from various sources.

## Overview

WeatherPy provides tools for working with weather data from different sources, including:

- Bureau of Meteorology (BOM) data
- National Oceanic and Atmospheric Administration (NOAA) data
- EPW files

The package includes functionality for:

- Importing weather data
- Cleaning and validating data
- Correcting data (e.g., terrain corrections)
- Analyzing data

## Project Structure

The project is organized into two main parts:

1. `weatherpy_legacy`: The original implementation with procedural code
2. `weatherpy`: The new implementation with object-oriented code

### Legacy Implementation

The legacy implementation is organized as follows:

- `weatherpy_legacy/data/initialization.py`: Main entry point for importing data
- `weatherpy_legacy/data/_bom_preparation.py`: Functions for preparing BOM data
- `weatherpy_legacy/data/_noaa_preparation.py`: Functions for preparing NOAA data
- `weatherpy_legacy/data/cleaning.py`: Functions for cleaning data
- `weatherpy_legacy/data/stations.py`: Functions for working with station data

### Object-Oriented Implementation

The new implementation is organized as follows:

- `weatherpy/data/wd_importer.py`: Classes for importing weather data
  - `WeatherDataImporter`: Base class for importing weather data
  - `BOMWeatherDataImporter`: Class for importing BOM data
  - `NOAAWeatherDataImporter`: Class for importing NOAA data
- `weatherpy/data/wd_base.py`: Base class for weather data
- `weatherpy/data/wd_cleaner.py`: Classes for cleaning weather data
- `weatherpy/data/wd_corrector.py`: Classes for correcting weather data
- `weatherpy/data/wd_unifier.py`: Classes for unifying weather data
- `weatherpy/data/wd_inferer.py`: Classes for inferring weather data
- `weatherpy/data/wd_stations.py`: Classes for working with station data

## Usage

### Importing Weather Data

```python
from weatherpy.data.wd_importer import BOMWeatherDataImporter, NOAAWeatherDataImporter

# Import BOM data
bom_importer = BOMWeatherDataImporter(
    station_id='066037',  # Sydney Airport
    time_zone='LocalTime',
    year_start=2010,
    year_end=2020,
    interval=60  # 60-minute intervals
)
bom_data, start_year, end_year = bom_importer.import_data()

# Import NOAA data
noaa_importer = NOAAWeatherDataImporter(
    station_id='72503014732',  # La Guardia Airport
    time_zone='LocalTime',
    year_start=2010,
    year_end=2020
)
noaa_data, start_year, end_year = noaa_importer.import_data()
```

## Examples

The `examples` directory contains scripts demonstrating how to use the package:

- `examples/compare_importer.py`: Compares the legacy and class-based implementations
- `examples/compare_importer_simple.py`: Tests the class-based implementation without comparison

## Development

### Requirements

- Python 3.6+
- pandas
- numpy
- pytz
- requests

### Testing

To run the comparison tests:

```bash
python examples/compare_importer.py
```

To run the simple tests:

```bash
python examples/compare_importer_simple.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 