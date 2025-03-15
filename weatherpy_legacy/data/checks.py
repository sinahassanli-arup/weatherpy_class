import platform, tempfile, os

def _get_weatherpy_temp_folder(print=False):
    """
    This is to get the proper temporary folder based on linux or windows system
    """
    current_system = platform.system()

    if current_system == 'Windows':
        temp_path = os.path.join(tempfile.gettempdir(),'weatherpy')
    elif current_system == 'Linux':
        temp_path = os.path.join('/tmp','weatherpy')  # Linux temporary folder
    else:
        raise NotImplementedError("This system is not recognized as Windows or Linux.")

    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    
    if print:
        print(temp_path)
        
    return temp_path


def _remove_local_weather_file(stationID=None, interval=None, remove_all=False):
    
    temp_path = _get_weatherpy_temp_folder()
    
    # location of weather data
    sourcedata = os.path.join(temp_path, 'sourcedata')
    
    # Get a list of all files in the sourcedata directory
    files = os.listdir(sourcedata)
    
    if remove_all:
        # Iterate over the files and remove each one
        for file in files:
            file_path = os.path.join(sourcedata, file)
            os.remove(file_path)
    
    else:
        matching_files = [file for file in files if stationID == file.split('_')[1] and f'{interval}minute.zip' == file.split('_')[3]]
        if matching_files:
            for file in matching_files:
                file_path = os.path.join(sourcedata, file)
                os.remove(file_path)
                print(f'{file} is removed from the local drive.')
        else:
            print(f'no local weather file with specified description is found.')
        