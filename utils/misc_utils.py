import os
import requests
import zipfile
import logging

logger = logging.getLogger(__name__)

def download_file(url, save_name):
    url = url
    logger.info(f"Downloading file from {url}...")
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, 'wb').write(file.content)

def unzip(zip_file=None):
    try:
        with zipfile.ZipFile(zipfile) as z:
            z.extractall("./")
            logger.info(f"Extracted all files...")
    except:
        logger.error(f"Couldn't extract: invalid file...")

def check_positive_integer(value, name):
    '''Checks if a value is a positive integer.'''
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")