import os
from box.exceptions import BoxValueError
import yaml
from src.textSummarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any, List
import yaml



#--------------------------------------READ YAML-----------------------------------------
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox object. 
    This allows for easy access to the configuration parameters defined in the YAML file.
    
    Args:
        path_to_yaml (Path): The path to the YAML file to be read.
    
    Returns:
        ConfigBox: A ConfigBox object containing the YAML file's contents.
    Raises:
        BoxValueError: If there is an error reading the YAML file, such as a syntax error or if the file is not found.
    """
    try:
        with open(path_to_yaml, "rb") as yaml_file:
            file_content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file '{path_to_yaml}' read successfully.")
            return ConfigBox(file_content)
    except BoxValueError as e:
        logger.error(f"Error reading YAML file '{path_to_yaml}': {e}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading YAML file '{path_to_yaml}': {e}")
        raise e
    
#--------------------------------------CREATE DIRECTORIES-----------------------------------------
def create_directories(path_to_directories: List[Path], verbose: bool = True) -> None:
    """
    Creates directories specified in the list of paths. If a directory already exists, it will not raise an error.
    
    Args:
        path_to_directories (list[Path]): A list of Path objects representing the directories to be created.
        verbose (bool): If True, logs the creation of each directory. Default is True.
    """
    for directory in path_to_directories:
        try:
            os.makedirs(directory, exist_ok=True)
            if verbose:
                logger.info(f"Directory '{directory}' created successfully.")
        except Exception as e:
            logger.error(f"An error occurred while creating directory '{directory}': {e}")
            raise e