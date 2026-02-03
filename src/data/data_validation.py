import pandas as pd
import os
import re
import json
import shutil
from datetime import datetime
from os import listdir
import logging
from src.logger import logging
from from_root import from_root

def fetch_values_from_schema():
    """  This method extracts all the relevant information from the pre-defined "Schema" file. """
    try:
        schema_path = os.path.join(from_root(), "configs/schema.json")
        with open(schema_path, 'r') as file:
            dict = json.load(file)
            file.close()

        pattern = dict["SampleFileName"]
        LengthOfDateStampInFile = dict['LengthOfDateStampInFile']
        LengthOfTimeStampInFile = dict['LengthOfTimeStampInFile']
        column_names = dict['ColName']
        NumberofColumns = dict['NumberofColumns']

        logging.info('LengthOfDateStampInFile: %s \t "LengthOfTimeStampInFile: %s \t "NumberofColumns: %s \n', LengthOfDateStampInFile, LengthOfTimeStampInFile, NumberofColumns) 
        
    except ValueError:
        logging.error('ValueError:Value not found inside schema.json')
        raise
    except KeyError:
        logging.error('KeyError:Key value error. Incorrect key passed')
        raise
    except Exception as e:
        logging.error('Unexpected error during fetching values from schema: %s', e)
        raise


def manual_regex_creation():
    """  This method contains a manually defined regex based on the "FileName" given in "Schema" file. This Regex is used to validate the filename of the training data. """

    regex = "['fraudDetection']+['\_'']+[\d_]+[\d]+\.csv"
    return regex


def create_directory_for_good_bad_raw_data():
    """This method creates directories to store the Good Data and Bad Data after validating the training data."""
    try:
        path = os.path.join("Raw_files_validation/", "Good_Raw/")
        os.makedirs(path, exist_ok=True)

        path = os.path.join("Raw_files_validation/", "Bad_Raw/")
        os.makedirs(path, exist_ok=True)

    except OSError as e:
        logging.error('Error while creating Directory %s', e)
        raise


def validate_file_name_raw(regex, filename, LengthOfDateStampInFile, LengthOfTimeStampInFile):
    """This method validates the name of the csv files as per given name in the schema!
       Regex pattern is used to do the validation. If name format do not match the file is moved to Bad Raw Data folder else in Good raw data.
    """
    try:
        if (re.match(regex, filename)): #fraudDetection_021119920_010222.csv
            splitAtDot = re.split('.csv', filename) #fraudDetection_021119920_010222, csv
            splitAtDash = (re.split('_', splitAtDot[0])) #fraudDetection 021119920 010222
            if len(splitAtDash[1]) == LengthOfDateStampInFile and len(splitAtDash[2]) == LengthOfTimeStampInFile:
                shutil.copy(filename, "Raw_files_validation/Good_Raw/")
                logging.info("Valid File name!! File moved to Good_Raw Folder: %s", filename)
            else:
                shutil.copy(filename, "Raw_files_validation/Bad_Raw/")
                logging.error("Invalid File name!! File moved to Bad_Raw Folder: %s", filename)  
    
    except Exception as e:
        logging.error('Error occured while validating FileName: %s', e)
        raise


def validate_column_length(NumberofColumns):
    """This function validates the number of columns in the csv files.
       It should be same as given in the schema file. If not, file is not suitable for processing and is moved to Bad Raw Data folder. If the column number matches, file is kept in Good Raw Data for processing.
    """
    try:
        good_raw_dir_path = 'Raw_files_validation/Good_Raw/'
        for file in listdir(good_raw_dir_path):
            csv = pd.read_csv(os.path.join(good_raw_dir_path), file)
            if csv.shape[1] == NumberofColumns:
                pass
            else:
                shutil.move("Raw_files_validation/Good_Raw/" + file, "Raw_files_validation/Bad_Raw")
                logging.error("Invalid Column Length for the file!! File moved to Bad Raw Folder : %s", file)
    
    except OSError as e:
        logging.error('Error Occured while moving the file: %s', e)
        raise
    except Exception as e:
        logging.error('Error occured while validating Column Length: %s', e)
        raise


def validate_missing_values_in_column():
    """This function validates if any column in the csv file has all values missing.
       If all the values are missing, the file is not suitable for processing and is moved to Bad Raw Data folder.
    """
    try:
        good_raw_dir_path = 'Raw_files_validation/Good_Raw/'
        for file in listdir(good_raw_dir_path):
            csv = pd.read_csv(os.path.join(good_raw_dir_path), file)
            count = 0
            for columns in csv:
                if (len(csv[columns]) - csv[columns].count()) != 0:
                    count+=1
                    shutil.move("Raw_files_validation/Good_Raw/" + file, "Raw_files_validation/Bad_Raw")
                    logging.error("Invalid Column for the file!! File moved to Bad Raw Folder : %s", file)
                    break

            if count==0:
                csv.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                csv.to_csv(os.path.join(good_raw_dir_path), file, index=False)
    
    except OSError as e:
        logging.error('Error Occured while moving the file: %s', e)
        raise
    except Exception as e:
        logging.error('Error occured while validating column missing values: %s', e)
        raise
