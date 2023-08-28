#to launch, type in the terminal (from root folder):
# uvicorn fastAPI:app --reload
# then in a browser: http://127.0.0.1:8000/sensors/1?lastN=5 
from fastapi import FastAPI

# import time
# from pydantic import BaseModel #for POST queries with JSON body
from pydantic import BaseModel

# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process                                                                                              
#  .venv\scripts\activate 
import numpy as np
import pandas as pd
# import itertools
import matplotlib.pyplot as plt #if needed for debug
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from datetime import timedelta
import itertools
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import model_selection
import joblib
import requests
import time
import json
import csv
from datetime import datetime, timedelta
import threading
import os
from typing import List
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import glob

####################################################  VARIABLEs #####################################################
NUMBER_DAYS_CONSIDERED = 7
AGGREGATION = '1H'
PROCESSED_DATA_FILE = 'processed_data'
TRAINING_HORIZON = '30D'
FORECAST_HORIZON = '1H' #'1D'   '1H'
FORECAST_HORIZON_LARGE = '24H' #'1D'   '1H'
TRAINING_FREQUENCY = '7D'
REQUESTED_INTERVAL_R2 = '7D'


MODEL_FILE_NAME = 'finalized_model'
CSV_FILE_NAME = ""
TYPES_OF_SENSOR = [  "GBOXLenval" ]

# TYPES_OF_SENSOR = [  "HOPac67b2ce5312_noise" ,"HOPac67b2cd1cfe_noise", "HOPac67b2d5c9ee_noise"]

# URL definition

URL_FORECAST  = "https://tip-imredd.unice.fr/data/imredd/nice/noisepollution/entityOperations/upsert?api-key=a1b4deee-008f-4161-ae24-4b7cf507107b"

TIME_TO_WAIT_FOR_TABLE_UPDATE = 15*60 #in seconds
TYPE_REQUEST = 2 #This is to indicate if your update of historical data use a Request to the Context Borker that use the AFTER (1) or the BEFORE (2) key word.
if (TYPE_REQUEST==1):
    URL_PART1 = 'https://tip-imredd.unice.fr/data/imredd/nice/noisepollution/temporal/entities/?api-key=a1b4deee-008f-4161-ae24-4b7cf507107b&type=https://smartdatamodels.org/dataModel.Environment/NoisePollution&id=https://api.nicecotedazur.org/nca/environment/air/noiselevel/AZIMUT/'
    URL_PART2 = '&timeproperty=modifiedAt&options=sysAttrs&lastN=200&attrs=Lamax2&timerel=after&timeAt='
else:        
    URL_PART1 = 'https://tip-imredd.unice.fr/data/imredd/nice/noisepollution/temporal/entities/?api-key=a1b4deee-008f-4161-ae24-4b7cf507107b&type=https://smartdatamodels.org/dataModel.Environment/NoisePollution&id=https://api.nicecotedazur.org/nca/environment/air/noiselevel/AZIMUT/'
    URL_PART2 = '&timeproperty=modifiedAt&options=sysAttrs&lastN=200&attrs=Lamax2&timerel=before&timeAt='


FORECAST_ID_PART1 = "https://api.nicecotedazur.org/nca/environment/air/noiselevel/AZIMUT/"
FORECAST_ID_PART2 = "/forecast"

CSV_TIME_FORMAT= '%m/%d/%Y %H:%M' # '%Y-%m-%d %H:%M:%S'
ATTRIBUTE_TO_FORECAST='Lamax2'

header = ["time", "historical value", "forecasts", "r2"]


#################################################### Functions Definitions  ###################################################


# get the last data and store it in the local csv file
def get_and_process_new_data_with_forecast_recurrent(time_to_wait_for_update,csv_file_name):
    
    print("starting update of local database")
    try:
        for type_of_sensor in TYPES_OF_SENSOR:
            URL_GREENMOV_NOISE =  URL_PART1 + type_of_sensor 
            URL_GREENMOV_NOISE = URL_GREENMOV_NOISE + URL_PART2
            
            '&timeproperty=modifiedAt&options=sysAttrs&lastN=20&timerel=before&timeAt=' 
            # [status] = get_new_data_once(URL_GREENMOV_NOISE,CSV_FILE_NAME+type_of_sensor+".csv")
            if (TYPE_REQUEST==1):
                [status] = get_new_data_once(URL_GREENMOV_NOISE,CSV_FILE_NAME+type_of_sensor+".csv")    
            else:        
                [status] = get_new_data_once_using_before_request(URL_GREENMOV_NOISE,CSV_FILE_NAME+type_of_sensor+".csv")            

        print("starting processing of data")
        if status == 0 :
            for type_of_sensor in TYPES_OF_SENSOR:
                process_data(CSV_FILE_NAME+type_of_sensor+".csv", AGGREGATION, PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ AGGREGATION +".csv", type_of_sensor, FORECAST_HORIZON)
            return {'could not update data but processed it'}
        else:
            for type_of_sensor in TYPES_OF_SENSOR:
                process_data(CSV_FILE_NAME+type_of_sensor+".csv", AGGREGATION, PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ AGGREGATION +".csv", type_of_sensor, FORECAST_HORIZON)

            print("data processed and enhanced dataframe stored")
        print("Generate forecast for next hour")    
        [merged_df1H,r2forecast1H]  = generate_new_forecasts(AGGREGATION,FORECAST_HORIZON, REQUESTED_INTERVAL_R2)
        print("Generate forecast for next 24 hour")    
        generate_new_forecasts(AGGREGATION,FORECAST_HORIZON_LARGE, REQUESTED_INTERVAL_R2, r2forecast1H)
    except Exception as e:
        print(f"An error occurred while getting and processing the data: {e}")
        threading.Timer(time_to_wait_for_update, get_and_process_new_data_with_forecast_recurrent, args=(time_to_wait_for_update, csv_file_name)).start()
        print ("Thread: %s, time: %s" % (threading.get_ident(), datetime.now()))
        print("end of thread")
    else:
        threading.Timer(time_to_wait_for_update, get_and_process_new_data_with_forecast_recurrent, args=(time_to_wait_for_update, csv_file_name)).start()
        print ("Thread: %s, time: %s" % (threading.get_ident(), datetime.now()))
        print("end of thread")







def get_new_data_once_using_before_request(url,csv_file_name):
    date_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    new_date_time = date_time.replace(":", "%3A" )
    url_full = url + new_date_time
    x = requests.get(url_full, verify=False)

    if (x.status_code <200 or x.status_code >299):
        print("Request Error to get last data")
        return [0]
    else :
        response = x.content
        jsondata = json.loads(response)
        array = []
        with open(csv_file_name, "r") as scraped:
            final_time = scraped.readlines()[-1].split(',')[0]
        final_datetime = datetime.strptime(final_time, CSV_TIME_FORMAT)  
        csvfile = open(csv_file_name, 'a', newline='')
        writer = csv.writer(csvfile) #, quoting=csv.QUOTE_NONE, delimiter=',', quotechar='')
        nb_rows_added = 0
        # print(jsondata[0]['Intensity'])
        # print(jsondata[0]['Intensity']['modifiedAt'])
        # print(isinstance(jsondata[0]['Intensity'], list))
        # print(jsondata[0]['Intensity'][0])

        if isinstance(jsondata[0][ATTRIBUTE_TO_FORECAST], list):
            for i in reversed(range(len(jsondata[0][ATTRIBUTE_TO_FORECAST]))):
                    booldate_ok=1
                    try:
                            datetime.strptime(jsondata[0][ATTRIBUTE_TO_FORECAST][i]['modifiedAt'], '%Y-%m-%dT%H:%M:%S.%fZ')
                    except ValueError:
                        try:
                            datetime.strptime(jsondata[0][ATTRIBUTE_TO_FORECAST][i]['modifiedAt'], '%Y-%m-%dT%H:%M:%SZ')
                            booldate_ok=2
                        except ValueError:
                            booldate_ok=0
                    if booldate_ok==1:
                            datetime_object = datetime.strptime(jsondata[0][ATTRIBUTE_TO_FORECAST][i]['modifiedAt'], '%Y-%m-%dT%H:%M:%S.%fZ') + timedelta(hours=2)
                    elif booldate_ok==2:
                            datetime_object = datetime.strptime(jsondata[0][ATTRIBUTE_TO_FORECAST][i]['modifiedAt'], '%Y-%m-%dT%H:%M:%SZ') + timedelta(hours=2)
                    if booldate_ok>0 and datetime_object>final_datetime+timedelta(minutes=1):
                            newtime = datetime_object.strftime('%m/%d/%Y %H:%M')
                            newvalue = jsondata[0][ATTRIBUTE_TO_FORECAST][i]['value']
                            writer.writerow([newtime,newvalue])
                            nb_rows_added = nb_rows_added+1
        else:
                i=0
                booldate_ok=1
                try:
                        datetime.strptime(jsondata[0][ATTRIBUTE_TO_FORECAST]['modifiedAt'], '%Y-%m-%dT%H:%M:%S.%fZ')
                except ValueError:
                    try:
                        datetime.strptime(jsondata[0][ATTRIBUTE_TO_FORECAST]['modifiedAt'], '%Y-%m-%dT%H:%M:%SZ')
                        booldate_ok=2
                    except ValueError:
                        booldate_ok=0
                if booldate_ok==1:
                        datetime_object = datetime.strptime(jsondata[0][ATTRIBUTE_TO_FORECAST]['modifiedAt'], '%Y-%m-%dT%H:%M:%S.%fZ') + timedelta(hours=2)
                elif booldate_ok==2:
                        datetime_object = datetime.strptime(jsondata[0]['Intensity']['modifiedAt'], '%Y-%m-%dT%H:%M:%SZ') + timedelta(hours=2)
                if booldate_ok>0 and datetime_object>final_datetime+timedelta(minutes=1):
                        newtime = datetime_object.strftime('%m/%d/%Y %H:%M')
                        newvalue = jsondata[0]['Intensity']['value']
                        writer.writerow([newtime,newvalue])
                        nb_rows_added = nb_rows_added+1
        csvfile.close()
        print("we filled: ", nb_rows_added, " rows")
        return [1]















# get the last data and store it in the local csv file
def get_new_data_once(url,csv_file_name):

    date_time_now = datetime.now()
    nb_rows_added=0
    lastest_time = datetime(2000, 1, 1, 0, 0, 0)
    booleen_update_incomplete = 1
    counter=1
    url_full=""
    while(booleen_update_incomplete):

        with open(csv_file_name, "r") as scraped:
            line = scraped.readlines()[-1].split(',')
        # index = line[0]
        # id_name = line[1]
        final_time = line[0]
        # longitude = line[3]
        # latitude = line[4]
        # la_eq =  line[5]

        final_datetime = datetime.strptime(final_time,CSV_TIME_FORMAT)- timedelta(hours=2)
        final_datetimestring = final_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')  
        final_datetimestring = final_datetimestring.replace(":", "%3A" )
        url_full_previous = url_full
        url_full = url + final_datetimestring


        booleen_update_incomplete = 0
        try:
            req = requests.get(url_full)  # Replace with your URL
            req.raise_for_status()  # Check if the request was successful
            print("Request successful! Status code:", req.status_code)
            response = req.content
            x=req
            booleen_update_incomplete = 1
            # print(response)
            print(url_full)
            # Process the response data here
        except requests.exceptions.RequestException as e:
            # Handle connection errors here
            print("Connection error:", e)
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors here (e.g., 404, 500, etc.)
            print("HTTP error:", e)
        except requests.exceptions.Timeout as e:
            # Handle timeout errors here
            print("Timeout error:", e)
        except requests.exceptions.TooManyRedirects as e:
            # Handle too many redirects errors here
            print("Too many redirects error:", e)
        except Exception as e:
            # Handle other unexpected errors here
            print("Unexpected error:", e)
        # x = response
        # x = requests.get(url_full)
        else:
            # print("time for request: ")
            # print(final_datetimestring)
            # print(x.status_code)
            if (x.status_code <200 or x.status_code >299):
                print("Request Error to get last data")
                booleen_update_incomplete = 0
                # return [0]
            else :
                
                jsondata = json.loads(response)
                Laeq_data = []
                if isinstance(jsondata, dict):
                    if len(jsondata.values()) > 0:
                        Laeq_data = jsondata[ATTRIBUTE_TO_FORECAST]
                elif isinstance(jsondata, list):
                    if len(jsondata) > 0:
                        Laeq_data = jsondata[0][ATTRIBUTE_TO_FORECAST]

                if len(Laeq_data)==0:
                    booleen_update_incomplete = 0
                else:
                    # print(Laeq_data)
#################################################################### we sort the data received ###########################################
                    # Create an empty DataFrame with columns
                    columns = ['time', 'value']
                    df_intermediary = pd.DataFrame(columns=columns)
                    # print('here1')
                    # print(df_intermediary)
                    if isinstance(Laeq_data, dict):
                            try:
                                datetime.strptime(Laeq_data['modifiedAt'], '%Y-%m-%dT%H:%M:%S.%fZ')
                                booldate_ok=1
                            except ValueError:
                                print(ValueError)
                                try:
                                    datetime.strptime(Laeq_data['modifiedAt'], '%Y-%m-%dT%H:%M:%SZ') 
                                    booldate_ok=2 
                                except ValueError:
                                    booldate_ok=0
                            if booldate_ok==1:
                                datetime_object = datetime.strptime(Laeq_data['modifiedAt'], '%Y-%m-%dT%H:%M:%S.%fZ') + timedelta(hours=2)
                            elif booldate_ok==2:
                                datetime_object = datetime.strptime(Laeq_data['modifiedAt'], '%Y-%m-%dT%H:%M:%SZ') + timedelta(hours=2)
                            if booldate_ok>0:
                                newvalue = Laeq_data['value']
                                new_row = {'time': datetime_object, 'value': newvalue}
                                # print(new_row)
                                df_intermediary.loc[len(df_intermediary)] = new_row
                            if datetime_object> lastest_time:
                                latest_time = datetime_object


                        # Do something for a list
                    elif isinstance(Laeq_data, list):
                        # Do something for an array


                        for i in (range(len(Laeq_data))): 
                            try:
                                datetime.strptime(Laeq_data[i]['modifiedAt'], '%Y-%m-%dT%H:%M:%S.%fZ')
                                booldate_ok=1
                            except ValueError:
                                print(ValueError)
                                try:
                                    datetime.strptime(Laeq_data[i]['modifiedAt'], '%Y-%m-%dT%H:%M:%SZ') 
                                    booldate_ok=2 
                                except ValueError:
                                    booldate_ok=0
                            if booldate_ok==1:
                                datetime_object = datetime.strptime(Laeq_data[i]['modifiedAt'], '%Y-%m-%dT%H:%M:%S.%fZ') + timedelta(hours=2)
                            elif booldate_ok==2:
                                datetime_object = datetime.strptime(Laeq_data[i]['modifiedAt'], '%Y-%m-%dT%H:%M:%SZ') + timedelta(hours=2)
                            if booldate_ok>0:
                                newvalue = Laeq_data[i]['value']
                                new_row = {'time': datetime_object, 'value': newvalue}
                                # print(new_row)
                                # print('------------------')
                                # print(df_intermediary)
                                # print('*****************')
                                # df_intermediary = df_intermediary.append(new_row, ignore_index=True)

                                df_intermediary.loc[len(df_intermediary)] = new_row
                            if datetime_object> lastest_time:
                                latest_time = datetime_object
                    if abs(final_datetime+ timedelta(hours=2)-df_intermediary['time'].min())< timedelta(minutes=1):
                        min_time_index = df_intermediary['time'].idxmin()

                        # Remove the row with the minimum datetime value
                        df_intermediary = df_intermediary.drop(min_time_index)
                    # Check if the DataFrame is empty
                    if df_intermediary.empty:
                        print("DataFrame is empty.")
                        booleen_update_incomplete = 0
                    else:

                        # Sort the DataFrame by the datetime column in ascending order
                        df_sorted = df_intermediary.sort_values(by='time')
                        print("storing new values locally")
                        with open(csv_file_name, 'a', newline='') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            
                            # Write CSV header
                            # csv_writer.writerow(['datetime_col', 'float_col'])
                            
                            # Iterate through DataFrame rows and process each row
                            for index, row in df_sorted.iterrows():
                                formatted_datetime = row['time'].strftime(CSV_TIME_FORMAT)
                                # row['datetime_col'].strftime('%Y-%m-%d %H:%M')
                                csv_writer.writerow([formatted_datetime, row['value']])
                                # row = str(int(index)+i+1) +"," +id_name+","+ newtime +","+longitude+","+latitude+"," + str(newvalue)+","+","
                                # row = newtime+","+newvalue #str(int(index)+i+1) +"," +id_name+","+ newtime +","+longitude+","+latitude+"," + str(newvalue)+","+","
                                nb_rows_added = nb_rows_added+1

                        
                        csvfile.close()
                        if latest_time > date_time_now - timedelta(hours=2+1):
                            booleen_update_incomplete = 0



                # # print(len(jsondata[0]['LAeq']))
                # # print(jsondata[0]['LAeq'][0]['value']) 
                #     array = []
                #     # with open(csv_file_name, "r") as scraped:
                #     #     final_time = scraped.readlines()[-1].split(',')[0]
                #     # final_datetime = datetime.strptime(final_time, '%Y-%m-%d %H:%M:%S')  
                #     csvfile = open(csv_file_name, 'a', newline='')
                #     writer = csv.writer(csvfile) #, quoting=csv.QUOTE_NONE, delimiter=',', quotechar='')
                #     for i in (range(len(Laeq_data))):
                #         booldate_ok=1
                #         try:
                #             datetime.strptime(Laeq_data[i]['modifiedAt'], '%Y-%m-%dT%H:%M:%S.%fZ')
                #         except ValueError:
                #             try:
                #                 datetime.strptime(Laeq_data[i]['modifiedAt'], '%Y-%m-%dT%H:%M:%SZ') 
                #                 booldate_ok=2 
                #             except ValueError:
                #                 booldate_ok=0
                #         if booldate_ok==1:
                #             datetime_object = datetime.strptime(Laeq_data[i]['modifiedAt'], '%Y-%m-%dT%H:%M:%S.%fZ') + timedelta(hours=2)
                #         elif booldate_ok==2:
                #             datetime_object = datetime.strptime(Laeq_data[i]['modifiedAt'], '%Y-%m-%dT%H:%M:%SZ') + timedelta(hours=2)
                            
                        
                #         if booldate_ok>0 and datetime_object>final_datetime+timedelta(minutes=1):
                #             newtime = datetime_object.strftime(CSV_TIME_FORMAT)
                #             newvalue = Laeq_data[i]['value']
                #             # row = str(int(index)+i+1) +"," +id_name+","+ newtime +","+longitude+","+latitude+"," + str(newvalue)+","+","
                #             # row = newtime+","+newvalue #str(int(index)+i+1) +"," +id_name+","+ newtime +","+longitude+","+latitude+"," + str(newvalue)+","+","
                #             writer.writerow([newtime,newvalue])
                #             nb_rows_added = nb_rows_added+1
                #         if datetime_object> lastest_time:
                #             latest_time = datetime_object
                #     if latest_time > date_time_now - timedelta(hours=2+1):
                #         booleen_update_incomplete = 0
                #     csvfile.close()
        counter = counter+1
        if url_full_previous == url_full or counter >30: #we stop requesting the server if we keep asking too much data
            booleen_update_incomplete = 0

    print("we filled: ", nb_rows_added, " rows")
    return [1]

        # if (x.status_code <200 or x.status_code >299):
        #     print("Request Error to get last data")
        #     booleen_update_incomplete = 0
        #     return [0]
        # else :











    # date_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    # new_date_time = date_time.replace(":", "%3A" )
    # url_full = url + new_date_time
    # x = requests.get(url_full)
    # print(url_full)
    # print(x.status_code)
    # if (x.status_code <200 or x.status_code >299):
    #     print("Request Error to get last data")
    #     return [0]
    # else :

    #     response = x.content
    #     jsondata = json.loads(response)
    #     print(len(jsondata[0]['Intensity']))
    #     print(jsondata[0]['Intensity'][0]['value']) 
    #     array = []
    #     with open(csv_file_name, "r") as scraped:
    #         final_time = scraped.readlines()[-1].split(',')[0]
    #     final_datetime = datetime.strptime(final_time, '%m/%d/%Y %H:%M')  
    #     csvfile = open(csv_file_name, 'a', newline='')
    #     writer = csv.writer(csvfile) #, quoting=csv.QUOTE_NONE, delimiter=',', quotechar='')
    #     nb_rows_added=0
    #     for i in reversed(range(len(jsondata[0]['LAeq']))):
    #         booldate_ok=1
    #         try:
    #             datetime.strptime(jsondata[0]['LAeq'][i]['modifiedAt'], '%Y-%m-%dT%H:%M:%S.%fZ')
    #         except ValueError:
    #             try:
    #                 datetime.strptime(jsondata[0]['LAeq'][i]['modifiedAt'], '%Y-%m-%dT%H:%M:%SZ') 
    #                 booldate_ok=2 
    #             except ValueError:
    #                 booldate_ok=0
    #         if booldate_ok==1:
    #             datetime_object = datetime.strptime(jsondata[0]['LAeq'][i]['modifiedAt'], '%Y-%m-%dT%H:%M:%S.%fZ') + timedelta(hours=2)
    #         elif booldate_ok==2:
    #             datetime_object = datetime.strptime(jsondata[0]['LAeq'][i]['modifiedAt'], '%Y-%m-%dT%H:%M:%SZ') + timedelta(hours=2)
                
            
    #         if booldate_ok>0 and datetime_object>final_datetime+timedelta(minutes=1):
    #             newtime = datetime_object.strftime('%m/%d/%Y %H:%M')
    #             newvalue = jsondata[0]['Lamax2'][i]['value']
    #             row = newtime +"," + str(newvalue)
    #             writer.writerow([newtime,newvalue])
    #             nb_rows_added = nb_rows_added+1
    #     csvfile.close()
    #     print("we filled: ", nb_rows_added, " rows")
    #     return [1]



############################  Analyse data  #################################
def analyse(df, plot=False):
    print("Analyzing Data")

    df = df.sort_values(['Time'], ascending=[True])
    df = clean_data(df)
    return df

def clean_data(df, quantile_value = 0.01):
   """
   Erase rows with NaN values inside any column(s)
   Erase (quantile_value%) left and (1 - quantile_value%) right, data value (get rid of eberrant data like lonely spikes) using quantile
   """
   print("***** Cleaning dataframe...")
   before = len(df)
   df.dropna(inplace=True)
   after = len(df)
   delta = before - after
   print((1-(after/before))*100, "% of NaN ==>", delta, "rows deleted")
   for col in df.columns.values[1:]:
      q1 = df[col].quantile(quantile_value)
      q3 = df[col].quantile(1 - quantile_value)
      df = df[(df[col] > q1) & (df[col] < q3)]
   after2 = len(df)
   delta = after - after2
   print((1-(after2/after))*100, "% deleted from", (quantile_value*100), "% quantile ==>", delta, "rows deleted")
   print("***** Cleaning done")
   return df


    
def simpleinference(model, x_train,  y_train, x_test,  y_test, numberrowsforinference, datasetindexbytime,  verbose=True):
        keywords_args = {}
        i = 0
        
        for parameter in model.params_name:
            keywords_args[parameter] = int(model.params_comb[i])
            i = i+1
        model = model.name(**keywords_args)
        # print("x train")
        # print(x_train)
        # print("y train")
        # print(y_train)
        # print("training size6")
        # print(x_train.size)
        model = model.fit(x_train, y_train) # Train model

        numberIndexShift1H =  len(datasetindexbytime.first('1H'))
        numberIndexShift7d =  len(datasetindexbytime.first('7D'))
        valuet_1 = y_test.tail(1).values[0][0]
        valuet_2 = x_test.tail(1)['t-1'].values[0]
        valuet_3 = x_test.tail(1)['t-2'].values[0]
        # if 't-1h' in x_test.columns:
        #     valuet_1h=(x_test[0:1]['t-1h'].values[0])
        # if 't-7days' in x_test.columns:
        #     valuet_7d=(x_test[0:1]['t-7days'].values[0])
        y_pred = []
        last_two_times = datasetindexbytime.tail(2).index
        timestepdelta = last_two_times[1] - last_two_times[0]

        # print("**")
        # print(numberrowsforinference)  
        # print(x_test)  
        # print(numberIndexShift7d)   
        # print(datasetindexbytime)
        for inference in range(numberrowsforinference):
            inference_time = last_two_times[1]+timestepdelta*(1+inference)

            X_test_unit = x_test.tail(1).copy()
     
            X_test_unit['DayOfWeek'] = min(NUMBER_DAYS_CONSIDERED,inference_time.dayofweek)
            X_test_unit['Hour'] = inference_time.hour
            X_test_unit['Minute'] = inference_time.minute
            X_test_unit['TimeSlot'] = time_slot_assigner(inference_time.hour)            
            if X_test_unit.empty == False:
                X_test_unit['t-1'] = valuet_1
                X_test_unit['t-2'] = valuet_2
                X_test_unit['t-3'] = valuet_3
                if (inference >= numberIndexShift1H) and ('t-1h' in x_train.columns):
                    print("here !!")
                    X_test_unit['t-1h'].values[0] = y_pred.copy()[inference-numberIndexShift1H:inference-numberIndexShift1H+1][0]
                elif ('t-1h' in x_train.columns):
                    X_test_unit['t-1h'].values[0] = y_test.loc[len(y_test)-numberIndexShift1H]
                if (inference >= numberIndexShift7d) and ('t-7days' in x_train.columns):

                    X_test_unit['t-7days'].values[0] = y_pred.copy()[inference-numberIndexShift7d:inference-numberIndexShift7d+1][0]             
                elif ('t-7days' in x_train.columns):
                    X_test_unit['t-7days'].values[0] = y_test.loc[min(len(y_test)-1,max(0,len(y_test)+inference-numberIndexShift7d-1))]
                valuet_3 = valuet_2
                valuet_2 = valuet_1
                # print("X_test_unit:")
                # print(X_test_unit)
                valuet_1 = model.predict(X_test_unit)

                if isinstance(valuet_1[0],float):
                    y_pred.append(valuet_1[0])
                elif isinstance(valuet_1[0][0],float):
                    y_pred.append(valuet_1[0][0])
                else:
                    print("error output not a float")
        # print(y_pred)
        return y_pred





def interpolationByInference(df, aggregation, forecasthorizon, typeofsensor, start, stop):
    
    # print("Interpolation")
    # print(df)

    # stop = pd.Timestamp.now()  # Get the current time
    # periods = pd.period_range(start=stop, periods=1, freq='H')  # Create a period range

    # # Set the minutes and seconds to 0
    # # periods = periods.map(lambda p: p.asfreq('H'))
    # # Set the seconds to 0
    # periods = periods.map(lambda p: p.asfreq('H').to_timestamp())

    # # Reset the seconds to 0
    # periods = periods.map(lambda ts: ts.replace(second=0))
    # time_inference1 = time_inference1.to_frame(index=False)
    # time_stop = periods.to_frame(index=False).tail(1).values
    # print(type(start))
    if type( start) is np.ndarray:
        start_time = start[0]
    else:
        start_time = start
    if isinstance(start_time, (pd.Timestamp, datetime)):
        # Convert the timestamp to NumPy datetime64
        start_time = np.datetime64(start_time)        
    if type(stop) is np.ndarray:
        time_stop = stop[0]
    else:
        time_stop = stop
    if isinstance(time_stop, (pd.Timestamp, datetime)):
        # Convert the timestamp to NumPy datetime64
        time_stop = np.datetime64(time_stop)
    # start_time = start # df.tail(1)['Time'].values  # Convert the last period to a timestamp
    # start_time = start_time[0]
    # time_stop = time_stop[0][0]
    # time_stop = stop
    # print(start_time)
    start_time = np.datetime64(start_time.astype('datetime64[h]'))
    time_stop = np.datetime64(time_stop.astype('datetime64[h]'))
    # print('**********************')
    # print(time_now)
    # print(last_time)
    hours_difference = (start_time - time_stop)
    hours_difference = np.abs(hours_difference.astype('int'))
    # print(hours_difference)







    upsampleddf2 = df.copy()
    enhancedf = enhanceDataSet(upsampleddf2) # Add other column (day, hour, minute...)
    enhancedf['t-1'] = enhancedf['value'].shift(periods=1)
    enhancedf['t-2'] = enhancedf['value'].shift(periods=2)
    enhancedf['t-3'] = enhancedf['value'].shift(periods=3)
    numberRowsAdded = 3
    intervalSeconds = enhancedf.loc[1,'Time']-enhancedf.loc[0,'Time']
    intervalSeconds = intervalSeconds.total_seconds()
    intervalHours = intervalSeconds/3600 #time between 2 rows in hours
    intervalDays = intervalSeconds/(3600*24)#time between 2 rows in days


    if intervalHours<1:
        enhancedf['t-1h'] = enhancedf['value'].shift(periods=int(1/intervalHours))
        numberRowsAdded = max(numberRowsAdded,int(1/intervalHours))


    if intervalDays<7:
        enhancedf['t-7days'] = enhancedf['value'].shift(periods=int(1/intervalHours*24*7))
        numberRowsAdded = max(numberRowsAdded,int(1/intervalHours*24*7))

    #if display:
        #show_trend(df, 'DayName')
    dfcopie = enhancedf.copy()
    enhancedf.drop(index=enhancedf.index[:numberRowsAdded], axis=0, inplace=True)

    # Drop 
    enhancedf.drop('Year', axis=1, inplace=True)
    enhancedf.drop('Month', axis=1, inplace=True)
    enhancedf.drop('DayName', axis=1, inplace=True)
    enhancedf.drop('Day', axis=1, inplace=True)
    enhancedf = enhancedf.reset_index(drop=True)
    # enhancedf.to_csv(processed_data_file,index=False)
    enhancedf['Time'] = enhancedf['Time'].astype('datetime64[s]')
    # Models = [ModelTesting(RandomForestRegressor, ["n_estimators", "random_state"], [1, 0], [101, 1], [20, 1]), ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [3], proportional=True)]
    # Models = [ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [10], proportional=True)]
    # Models = [ModelTesting(GradientBoostingRegressor, ["n_estimators"], [60], [61], [20])] #, ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [3], proportional=True)]
    # model_list = ['GradientBoost', 'kNN']


    # we now fill the data until now with forecasts (in case there is a lot of data missing)
    #check if model exists
    dirname = os.getcwd()
    ext = ('.pkl')
    name_str = typeofsensor+"_aggregation_"+aggregation + "_forecast_horizon_" + forecasthorizon +".pkl"
    # print("looking for model : ", PROCESSED_DATA_FILE+typeofsensor+"_aggregation_"+ aggregation + "_forecast_horizon_" + forecasthorizon +".pkl", "in : ", dirname)
    # check if data file exists with that aggregation 
    booleen = 0
    # print(MODEL_FILE_NAME)
    for file in os.listdir(dirname):
        if file.endswith(ext):
            if name_str in file:
                booleen = 1
                print("existing trained model found")
                modelfilename = MODEL_FILE_NAME+typeofsensor+ "_aggregation_"+ aggregation + "_forecast_horizon_" + forecasthorizon +".pkl"
    # if booleen == 0: #if the model does not exist
    #     if 'H' in forecasthorizon:
    #         name_str = "_aggregation_"+aggregation + "_forecast_horizon_1H.pkl"
    #         booleen = 0
    #         for file in os.listdir(dirname):
    #             if file.endswith(ext):
    #                 if name_str in file:
    #                     booleen = 1
    #                     print("We will use model for 1H instead")
    #                     modelfilename = MODEL_FILE_NAME+typeofsensor+ "_aggregation_"+ aggregation + "_forecast_horizon_" + "1H" +".pkl"
    #     elif 'D' in forecasthorizon:
    #         name_str = "_aggregation_"+aggregation + "_forecast_horizon_1D.pkl"
    #         booleen = 0
    #         for file in os.listdir(dirname):
    #             if file.endswith(ext):
    #                 if name_str in file:
    #                     booleen = 1
    #                     print("We will use model for 1D instead")
    #                     forecasthorizon = '1D'           
    #                     modelfilename = MODEL_FILE_NAME+typeofsensor+ "_aggregation_"+ aggregation + "_forecast_horizon_" + "1D" +".pkl"
# time_inference1 = pd.period_range(start=DataSetIndexbyTime.tail(1).index[0]+pd.Timedelta(hours=1), end=DataSetIndexbyTime.tail(1).index[0]+pd.Timedelta(hours=numberofhoursfortime), freq='H').start_time
    if booleen == 1:
        # print("ici")
        # # load the model from disk
        loadedModel = joblib.load(modelfilename)
        # result = loaded_model.score(X_test, Y_test)
        # print(result)
        if aggregation =='1H':
            numberRowsForecastHorizon = hours_difference
        elif aggregation =='1D':
            numberRowsForecastHorizon = round(hours_difference/24)
            # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ ", numberRowsForecastHorizon)

        else:
            print("case not considered yet")

        """
        This function will:
        - Try different size of df (Whole df (10 years), 1 year, 6 months, 1 month, 2 weeks) according to self.time_ranges
        - Try different regressor with different params (see benchmark_regressor) according to self.benchs
        """
        print("$$$$$ STARTING INFERENCE $$$$$")
        # Different size of DataFrame
        index = []
        listBestR2 = []
        indexmodel =0
        # self.logs[1][time_range] = [] # This array will be filled soon with regressor's r2 values

        initialDataSet = enhancedf[enhancedf.DayOfWeek<=NUMBER_DAYS_CONSIDERED].copy().reset_index(drop=True)  # we remove Week Ends
        # initialDataSet = enhancedf.reset_index(drop=True)  # we remove Week Ends

        initialDataSetCopy=initialDataSet.copy()
        upsampledDataSet = initialDataSetCopy.set_index('Time') # Needed for resample
        # upsampledDataSet = upsampledDataSet.resample(aggregation).mean().bfill() # At this point, df is only composed of Time as index (every 15mins) and dB as column
        upsampledDataSet = upsampledDataSet.reset_index(drop=False) # Get back Time column

        DataSetIndexbyTime = upsampledDataSet.set_index('Time') # Have to do this in order to make the next line work ("last" function)
        numberRowsfor1day = len(DataSetIndexbyTime.first('1D'))
        if len(DataSetIndexbyTime) <= len(DataSetIndexbyTime.first(TRAINING_HORIZON)): #if the data set is too small, we must select as much as possible
            numberRowsforR2 = max(len(DataSetIndexbyTime.first(TRAINING_FREQUENCY)),len(DataSetIndexbyTime.first(REQUESTED_INTERVAL_R2)))
            numberRowsinTrainDataSet =max(len(DataSetIndexbyTime)-numberRowsforR2,len(DataSetIndexbyTime.first(TRAINING_HORIZON)))
            numberRowsforTraining = len(DataSetIndexbyTime.first(TRAINING_FREQUENCY))
            # numberRowsinTrainDataSet = min(len(DataSetIndexbyTime)-numberRowsforR2,len(DataSetIndexbyTime.first(TRAINING_HORIZON)))
            currentDataSet = upsampledDataSet.copy().tail(numberRowsinTrainDataSet+numberRowsforR2)
            currentDataSet = currentDataSet.reset_index(drop=True)
            trainDataSet = currentDataSet.copy().head(numberRowsinTrainDataSet)
            indexmodel =0

            # modelname = model_list[indexmodel]
            Bestr2Model = []
            BestparamModel = []
            averageR2 = 0
            bestAverageR2 = 0
            # print(currentDataSet)
            # print(numberRowsinTrainDataSet)
            # for day in range(numberofdaysinDataSet):
            #     if day%50==0:
            #         print(day)
            #     currentDataSet = upsampledDataSet.copy()[day*numberRowsfor1day:day*numberRowsfor1day+numberRowsinTrainDataSet+numberRowsforR2]
            #     currentDataSet = currentDataSet.reset_index(drop=True)
            trainDataSet = currentDataSet.copy().head(numberRowsinTrainDataSet)
            testDataSet = currentDataSet.copy().tail(max(numberRowsfor1day*7+1,len(currentDataSet)-len(trainDataSet)))
        else:
            numberRowsforR2 = max(len(DataSetIndexbyTime.first(TRAINING_FREQUENCY)),len(DataSetIndexbyTime.first(REQUESTED_INTERVAL_R2)))
            numberRowsinTrainDataSet =min(len(DataSetIndexbyTime)-numberRowsforR2,len(DataSetIndexbyTime.first(TRAINING_HORIZON)))
            numberRowsforTraining = len(DataSetIndexbyTime.first(TRAINING_FREQUENCY))

            # numberRowsinTrainDataSet = min(len(DataSetIndexbyTime)-numberRowsforR2,len(DataSetIndexbyTime.first(TRAINING_HORIZON)))
            currentDataSet = upsampledDataSet.copy().tail(numberRowsinTrainDataSet+numberRowsforR2)
            currentDataSet = currentDataSet.reset_index(drop=True)
            trainDataSet = currentDataSet.copy().head(numberRowsinTrainDataSet)
            indexmodel =0

            # modelname = model_list[indexmodel]
            Bestr2Model = []
            BestparamModel = []
            averageR2 = 0
            bestAverageR2 = 0
            # print(currentDataSet)
            # print(numberRowsinTrainDataSet)
            # for day in range(numberofdaysinDataSet):
            #     if day%50==0:
            #         print(day)
            #     currentDataSet = upsampledDataSet.copy()[day*numberRowsfor1day:day*numberRowsfor1day+numberRowsinTrainDataSet+numberRowsforR2]
            #     currentDataSet = currentDataSet.reset_index(drop=True)
            trainDataSet = currentDataSet.copy().head(numberRowsinTrainDataSet)
            testDataSet = currentDataSet.copy().tail(max(numberRowsfor1day*7+1,len(currentDataSet)-len(trainDataSet)))

        BEST_BENCH = None
        # X_train, X_test, Y_train, Y_test = train_test_split(input, target, test_size=int(numberRowsinTrainDataSet*0.2), shuffle=False)
        Y_train = trainDataSet.copy()[['value']] # Expected results
        X_train = trainDataSet.copy().drop(columns=['value', 'Time'])
        Y_test = testDataSet.copy()[['value']] # Expected results
        X_test = testDataSet.copy().drop(columns=['value', 'Time'])
        Y_train = Y_train.reset_index(drop=True)
        X_train = X_train.reset_index(drop=True)
        Y_test = Y_test.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        # plt.plot(Yout, label = "Pred")
        # plt.plot(Y_test,  label = "Truth")
        # plt.legend()
        # plt.show()

        averageparam = []            
        averageR2 = sum(Bestr2Model)/max(1,len(Bestr2Model))
        arrayParam = np.array(BestparamModel)
        if arrayParam.any():
            for ituple in range(arrayParam.shape[1]):
                averageparam.append(np.average(arrayParam[:,ituple]))

        # self.logs[1][list(self.logs[1].keys())[len(self.logs[1])-1]].append(model.best[1]) # Add r2 score for current regressor and time_range
        # if BEST_BENCH == None or model.best[1] > BEST_BENCH.best[1]:
        #     BEST_BENCH = model
        if averageR2 >= bestAverageR2:
            bestAverageR2 = averageR2
            # bestparam = averageparam
            # BEST_MODEL = loadedModel
        # print("training horizon: ", TRAINING_HORIZON, "forecast horizon :", forecasthorizon, " best model : ", averageparam, "r2 average: ", bestAverageR2, "number of r2: ", len(Bestr2Model), " Lowest r2: ", sum(Lowestr2Model)/max(1,len(Lowestr2Model)), "number of low r2: ", len(Lowestr2Model))   
        listBestR2.append(bestAverageR2)
        indexmodel=indexmodel+1
        # print(X_train)
        # print(numberRowsForecastHorizon)
        y = simpleinference(loadedModel, X_train, Y_train, X_test, Y_test, numberRowsForecastHorizon, DataSetIndexbyTime) 
        # print(numberRowsForecastHorizon)
        # print(len(y))

        # y = pd.DataFrame(y)
    # print(df.tail(1)['Time'].values[0])
    # print(type(df.tail(1)['Time'].values[0]))
    # start_time = np.datetime64(df.tail(1)['Time'].values[0], 's')
    else:
        y=0
    # # periods = pd.period_range(start=start_time, periods=numberRowsForecastHorizon, freq='H')  # pd.period_range(start=start_time, periods=numberRowsForecastHorizon, freq='15T') # Create a period range

    # # # Set the minutes and seconds to 0
    # # # periods = periods.map(lambda p: p.asfreq('H'))
    # # # Set the seconds to 0
    # # periods = periods.map(lambda p: p.asfreq('H').to_timestamp())

    # # # Reset the seconds to 0
    # # periods = periods.map(lambda ts: ts.replace(second=0))
    # # # time_inference1 = time_inference1.to_frame(index=False)
    # # time_infer = periods.to_frame(index=False).tail(numberRowsForecastHorizon).values
    # # time_infer = pd.DataFrame(time_infer)
    # # print(time_infer)
    # # y = pd.DataFrame(y)
    # # df_combined = pd.concat([time_infer, y], axis=1)

    # # df_combined.columns=['Time', 'value']
    # # print("prediction y:")
    # # # print(type(y))
    # # print(y)
    # # print("dataframe ")
    # # # print(type(upsampleddf))
    # # print(df)

    # # df_final= pd.concat([df, df_combined], ignore_index=True)
    # # print("df_final: ")

    # # print(df_final)



    # # df_final.to_csv("df_final.csv",index=False)
    return y






############################  Analyse data  #################################
def fill_gap(df, max_time_interval_second, type_of_sensor, aggregation, forecast_horizon, plot=False):
    # we move everything into numpy, because pandas ne gere pas bien les calculs.
    ValueArr = df['value'].to_numpy()
    #on recupere le temps dans timearray
    df['Time'] = df['Time'].astype('datetime64[s]') 
    timearray = df['Time'].to_numpy()
    print("pre-processing data")
    # conversion des dates, et on enleve les lignes qui n'etaient pas bonnes (time not in a date format, or just some space instaead of time)
    count = 0 
    min_timeinterval = 10000000000000000000 # just a large number as we will look for the min
    array_min_timeintervals = {} # we create an array that will have the time intervals and the number of occurence for each time interval
    for idx in range(0,min(30000,timearray.size)): # now we check the most usual steps.
        tmp = {}
        delta = abs(timearray[idx-count] - timearray[idx-count-1])/ np.timedelta64(1, 's')
        bool = 0
        for idxarr in range(len(array_min_timeintervals)):
            if(abs(delta-array_min_timeintervals[idxarr][0])<(1)): #if we find a delta in time that is already in our list of deltas
                array_min_timeintervals[idxarr][1]=array_min_timeintervals[idxarr][1]+1
                bool=1
        if bool==0:
            tmp[0] = delta
            tmp[1] = 1
            array_min_timeintervals[len(array_min_timeintervals)] = tmp
    most_frequent_delta = 0
    highest_frequency = 0
    for idx in range(len(array_min_timeintervals)):
        if array_min_timeintervals[idx][1] >highest_frequency:
            most_frequent_delta = array_min_timeintervals[idx][0] 
            highest_frequency = array_min_timeintervals[idx][1] 
    max_time_interval_second = max(max_time_interval_second,most_frequent_delta)

    booleen10=0
    for idx in range(0,timearray.size): # now we delete the necessary lines and convert others in datetime.
        progress = idx/timearray.size*100
        if np.floor(progress)%20 ==0 and booleen10 ==0:
            print(np.floor(progress),"%")
            booleen10=1
        if np.floor(progress)%20 ==1 and booleen10 ==1:
            booleen10 = 0
        # if idx>100000:
        #     temp=0
        if isinstance(timearray[idx-count], float) or timearray[idx-count] ==' ':
            timearray = np.delete(timearray,idx-count,0)
            ValueArr = np.delete(ValueArr,idx-count,0)
            count = count+1
        # elif abs(timearray[idx-count-1]-timearray[idx-count])<int(max_time_interval_second):  # if the data is redundant we remove one of both entries.
        #     timearray = np.delete(timearray,idx-count,0)
        #     ValueArr[idx-count-1]= (ValueArr[idx-count-1]+ValueArr[idx-count])/2
        #     ValueArr = np.delete(ValueArr,idx-count,0)
        #     count = count+1        
        elif isinstance(timearray[idx-count], str):
            timearray[idx-count] = datetime.strptime(timearray[idx-count], '%Y-%m-%dT%H:%M:%S') #'%d/%m/%Y %H:%M') # on ajoute le count car on a supprimé des lignes
            if idx-count >1:
                delta = abs(timearray[idx-count] - timearray[idx-count-1])
                #we keep it only if it is well spread in the dataset
                for idx5 in range(len(array_min_timeintervals)):
                    if array_min_timeintervals[idx5][0] == delta.total_seconds():
                        if array_min_timeintervals[idx5][1]>(1/3*min(30000,timearray.size)): #we keep this interval if it represents more than 1/3 of the dataset
                            min_timeinterval = min(min_timeinterval, delta.total_seconds())
        # elif  isinstance(timearray[idx-count], np.datetime64):
        #     # timearray[idx-count] = datetime.strptime(timearray[idx-count], '%Y-%m-%dT%H:%M:%S') #'%d/%m/%Y %H:%M') # on ajoute le count car on a supprimé des lignes
        #     if idx-count >1:
        #         delta = abs(timearray[idx-count] - timearray[idx-count-1])
        #         #we keep it only if it is well spread in the dataset
        #         for idx5 in range(len(array_min_timeintervals)):
        #             if array_min_timeintervals[idx5][0] == delta.astype('timedelta64[s]').astype(int):
        #                 if array_min_timeintervals[idx5][1]>(1/3*len(timearray)): #we keep this interval if it represents more than 1/3 of the dataset
        #                     min_timeinterval = min(min_timeinterval, delta.astype('timedelta64[s]').astype(int))
        elif isinstance(timearray[idx-count], np.datetime64):
            # timearray[idx-count] = datetime.strptime(timearray[idx-count], '%Y-%m-%dT%H:%M:%S') #'%d/%m/%Y %H:%M') # on ajoute le count car on a supprimé des lignes
            if idx-count >1:
                delta = abs(timearray[idx-count] - timearray[idx-count-1])/ np.timedelta64(1, 's')
                if int(delta) == int(max_time_interval_second):
                              min_timeinterval = min(min_timeinterval, delta)
                else:
                    #we keep it only if it is well spread in the dataset
                    for idx2 in range(len(array_min_timeintervals)):
                        if int(array_min_timeintervals[idx2][0]) == int(delta) :
                            if array_min_timeintervals[idx2][1]>(1/3*min(30000,timearray.size)): #we keep this interval if it represents more than 1/3 of the dataset
                                min_timeinterval = min(min_timeinterval, delta)
        if min_timeinterval == 0:
            print("Error: a minimum time interval of 0 was found ###############")
    Value_original = ValueArr
    if min_timeinterval ==10000000000000000000:
        print("Error: did not find any time format ######################")

    if min_timeinterval<max_time_interval_second:
        print("beware that the smallest time step found : ",min_timeinterval, "adopted timestep: ", max_time_interval_second)
        min_timeinterval=max_time_interval_second

    for idx in range(ValueArr.size-1):
        if ValueArr[idx] == ' ' or ValueArr[idx] == '':
            ValueArr[idx]  = 0
        
 



    print("processing data")

    #################### # Creation of long arrays that will include all data at the smallest timestep => Interpolation ##########################################################
    #we should extract the minimum timedelta betwen dates in order to create a 
    min_delta_in_min =  divmod(min_timeinterval, 60)[0]
    # %reset_selective -f "complete_time" 

    #now we create an array of date time with the minmum time interval

    complete_time = np.arange(timearray[0] , timearray[timearray.size-1],  np.timedelta64(int(min_delta_in_min), 'm'), dtype='datetime64[m]')
    print("number of rows addedd due to holes in data: ", complete_time.size-timearray.size)
    # convert timearray into datetime64
    if min_timeinterval >=3600*24:
        for i in range(timearray.size):
            timearray[i] = np.datetime64(timearray[i],'h')
    elif min_timeinterval >=3600:
        for i in range(timearray.size):
            timearray[i] = np.datetime64(timearray[i], 'm') 
    elif min_timeinterval >=60:
        for i in range(timearray.size):
            timearray[i] = np.datetime64(timearray[i],'s')
    # compute the duration of the whole dataset
    duration =  (timearray[timearray.size-1]-timearray[1] )#/ np.timedelta64(1, 's')

    duration_in_min = duration/np.timedelta64(1,'m')


    ValueArr = Value_original.astype(float) # convert ValueArr into floats

    ValueArr_long = np.ones(complete_time.size, dtype=float)
    ValueArr_long[0] = ValueArr[0]

    #let's fill our reference arrays (time and ValueArr) at the smallest timestep
    idx2 = 0
    booleen = 0
    coeff_dir = 0
    idx4=0
    idx3=0
    idx4_start=0
    booleen4=0
    booleen10 = 0
    for i in range(complete_time.size):
        if i==232:
            temp=1
        temp_value = 0
        countervalue=0
        progress = i/complete_time.size*100
        if np.floor(progress)%20 ==0 and booleen10 ==0:
            print(np.floor(progress),"%")
            booleen10=1
        if np.floor(progress)%20 ==1 and booleen10 ==1:
            booleen10 = 0

        # print(complete_time[i])
        # print(timearray[idx2])
        if(i>4862):
            a = 0
        if idx2 > 1 and (idx2< timearray.size) and timearray[idx2] - timearray[idx2-1] < np.timedelta64(1,'s'): # for time shift in october (heure d'hiver)
            idx2_shift = 0
            timereference = timearray[idx2-1]

            while complete_time[i] - timearray[min(idx2+idx2_shift,len(timearray)-1)] >= np.timedelta64(1,'m') :
            # timearray[idx2+idx2_shift] - timearray[idx2-1] < 0:
                idx2_shift = idx2_shift+1
                if idx2_shift >= idx2:
                    idx2_shift = 0
                    break
            idx2 = idx2 + idx2_shift
            # if idx2 < timearray.size and abs(complete_time[i] - timearray[idx2]) < np.timedelta64(1,'m') :  #if the current time is represented both in comprehensive time and in timearray (original time)
        if idx2 < timearray.size and abs(complete_time[i] - timearray[idx2]) < np.timedelta64(int(min_delta_in_min),'m') :  #if the current time is represented both in comprehensive time and in timearray (original time)
            # ValueArr_long[i] = ValueArr[idx2]
            temp_value = temp_value+ValueArr[idx2]
            countervalue=countervalue+1
            idx2 = idx2+1
            booleen = 0
            idx4_start = idx2
            if idx4+idx4_start>=timearray.size-1:
                ValueArr_long[i] = temp_value/countervalue
            for idx4 in range(timearray.size-idx4_start):   
                if abs(complete_time[i] - timearray[idx4+idx4_start]) < np.timedelta64(int(min_delta_in_min),'m') :  #if the current time is represented both in comprehensive time and in timearray (original time)
                    # ValueArr_long[i] = ValueArr[idx4+idx4_start]
                    temp_value = temp_value+ValueArr[idx4+idx4_start]
                    countervalue=countervalue+1
                    idx2 = idx4+idx4_start+1
                    booleen = 0
                    booleen4 = 1
                if timearray[idx4+idx4_start] - complete_time[i] > np.timedelta64(int(min_delta_in_min),'m') or  idx4+idx4_start==timearray.size-1:
                    ValueArr_long[i] = temp_value/countervalue
                    break

        elif idx2 < timearray.size:
        # Linear interpolation
                # if abs(timearray[idx2] - timearray[idx2-1]) < np.timedelta64(60,'m'): # if the missing time is below 1hour, we say ValueArr at t (i) = averagrde between the 2 times
                #     ValueArr_long[i] = (ValueArr[idx2]+ValueArr[idx2-1])/2
                #     # booleen = 0
                # else:
                    # ValueArr_long[i] = 0
            # start =   timearray[idx2-1] #dfinterm.tail(1)['Time'].values  
            # stop = timearray[idx2]   

            booleen4=0
            for idx4 in range(timearray.size-idx4_start):   
                if abs(complete_time[i] - timearray[idx4+idx4_start]) < np.timedelta64(int(min_delta_in_min),'m') :  #if the current time is represented both in comprehensive time and in timearray (original time)
                    # ValueArr_long[i] = ValueArr[idx4+idx4_start]
                    temp_value = temp_value+ValueArr[idx4+idx4_start]
                    countervalue=countervalue+1
                    idx2 = idx4+idx4_start+1
                    booleen = 0
                    booleen4 = 1
                if booleen4==1 and timearray[idx4+idx4_start] - complete_time[i] > np.timedelta64(int(min_delta_in_min),'m') :
                    ValueArr_long[i] = temp_value/countervalue
                    break
                if booleen4 ==0 and timearray[idx4+idx4_start] - complete_time[i] > np.timedelta64(int(min_delta_in_min),'m') :
                    break

            if booleen==0 and booleen4==0:
                dfinterm = pd.DataFrame({'Time': complete_time[0:i], 'value': ValueArr_long[0:i]})
                # start = dfinterm.tail(1)['Time'].values 
                if type( timearray[idx2-1]) is np.ndarray:
                    start = timearray[idx2-1][0]
                else:
                    start = timearray[idx2-1]

                if type(timearray[idx2]) is np.ndarray:
                    stop = timearray[idx2][0]
                else:
                    stop = timearray[idx2]
                # stop = timearray[idx2]
                # print(idx2)
                # print("start")
                # print(start)
                # print("stop")
                # print(stop)
                interpolatedvalues = interpolationByInference(dfinterm, aggregation, forecast_horizon, type_of_sensor, start, stop)
                # print("voila")
                booleen = 1
                coeff_dir = (ValueArr[idx2]-ValueArr[idx2-1])/(int((timearray[idx2]-timearray[idx2-1])/np.timedelta64(1,'m'))/min_delta_in_min)
                idx3 = 0
            if 'interpolatedvalues' in locals() and interpolatedvalues !=0:    
                ValueArr_long[i] = interpolatedvalues[min(idx3+1,len(interpolatedvalues)-1)]
                # if idx3 == len(interpolatedvalues)-1:
                    # print("we have exceeded  indexs")
                    # print(interpolatedvalues)
                    # print(len(interpolatedvalues))
                    # print(idx3)
                idx3= idx3+1
            else:
                ValueArr_long[i] = ValueArr_long[i-1] +coeff_dir

            # if ValueArr_long[i]<50:
            #     print("break")
        else:
            ValueArr_long[i] = ValueArr[idx2-1]
        #We need to deal with the last data as well.
        # plt.plot(a,  label = "Truth")
    # plt.plot(timearray, Value_original, label="real")
    # plt.plot(complete_time, ValueArr_long,  label = "interpolated")
    # plt.legend()
    # plt.show()  
    
    df2 = pd.DataFrame(ValueArr_long,columns=['value'])
    df1 = pd.DataFrame(complete_time,columns=['Time'])
    df = pd.concat([df1,df2], axis = 1)


    start = df.tail(1)['Time'].values  # Convert the last period to a timestamp
    stop = pd.Timestamp.now() 
    # periods = pd.period_range(start=stop, periods=1, freq='H')  # Create a period range

    # # Set the minutes and seconds to 0
    # periods = periods.map(lambda p: p.asfreq('H'))
    # # Set the seconds to 0
    # periods = periods.map(lambda p: p.asfreq('H').to_timestamp())

    # # Reset the seconds to 0
    # periods = periods.map(lambda ts: ts.replace(second=0))
    # time_inference1 = time_inference1.to_frame(index=False)
    # time_stop = periods.to_frame(index=False).tail(1).values
    start_time = start # df.tail(1)['Time'].values  # Convert the last period to a timestamp
    start_time = start_time[0]

    dfinterm = pd.DataFrame({'Time': complete_time[0:i], 'value': ValueArr_long[0:i]})

    if aggregation.endswith('m'):
        # duration_type = 'minutes'
        duration_value = int(aggregation[:-1])
        next_stop = start_time + np.timedelta64(duration_value, 'm')
    elif aggregation.endswith('H'):
        duration_value = int(aggregation[:-1])
        next_stop = start_time + np.timedelta64(duration_value, 'h')
    elif aggregation.endswith('D'):
        # duration_type = 'days'
        duration_value = int(aggregation[:-1])
        next_stop = start_time + np.timedelta64(duration_value, 'D')
    else:
        print("Invalid aggregation string")
    


    next_stop_datetime = next_stop.astype(datetime)
    stop_timestamp = int(stop.timestamp())

    try:
        next_stop_datetime.timestamp()
    except Exception as e:
        next_stop_datetimeint = next_stop_datetime

    else:
        next_stop_datetimeint= int(next_stop_datetime.timestamp())

    if stop_timestamp > next_stop_datetimeint: # if we are missing at least one data between the end of the database and now

        if min_timeinterval >=3600*24:
            time_stop = stop - np.timedelta64(stop.minute, 'h')- np.timedelta64(stop.minute, 'm') - np.timedelta64(stop.second, 's')
            time_stop = np.datetime64(time_stop,'s')
            
        elif min_timeinterval >=3600:
            time_stop = stop - np.timedelta64(stop.minute, 'm') - np.timedelta64(stop.second, 's')
            time_stop = np.datetime64(time_stop,'s')
        elif min_timeinterval >=60:
            time_stop = stop  - np.timedelta64(stop.second, 's')
            time_stop = np.datetime64(time_stop,'s')

        interpolatedvalues = interpolationByInference(dfinterm, aggregation, forecast_horizon, type_of_sensor, start_time, time_stop)
    else:
        interpolatedvalues=0
    if(interpolatedvalues!=0):

        start_time = np.datetime64(start_time.astype('datetime64[h]'))
        time_stop = np.datetime64(time_stop.astype('datetime64[h]'))
        # print(time_now)
        # print(last_time)
        hours_difference = (start_time - time_stop)
        numberRowsForecastHorizon = np.abs(hours_difference.astype('int'))
        periods = pd.period_range(start=start_time, periods=numberRowsForecastHorizon, freq='H')  # pd.period_range(start=start_time, periods=numberRowsForecastHorizon, freq='15T') # Create a period range

        # Set the minutes and seconds to 0
        # periods = periods.map(lambda p: p.asfreq('H'))
        # Set the seconds to 0
        periods = periods.map(lambda p: p.asfreq('H').to_timestamp())

        # Reset the seconds to 0
        periods = periods.map(lambda ts: ts.replace(second=0))
        # time_inference1 = time_inference1.to_frame(index=False)
        time_infer = periods.to_frame(index=False).tail(numberRowsForecastHorizon).values
        time_infer = pd.DataFrame(time_infer)
        # print(time_infer)
        # print(interpolatedvalues)
        # print(type(interpolatedvalues))
        y = pd.DataFrame(interpolatedvalues)
        if time_infer.shape[0]==y.shape[0]:
         df_combined = pd.concat([time_infer, y], axis=1)

         df_combined.columns=['Time', 'value']
        else:
         print("cannot concatenate df_combined due to size issue:")
         print(time_infer)
         print("-------")
         print(y)
         df_combined=pd.DataFrame()
        # print("prediction y:")
        # # print(type(y))
        # print(y)
        # print("dataframe ")
        # # print(type(upsampleddf))
        # print(df)
        # plt.figure(figsize=(10, 6))  # Set the size of the plot (optional)
        # plt.plot(df_combined['Time'], df_combined['value'])  # Plot the data
        # plt.xlabel('Time')  # Set the label for the x-axis
        # plt.ylabel('Value')  # Set the label for the y-axis
        # plt.title('Plot of interpolated data')  # Set the title of the plot
        # plt.grid(True)  # Show gridlines (optional)
        # plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility (optional)
        # plt.tight_layout()  # Adjust layout to prevent clipping of labels (optional)
        # plt.show()  # Show the plot
        if df.shape[0]==df_combined.shape[0]:
         df_final= pd.concat([df, df_combined], ignore_index=True)
        else:
         print("cannot concatenate df_final due to size issue")
         print(df.columns)
         print("-------")
         print(df_combined)
         df_final = df
        # print("df_final: ")

        # print(df_final)
        # df_final.to_csv("df_final_test.csv",index=False)
        df = df_final
        interpolateddf = df_final

        # # Plot the DataFrame
        # plt.figure(figsize=(10, 6))  # Set the size of the plot (optional)
        # plt.plot(df['Time'], df['value'])  # Plot the data
        # plt.xlabel('Time')  # Set the label for the x-axis
        # plt.ylabel('Value')  # Set the label for the y-axis
        # plt.title('Plot of interpolated data')  # Set the title of the plot
        # plt.grid(True)  # Show gridlines (optional)
        # plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility (optional)
        # plt.tight_layout()  # Adjust layout to prevent clipping of labels (optional)
        # plt.show()  # Show the plot
    # df.to_csv("df_final_test.csv",index=False)


    return df



def enhanceDataSet(df):
    df['Year'] = df['Time'].dt.year
    df['Month'] = df['Time'].dt.month
    df['Day'] = df['Time'].dt.day
    df['DayOfWeek'] = df['Time'].dt.dayofweek
    df['DayName'] = df['Time'].dt.day_name()
    df['Hour'] = df['Time'].dt.hour
    df['Minute'] = df['Time'].dt.minute
    df['TimeSlot'] = df['Hour'].map(lambda x: time_slot_assigner(x))


    return df


def time_slot_assigner(hour):
    if 2 <= hour <= 5:
        return 1
    elif 6 <= hour <= 9:
        return 2
    elif 10 <= hour <= 13:
        return 3
    elif 14 <= hour <= 17:
        return 4
    elif 18 <= hour <= 21:
        return 5
    else:
        return 0
    
def process_data(csv_file_name, aggregation, processed_data_file,typeofsensor, forecast_horizon):
    df = pd.read_csv(csv_file_name, sep=',', skiprows=[0, 1], usecols=[0,1], names=["Time", "value"])
    # df = pd.read_csv(csv_file_name, sep=',', skiprows=[0], usecols=[2,5], names=["Time", "value"])
    df['Time'] = pd.to_datetime(df['Time'], dayfirst=False)
    # df = pd.read_csv(file, sep=';', skiprows=[0, 1, 2, 3], usecols=[0, 1], names=["Time", "value"])
    # df['Time'] = pd.to_datetime(df['Time'], dayfirst=True)
    df = df.sort_values(by='Time')
    df = df.reset_index(drop=True)
    # df["value"] = df["value"].str.replace(',', '.').astype(float)
    display = False

    # df = analyse(df, display)
    df.dropna(inplace=True)
    df.reset_index()
    saved = df.copy() # Used for aggregation later on
    # print(df)
    print("will fill data, with aggration = ", aggregation)
    if aggregation=='1H':
       df = fill_gap(df, 60*60,typeofsensor,aggregation,forecast_horizon,display)
    elif aggregation=='1D':   
       df = fill_gap(df, 3600*24,typeofsensor,aggregation,forecast_horizon,display)
    else:
       df = fill_gap(df, 3600*48,typeofsensor,aggregation,forecast_horizon,display)
    # print(df)
    dfcopy=df.copy()

    if aggregation == 'none':
        upsampleddf = dfcopy.set_index('Time') # Needed for resample
        upsampleddf = upsampleddf.reset_index(drop=False) # Get back Time column
    else:

        upsampleddf = dfcopy.set_index('Time') # Needed for resample
        upsampleddf = upsampleddf.resample(aggregation).mean().bfill() # At this point, df is only composed of Time as index (every 15mins) and dB as column
        if upsampleddf.size >= dfcopy.size:
            upsampleddf = dfcopy.set_index('Time') # Needed for resample
        upsampleddf = upsampleddf.reset_index(drop=False) # Get back Time column





    enhancedf = enhanceDataSet(upsampleddf) # Add other column (day, hour, minute...)
    enhancedf['t-1'] = enhancedf['value'].shift(periods=1)
    enhancedf['t-2'] = enhancedf['value'].shift(periods=2)
    enhancedf['t-3'] = enhancedf['value'].shift(periods=3)
    numberRowsAdded = 3
    intervalSeconds = enhancedf.loc[1,'Time']-enhancedf.loc[0,'Time']
    intervalSeconds = intervalSeconds.total_seconds()
    intervalHours = intervalSeconds/3600 #time between 2 rows in hours
    intervalDays = intervalSeconds/(3600*24)#time between 2 rows in days


    if intervalHours<1:
        enhancedf['t-1h'] = enhancedf['value'].shift(periods=int(1/intervalHours))
        numberRowsAdded = max(numberRowsAdded,int(1/intervalHours))


    if intervalDays<7:
        enhancedf['t-7days'] = enhancedf['value'].shift(periods=int(1/intervalHours*24*7))
        numberRowsAdded = max(numberRowsAdded,int(1/intervalHours*24*7))

    #if display:
        #show_trend(df, 'DayName')
    dfcopie = enhancedf.copy()
    enhancedf.drop(index=enhancedf.index[:numberRowsAdded], axis=0, inplace=True)

    # Drop 
    enhancedf.drop('Year', axis=1, inplace=True)
    enhancedf.drop('Month', axis=1, inplace=True)
    enhancedf.drop('DayName', axis=1, inplace=True)
    enhancedf.drop('Day', axis=1, inplace=True)
    enhancedf = enhancedf.reset_index(drop=True)
    enhancedf.to_csv(processed_data_file,index=False)


    
def process_data2(csv_file_name, aggregation, processed_data_file, typeofsensor, forecast_horizon):
    df = pd.read_csv(csv_file_name, sep=',', skiprows=[0, 1], usecols=[2, 5], names=["Time", "value"])
    df['Time'] = pd.to_datetime(df['Time'], dayfirst=False)
    # df = pd.read_csv(file, sep=';', skiprows=[0, 1, 2, 3], usecols=[0, 1], names=["Time", "value"])
    # df['Time'] = pd.to_datetime(df['Time'], dayfirst=True)
    df = df.sort_values(by='Time')
    df = df.reset_index(drop=True)
    # df["value"] = df["value"].str.replace(',', '.').astype(float)
    display = False

    # df = analyse(df, display)
    df.dropna(inplace=True)
    df.reset_index()
    saved = df.copy() # Used for aggregation later on

    if aggregation=='1H':
       df = fill_gap(df, 3600,typeofsensor,aggregation,forecast_horizon,display)
    elif aggregation=='1D':   
       df = fill_gap(df, 3600*24,typeofsensor,aggregation,forecast_horizon,display)
    else:
       df = fill_gap(df, 3600*24*2,typeofsensor,aggregation,forecast_horizon,display)

    dfcopy=df.copy()

    if aggregation == 'none':
        upsampleddf = dfcopy.set_index('Time') # Needed for resample
        upsampleddf = upsampleddf.reset_index(drop=False) # Get back Time column
    else:

        upsampleddf = dfcopy.set_index('Time') # Needed for resample
        upsampleddf = upsampleddf.resample(aggregation).mean().bfill() # At this point, df is only composed of Time as index (every 15mins) and dB as column
        if upsampleddf.size >= dfcopy.size:
            upsampleddf = dfcopy.set_index('Time') # Needed for resample
        upsampleddf = upsampleddf.reset_index(drop=False) # Get back Time column



    enhancedf = enhanceDataSet(upsampleddf) # Add other column (day, hour, minute...)
    enhancedf['t-1'] = enhancedf['value'].shift(periods=1)
    enhancedf['t-2'] = enhancedf['value'].shift(periods=2)
    enhancedf['t-3'] = enhancedf['value'].shift(periods=3)
    numberRowsAdded = 3
    intervalSeconds = enhancedf.loc[1,'Time']-enhancedf.loc[0,'Time']
    intervalSeconds = intervalSeconds.total_seconds()
    intervalHours = intervalSeconds/3600 #time between 2 rows in hours
    intervalDays = intervalSeconds/(3600*24)#time between 2 rows in days


    if intervalHours<1:
        enhancedf['t-1h'] = enhancedf['value'].shift(periods=int(1/intervalHours))
        numberRowsAdded = max(numberRowsAdded,int(1/intervalHours))


    if intervalDays<7:
        enhancedf['t-7days'] = enhancedf['value'].shift(periods=int(1/intervalHours*24*7))
        numberRowsAdded = max(numberRowsAdded,int(1/intervalHours*24*7))

    #if display:
        #show_trend(df, 'DayName')
    dfcopie = enhancedf.copy()
    enhancedf.drop(index=enhancedf.index[:numberRowsAdded], axis=0, inplace=True)

    # Drop 
    enhancedf.drop('Year', axis=1, inplace=True)
    enhancedf.drop('Month', axis=1, inplace=True)
    enhancedf.drop('DayName', axis=1, inplace=True)
    enhancedf.drop('Day', axis=1, inplace=True)
    enhancedf = enhancedf.reset_index(drop=True)
    enhancedf.to_csv(processed_data_file,index=False)
  







class ModelTesting:
    def __init__(self, regressor, params_name, params_start, params_stop, params_steps, proportional=False):
        self.name = regressor
        self.regressor = regressor
        self.params_name = params_name
        self.params_start = params_start
        self.params_stop = params_stop
        self.params_steps = params_steps
        self.params_comb = []
        for i, param in enumerate(self.params_start):
            self.params_comb.append(list(np.arange(param, self.params_stop[i], self.params_steps[i])))
        self.proportional = proportional
        self.best = None


    def testParam(self, x_train, x_test, y_train, y_test, numberrowsforr2, datasetindexbytime, numberrowsforecasthorizon, numberrowsfortraining, verbose=True):
        r2previous1 = 0
        r2previous2 = 0
        # print("strating testing of parameters")
        if verbose: print("** * ** Bench for", self.regressor.__name__, "** * **")
        BEST = (None, None) # [0] -> model, [1] -> r2_score
        # print(itertools.product(*self.params_comb))
        for i, combination in enumerate(itertools.product(*self.params_comb)):
            keywords_args = {}
            for n, param_name in enumerate(self.params_name):
                if self.proportional and self.params_stop[n] > len(x_train):
                    s = self.params_start[n]
                    gr = self.params_steps[n] / self.params_stop[n]
                    max = len(x_train)
                    step = gr * max
                    next = int(s + (step * i))
                    keywords_args[param_name] = next
                else:
                    keywords_args[param_name] = combination[n]
            model = self.regressor(**keywords_args)

            model = model.fit(x_train, y_train) # Train model

            numberIndexShift1H =  len(datasetindexbytime .first('1H'))
            numberIndexShift7d =  len(datasetindexbytime .first('7D'))
            r2=[]
            valuet_1h = 0
            valuet_7d = 0
            valuet_1 = x_test[0:1]['t-1'].values[0]
            valuet_2 = x_test[0:1]['t-2'].values[0]
            valuet_3 = x_test[0:1]['t-3'].values[0]
            if 't-1h' in x_test.columns:
                valuet_1h=(x_test[0:1]['t-1h'].values[0])
            if 't-7days' in x_test.columns:
                valuet_7d=(x_test[0:1]['t-7days'].values[0])
            y_output = y_test.copy()
            y_pred = []
            y_true= []
            
            for inference in range(numberrowsforr2):
                if inference%(numberrowsfortraining) == 0 and inference >0:  #we retrain every day
                    x_train2 = pd.concat([x_train, x_test.head(inference)], ignore_index=True)
                    y_train2 = pd.concat([y_train, y_test.head(inference)], ignore_index=True)
                    # print("training size2")
                    # print(x_train2.size)
                    model = model.fit(x_train2, y_train2) # Train model

                y_test_unit = y_test.copy()[inference:inference+1]
                X_test_unit = x_test.copy()[inference:inference+1]
                if X_test_unit.empty == False:
                    if inference%numberrowsforecasthorizon == 0 and inference < len(x_test):
                        valuet_1 = X_test_unit['t-1'].values[0]
                        valuet_2 = X_test_unit['t-2'].values[0]
                        valuet_3 = X_test_unit['t-3'].values[0]
                    elif inference < len(x_test):
                        X_test_unit['t-1'] = valuet_1
                        X_test_unit['t-2'] = valuet_2
                        X_test_unit['t-3'] = valuet_3
                    if (inference >= numberIndexShift1H) and ('t-1h' in x_test.columns) and (numberrowsforecasthorizon > numberIndexShift1H ) and (inference < len(x_test)):
                        X_test_unit['t-1h'].values[0] = y_output.copy()[inference-numberIndexShift1H:inference-numberIndexShift1H+1]['value'].values[0]
                    if (inference >= numberIndexShift7d) and ('t-7days' in x_test.columns) and (numberrowsforecasthorizon > numberIndexShift7d ) and (inference < len(x_test)):
                        X_test_unit['t-7days'].values[0] = y_output.copy()[inference-numberIndexShift7d:inference-numberIndexShift7d+1]['value'].values[0]               
                    valuet_3 = valuet_2
                    valuet_2 = valuet_1
                    valuet_1 = model.predict(X_test_unit)
                    y_true.append(y_test_unit.values[0][0])
                    y_output[inference:inference+1]['value'].values[0] = valuet_1
                    # y_pred.append(valuet_1[0][0])
                    if isinstance(valuet_1[0],float):
                        y_pred.append(valuet_1[0])
                    elif isinstance(valuet_1[0][0],float):
                        y_pred.append(valuet_1[0][0])
                    else:
                        print("error output not a float")
            # r2 = (r2_score(y_true, y_pred))

            r2 = (r2_score(y_true, y_pred))

            # print("Iteration n°" + str(i) + ", params:", keywords_args ,", r2:", r2)
            if BEST[1] == None or r2 > BEST[1]:
                # if verbose: print("New best r2:", r2, "at iteration", i)
                BEST = (model, r2, combination) # keywords_args)
                y_pred_out = y_output.copy()

            if (r2 < r2previous1) and (r2previous1 < r2previous2):
                break
            r2previous2 = r2previous1
            r2previous1 = r2               
            # r2 = r2_score(y_test, model.predict(X_test)) # Coefficient of determination
        if verbose: print("Ending benching for", self.regressor.__name__, "with best r2:", BEST[1])
        self.best = BEST
        # print(BEST[1])
        # plt.plot(y_pred, label = "Pred")
        # plt.plot(y_true,  label = "Truth")
        # plt.legend()
        # plt.show()

        # a = pd.concat([Y_train,Y_test[0:96]],ignore_index=True)
        # b = pd.concat([Y_train,y_pred_out[0:96]],ignore_index=True)
        # plt.plot(a,  label = "Truth")
        # plt.plot(b, label="pred")
        # plt.plot(y_train,  label = "train")
        # plt.legend()
        # plt.show()  
        return [BEST[1], BEST[2], y_pred_out]


def  train_forecasting_model(models,model_list,processed_data_file, forecasthorizon, requested_interval_r2):
    enhancedf = pd.read_csv(processed_data_file, parse_dates=['Time'])
    enhancedf['Time'] = enhancedf['Time'].astype('datetime64[s]')
    Models = [ModelTesting(RandomForestRegressor, ["n_estimators", "random_state"], [1, 0], [101, 1], [20, 1]), ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [3], proportional=True)]
    Models = [ModelTesting(KNeighborsRegressor, ["n_neighbors"], [5], [16], [5], proportional=True)]
    # Models = [ModelTesting(GradientBoostingRegressor, ["n_estimators"], [1], [101], [20])] #, ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [3], proportional=True)]
    # model_list = ['GradientBoost', 'kNN']
    model_list = ['KNN']

    # # load the model from disk
    # loaded_model = joblib.load(filename)
    # result = loaded_model.score(X_test, Y_test)
    # print(result)

    """
    This function will:
    - Try different size of df (Whole df (10 years), 1 year, 6 months, 1 month, 2 weeks) according to self.time_ranges
    - Try different regressor with different params (see benchmark_regressor) according to self.benchs
    """
    print("$$$$$ STARTING BENCHMARK $$$$$")
    # Different size of DataFrame
    index = []
    listBestR2 = []
    indexmodel =0
    # self.logs[1][time_range] = [] # This array will be filled soon with regressor's r2 values

    initialDataSet = enhancedf[enhancedf.DayOfWeek<=NUMBER_DAYS_CONSIDERED].copy().reset_index(drop=True)  # we remove Week Ends
    # initialDataSet = enhancedf.reset_index(drop=True)  # we remove Week Ends

    initialDataSetCopy=initialDataSet.copy()
    upsampledDataSet = initialDataSetCopy.set_index('Time') # Needed for resample
    # upsampledDataSet = upsampledDataSet.resample(aggregation).mean().bfill() # At this point, df is only composed of Time as index (every 15mins) and dB as column
    upsampledDataSet = upsampledDataSet.reset_index(drop=False) # Get back Time column

    DataSetIndexbyTime = upsampledDataSet.set_index('Time') # Have to do this in order to make the next line work ("last" function)
    numberRowsfor1day = len(DataSetIndexbyTime.first('1D'))

    numberRowsforR2 = max(len(DataSetIndexbyTime.first(TRAINING_FREQUENCY)),len(DataSetIndexbyTime.first(requested_interval_r2)))
    numberRowsinTrainDataSet =min(len(DataSetIndexbyTime)-numberRowsforR2,len(DataSetIndexbyTime.first(TRAINING_HORIZON)))
    numberRowsforTraining = len(DataSetIndexbyTime.first(TRAINING_FREQUENCY))
    numberRowsForecastHorizon = len(DataSetIndexbyTime.first(forecasthorizon))
    numberofdaysinDataSet = int(len(DataSetIndexbyTime)/numberRowsfor1day)

    # numberRowsinTrainDataSet = min(len(DataSetIndexbyTime)-numberRowsforR2,len(DataSetIndexbyTime.first(TRAINING_HORIZON)))
    currentDataSet = upsampledDataSet.copy().tail(numberRowsinTrainDataSet+numberRowsforR2)
    currentDataSet = currentDataSet.reset_index(drop=True)
    trainDataSet = currentDataSet.copy().head(numberRowsinTrainDataSet)
    numberofdaysintraindataset = int(len(trainDataSet)/numberRowsfor1day)
    indexmodel =0
    r2 = 0
    a=0
    for model in Models:
        modelname = model_list[indexmodel]
        Bestr2Model = []
        Lowestr2Model=[]
        BestparamModel = []
        worstdays = []
        averageR2 = 0
        bestAverageR2 = -50
        # print("@@@@@@@@@@@@@@")
        # print(numberRowsforTraining)
        booleen10 =0
        for day in range(numberofdaysinDataSet-numberofdaysintraindataset):
            # if day%50==0:
                # print(day)
                # print(r2)
            progress = day/(numberofdaysinDataSet-numberofdaysintraindataset)*100
            if np.floor(progress)%20 ==0 and booleen10 ==0:
                print(np.floor(progress),"%")
                booleen10=1
            if np.floor(progress)%20 ==1 and booleen10 ==1:
                booleen10 = 0

            currentDataSet = upsampledDataSet.copy()[day*numberRowsfor1day:day*numberRowsfor1day+numberRowsinTrainDataSet+numberRowsforR2]
            currentDataSet = currentDataSet.reset_index(drop=True)
            trainDataSet = currentDataSet.copy().head(numberRowsinTrainDataSet)
            testDataSet = currentDataSet.copy().tail(len(currentDataSet)-len(trainDataSet)).head(numberRowsforR2)
            target = trainDataSet[['value']] # Expected results
            input = trainDataSet.drop(columns=['value', 'Time'])
            BEST_BENCH = None
            # X_train, X_test, Y_train, Y_test = train_test_split(input, target, test_size=int(numberRowsinTrainDataSet*0.2), shuffle=False)
            Y_train = trainDataSet.copy()[['value']] # Expected results
            X_train = trainDataSet.copy().drop(columns=['value', 'Time'])
            Y_test = testDataSet.copy()[['value']] # Expected results
            X_test = testDataSet.copy().drop(columns=['value', 'Time'])
            Y_train = Y_train.reset_index(drop=True)
            X_train = X_train.reset_index(drop=True)
            Y_test = Y_test.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            if X_test.empty == False:
                
                [r2, param, Yout] = model.testParam(X_train, X_test, Y_train, Y_test,numberRowsforR2, DataSetIndexbyTime, numberRowsForecastHorizon, numberRowsforTraining, verbose=False)
            
                if np.isnan(r2)==False:
                    Bestr2Model.append(min(max(r2,-1),1))
                    BestparamModel.append(param)
                    # print(r2)
            # plt.plot(Yout, label = "Pred")
            # plt.plot(Y_test,  label = "Truth")
            # plt.legend()
            # plt.show()
            # worstdays.append(testDataSet[0:1]['Time'])
            # print("worst days:", testDataSet[0:1]['Time'])
            # print(r2)
            # print(len(Yout))
            #if r2 < 0.6: #we remove the weeks where there is a day without data (linear interpolation)
            #  a = a+1    
            #else:   
            #Bestr2Model.append(r2)
            #BestparamModel.append(param)
            # print(r2)
        averageparam = []            
        index.append((modelname,TRAINING_HORIZON, forecasthorizon))
        averageR2 = sum(Bestr2Model)/max(1,len(Bestr2Model))
        arrayParam = np.array(BestparamModel)
        if arrayParam.any():
            for ituple in range(arrayParam.shape[1]):
                averageparam.append(np.average(arrayParam[:,ituple]))

        # self.logs[1][list(self.logs[1].keys())[len(self.logs[1])-1]].append(model.best[1]) # Add r2 score for current regressor and time_range
        # if BEST_BENCH == None or model.best[1] > BEST_BENCH.best[1]:
        #     BEST_BENCH = model
        print(averageR2)
        if averageR2 >= bestAverageR2:
            bestAverageR2 = averageR2
            bestparam = averageparam
            model.params_comb = bestparam
            BEST_MODEL = model
        print("best for Model: ", model_list[indexmodel] , " training horizon: ", TRAINING_HORIZON, "forecast horizon :", forecasthorizon, " best model : ", averageparam, "r2 average: ", bestAverageR2, "number of r2: ", len(Bestr2Model), " Lowest r2: ", sum(Lowestr2Model)/max(1,len(Lowestr2Model)), "number of low r2: ", a)   
        listBestR2.append(bestAverageR2)
        indexmodel=indexmodel+1

    # s = pd.Series(listBestR2,index=index)
    # m_index = pd.MultiIndex.from_tuples(index)
    # s = s.reindex(m_index)
    # print(s)
    return [BEST_MODEL,bestAverageR2]



###### function API inference  #############
def testandupdatemodel(model, x_train, x_test, y_train, y_test, numberrowsforr2, datasetindexbytime, numberrowsforecasthorizon, numberrowsfortraining, verbose=True):
        keywords_args = {}
        i = 0
        for parameter in model.params_name:
            keywords_args[parameter] = int(model.params_comb[i])
            i = i+1
        model = model.name(**keywords_args)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! training size3/////////////////////////////////")
        # print(numberrowsfortraining)
        model = model.fit(x_test.tail(numberrowsfortraining), y_test.tail(numberrowsfortraining)) # Train model

        # print(datasetindexbytime)
        numberIndexShift1H =  len(datasetindexbytime .first('1H'))
        numberIndexShift7d =  len(datasetindexbytime .first('7D'))
        r2=[]
        valuet_1h = 0
        valuet_7d = 0
        valuet_1 = x_test[0:1]['t-1'].values[0]
        valuet_2 = x_test[0:1]['t-2'].values[0]
        valuet_3 = x_test[0:1]['t-3'].values[0]
        if 't-1h' in x_test.columns:
            valuet_1h=(x_test[0:1]['t-1h'].values[0])
        if 't-7days' in x_test.columns:
            valuet_7d=(x_test[0:1]['t-7days'].values[0])
        y_output = y_test.copy()
        y_pred = []
        y_true= []
        
        for inference in range(numberrowsforr2):
            if inference%(numberrowsfortraining) == 0 and inference >0:  #we retrain every day
                x_train2 = pd.concat([x_train, x_test.head(inference)], ignore_index=True) #change x_train to take the tail (len - inference)
                y_train2 = pd.concat([y_train, y_test.head(inference)], ignore_index=True)
                # print("training size3")
                # print(x_train.size)
                model = model.fit(x_train2, y_train2) # Train model

            y_test_unit = y_test.copy()[inference:inference+1]
            X_test_unit = x_test.copy()[inference:inference+1]
            if X_test_unit.empty == False:
                if inference%numberrowsforecasthorizon == 0 and inference < len(x_test):
                    valuet_1 = X_test_unit['t-1'].values[0]
                    valuet_2 = X_test_unit['t-2'].values[0]
                    valuet_3 = X_test_unit['t-3'].values[0]
                elif inference < len(x_test):
                    X_test_unit['t-1'] = valuet_1
                    X_test_unit['t-2'] = valuet_2
                    X_test_unit['t-3'] = valuet_3
                if (inference >= numberIndexShift1H) and ('t-1h' in x_test.columns) and (numberrowsforecasthorizon > numberIndexShift1H ) and (inference < len(x_test)):
                    X_test_unit['t-1h'].values[0] = y_output.copy()[inference-numberIndexShift1H:inference-numberIndexShift1H+1]['value'].values[0]
                if (inference >= numberIndexShift7d) and ('t-7days' in x_test.columns) and (numberrowsforecasthorizon > numberIndexShift7d ) and (inference < len(x_test)):
                    X_test_unit['t-7days'].values[0] = y_output.copy()[inference-numberIndexShift7d:inference-numberIndexShift7d+1]['value'].values[0]               
                valuet_3 = valuet_2
                valuet_2 = valuet_1
                # print(X_test_unit)
                valuet_1 = model.predict(X_test_unit)
                # print(y_test_unit.values[0][0])
                y_true.append(y_test_unit.values[0][0])
                y_output[inference:inference+1]['value'].values[0] = valuet_1
                # y_pred.append(valuet_1[0][0])
                if isinstance(valuet_1[0],float):
                    y_pred.append(valuet_1[0])
                elif isinstance(valuet_1[0][0],float):
                    y_pred.append(valuet_1[0][0])
                else:
                    print("error output not a float")
        # r2 = (r2_score(y_true, y_pred))
        # print(y_pred)
        # print(y_true)
        r2 = (r2_score(y_true, y_pred))

        # if verbose: print("Iteration n°" + str(i) + ", params:", keywords_args ,", r2:", r2)
        y_pred_out = y_output.copy()

               
        # r2 = r2_score(y_test, model.predict(X_test)) # Coefficient of determination
        if verbose: print("r2:", r2)
    # print(BEST[1])
    # plt.plot(y_pred, label = "Pred")
    # plt.plot(y_true,  label = "Truth")
    # plt.legend()
    # plt.show()



    # a = pd.concat([Y_train,Y_test[0:96]],ignore_index=True)
    # b = pd.concat([Y_train,y_pred_out[0:96]],ignore_index=True)
    # plt.plot(a,  label = "Truth")
    # plt.plot(b, label="pred")
    # plt.plot(y_train,  label = "train")
    # plt.legend()
    # plt.show()  
        return [r2,  y_pred_out]


    
def inference(model, x_train,  y_train, numberrowsforinference, datasetindexbytime,  verbose=True):
        keywords_args = {}
        i = 0
        for parameter in model.params_name:
            keywords_args[parameter] = int(model.params_comb[i])
            i = i+1
        model = model.name(**keywords_args)
        # print("training size5")
        # print(x_train.size)
        model = model.fit(x_train, y_train) # Train model






        numberIndexShift1H =  len(datasetindexbytime.first('1H'))
        numberIndexShift7d =  len(datasetindexbytime.first('7D'))
        valuet_1 = y_train.tail(1).values[0][0]
        valuet_2 = x_train.tail(1)['t-1'].values[0]
        valuet_3 = x_train.tail(1)['t-2'].values[0]
        # if 't-1h' in x_test.columns:
        #     valuet_1h=(x_test[0:1]['t-1h'].values[0])
        # if 't-7days' in x_test.columns:
        #     valuet_7d=(x_test[0:1]['t-7days'].values[0])
        y_pred = []
        last_two_times = datasetindexbytime.tail(2).index
        timestepdelta = last_two_times[1] - last_two_times[0]

        # print("**")
        # print(numberrowsforinference)  
        # print(y_train)  
        # print(numberIndexShift7d)   
        for inference in range(numberrowsforinference):
            inference_time = last_two_times[1]+timestepdelta*(1+inference)

            X_test_unit = x_train.head(1).copy()
     
            X_test_unit['DayOfWeek'] = min(NUMBER_DAYS_CONSIDERED,inference_time.dayofweek)
            X_test_unit['Hour'] = inference_time.hour
            X_test_unit['Minute'] = inference_time.minute
            X_test_unit['TimeSlot'] = time_slot_assigner(inference_time.hour)            
            if X_test_unit.empty == False:
                X_test_unit['t-1'] = valuet_1
                X_test_unit['t-2'] = valuet_2
                X_test_unit['t-3'] = valuet_3
                if (inference >= numberIndexShift1H) and ('t-1h' in x_train.columns):
                    print("here !!")
                    X_test_unit['t-1h'].values[0] = y_pred.copy()[inference-numberIndexShift1H:inference-numberIndexShift1H+1][0]
                elif ('t-1h' in x_train.columns):
                    X_test_unit['t-1h'].values[0] = y_train.loc[len(y_train)-numberIndexShift1H]
                if (inference >= numberIndexShift7d) and ('t-7days' in x_train.columns):

                    X_test_unit['t-7days'].values[0] = y_pred.copy()[inference-numberIndexShift7d:inference-numberIndexShift7d+1][0]             
                elif ('t-7days' in x_train.columns):
                    X_test_unit['t-7days'].values[0] = y_train.loc[len(y_train)+inference-numberIndexShift7d-1]
                valuet_3 = valuet_2
                valuet_2 = valuet_1
                valuet_1 = model.predict(X_test_unit)

                if isinstance(valuet_1[0],float):
                    y_pred.append(valuet_1[0])
                elif isinstance(valuet_1[0][0],float):
                    y_pred.append(valuet_1[0][0])
                else:
                    print("error output not a float")
        # print(y_pred)
        return y_pred



   
def inferencenow(model, x_train,  y_train, x_test,  y_test, numberrowsforinference, datasetindexbytime, aggregation,  verbose=True):
        keywords_args = {}
        i = 0
        for parameter in model.params_name:
            keywords_args[parameter] = int(model.params_comb[i])
            i = i+1
        model = model.name(**keywords_args)
        # print("x train")
        # print(x_train)
        # print("y train")
        # print(y_train)
        # print("testing size6")
        # print(x_test.size)
        model = model.fit(x_test, y_test) # Train model
        last_two_times = datasetindexbytime.tail(2).index
        timestepdelta = last_two_times[1] - last_two_times[0]
        numberIndexShift1H =  len(datasetindexbytime.first('1H'))
        numberIndexShift7d =  len(datasetindexbytime.first('7D'))
        timenow = pd.Timestamp.now()
        inference_time = timenow+timestepdelta*(1)
        # print(inference_time)

        if aggregation.endswith('m'):

            print('aggregation time not considered yet')
        elif aggregation.endswith('H'):
            last_row_with_desired_values = x_test.loc[(x_test['DayOfWeek'] == min(NUMBER_DAYS_CONSIDERED,inference_time.dayofweek)) & (x_test['Hour'] == inference_time.hour) ].tail(1)

        elif aggregation.endswith('D'):
            last_row_with_desired_values = x_test.loc[(x_test['DayOfWeek'] == min(NUMBER_DAYS_CONSIDERED,inference_time.dayofweek))  ].tail(1)
        else:
            print("Invalid duration string")
        
        # print(last_row_with_desired_values)
        # print(last_row_with_desired_values['t-1'])
        valuet_1 = last_row_with_desired_values['t-1'].values
        # print(last_row_with_desired_values)
        valuet_2 =  last_row_with_desired_values['t-2'].values
        valuet_3 = last_row_with_desired_values['t-3'].values
        # if 't-1h' in x_test.columns:
        #     valuet_1h=(x_test[0:1]['t-1h'].values[0])
        # if 't-7days' in x_test.columns:
        #     valuet_7d=(x_test[0:1]['t-7days'].values[0])
        y_pred = []


        # print("**")
        # print(numberrowsforinference)  
        # print(x_test)  
        # print(numberIndexShift7d)   
        # print(datasetindexbytime)
        for inference in range(numberrowsforinference):
            inference_time = timenow+timestepdelta*(1+inference)

            X_test_unit = x_test.tail(1).copy()
     
            X_test_unit['DayOfWeek'] = min(NUMBER_DAYS_CONSIDERED,inference_time.dayofweek)
            X_test_unit['Hour'] = inference_time.hour
            X_test_unit['Minute'] = 0# inference_time.minute
            X_test_unit['TimeSlot'] = time_slot_assigner(inference_time.hour) 
            # print(X_test_unit)           
            if X_test_unit.empty == False:
                # print(valuet_1)           

                X_test_unit['t-1'] = valuet_1
                X_test_unit['t-2'] = valuet_2
                X_test_unit['t-3'] = valuet_3
                if (inference >= numberIndexShift1H) and ('t-1h' in x_train.columns):
                    # print("here !!")
                    X_test_unit['t-1h'].values[0] = y_pred.copy()[inference-numberIndexShift1H:inference-numberIndexShift1H+1][0]
                elif ('t-1h' in x_train.columns):
                    X_test_unit['t-1h'].values[0] = y_test.loc[len(y_test)-numberIndexShift1H]
                if (inference >= numberIndexShift7d) and ('t-7days' in x_train.columns):

                    X_test_unit['t-7days'].values[0] = y_pred.copy()[inference-numberIndexShift7d:inference-numberIndexShift7d+1][0]             
                elif ('t-7days' in x_train.columns):
                    X_test_unit['t-7days'].values[0] = y_test.loc[min(len(y_test)-1,max(0,len(y_test)+inference-numberIndexShift7d-1))]
                valuet_3 = valuet_2
                valuet_2 = valuet_1
                # print("X_test_unit:")
                # print(X_test_unit)
                valuet_1 = model.predict(X_test_unit)

                if isinstance(valuet_1[0],float):
                    y_pred.append(valuet_1[0])
                elif isinstance(valuet_1[0][0],float):
                    y_pred.append(valuet_1[0][0])
                else:
                    print("error output not a float")
        # print(y_pred)
        return y_pred




def infer_forecastnow(modelfilename,processed_data_file, aggregation, forecasthorizon, requested_interval_r2):
    enhancedf = pd.read_csv(processed_data_file, parse_dates=['Time'])
    enhancedf['Time'] = enhancedf['Time'].astype('datetime64[s]')
    # Models = [ModelTesting(RandomForestRegressor, ["n_estimators", "random_state"], [1, 0], [101, 1], [20, 1]), ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [3], proportional=True)]
    # Models = [ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [10], proportional=True)]
    # Models = [ModelTesting(GradientBoostingRegressor, ["n_estimators"], [60], [61], [20])] #, ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [3], proportional=True)]
    # model_list = ['GradientBoost', 'kNN']



    # # load the model from disk
    loadedModel = joblib.load(modelfilename)
    # result = loaded_model.score(X_test, Y_test)
    # print(result)

    """
    This function will:
    - Try different size of df (Whole df (10 years), 1 year, 6 months, 1 month, 2 weeks) according to self.time_ranges
    - Try different regressor with different params (see benchmark_regressor) according to self.benchs
    """
    print("$$$$$ STARTING INFERENCE $$$$$")
    # Different size of DataFrame
    index = []
    listBestR2 = []
    indexmodel =0
    # self.logs[1][time_range] = [] # This array will be filled soon with regressor's r2 values

    initialDataSet = enhancedf[enhancedf.DayOfWeek<=NUMBER_DAYS_CONSIDERED].copy().reset_index(drop=True)  # we remove Week Ends
    # initialDataSet = enhancedf.reset_index(drop=True)  # we remove Week Ends

    initialDataSetCopy=initialDataSet.copy()
    upsampledDataSet = initialDataSetCopy.set_index('Time') # Needed for resample
    # upsampledDataSet = upsampledDataSet.resample(aggregation).mean().bfill() # At this point, df is only composed of Time as index (every 15mins) and dB as column
    upsampledDataSet = upsampledDataSet.reset_index(drop=False) # Get back Time column

    DataSetIndexbyTime = upsampledDataSet.set_index('Time') # Have to do this in order to make the next line work ("last" function)
    numberRowsfor1day = len(DataSetIndexbyTime.first('1D'))

    numberRowsforR2 = max(len(DataSetIndexbyTime.first(TRAINING_FREQUENCY)),len(DataSetIndexbyTime.first(requested_interval_r2)))
    numberRowsinTrainDataSet =min(len(DataSetIndexbyTime)-numberRowsforR2,len(DataSetIndexbyTime.first(TRAINING_HORIZON)))
    numberRowsforTraining = len(DataSetIndexbyTime.first(TRAINING_HORIZON)) ###############!!!!!!!!!!!!!!!!!!
    numberRowsForecastHorizon = len(DataSetIndexbyTime.first(forecasthorizon))
    numberofdaysinDataSet = int(len(DataSetIndexbyTime)/numberRowsfor1day)
    time_testr2_df = DataSetIndexbyTime.copy().tail(numberRowsforR2).index
    # numberRowsinTrainDataSet = min(len(DataSetIndexbyTime)-numberRowsforR2,len(DataSetIndexbyTime.first(TRAINING_HORIZON)))
    currentDataSet = upsampledDataSet.copy().tail(numberRowsinTrainDataSet+numberRowsforR2)
    currentDataSet = currentDataSet.reset_index(drop=True)
    trainDataSet = currentDataSet.copy().head(numberRowsinTrainDataSet)
    indexmodel =0

    # modelname = model_list[indexmodel]
    Bestr2Model = []
    Lowestr2Model=[]
    BestparamModel = []
    worstdays = []
    averageR2 = 0
    bestAverageR2 = 0

    # for day in range(numberofdaysinDataSet):
    #     if day%50==0:
    #         print(day)
    #     currentDataSet = upsampledDataSet.copy()[day*numberRowsfor1day:day*numberRowsfor1day+numberRowsinTrainDataSet+numberRowsforR2]
    #     currentDataSet = currentDataSet.reset_index(drop=True)
    trainDataSet = currentDataSet.copy().head(numberRowsinTrainDataSet)
    testDataSet = currentDataSet.copy().tail(len(currentDataSet)-len(trainDataSet)).head(numberRowsforR2)
    target = trainDataSet[['value']] # Expected results
    input = trainDataSet.drop(columns=['value', 'Time'])
    BEST_BENCH = None
    # X_train, X_test, Y_train, Y_test = train_test_split(input, target, test_size=int(numberRowsinTrainDataSet*0.2), shuffle=False)
    Y_train = trainDataSet.copy()[['value']] # Expected results
    X_train = trainDataSet.copy().drop(columns=['value', 'Time'])
    Y_test = testDataSet.copy()[['value']] # Expected results
    X_test = testDataSet.copy().drop(columns=['value', 'Time'])
    Y_train = Y_train.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    if X_test.empty == False:

        [r2,  Yout] = testandupdatemodel(loadedModel,X_train, X_test, Y_train, Y_test,numberRowsforR2, DataSetIndexbyTime, numberRowsForecastHorizon, numberRowsforTraining, verbose=False)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(Yout)
        # print(Y_test)
        # print(DataSetIndexbyTime)
        # plt.plot(Yout, label = "Pred")
    # plt.plot(Y_test,  label = "Truth")
    # plt.legend()
    # plt.show()
    # worstdays.append(testDataSet[0:1]['Time'])
    # print("worst days:", testDataSet[0:1]['Time'])
    if r2 < 0.6: #we remove the weeks where there is a day without data (linear interpolation)
        a = 0     
    else:   
        Bestr2Model.append(r2)
        # BestparamModel.append(param)


    averageparam = []            
    averageR2 = sum(Bestr2Model)/max(1,len(Bestr2Model))
    arrayParam = np.array(BestparamModel)
    if arrayParam.any():
        for ituple in range(arrayParam.shape[1]):
            averageparam.append(np.average(arrayParam[:,ituple]))

    # self.logs[1][list(self.logs[1].keys())[len(self.logs[1])-1]].append(model.best[1]) # Add r2 score for current regressor and time_range
    # if BEST_BENCH == None or model.best[1] > BEST_BENCH.best[1]:
    #     BEST_BENCH = model
    if averageR2 >= bestAverageR2:
        bestAverageR2 = averageR2
        # bestparam = averageparam
        # BEST_MODEL = loadedModel
    print("training horizon: ", TRAINING_HORIZON, "forecast horizon :", forecasthorizon, " best model : ", averageparam, "r2 average: ", bestAverageR2, "number of r2: ", len(Bestr2Model), " Lowest r2: ", sum(Lowestr2Model)/max(1,len(Lowestr2Model)), "number of low r2: ", len(Lowestr2Model))   
    listBestR2.append(bestAverageR2)
    indexmodel=indexmodel+1
    # print(X_test)
    # print(DataSetIndexbyTime)
    y = inferencenow(loadedModel, X_train, Y_train, X_test, Y_test, numberRowsForecastHorizon, DataSetIndexbyTime, aggregation) 
    print("r2 on last week: ", r2 , " inference: ", y)
    if aggregation =='1H':
        if 'D' in forecasthorizon:
            tmp = forecasthorizon
            tmp = int(tmp.replace('D',""))
            numberofhoursfortime = tmp*24
        elif  'H' in forecasthorizon:
            tmp = forecasthorizon
            tmp = int(tmp.replace('H',""))
            numberofhoursfortime = tmp
        else: print("Sorry, issue on aggregation: case NOT CONSIDERED YET")
        
        # time_inference1 = pd.period_range(start=DataSetIndexbyTime.tail(1).index[0]+pd.Timedelta(hours=1), end=DataSetIndexbyTime.tail(1).index[0]+pd.Timedelta(hours=numberofhoursfortime), freq='H').start_time
        now = pd.Timestamp.now()  # Get the current time
        periods = pd.period_range(start=now, periods=numberofhoursfortime, freq='H')  # Create a period range

        # Set the minutes and seconds to 0
        # periods = periods.map(lambda p: p.asfreq('H'))
        # Set the seconds to 0
        periods = periods.map(lambda p: p.asfreq('H').to_timestamp())

        # Reset the seconds to 0
        periods = periods.map(lambda ts: ts.replace(second=0))
        # time_inference1 = time_inference1.to_frame(index=False)
        time_inference1 = periods.to_frame(index=False)
    elif aggregation =='1D':
        if 'D' in forecasthorizon:
            tmp = forecasthorizon
            tmp = int(tmp.replace('D',""))
            numberofhoursfortime = tmp
        else: print("Sorry, issue on aggregation: case NOT CONSIDERED YET")
        
        time_inference1 = pd.period_range(start=DataSetIndexbyTime.tail(1).index[0]+pd.Timedelta(days=1), end=DataSetIndexbyTime.tail(1).index[0]+pd.Timedelta(days=numberofhoursfortime), freq='D').start_time
        time_inference1 = time_inference1.to_frame(index=False)
        print(time_inference1)
    else: 
        print("Sorry, issue on aggregation: case NOT CONSIDERED YET")
    # print(time_inference1)
    return [r2, y, Yout, Y_test, time_testr2_df, time_inference1]



def infer_forecast(modelfilename,processed_data_file, aggregation, forecasthorizon, requested_interval_r2):
    enhancedf = pd.read_csv(processed_data_file, parse_dates=['Time'])
    enhancedf['Time'] = enhancedf['Time'].astype('datetime64[s]')
    # Models = [ModelTesting(RandomForestRegressor, ["n_estimators", "random_state"], [1, 0], [101, 1], [20, 1]), ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [3], proportional=True)]
    # Models = [ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [10], proportional=True)]
    # Models = [ModelTesting(GradientBoostingRegressor, ["n_estimators"], [60], [61], [20])] #, ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [3], proportional=True)]
    # model_list = ['GradientBoost', 'kNN']



    # # load the model from disk
    loadedModel = joblib.load(modelfilename)
    # result = loaded_model.score(X_test, Y_test)
    # print(result)

    """
    This function will:
    - Try different size of df (Whole df (10 years), 1 year, 6 months, 1 month, 2 weeks) according to self.time_ranges
    - Try different regressor with different params (see benchmark_regressor) according to self.benchs
    """
    print("$$$$$ STARTING INFERENCE $$$$$")
    # Different size of DataFrame
    index = []
    listBestR2 = []
    indexmodel =0
    # self.logs[1][time_range] = [] # This array will be filled soon with regressor's r2 values

    initialDataSet = enhancedf[enhancedf.DayOfWeek<=NUMBER_DAYS_CONSIDERED].copy().reset_index(drop=True)  # we remove Week Ends
    # initialDataSet = enhancedf.reset_index(drop=True)  # we remove Week Ends

    initialDataSetCopy=initialDataSet.copy()
    upsampledDataSet = initialDataSetCopy.set_index('Time') # Needed for resample
    # upsampledDataSet = upsampledDataSet.resample(aggregation).mean().bfill() # At this point, df is only composed of Time as index (every 15mins) and dB as column
    upsampledDataSet = upsampledDataSet.reset_index(drop=False) # Get back Time column

    DataSetIndexbyTime = upsampledDataSet.set_index('Time') # Have to do this in order to make the next line work ("last" function)
    numberRowsfor1day = len(DataSetIndexbyTime.first('1D'))

    numberRowsforR2 = max(len(DataSetIndexbyTime.first(TRAINING_FREQUENCY)),len(DataSetIndexbyTime.first(requested_interval_r2)))
    numberRowsinTrainDataSet =min(len(DataSetIndexbyTime)-numberRowsforR2,len(DataSetIndexbyTime.first(TRAINING_HORIZON)))
    numberRowsforTraining = len(DataSetIndexbyTime.first(TRAINING_FREQUENCY))
    numberRowsForecastHorizon = len(DataSetIndexbyTime.first(forecasthorizon))
    numberofdaysinDataSet = int(len(DataSetIndexbyTime)/numberRowsfor1day)
    time_testr2_df = DataSetIndexbyTime.copy().tail(numberRowsforR2).index
    # numberRowsinTrainDataSet = min(len(DataSetIndexbyTime)-numberRowsforR2,len(DataSetIndexbyTime.first(TRAINING_HORIZON)))
    currentDataSet = upsampledDataSet.copy().tail(numberRowsinTrainDataSet+numberRowsforR2)
    currentDataSet = currentDataSet.reset_index(drop=True)
    trainDataSet = currentDataSet.copy().head(numberRowsinTrainDataSet)
    indexmodel =0

    # modelname = model_list[indexmodel]
    Bestr2Model = []
    Lowestr2Model=[]
    BestparamModel = []
    worstdays = []
    averageR2 = 0
    bestAverageR2 = 0

    # for day in range(numberofdaysinDataSet):
    #     if day%50==0:
    #         print(day)
    #     currentDataSet = upsampledDataSet.copy()[day*numberRowsfor1day:day*numberRowsfor1day+numberRowsinTrainDataSet+numberRowsforR2]
    #     currentDataSet = currentDataSet.reset_index(drop=True)
    trainDataSet = currentDataSet.copy().head(numberRowsinTrainDataSet)
    testDataSet = currentDataSet.copy().tail(len(currentDataSet)-len(trainDataSet)).head(numberRowsforR2)
    target = trainDataSet[['value']] # Expected results
    input = trainDataSet.drop(columns=['value', 'Time'])
    BEST_BENCH = None
    # X_train, X_test, Y_train, Y_test = train_test_split(input, target, test_size=int(numberRowsinTrainDataSet*0.2), shuffle=False)
    Y_train = trainDataSet.copy()[['value']] # Expected results
    X_train = trainDataSet.copy().drop(columns=['value', 'Time'])
    Y_test = testDataSet.copy()[['value']] # Expected results
    X_test = testDataSet.copy().drop(columns=['value', 'Time'])
    Y_train = Y_train.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    if X_test.empty == False:
        [r2,  Yout] = testandupdatemodel(loadedModel,X_train, X_test, Y_train, Y_test,numberRowsforR2, DataSetIndexbyTime, numberRowsForecastHorizon, numberRowsforTraining, verbose=False)
    # plt.plot(Yout, label = "Pred")
    # plt.plot(Y_test,  label = "Truth")
    # plt.legend()
    # plt.show()
    # worstdays.append(testDataSet[0:1]['Time'])
    # print("worst days:", testDataSet[0:1]['Time'])
    if r2 < 0.6: #we remove the weeks where there is a day without data (linear interpolation)
        a = 0     
    else:   
        Bestr2Model.append(r2)
        # BestparamModel.append(param)


    averageparam = []            
    averageR2 = sum(Bestr2Model)/max(1,len(Bestr2Model))
    arrayParam = np.array(BestparamModel)
    if arrayParam.any():
        for ituple in range(arrayParam.shape[1]):
            averageparam.append(np.average(arrayParam[:,ituple]))

    # self.logs[1][list(self.logs[1].keys())[len(self.logs[1])-1]].append(model.best[1]) # Add r2 score for current regressor and time_range
    # if BEST_BENCH == None or model.best[1] > BEST_BENCH.best[1]:
    #     BEST_BENCH = model
    if averageR2 >= bestAverageR2:
        bestAverageR2 = averageR2
        # bestparam = averageparam
        # BEST_MODEL = loadedModel
    # print("training horizon: ", TRAINING_HORIZON, "forecast horizon :", forecasthorizon, " best model : ", averageparam, "r2 average: ", bestAverageR2, "number of r2: ", len(Bestr2Model), " Lowest r2: ", sum(Lowestr2Model)/max(1,len(Lowestr2Model)), "number of low r2: ", len(Lowestr2Model))   
    listBestR2.append(bestAverageR2)
    indexmodel=indexmodel+1
    # print(X_test)
    # print(DataSetIndexbyTime)
    y = simpleinference(loadedModel, X_train, Y_train, X_test, Y_test, numberRowsForecastHorizon, DataSetIndexbyTime) 
    # print("r2 on last week: ", r2 , " inference: ", y)
    if aggregation =='1H':
        if 'D' in forecasthorizon:
            tmp = forecasthorizon
            tmp = int(tmp.replace('D',""))
            numberofhoursfortime = tmp*24
        elif  'H' in forecasthorizon:
            tmp = forecasthorizon
            tmp = int(tmp.replace('H',""))
            numberofhoursfortime = tmp
        else: print("Sorry, issue on aggregation: case NOT CONSIDERED YET")
        
        # time_inference1 = pd.period_range(start=DataSetIndexbyTime.tail(1).index[0]+pd.Timedelta(hours=1), end=DataSetIndexbyTime.tail(1).index[0]+pd.Timedelta(hours=numberofhoursfortime), freq='H').start_time
        now = pd.Timestamp.now()  # Get the current time
        periods = pd.period_range(start=now, periods=numberofhoursfortime, freq='H')  # Create a period range

        # Set the minutes and seconds to 0
        # periods = periods.map(lambda p: p.asfreq('H'))
        # Set the seconds to 0
        periods = periods.map(lambda p: p.asfreq('H').to_timestamp())

        # Reset the seconds to 0
        periods = periods.map(lambda ts: ts.replace(second=0))
        # time_inference1 = time_inference1.to_frame(index=False)
        time_inference1 = periods.to_frame(index=False)
    elif aggregation =='1D':
        if 'D' in forecasthorizon:
            tmp = forecasthorizon
            tmp = int(tmp.replace('D',""))
            numberofhoursfortime = tmp
        else: print("Sorry, issue on aggregation: case NOT CONSIDERED YET")
        
        time_inference1 = pd.period_range(start=DataSetIndexbyTime.tail(1).index[0]+pd.Timedelta(days=1), end=DataSetIndexbyTime.tail(1).index[0]+pd.Timedelta(days=numberofhoursfortime), freq='D').start_time
        time_inference1 = time_inference1.to_frame(index=False)
        # print(time_inference1)
    else: 
        print("Sorry, issue on aggregation: case NOT CONSIDERED YET")
    return [r2, y, Yout, Y_test, time_testr2_df, time_inference1]

def send_to_context_broker(url_forecast, forecast_dataframe,typeofsensor):
    response = [0]
    current_dateTime = datetime.now()
    date_now_ISO=current_dateTime.strftime('%Y-%m-%dT%H:%M:%S.00Z')

    # co2=0
    # for typeofsensor in TYPES_OF_SENSOR:
    #     co2=co2+co2_of_typeofsensor*noise_forecast
        
    # noise = ''
    # for typeofsensor in TYPES_OF_SENSOR:
        # noise = noise + '{
        #     "vehicleClass": typeofsensor,
        #     "intensityExpected": noise_forecast,
        #     "occupancyExpected": 0.2,
        #     "averageVehicleSpeedExpected": 45
        # },'
                
    for index, row in forecast_dataframe.iterrows():
        date_forecast = row['time'].strftime('%Y-%m-%dT%H:%M:%S.00Z')
        date_end_forecast = (row['time']+pd.Timedelta(minutes=59)).strftime('%Y-%m-%dT%H:%M:%S.00Z')
        Noise_forecast = row['forecast']
        noise_annoyance=0
        if Noise_forecast > 65:
            noise_annoyance =5
        elif Noise_forecast > 62:
            noise_annoyance =4

        elif Noise_forecast > 60:
            noise_annoyance =3

        elif Noise_forecast > 58:
            noise_annoyance =2

        elif Noise_forecast > 56:
            noise_annoyance =1
        # "type": "https://uri.fiware.org/ns/data-models#NoisePollutionForecast",

        my_json = {
        "id": FORECAST_ID_PART1+typeofsensor+ FORECAST_ID_PART2, 
        "type": "NoisePollutionForecast",
        "address": {
            "type": "Property",
            "value": {
            "addressCountry": "France",
            "postalCode": "06200",
            "addressLocality": "Nice",
            "type": "PostalAddress"
            }
        },
        "location": {
            "type": "GeoProperty",
            "value": {
            "type": "Point",
            "coordinates": [
                7.2032497427380235,
                43.68056738083439
            ]
            }
        },
        "dataProvider": {
            "type": "Property",
            "value": "IMREDD_UCA_Nice"
        },
        "dateCreated": {
            "type": "Property",
            "value": {
            "@type": "DateTime",
            "@value": date_now_ISO
            }
        },
        "LAmax": {
            "type": "Property",
            "value": Noise_forecast 
        },
        "NoiseAnnoyanceIndex": {
            "type": "Property",
            "value": noise_annoyance
        },
        "noiseOrigin": {
            "type": "Property",
            "value": "traffic"
        },
        "exposureType": {
            "type": "Property",
            "value": "short term exposure"
        },
        "buildingsType": {
            "type": "Property",
            "value": "residential"
        },
        "groundType": {
            "type": "Property",
            "value": "concrete"
        },
        "wallsType": {
            "type": "Property",
            "value": "glass"
        },
         "validFrom": {
            "type": "Property",
            "value": {
            "@type": "date-time",
            "@value": date_forecast
            }
        },
        "validTo": {
            "type": "Property",
            "value": {
            "@type": "date-time",
            "@value": date_end_forecast
            }
        },
        "dateIssued": {
            "type": "Property",
            "value": {
            "@type": "date-time",
            "@value": date_now_ISO
            }
        },
        "@context": [
            "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
            "https://raw.githubusercontent.com/smart-data-models/dataModel.Environment/master/context.jsonld"
        ]
        }

        # print([my_json])
        # URL = "https://tip-imredd.unice.fr/data/imredd/nice/noisepollution/entityOperations/upsert?api-key=a1b4deee-008f-4161-ae24-4b7cf507107b"
        newHeaders = {'Content-Type': 'application/ld+json'}

        timeout_seconds = 8
        try:
            response = requests.post(url_forecast, json=[my_json], headers=newHeaders, timeout=timeout_seconds)

            if response.status_code>=200 and response.status_code<300:  # Successful response
                print("forecast sent to the CB", response.status_code)

            else:
                print("POST request failed with status code:", response.status_code)

        except requests.Timeout:
            print("Request timed out")

        except requests.RequestException as e:
            print("An error occurred:", e)

















        # response = requests.post(url_forecast, json=[my_json], headers=newHeaders)
        # if response.status_code>=200 and response.status_code<300:
        #     print("forecast sent to the CB", response.status_code)
        #     # print("forecast sent to the CB", response.text)

        # else:
        #     print("Could not send forecast to the CB", response.status_code)

    return response


def generate_new_forecasts(aggregation,forecast_horizon_func, intervalr2, r2input=0):
            
        
        
        # Models = [ModelTesting(RandomForestRegressor, ["n_estimators", "random_state"], [1, 0], [101, 1], [20, 1]), ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [3], proportional=True)]
        Models = [ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [10], proportional=True)]
        # Models = [ModelTesting(GradientBoostingRegressor, ["n_estimators"], [60], [61], [20])] #, ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [3], proportional=True)]
        # model_list = ['GradientBoost', 'kNN']
        # giving directory name
        dirname = os.getcwd()
        # giving file extension
        ext = ('.csv')
        name_str = "_aggregation_"+aggregation
        for type_of_sensor in TYPES_OF_SENSOR:
            print("looking for file : ", PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ aggregation +".csv", "in : ", dirname)
        # check if data file exists with that aggregation 
        booleen = 0
        for file in os.listdir(dirname):
            if file.endswith(ext):
                if name_str in file:
                    booleen = 1
                    print("existing processed file found")
        if booleen == 0:
            for type_of_sensor in TYPES_OF_SENSOR:
                process_data(CSV_FILE_NAME+type_of_sensor+".csv", aggregation, PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ aggregation +".csv", type_of_sensor, forecast_horizon_func)
                print("started creation of processed file")


    #check if model exists
        dirname = os.getcwd()
        ext = ('.pkl')
        name_str = "_aggregation_"+aggregation + "_forecast_horizon_" + forecast_horizon_func +".pkl"
        # for type_of_sensor in TYPES_OF_SENSOR:
            # print("looking for model : ", PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ aggregation + "_forecast_horizon_" + forecast_horizon_func +".pkl", "in : ", dirname)
        # check if data file exists with that aggregation 
        booleen = 0
        # print(MODEL_FILE_NAME)
        for file in os.listdir(dirname):
            if file.endswith(ext):
                if name_str in file:
                    booleen = 1
                    print("existing trained model found")
                    for type_of_sensor in TYPES_OF_SENSOR:
                        modelfilename = MODEL_FILE_NAME+type_of_sensor+ "_aggregation_"+ aggregation + "_forecast_horizon_" + forecast_horizon_func +".pkl"
        if booleen == 0: #if the model does not exist
            if 'H' in forecast_horizon_func:
                name_str = "_aggregation_"+aggregation + "_forecast_horizon_1H.pkl"
                booleen = 0
                for file in os.listdir(dirname):
                    if file.endswith(ext):
                        if name_str in file:
                            booleen = 1
                            print("We will use model for 1H instead")
                            for type_of_sensor in TYPES_OF_SENSOR:
                                modelfilename = MODEL_FILE_NAME+type_of_sensor+ "_aggregation_"+ aggregation + "_forecast_horizon_" + "1H" +".pkl"
                    
            elif 'D' in forecast_horizon_func:
                name_str = "_aggregation_"+aggregation + "_forecast_horizon_1D.pkl"
                booleen = 0
                for file in os.listdir(dirname):
                    if file.endswith(ext):
                        if name_str in file:
                            booleen = 1
                            print("We will use model for 1D instead")
                            for type_of_sensor in TYPES_OF_SENSOR:
                                modelfilename = MODEL_FILE_NAME+type_of_sensor+ "_aggregation_"+ aggregation + "_forecast_horizon_" + "1D" +".pkl"

        if booleen==0:
            print("We need to create a new model") 
            Models = [ModelTesting(KNeighborsRegressor, ["n_neighbors"], [5], [16], [5], proportional=True)]
            model_list = ['KNN']
            
            for type_of_sensor in TYPES_OF_SENSOR:
                Models = [ModelTesting(KNeighborsRegressor, ["n_neighbors"], [5], [16], [5], proportional=True)]
                model_list = ['KNN']
                [Best_model , r2]= train_forecasting_model(Models,model_list,PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ aggregation +".csv", forecast_horizon_func,intervalr2)
                print("ok")
            # save the model to disk
                joblib.dump(Best_model, MODEL_FILE_NAME +type_of_sensor+ "_aggregation_"+ aggregation + "_forecast_horizon_" + forecast_horizon_func +".pkl")
            print("created and trained new model")
            
            for type_of_sensor in TYPES_OF_SENSOR:
                modelfilename = MODEL_FILE_NAME+type_of_sensor + "_aggregation_"+ aggregation + "_forecast_horizon_" + forecast_horizon_func +".pkl"


        model_list = ['KNN']
        allforecast = pd.DataFrame()
        #columns_manquantes = []
        i=0
        r2_output=0
        for type_of_sensor in TYPES_OF_SENSOR:
            i=i+1
            [r2_average, inference_result, forecast_example, target_example, time_r2_df, time_inference]= infer_forecastnow(modelfilename, PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ aggregation +".csv", aggregation, forecast_horizon_func, intervalr2) 
            # save the model to disk
            # joblib.dump(Best_model, MODEL_FILE_NAME + "_aggregation_"+ aggregation + "_forecast_horizon" + forecast_horizon_func +".pkl")
            r2_output = max(r2_output,r2_average)
            #we prepare the output:
            now = datetime.now()
            time_inference.columns = ['time']

        #     step
        #    id: str
        #    date: str
        #    step_time: str
        #    prediction: List[float] = []
        #    average_r2: float
        #    historic_prediction: List[float]=[]
        #    historic_target: List[float] = []
            # print(type(forecast_example.values.tolist()))
            # print(Forecast(id="noise Forecasting Greenmov", date = now.strftime("%m/%d/%Y, %H:%M:%S"), step_time = aggregation, prediction= inference_result,average_r2= r2_average, historic_prediction= forecast_example,historic_target=target_example ))
            # print(forecast_example)
            forecast_example.columns=["forecast "+type_of_sensor+  " with r2: "+str(r2_average)]
            #inference_forecast = inference_result + type_of_sensor
            inference_forecast = pd.DataFrame(inference_result)

            # ----------------------------------------
            #    we prepare a display of the forecasts including forecast on historical data and we add the r2
            # ----------------------------------------
            inference_forecast.columns=["forecast "+type_of_sensor+ " with r2: "+str(r2_average)]
            target_example.columns=["historical value " + type_of_sensor]
            #print("test :",target_example.columns)
            allforecast = pd.concat([forecast_example,inference_forecast],axis=0, ignore_index=True)
            #allforecast = pd.concat([allforecast,forecast_example,inference_forecast],axis=0, ignore_index=True)
            time_r2_df = time_r2_df.to_frame(index=False)
            time_r2_df.columns = ["time"]
            alltime = pd.concat([time_r2_df,time_inference],axis=0, ignore_index=True)
            df_tosave=alltime.join(target_example)
            df_tosave =  df_tosave.join(allforecast)
            if (r2input==0):
                df_tosave['r2'] = r2_average
            else:
                df_tosave['r2']= r2input
            if i==1 :
                merged_df_func = alltime.join(target_example)
            else:
                merged_df_func = merged_df_func.join(target_example)  
        
            #alltime_ = pd.DataFrame({'"historical value " + type_of_sensor':target_example["historical value " + str(TYPES_OF_SENSOR[i])]})
            #columns_manquantes.append(target_example["historical value " + str(TYPES_OF_SENSOR[i])].tolist())
            

            merged_df_func =  merged_df_func.join(allforecast)
            
            # ----------------------------------------
            #    we will now store the data into the csv file that gathers all the last forecasts
            # ----------------------------------------
            # new_df = pd.concat([time_inference,inference_forecast],axis=1, ignore_index=True)
            # print(df_tosave)
            df_tosave.columns = ['time','historical value', 'forecasts', 'r2']
            # Read the existing CSV file
            existing_df = pd.read_csv('forecast_'+type_of_sensor+'.csv')
            existing_df.columns = ['time','historical value', 'forecasts', 'r2']

            if existing_df.empty:
                # print("no data yet so far")
                df_tosave['time'] = pd.to_datetime(df_tosave['time'])
                df_tosave.to_csv('forecast_'+type_of_sensor+'.csv', mode='a', header = False, index=False)
            else:
                existing_df['time'] = pd.to_datetime(existing_df['time'])              
                # Convert the Timestamp column to datetime format
                # print(new_df)
                df_tosave['time'] = pd.to_datetime(df_tosave['time'])

                # Define the time window (less than one hour difference)
                time_window = timedelta(hours=1)

                # Iterate through rows of df1 in reverse and update with matching rows from df2
                time_start_forecast = df_tosave.loc[0]['time']
                matching_index=-1
                # print("existing df before")
                # print(existing_df)
                for idx1 in reversed(existing_df.index):
                    row1 = existing_df.loc[idx1]
                    time1 = row1['time']
                    if ( time_start_forecast> time1 - time_window) & (time_start_forecast < time1 + time_window):
                        matching_index = idx1
                        # print("existing df before")
                        # print(existing_df)
                        # print("df_tosave")
                        # print(df_tosave)
                        df1_part1 = existing_df.loc[:matching_index-1] #= df_tosave
                        # print("#########################here ###########################")
                        # print(existing_df)
                        # print(df1_part1)
                        # print(df_tosave)
                        # Join df1_part1 and df2 using pd.concat()
                        result_df = pd.concat([df1_part1, df_tosave], ignore_index=True)

                        # print("index:", matching_index)
                        # print("existing df after")
                        # print(result_df)
                        result_df.to_csv('forecast_'+type_of_sensor+'.csv',  index=False)
                        break
                if matching_index == -1:
                    df_tosave.to_csv('forecast_'+type_of_sensor+'.csv', mode='a', header = False, index=False)
            # print("existing df after ")
            # print(existing_df)


            # Find the latest timestamp in the existing CSV file
            # latest_timestamp = existing_df['time'].max()
            # # Filter new_df to include only rows with timestamps beyond the latest_timestamp
            # new_rows = new_df[new_df['time'] > latest_timestamp]
            # # Append the new rows to the existing CSV file
            




            # print(merged_df_func.to_csv())
            
            forecast_df = pd.concat([time_inference, inference_forecast],axis=1, ignore_index=True)
            forecast_df.columns=["time", "forecast"]
            # URL_FORECAST  = "https://tip-imredd.unice.fr/data/imredd/nice/noisepollution/entityOperations/upsert?api-key=a1b4deee-008f-4161-ae24-4b7cf507107b"
            
            send_to_context_broker(URL_FORECAST, forecast_df, type_of_sensor)
        return [merged_df_func, r2_output]



# ###########################  MAIN  ########################################z
print("Calling thread: %s" % (threading.get_ident()))
for typeofsensor in TYPES_OF_SENSOR:
    if not os.path.exists('forecast_'+typeofsensor+'.csv'):
        with open('forecast_'+typeofsensor+'.csv', "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)
        print("CSV file created with header.")
    else:
        print("CSV file already exists.")

dirname = os.getcwd()
print(dirname)
print("starting first Thread. Will start server shortly after...")
# get_and_process_new_data_with_forecast_recurrent(TIME_TO_WAIT_FOR_TABLE_UPDATE,CSV_FILE_NAME)
print("Server is now ready to get started!")


# lancer forecast avec 1H, garder le R2, puis relancer le forecast avec 24h


############################################"  API Definition ###########################################


app = FastAPI()

# Configure CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/home")
async def index():
    return {"message": "Hello World "}

@app.get("/process/greenmov/noise")
async def processdb_noise(aggregation: str=AGGREGATION):  #aggregation can be '1D', '1H', ... 
    # for type_of_sensor in TYPES_OF_SENSOR:
    #     URL_GREENMOV_NOISE = 'https://tip-imredd.unice.fr/data/imredd/nice/noise/temporal/entities/?api-key=978c0962-c4e3-46f7-867d-9932d05d7987&type=https://smart-data-models.org/dataModel.Transportation/noiseFlowObserved&id=https://api.nicecotedazur.org/nca/mobility/noise/Promenade/manuel/'+type_of_sensor
    #     URL_GREENMOV_NOISE = URL_GREENMOV_NOISE + '&timeproperty=modifiedAt&options=sysAttrs&lastN=20&timerel=before&timeAt=' 
    #     [status] = get_new_data_once(URL_GREENMOV_NOISE,CSV_FILE_NAME+type_of_sensor+".csv")
    
        print("received request to process data")
    # if status == 0 :
    #     return {'could not update data'}
    # else:
        for type_of_sensor in TYPES_OF_SENSOR:
            process_data2(CSV_FILE_NAME+type_of_sensor+".csv", aggregation, PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ aggregation +".csv", type_of_sensor, FORECAST_HORIZON)
# def process_data(csv_file_name, aggregation, processed_data_file,typeofsensor, forecast_horizon)
        return {"data processed"}

@app.get("/updatedb/greenmov/noise")
async def updatedb_noise(aggregation: str=AGGREGATION):  #aggregation can be '1D', '1H', ... 
    print("received request to get - update- process data")
    print("starting getting data")
    for type_of_sensor in TYPES_OF_SENSOR:
        URL_GREENMOV_NOISE =  URL_PART1 + type_of_sensor 
        URL_GREENMOV_NOISE = URL_GREENMOV_NOISE + URL_PART2
        
        '&timeproperty=modifiedAt&options=sysAttrs&lastN=20&timerel=before&timeAt=' 
        # [status] = get_new_data_once(URL_GREENMOV_NOISE,CSV_FILE_NAME+type_of_sensor+".csv")
        if (TYPE_REQUEST==1):
            [status] = get_new_data_once(URL_GREENMOV_NOISE,CSV_FILE_NAME+type_of_sensor+".csv")    
        else:        
            [status] = get_new_data_once_using_before_request(URL_GREENMOV_NOISE,CSV_FILE_NAME+type_of_sensor+".csv")            


    if status == 0 :
        for type_of_sensor in TYPES_OF_SENSOR:
            process_data(CSV_FILE_NAME+type_of_sensor+".csv", aggregation, PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ aggregation +".csv", type_of_sensor, FORECAST_HORIZON)
        return {'could not update data but processed it'}
    else:
        for type_of_sensor in TYPES_OF_SENSOR:
            process_data(CSV_FILE_NAME+type_of_sensor+".csv", aggregation, PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ aggregation +".csv", type_of_sensor, FORECAST_HORIZON)

        return {"data updated and processed"}
@app.get("/trainforecasting/greenmov/noise")
async def trainmodel_noise(aggregation: str=AGGREGATION, forecast_horizon: str=FORECAST_HORIZON, intervalr2=REQUESTED_INTERVAL_R2 ):  #aggregation can be '1D', '1H', ... 
    # Models = [ModelTesting(RandomForestRegressor, ["n_estimators", "random_state"], [1, 0], [101, 1], [20, 1]), ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [3], proportional=True)]
    print("received request to train model")
    Models = [ModelTesting(KNeighborsRegressor, ["n_neighbors"], [5], [16], [5], proportional=True)]
    # Models = [ModelTesting(GradientBoostingRegressor, ["n_estimators"], [60], [61], [20])] #, ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [3], proportional=True)]
    # model_list = ['GradientBoost', 'kNN']
    # giving directory name
    dirname = os.getcwd()
    # giving file extension



    modelfiles = glob.glob(os.path.join(dirname, "*.pkl"))

    # Remove each pickle file
    for file_path in modelfiles:
        os.remove(file_path)
    print("generating new processed file without any interpolation data")
    for type_of_sensor in TYPES_OF_SENSOR:
                URL_GREENMOV_NOISE =  URL_PART1 + type_of_sensor 
                URL_GREENMOV_NOISE = URL_GREENMOV_NOISE + URL_PART2
                
                '&timeproperty=modifiedAt&options=sysAttrs&lastN=20&timerel=before&timeAt=' 
                if (TYPE_REQUEST==1):
                    [status] = get_new_data_once(URL_GREENMOV_NOISE,CSV_FILE_NAME+type_of_sensor+".csv")    
                else:        
                    [status] = get_new_data_once_using_before_request(URL_GREENMOV_NOISE,CSV_FILE_NAME+type_of_sensor+".csv")            
                process_data(CSV_FILE_NAME+type_of_sensor+".csv", aggregation, PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ aggregation +".csv", type_of_sensor, forecast_horizon)


    # ext = ('.csv')
    # name_str = "_aggregation_"+aggregation
    # for type_of_sensor in TYPES_OF_SENSOR:
    #     print("looking for file : ", PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ aggregation +".csv", "in : ", dirname)
   
    # # check if data file exists with that aggregation 
    # booleen = 0
    # for file in os.listdir(dirname):
    #     if file.endswith(ext):
    #         if name_str in file:
    #             booleen = 1
    #             print("existing processed file found")
    # if booleen == 0:
    #     for type_of_sensor in TYPES_OF_SENSOR:
    #         URL_GREENMOV_NOISE = 'https://tip-imredd.unice.fr/data/imredd/nice/noise/temporal/entities/?api-key=978c0962-c4e3-46f7-867d-9932d05d7987&type=https://smart-data-models.org/dataModel.Transportation/noiseFlowObserved&id=https://api.nicecotedazur.org/nca/mobility/noise/Promenade/manuel/'+type_of_sensor
    #         URL_GREENMOV_NOISE = URL_GREENMOV_NOISE + '&timeproperty=modifiedAt&options=sysAttrs&lastN=20&timerel=before&timeAt=' 
    #         [status] = get_new_data_once(URL_GREENMOV_NOISE,CSV_FILE_NAME+type_of_sensor+".csv")             
    #         process_data(CSV_FILE_NAME+type_of_sensor+".csv", aggregation, PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ aggregation +".csv", type_of_sensor, forecast_horizon)
   
    #     print("started creation of processed file")
    model_list = ['KNN']
    bestr2=-10

    for type_of_sensor in TYPES_OF_SENSOR:
        print ('training of :',type_of_sensor)
        Models = [ModelTesting(KNeighborsRegressor, ["n_neighbors"], [5], [16], [5], proportional=True)]
        [Best_model , r2]= train_forecasting_model(Models,model_list,PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ aggregation +".csv", forecast_horizon,intervalr2)
        r2 = max(0,r2)
        if r2>bestr2:
            bestr2=r2
        joblib.dump(Best_model, MODEL_FILE_NAME+type_of_sensor+ "_aggregation_"+ aggregation + "_forecast_horizon_" + forecast_horizon +".pkl")
        print ('saved model of :',type_of_sensor)

    # save the model to disk
    for type_of_sensor in TYPES_OF_SENSOR:
        joblib.dump(Best_model, MODEL_FILE_NAME+type_of_sensor+ "_aggregation_"+ aggregation + "_forecast_horizon_" + forecast_horizon +".pkl")
    
    return {"model trained, and average accuracy over the period (in %): ":  bestr2*100}




class Forecast(BaseModel):
   id: str
   date: str
   step_time: str
   prediction: List[float] = []
   average_r2: float
   historic_prediction: List[float]=[]
   historic_target: List[float] = []




@app.get("/forecast/greenmov/noise")
async def infer_forecast_noise(aggregation: str=AGGREGATION, forecast_horizon: str=FORECAST_HORIZON, intervalr2=REQUESTED_INTERVAL_R2 ):  #aggregation can be '1D', '1H', ... 
    print("received request to forecast data")
    #------------------------------------------------
    # First we check if the forecast is already available
    #-------------------------------------------------
    if aggregation =='1H':
        if 'D' in forecast_horizon:
            tmp = forecast_horizon
            tmp = int(tmp.replace('D',""))
            numberofhoursfortime = tmp*24
        elif  'H' in forecast_horizon:
            tmp = forecast_horizon
            tmp = int(tmp.replace('H',""))
            numberofhoursfortime = tmp
        else: print("Sorry, issue on aggregation: case NOT CONSIDERED YET")
        
        # time_inference1 = pd.period_range(start=DataSetIndexbyTime.tail(1).index[0]+pd.Timedelta(hours=1), end=DataSetIndexbyTime.tail(1).index[0]+pd.Timedelta(hours=numberofhoursfortime), freq='H').start_time
        now = pd.Timestamp.now()  # Get the current time
        periods = pd.period_range(start=now, periods=numberofhoursfortime, freq='H')  # Create a period range

        # Set the minutes and seconds to 0
        # periods = periods.map(lambda p: p.asfreq('H'))
        # Set the seconds to 0
        periods = periods.map(lambda p: p.asfreq('H').to_timestamp())

        # Reset the seconds to 0
        periods = periods.map(lambda ts: ts.replace(second=0))
        # time_inference1 = time_inference1.to_frame(index=False)
        time_inference1 = periods.to_frame(index=False)
    elif aggregation =='1D':
        now = pd.Timestamp.now()  # Get the current time

        if 'D' in forecast_horizon:
            tmp = forecast_horizon
            tmp = int(tmp.replace('D',""))
            numberofhoursfortime = tmp
        else: print("Sorry, issue on aggregation: case NOT CONSIDERED YET")
        
        periods = pd.period_range(start=now, periods=numberofhoursfortime, freq='D')
        periods = periods.map(lambda p: p.asfreq('D').to_timestamp())
        # print(time_inference1)
    else: 
        print("Sorry, issue on aggregation: case NOT CONSIDERED YET")
    
    
    # print(numberofhoursfortime)
    #we check if the requested forecast is already in the file of the forecasts:
    
    # ----------------------------------------
    #    we will now store the data into the csv file that gathers all the last forecasts
    # ----------------------------------------
    periods = pd.DataFrame(periods)
    # print(periods)
    periods.columns=['time']
    for type_of_sensor in TYPES_OF_SENSOR:
        existing_df = pd.read_csv('forecast_'+type_of_sensor+'.csv')
        existing_df['time'] = pd.to_datetime(existing_df['time'])  
        print(existing_df)
        if existing_df.empty:
            print("no data yet so far")
            bool_need_to_forecast = 1
        else:
            # Get the last timestamp from each DataFrame
            last_timestamp_df1 = existing_df['time'].iloc[-1]
            last_timestamp_df2 = periods['time'].iloc[-1]
            print(last_timestamp_df1)
            print(last_timestamp_df2)
            # Calculate the time difference
            time_difference = last_timestamp_df2 - last_timestamp_df1
            bool_need_to_forecast = 0
            # print(time_difference)
            number_rowsr2 = 0
            if aggregation =='1H':
                time_difference_int = int(abs(time_difference.days*24+time_difference.seconds/3600))
                tmp = intervalr2
                tmp = int(tmp.replace('D',""))
                number_rowsr2 = tmp*24                # print(time_difference_int)
            # Check if the time difference is more than 1 hour
                if time_difference >= timedelta(hours=1):
                    print("Time difference is more than 1 hour.")
                    bool_need_to_forecast = 1
                else:
                    print("Time difference is not more than 1 hour.")
                    bool_need_to_forecast = 0
            elif aggregation =='1D':
                tmp = intervalr2
                tmp = int(tmp.replace('D',""))
                number_rowsr2 = tmp
                time_difference_int = int(abs(time_difference.days+time_difference.seconds/(24*3600)))
                # Check if the time difference is more than 1 day
                print("time_diff")
                print(time_difference_int)
                if time_difference >= timedelta(days=1):
                    print("Time difference is more than 1 day.")
                    bool_need_to_forecast = 1
                else:
                    print("Time difference is not more than 1 day.")
                    bool_need_to_forecast = 0
            else:
                print("case of aggregation not considered yet")
                bool_need_to_forecast = 1
    # print("##############################################################################################################################")
    # print(bool_need_to_forecast)
    # # Convert the Timestamp column to datetime format
    # # print(new_df)
    # new_df['time'] = pd.to_datetime(new_df['time'])
    # # Find the latest timestamp in the existing CSV file
    # latest_timestamp = existing_df['time'].max()
    # # Filter new_df to include only rows with timestamps beyond the latest_timestamp
    # new_rows = new_df[new_df['time'] > latest_timestamp]
    # # Append the new rows to the existing CSV file
    # new_rows.to_csv('forecast_'+type_of_sensor+'.csv', mode='a', header=False, index=False)
    
    
    if bool_need_to_forecast==0:
        i = 0
        for type_of_sensor in TYPES_OF_SENSOR:
            i=i+1
            existing_df = pd.read_csv('forecast_'+type_of_sensor+'.csv')
            r2_value = existing_df['r2'].iloc[-1]
            existing_df=existing_df.drop(columns=['r2'])

            existing_df.columns = ['time','historical value '+type_of_sensor, 'forecasts '+type_of_sensor +' r2:' + str(r2_value)]
            df_tmp = existing_df.copy()
            df_tmp['time'] = pd.to_datetime(df_tmp['time'])
            df_tmp.set_index('time', inplace=True)
            # print(df_tmp)
            number_of_rows = number_rowsr2+numberofhoursfortime #len(df_tmp.first(intervalr2, errors='ignore'))+numberofhoursfortime
            # print(number_rowsr2)
            # print(numberofhoursfortime)
            # print(existing_df)
            # print(existing_df.tail(number_of_rows+time_difference_int))
            if i==1 :
                merged_df = existing_df.tail(number_of_rows+time_difference_int).head(number_of_rows)
            else:
                existing_df=existing_df.drop(columns=['time'])
                merged_df = merged_df.join(existing_df.tail(number_of_rows+time_difference_int).head(number_of_rows))  
        
        # Remove rows with NaN values only in the first 10 rows
        rows_to_keep_wo_na = merged_df.iloc[:number_rowsr2].dropna()
        merged_df = pd.concat([rows_to_keep_wo_na, merged_df.iloc[number_rowsr2:]])
    
    else:
        print("we need to compute new forecasts")

        #------------------------------------------------
        # If the forecast is not available, we will compute it
        #-------------------------------------------------
        
        
        
        [merged_df,r2forecast] = generate_new_forecasts(aggregation,forecast_horizon, intervalr2)
        # print(r2forecast)
        # print("###################################################################")
     
    return HTMLResponse(content=merged_df.to_html(), status_code=200)
    # return merged_df.to_csv(lineterminator="\n")




@app.get("/forecast/greenmov/noise/single")
async def infer_forecast_noise_single(aggregation: str=AGGREGATION, forecast_horizon: str=FORECAST_HORIZON, intervalr2='0D' ):  #aggregation can be '1D', '1H', ... 
    print("received request to forecast data")
    #------------------------------------------------
    # First we check if the forecast is already available
    #-------------------------------------------------
    if aggregation =='1H':
        if 'D' in forecast_horizon:
            tmp = forecast_horizon
            tmp = int(tmp.replace('D',""))
            numberofhoursfortime = tmp*24
        elif  'H' in forecast_horizon:
            tmp = forecast_horizon
            tmp = int(tmp.replace('H',""))
            numberofhoursfortime = tmp
        else: print("Sorry, issue on aggregation: case NOT CONSIDERED YET")
        
        # time_inference1 = pd.period_range(start=DataSetIndexbyTime.tail(1).index[0]+pd.Timedelta(hours=1), end=DataSetIndexbyTime.tail(1).index[0]+pd.Timedelta(hours=numberofhoursfortime), freq='H').start_time
        now = pd.Timestamp.now()  # Get the current time
        periods = pd.period_range(start=now, periods=numberofhoursfortime, freq='H')  # Create a period range

        # Set the minutes and seconds to 0
        # periods = periods.map(lambda p: p.asfreq('H'))
        # Set the seconds to 0
        periods = periods.map(lambda p: p.asfreq('H').to_timestamp())

        # Reset the seconds to 0
        periods = periods.map(lambda ts: ts.replace(second=0))
        # time_inference1 = time_inference1.to_frame(index=False)
        time_inference1 = periods.to_frame(index=False)
    elif aggregation =='1D':
        now = pd.Timestamp.now()  # Get the current time

        if 'D' in forecast_horizon:
            tmp = forecast_horizon
            tmp = int(tmp.replace('D',""))
            numberofhoursfortime = tmp
        else: print("Sorry, issue on aggregation: case NOT CONSIDERED YET")
        
        periods = pd.period_range(start=now, periods=numberofhoursfortime, freq='D')
        periods = periods.map(lambda p: p.asfreq('D').to_timestamp())
        # print(time_inference1)
    else: 
        print("Sorry, issue on aggregation: case NOT CONSIDERED YET")
    
    
    # print(numberofhoursfortime)
    #we check if the requested forecast is already in the file of the forecasts:
    
    # ----------------------------------------
    #    we will now store the data into the csv file that gathers all the last forecasts
    # ----------------------------------------
    periods = pd.DataFrame(periods)
    # print(periods)
    periods.columns=['time']
    for type_of_sensor in TYPES_OF_SENSOR:
        existing_df = pd.read_csv('forecast_'+type_of_sensor+'.csv')
        existing_df['time'] = pd.to_datetime(existing_df['time'])  
        # print(existing_df)
        if existing_df.empty:
            print("no data yet so far")
            bool_need_to_forecast = 1
        else:
            # Get the last timestamp from each DataFrame
            last_timestamp_df1 = existing_df['time'].iloc[-1]
            last_timestamp_df2 = periods['time'].iloc[-1]
            # print(last_timestamp_df1)
            # print(last_timestamp_df2)
            # Calculate the time difference
            time_difference = last_timestamp_df2 - last_timestamp_df1
            bool_need_to_forecast = 0
            # print(time_difference)
            number_rowsr2 = 0
            if aggregation =='1H':
                time_difference_int = int(abs(time_difference.days*24+time_difference.seconds/3600))
                tmp = intervalr2
                tmp = int(tmp.replace('D',""))
                number_rowsr2 = tmp*24                # print(time_difference_int)
            # Check if the time difference is more than 1 hour
                if time_difference >= timedelta(hours=1):
                    print("Time difference is more than 1 hour.")
                    bool_need_to_forecast = 1
                else:
                    print("Time difference is not more than 1 hour.")
                    bool_need_to_forecast = 0
            elif aggregation =='1D':
                tmp = intervalr2
                tmp = int(tmp.replace('D',""))
                number_rowsr2 = tmp
                time_difference_int = int(abs(time_difference.days+time_difference.seconds/(24*3600)))
                # Check if the time difference is more than 1 day
                if time_difference >= timedelta(days=1):
                    print("Time difference is more than 1 day.")
                    bool_need_to_forecast = 1
                else:
                    print("Time difference is not more than 1 day.")
                    bool_need_to_forecast = 0
            else:
                print("case of aggregation not considered yet")
                bool_need_to_forecast = 1
    # print("##############################################################################################################################")
    # print(bool_need_to_forecast)
    # # Convert the Timestamp column to datetime format
    # # print(new_df)
    # new_df['time'] = pd.to_datetime(new_df['time'])
    # # Find the latest timestamp in the existing CSV file
    # latest_timestamp = existing_df['time'].max()
    # # Filter new_df to include only rows with timestamps beyond the latest_timestamp
    # new_rows = new_df[new_df['time'] > latest_timestamp]
    # # Append the new rows to the existing CSV file
    # new_rows.to_csv('forecast_'+type_of_sensor+'.csv', mode='a', header=False, index=False)
    
    
    if bool_need_to_forecast==0:
        i = 0
        for type_of_sensor in TYPES_OF_SENSOR:
            i=i+1
            existing_df = pd.read_csv('forecast_'+type_of_sensor+'.csv')
            r2_value = existing_df['r2'].iloc[-1]
            existing_df=existing_df.drop(columns=['r2'])

            existing_df.columns = ['time','historical value '+type_of_sensor, 'forecasts '+type_of_sensor +' r2:' + str(r2_value)]
            df_tmp = existing_df.copy()
            df_tmp['time'] = pd.to_datetime(df_tmp['time'])
            df_tmp.set_index('time', inplace=True)
            # print(df_tmp)
            number_of_rows = number_rowsr2+numberofhoursfortime #len(df_tmp.first(intervalr2, errors='ignore'))+numberofhoursfortime
            if i==1 :
                merged_df = existing_df.tail(number_of_rows+time_difference_int).head(number_of_rows)
            else:
                existing_df=existing_df.drop(columns=['time'])
                merged_df = merged_df.join(existing_df.tail(number_of_rows+time_difference_int).head(number_of_rows))  
        
        # Remove rows with NaN values only in the first 10 rows
        rows_to_keep_wo_na = merged_df.iloc[:number_rowsr2].dropna()
        merged_df = pd.concat([rows_to_keep_wo_na, merged_df.iloc[number_rowsr2:]])
        
    
    else:

        #------------------------------------------------
        # If the forecast is not available, we will compute it
        #-------------------------------------------------
        
        
        
        [merged_df,r2forecast] = generate_new_forecasts(aggregation,forecast_horizon, intervalr2)
        # print(r2forecast)
        # print("###################################################################")
    merged_df=merged_df.tail(1) 
    return HTMLResponse(content=merged_df.to_html(), status_code=200)



@app.get("/forecast/greenmov/noise/json")
async def infer_forecast_noisejson(aggregation: str=AGGREGATION, forecast_horizon: str=FORECAST_HORIZON, intervalr2=REQUESTED_INTERVAL_R2 ):  #aggregation can be '1D', '1H', ... 
    print("received request to provide forecast in json")
#     # Models = [ModelTesting(RandomForestRegressor, ["n_estimators", "random_state"], [1, 0], [101, 1], [20, 1]), ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [3], proportional=True)]
#     Models = [ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [10], proportional=True)]
#     # Models = [ModelTesting(GradientBoostingRegressor, ["n_estimators"], [60], [61], [20])] #, ModelTesting(KNeighborsRegressor, ["n_neighbors"], [1], [41], [3], proportional=True)]
#     # model_list = ['GradientBoost', 'kNN']
#     # giving directory name
#     dirname = os.getcwd()
#     # giving file extension
#     ext = ('.csv')
#     name_str = "_aggregation_"+aggregation
#     for type_of_sensor in TYPES_OF_SENSOR:
#         print("looking for file : ", PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ aggregation +".csv", "in : ", dirname)
#     # check if data file exists with that aggregation 
#     booleen = 0
#     for file in os.listdir(dirname):
#         if file.endswith(ext):
#             if name_str in file:
#                 booleen = 1
#                 print("existing processed file found")
#     if booleen == 0:
#         for type_of_sensor in TYPES_OF_SENSOR:
#             process_data(CSV_FILE_NAME+type_of_sensor+".csv", aggregation, PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ aggregation +".csv", type_of_sensor, forecast_horizon)
#             print("started creation of processed file")


# #check if model exists
#     dirname = os.getcwd()

#     ext = ('.pkl')
#     name_str = "_aggregation_"+aggregation + "_forecast_horizon_" + forecast_horizon +".pkl"
#     # for type_of_sensor in TYPES_OF_SENSOR:
#         # print("looking for model : ", PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ aggregation + "_forecast_horizon_" + forecast_horizon +".pkl", "in : ", dirname)
#     # check if data file exists with that aggregation 
#     booleen = 0
#     # print(MODEL_FILE_NAME)
#     for file in os.listdir(dirname):
#         if file.endswith(ext):
#             if name_str in file:
#                 booleen = 1
#                 print("existing trained model found")
#                 for type_of_sensor in TYPES_OF_SENSOR:
#                     modelfilename = MODEL_FILE_NAME+type_of_sensor+ "_aggregation_"+ aggregation + "_forecast_horizon_" + forecast_horizon +".pkl"
#     if booleen == 0: #if the model does not exist
#         if 'H' in forecast_horizon:
#             name_str = "_aggregation_"+aggregation + "_forecast_horizon_1H.pkl"
#             booleen = 0
#             for file in os.listdir(dirname):
#                 if file.endswith(ext):
#                     if name_str in file:
#                         booleen = 1
#                         print("We will use model for 1H instead")
#                         for type_of_sensor in TYPES_OF_SENSOR:
#                             modelfilename = MODEL_FILE_NAME+type_of_sensor+ "_aggregation_"+ aggregation + "_forecast_horizon_" + "1H" +".pkl"
                  
#         elif 'D' in forecast_horizon:
#             name_str = "_aggregation_"+aggregation + "_forecast_horizon_1D.pkl"
#             booleen = 0
#             for file in os.listdir(dirname):
#                 if file.endswith(ext):
#                     if name_str in file:
#                         booleen = 1
#                         print("We will use model for 1D instead")
#                         forecast_horizon = '1D'           
#                         for type_of_sensor in TYPES_OF_SENSOR:
#                             modelfilename = MODEL_FILE_NAME+type_of_sensor+ "_aggregation_"+ aggregation + "_forecast_horizon_" + "1D" +".pkl"

#     if booleen==0:
#         print("We need to create a new model") 
#         Models = [ModelTesting(KNeighborsRegressor, ["n_neighbors"], [5], [16], [5], proportional=True)]
#         model_list = ['KNN']
#         for type_of_sensor in TYPES_OF_SENSOR:
#             Models = [ModelTesting(KNeighborsRegressor, ["n_neighbors"], [5], [16], [5], proportional=True)]
#             model_list = ['KNN']
#             [Best_model , r2]= train_forecasting_model(Models,model_list,PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ aggregation +".csv", forecast_horizon,intervalr2)
#             print("ok")
#         # save the model to disk
#             joblib.dump(Best_model, MODEL_FILE_NAME +type_of_sensor+ "_aggregation_"+ aggregation + "_forecast_horizon_" + forecast_horizon +".pkl")
#         print("created and trained new model")
        
#         for type_of_sensor in TYPES_OF_SENSOR:
#             modelfilename = MODEL_FILE_NAME+type_of_sensor + "_aggregation_"+ aggregation + "_forecast_horizon_" + forecast_horizon +".pkl"


#     model_list = ['KNN']
#     allforecast = pd.DataFrame()
#     #columns_manquantes = []
#     i=0
#     for type_of_sensor in TYPES_OF_SENSOR:
#         i=i+1
#         [r2_average, inference_result, forecast_example, target_example, time_r2_df, time_inference]= infer_forecastnow(modelfilename, PROCESSED_DATA_FILE+type_of_sensor+"_aggregation_"+ aggregation +".csv", aggregation, forecast_horizon, intervalr2) 
#         # save the model to disk
#         # joblib.dump(Best_model, MODEL_FILE_NAME + "_aggregation_"+ aggregation + "_forecast_horizon" + forecast_horizon +".pkl")

#         #we prepare the output:
#         now = datetime.now()
#     #     step
#     #    id: str
#     #    date: str
#     #    step_time: str
#     #    prediction: List[float] = []
#     #    average_r2: float
#     #    historic_prediction: List[float]=[]
#     #    historic_target: List[float] = []
#         # print(type(forecast_example.values.tolist()))
#         # print(Forecast(id="noise Forecasting Greenmov", date = now.strftime("%m/%d/%Y, %H:%M:%S"), step_time = aggregation, prediction= inference_result,average_r2= r2_average, historic_prediction= forecast_example,historic_target=target_example ))
#         # print(forecast_example)
#         forecast_example.columns=["forecast "+type_of_sensor+  " with r2: "+str(r2_average)]
#         #inference_forecast = inference_result + type_of_sensor
#         inference_forecast = pd.DataFrame(inference_result)
#         inference_forecast.columns=["forecast "+type_of_sensor+ " with r2: "+str(r2_average)]
#         target_example.columns=["historical value " + type_of_sensor]
#         #print("test :",target_example.columns)
#         allforecast = pd.concat([forecast_example,inference_forecast],axis=0, ignore_index=True)
#         #allforecast = pd.concat([allforecast,forecast_example,inference_forecast],axis=0, ignore_index=True)
#         time_r2_df = time_r2_df.to_frame(index=False)
#         time_r2_df.columns = ["time"]
#         time_inference.columns = ["time"]
#         alltime = pd.concat([time_r2_df,time_inference],axis=0, ignore_index=True)
#         if i==1 :
#             merged_df = alltime.join(target_example)
#         else:
#             merged_df = merged_df.join(target_example)  

#         #alltime_ = pd.DataFrame({'"historical value " + type_of_sensor':target_example["historical value " + str(TYPES_OF_SENSOR[i])]})
#         #columns_manquantes.append(target_example["historical value " + str(TYPES_OF_SENSOR[i])].tolist())
        

#         merged_df =  merged_df.join(allforecast)
#         # print(merged_df.to_csv())
        
#         forecast_df = pd.concat([time_inference, inference_forecast],axis=1, ignore_index=True)
#         forecast_df.columns=["time", "forecast"]
#         URL_FORECAST  = "https://tip-imredd.unice.fr/data/imredd/nice/noisepollution/entityOperations/upsert?api-key=a1b4deee-008f-4161-ae24-4b7cf507107b"
        
#         send_to_context_broker(URL_FORECAST, forecast_df, type_of_sensor)
#     #print("columns_manquantes: ",columns_manquantes)
#     #columns = ["thermal", "elec", "bus", "moto"]
    
#     #df = pd.DataFrame(columns_manquantes)
#     #df.columns =columns
#     #print(df)
    #------------------------------------------------
    # First we check if the forecast is already available
    #-------------------------------------------------
    if aggregation =='1H':
        if 'D' in forecast_horizon:
            tmp = forecast_horizon
            tmp = int(tmp.replace('D',""))
            numberofhoursfortime = tmp*24
        elif  'H' in forecast_horizon:
            tmp = forecast_horizon
            tmp = int(tmp.replace('H',""))
            numberofhoursfortime = tmp
        else: print("Sorry, issue on aggregation: case NOT CONSIDERED YET")
        
        # time_inference1 = pd.period_range(start=DataSetIndexbyTime.tail(1).index[0]+pd.Timedelta(hours=1), end=DataSetIndexbyTime.tail(1).index[0]+pd.Timedelta(hours=numberofhoursfortime), freq='H').start_time
        now = pd.Timestamp.now()  # Get the current time
        periods = pd.period_range(start=now, periods=numberofhoursfortime, freq='H')  # Create a period range

        # Set the minutes and seconds to 0
        # periods = periods.map(lambda p: p.asfreq('H'))
        # Set the seconds to 0
        periods = periods.map(lambda p: p.asfreq('H').to_timestamp())

        # Reset the seconds to 0
        periods = periods.map(lambda ts: ts.replace(second=0))
        # time_inference1 = time_inference1.to_frame(index=False)
        time_inference1 = periods.to_frame(index=False)
    elif aggregation =='1D':
        now = pd.Timestamp.now()  # Get the current time

        if 'D' in forecast_horizon:
            tmp = forecast_horizon
            tmp = int(tmp.replace('D',""))
            numberofhoursfortime = tmp
        else: print("Sorry, issue on aggregation: case NOT CONSIDERED YET")
        
        periods = pd.period_range(start=now, periods=numberofhoursfortime, freq='D')
        periods = periods.map(lambda p: p.asfreq('D').to_timestamp())
        # print(time_inference1)
    else: 
        print("Sorry, issue on aggregation: case NOT CONSIDERED YET")
    
    
    # print(numberofhoursfortime)
    #we check if the requested forecast is already in the file of the forecasts:
    
    # ----------------------------------------
    #    we will now store the data into the csv file that gathers all the last forecasts
    # ----------------------------------------
    periods = pd.DataFrame(periods)
    # print(periods)
    periods.columns=['time']
    for type_of_sensor in TYPES_OF_SENSOR:
        existing_df = pd.read_csv('forecast_'+type_of_sensor+'.csv')
        existing_df['time'] = pd.to_datetime(existing_df['time'])  
        print(existing_df)
        if existing_df.empty:
            print("no data yet so far")
            bool_need_to_forecast = 1
        else:
            # Get the last timestamp from each DataFrame
            last_timestamp_df1 = existing_df['time'].iloc[-1]
            last_timestamp_df2 = periods['time'].iloc[-1]
            print(last_timestamp_df1)
            print(last_timestamp_df2)
            # Calculate the time difference
            time_difference = last_timestamp_df2 - last_timestamp_df1
            bool_need_to_forecast = 0
            # print(time_difference)
            number_rowsr2 = 0
            if aggregation =='1H':
                time_difference_int = int(abs(time_difference.days*24+time_difference.seconds/3600))
                tmp = intervalr2
                tmp = int(tmp.replace('D',""))
                number_rowsr2 = tmp*24                # print(time_difference_int)
            # Check if the time difference is more than 1 hour
                if time_difference >= timedelta(hours=1):
                    print("Time difference is more than 1 hour.")
                    bool_need_to_forecast = 1
                else:
                    print("Time difference is not more than 1 hour.")
                    bool_need_to_forecast = 0
            elif aggregation =='1D':
                tmp = intervalr2
                tmp = int(tmp.replace('D',""))
                number_rowsr2 = tmp
                time_difference_int = int(abs(time_difference.days+time_difference.seconds/(24*3600)))
                # Check if the time difference is more than 1 day
                if time_difference >= timedelta(days=1):
                    print("Time difference is more than 1 day.")
                    bool_need_to_forecast = 1
                else:
                    print("Time difference is not more than 1 day.")
                    bool_need_to_forecast = 0
            else:
                print("case of aggregation not considered yet")
                bool_need_to_forecast = 1
    # print("##############################################################################################################################")
    # print(bool_need_to_forecast)
    # # Convert the Timestamp column to datetime format
    # # print(new_df)
    # new_df['time'] = pd.to_datetime(new_df['time'])
    # # Find the latest timestamp in the existing CSV file
    # latest_timestamp = existing_df['time'].max()
    # # Filter new_df to include only rows with timestamps beyond the latest_timestamp
    # new_rows = new_df[new_df['time'] > latest_timestamp]
    # # Append the new rows to the existing CSV file
    # new_rows.to_csv('forecast_'+type_of_sensor+'.csv', mode='a', header=False, index=False)
    
    
    if bool_need_to_forecast==0:
        i = 0
        for type_of_sensor in TYPES_OF_SENSOR:
            i=i+1
            existing_df = pd.read_csv('forecast_'+type_of_sensor+'.csv')
            r2_value = existing_df['r2'].iloc[-1]
            existing_df=existing_df.drop(columns=['r2'])

            existing_df.columns = ['time','historical value '+type_of_sensor, 'forecasts '+type_of_sensor +' r2:' + str(r2_value)]
            df_tmp = existing_df.copy()
            df_tmp['time'] = pd.to_datetime(df_tmp['time'])
            df_tmp.set_index('time', inplace=True)
            # print(df_tmp)
            number_of_rows = number_rowsr2+numberofhoursfortime #len(df_tmp.first(intervalr2, errors='ignore'))+numberofhoursfortime
            if i==1 :
                merged_df = existing_df.tail(number_of_rows+time_difference_int).head(number_of_rows)
            else:
                existing_df=existing_df.drop(columns=['time'])
                merged_df = merged_df.join(existing_df.tail(number_of_rows+time_difference_int).head(number_of_rows))  
        
        
        # Remove rows with NaN values only in the first 10 rows
        rows_to_keep_wo_na = merged_df.iloc[:number_rowsr2].dropna()
        merged_df = pd.concat([rows_to_keep_wo_na, merged_df.iloc[number_rowsr2:]])    
    
    else:

        #------------------------------------------------
        # If the forecast is not available, we will compute it
        #-------------------------------------------------
        
        
        
        [merged_df,r2forecast]  = generate_new_forecasts(aggregation,forecast_horizon, intervalr2)
        # Convert DataFrame to dictionary
    merged_df['time'] = pd.to_datetime(merged_df['time'])
    merged_df['time'] = merged_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    columns_to_round = merged_df.columns[1:]  # Exclude the first column
    merged_df[columns_to_round] = merged_df[columns_to_round].round(4)
    merged_df = merged_df.fillna(0)
    df_dict = merged_df.to_dict(orient='records')  # Choose 'records' or 'list' depending on your use case
  
    # print(df_dict)
    # Return the dictionary as a JSON response
    return JSONResponse(content=df_dict, status_code=200)





@app.get("/forecast/greenmov/noise/json/single")
async def infer_forecast_noisejsonsingle(aggregation: str=AGGREGATION, forecast_horizon: str=FORECAST_HORIZON, intervalr2='0D' ):  #aggregation can be '1D', '1H', ... 
    print("received request to provide forecast in json")
#
    #------------------------------------------------
    # First we check if the forecast is already available
    #-------------------------------------------------
    if aggregation =='1H':
        if 'D' in forecast_horizon:
            tmp = forecast_horizon
            tmp = int(tmp.replace('D',""))
            numberofhoursfortime = tmp*24
        elif  'H' in forecast_horizon:
            tmp = forecast_horizon
            tmp = int(tmp.replace('H',""))
            numberofhoursfortime = tmp
        else: print("Sorry, issue on aggregation: case NOT CONSIDERED YET")
        
        # time_inference1 = pd.period_range(start=DataSetIndexbyTime.tail(1).index[0]+pd.Timedelta(hours=1), end=DataSetIndexbyTime.tail(1).index[0]+pd.Timedelta(hours=numberofhoursfortime), freq='H').start_time
        now = pd.Timestamp.now()  # Get the current time
        periods = pd.period_range(start=now, periods=numberofhoursfortime, freq='H')  # Create a period range

        # Set the minutes and seconds to 0
        # periods = periods.map(lambda p: p.asfreq('H'))
        # Set the seconds to 0
        periods = periods.map(lambda p: p.asfreq('H').to_timestamp())

        # Reset the seconds to 0
        periods = periods.map(lambda ts: ts.replace(second=0))
        # time_inference1 = time_inference1.to_frame(index=False)
        time_inference1 = periods.to_frame(index=False)
    elif aggregation =='1D':
        now = pd.Timestamp.now()  # Get the current time

        if 'D' in forecast_horizon:
            tmp = forecast_horizon
            tmp = int(tmp.replace('D',""))
            numberofhoursfortime = tmp
        else: print("Sorry, issue on aggregation: case NOT CONSIDERED YET")
        
        periods = pd.period_range(start=now, periods=numberofhoursfortime, freq='D')
        periods = periods.map(lambda p: p.asfreq('D').to_timestamp())
        # print(time_inference1)
    else: 
        print("Sorry, issue on aggregation: case NOT CONSIDERED YET")
    
    
    # print(numberofhoursfortime)
    #we check if the requested forecast is already in the file of the forecasts:
    
    # ----------------------------------------
    #    we will now store the data into the csv file that gathers all the last forecasts
    # ----------------------------------------
    periods = pd.DataFrame(periods)
    # print(periods)
    periods.columns=['time']
    for type_of_sensor in TYPES_OF_SENSOR:
        existing_df = pd.read_csv('forecast_'+type_of_sensor+'.csv')
        existing_df['time'] = pd.to_datetime(existing_df['time'])  
        # print(existing_df)
        if existing_df.empty:
            print("no data yet so far")
            bool_need_to_forecast = 1
        else:
            # Get the last timestamp from each DataFrame
            last_timestamp_df1 = existing_df['time'].iloc[-1]
            last_timestamp_df2 = periods['time'].iloc[-1]
            # print(last_timestamp_df1)
            # print(last_timestamp_df2)
            # Calculate the time difference
            time_difference = last_timestamp_df2 - last_timestamp_df1
            bool_need_to_forecast = 0
            # print(time_difference)
            number_rowsr2 = 0
            if aggregation =='1H':
                time_difference_int = int(abs(time_difference.days*24+time_difference.seconds/3600))
                tmp = intervalr2
                tmp = int(tmp.replace('D',""))
                number_rowsr2 = tmp*24                # print(time_difference_int)
            # Check if the time difference is more than 1 hour
                if time_difference >= timedelta(hours=1):
                    print("Time difference is more than 1 hour.")
                    bool_need_to_forecast = 1
                else:
                    print("Time difference is not more than 1 hour.")
                    bool_need_to_forecast = 0
            elif aggregation =='1D':
                tmp = intervalr2
                tmp = int(tmp.replace('D',""))
                number_rowsr2 = tmp
                time_difference_int = int(abs(time_difference.days+time_difference.seconds/(24*3600)))

                # Check if the time difference is more than 1 day
                if time_difference >= timedelta(days=1):
                    print("Time difference is more than 1 day.")
                    bool_need_to_forecast = 1
                else:
                    print("Time difference is not more than 1 day.")
                    bool_need_to_forecast = 0
            else:
                print("case of aggregation not considered yet")
                bool_need_to_forecast = 1
    # print("##############################################################################################################################")
    # print(bool_need_to_forecast)
    # # Convert the Timestamp column to datetime format
    # # print(new_df)
    # new_df['time'] = pd.to_datetime(new_df['time'])
    # # Find the latest timestamp in the existing CSV file
    # latest_timestamp = existing_df['time'].max()
    # # Filter new_df to include only rows with timestamps beyond the latest_timestamp
    # new_rows = new_df[new_df['time'] > latest_timestamp]
    # # Append the new rows to the existing CSV file
    # new_rows.to_csv('forecast_'+type_of_sensor+'.csv', mode='a', header=False, index=False)
    
    
    if bool_need_to_forecast==0:
        i = 0
        for type_of_sensor in TYPES_OF_SENSOR:
            i=i+1
            existing_df = pd.read_csv('forecast_'+type_of_sensor+'.csv')
            r2_value = existing_df['r2'].iloc[-1]
            existing_df=existing_df.drop(columns=['r2'])

            existing_df.columns = ['time','historical value '+type_of_sensor, 'forecasts '+type_of_sensor +' r2:' + str(r2_value)]
            df_tmp = existing_df.copy()
            df_tmp['time'] = pd.to_datetime(df_tmp['time'])
            df_tmp.set_index('time', inplace=True)
            # print(df_tmp)
            number_of_rows = number_rowsr2+numberofhoursfortime #len(df_tmp.first(intervalr2, errors='ignore'))+numberofhoursfortime
            # print(number_of_rows)
            if i==1 :
                merged_df = existing_df.tail(number_of_rows+time_difference_int).head(number_of_rows)
            else:
                existing_df=existing_df.drop(columns=['time'])
                merged_df = merged_df.join(existing_df.tail(number_of_rows+time_difference_int).head(number_of_rows))  
        
        
        # Remove rows with NaN values only in the first 10 rows
        rows_to_keep_wo_na = merged_df.iloc[:number_rowsr2].dropna()
        merged_df = pd.concat([rows_to_keep_wo_na, merged_df.iloc[number_rowsr2:]])    
    else:

        #------------------------------------------------
        # If the forecast is not available, we will compute it
        #-------------------------------------------------
        
        
        
        [merged_df,r2forecast]  = generate_new_forecasts(aggregation,forecast_horizon, intervalr2)
        # Convert DataFrame to dictionary
    merged_df['time'] = pd.to_datetime(merged_df['time'])
    merged_df['time'] = merged_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    merged_df = merged_df.tail(1)
    print(merged_df)
    for type_of_sensor in TYPES_OF_SENSOR:
        merged_df=merged_df.drop(columns=['historical value '+type_of_sensor])
    columns_to_round = merged_df.columns[1:]  # Exclude the first column
    merged_df[columns_to_round] = merged_df[columns_to_round].round(4)
    merged_df = merged_df.fillna(0)

    df_dict = merged_df.to_dict(orient='records')  # Choose 'records' or 'list' depending on your use case




    # print(df_dict)
    # Return the dictionary as a JSON response
    return JSONResponse(content=df_dict, status_code=200)



    # return HTMLResponse(content=merged_df.to_html(), status_code=200)
    # return merged_df.to_csv(lineterminator="\n")
# @app.get("/noise/trainforecast")
# async def trainandgetforecast():
#     ###################### GET DATA ##############################
#     #connect to the database
#     print("received a GET request")
#     db = psycopg2.connect(database="postgres", user = "postgres", password = "root", host = "sensorDB", port = "5432")
#     #create a cursor to navigate in the database:
#     cursor = db.cursor()
#     ## creating a table called 'sensordata' in the 'postgres' database
#     cursor.execute("CREATE TABLE IF NOT EXISTS sensordata (id VARCHAR(255), value FLOAT, timestamp BIGINT);")
#     query = "SELECT * FROM sensordata order by timestamp desc limit 1;"
#     # query = "SELECT * FROM sensordata;"
#     # ## getting records from the table
#     cursor.execute(query)
#     # ## fetching all records from the 'cursor' object
#     records = cursor.fetchall()
#     db.close()
#     # ## returning the data
#     return records






