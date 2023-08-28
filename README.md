# Noise_TimeSeries_Forecasting
Solution to do short term forecast of timeseries such as noise 
To deploy the service:
DOwnload the files:
wget -O DockerfileNoise.zip 'https://unice-my.sharepoint.com/:u:/g/personal/benoit_couraud_unice_fr/EZgy3MsEZ7VPiFt0fXliU9YBj-NVqRaaieFuRByn2F2qBA?download=1'
unzip DockerfileNoise.zip
#move into the unzipped folder:
cd DockerFolder
#then, in "docker-compose.yml", change the PATH to the path of the location of the current unzipped folder.
        volumes:
        - "C:/Users/Benoit Couraud/OneDrive - Université Nice Sophia Antipolis/projet/Greenmov/Activity 3/NoiseForecasting/DeploymentDocker/DockerFolder:/usr/src/appnoise"
Indeed, in the docker-compose.yml, You should change the path of the volumes to match your own directory: replace "C:/Users/Benoit Couraud/OneDrive - Université Nice Sophia Antipolis/projet/Greenmov/Activity 3/NoiseForecasting/Final_2023_08_21/DockerFolder" with the path to your directory:



In the code, you should change:
- TYPES_OF_SENSOR = [  "Lenval_Noise" ] --> put the name of your SENSOR, e.g: "YOUR_SENSOR_NAME"
In the folder, you should add a csv file with historical data that is named "YOUR_SENSOR_NAME.csv" where you should change the name to match what is in the array "TYPES_OF_SENSOR".
In case you want You can change "CSV_FILE_NAME" in case the name of your csv files is "CSV_NAME_type_of_sensor.csv". In this example, there is no CSV_NAME except the name of the sensor.

You should change the URLs as follows:
- URL_FORECAST  = "https://tip-imredd.unice.fr/data/imredd/nice/noisepollution/entityOperations/upsert?api-key=a1b4deee-008f-4161-ae24-4b7cf507107b"
- URL_PART1 = 'https://tip-imredd.unice.fr/data/imredd/nice/noisepollution/temporal/entities/?api-key=a1b4deee-008f-4161-ae24-4b7cf507107b&type=https://smartdatamodels.org/dataModel.Environment/NoisePollution&id=https://api.nicecotedazur.org/nca/environment/air/noiselevel/AZIMUT/'
- URL_PART2 = '?timeproperty=modifiedAt&options=sysAttrs&lastN=200&attrs=Lamax2&timerel=before&timeAt='
Where URL_PART1 should stop where the names in "TYPES_OF_SENSOR" will be inserted.
You should also update the following variable:
- FORECAST_ID_PART1 = "https://api.nicecotedazur.org/nca/environment/air/noiselevel/AZIMUT/" as the id of the forecast you will compute is constructed as follows:  "id": FORECAST_ID_PART1+typeofsensor+ FORECAST_ID_PART2, 
- FORECAST_ID_PART2 = "/forecast"
- TYPE_REQUEST = 2 #This is to indicate if your update of historical data use a Request to the Context Borker that use the AFTER (1) or the BEFORE (2) key word.

Then, in the function: 
def process_data(csv_file_name, aggregation, processed_data_file,typeofsensor, forecast_horizon):
    df = pd.read_csv(csv_file_name, sep=',', skiprows=[0, 1], usecols=[0, 1], names=["Time", "value"])
In our case, columns 0 and 1 are used for time and the value respectively.
you should change it to match your own data format. 
-CSV_TIME_FORMAT= '%Y-%m-%d %H:%M:%S'  you should change it so it matches the format you chose for the time in your csv file
You should also specify the Attribute that you want to forecast, in this case, Lamax2 for example, but could be LAeq
-ATTRIBUTE_TO_FORECAST='Lamax2'

 You also might have to change the data model. This is defined in the send_to_context_broker function.
 
Finally, you can run: docker compose up -d


Then, to make it run properly:
1. Verify that everything is working properly by opening a browser and open "http://127.0.0.1:8910/home" (replace with your IP and the right port)
2. Update the database: "http://127.0.0.1:8910/updatedb/greenmov/noise" (replace with your IP and the right port)
3. Train a first model (this should take some time): "http://127.0.0.1:8910/trainforecasting/greenmov/noise?aggregation=1H&forecast_horizon=1H"
4. Try it out: http://127.0.0.1:8000/forecast/greenmov/noise?aggregation=1H&forecast_horizon=1H

You can query for larger time horizons by changing the last parameter of the query: http://127.0.0.1:8910/forecast/greenmov/noise?aggregation=1H&forecast_horizon=24H
You can also query for json data : http://127.0.0.1:8910/forecast/greenmov/noise/json?aggregation=1H&forecast_horizon=1H
If you do not need to have the forecast of the last 7 days along with historical data, you can run the following requests:
http://127.0.0.1:8910/forecast/greenmov/noise/single?aggregation=1H&forecast_horizon=1H
to get json data: http://127.0.0.1:8910/forecast/greenmov/noise/json/single?aggregation=1H&forecast_horizon=1H

Finally, you can also query for a specific time: a swagger is available at 127.0.0.1:8910/docs

