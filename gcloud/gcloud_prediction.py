#ref : https://medium.com/google-cloud/online-predictions-api-using-scikit-learn-and-cloud-machine-learning-engine-on-gcp-10f86585e11e
#https://cloud.google.com/ml-engine/docs/python-client-library
#https://cloud.google.com/ml-engine/docs/scikit/custom-prediction-routine-scikit-learn
#https://cloud.google.com/ml-engine/docs/scikit/quickstart
#pip install google-api-python-client

import googleapiclient.discovery
import datetime
import numpy as np
import os

def predict_trip_time(project, model, start_lat, start_long, end_lat, end_long, date = 0, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        start_lat : starting latitude
        start_long : start longitude
        end_lat : ending latitude
        end_long : ending longitude
        date : if date other than current date
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'isb-cab-data-analysis-195630b10717.json'
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)
        
    if(date == 0) :
        date = datetime.datetime.now()
        hour = date.hour
    else :
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
        hour = 15
    dayofweek = date.weekday()
    
    earth_radius = 6371
    lat1, lon1, lat2, lon2 = np.radians([start_lat, start_long, end_lat, end_long])
    a = np.sin((lat2-lat1)/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2
    distance = earth_radius * 2 * np.arcsin(np.sqrt(a))
    
    data = [[start_lat, start_long, end_lat, end_long, distance, hour, dayofweek]]
    
    response = service.projects().predict(
        name=name,
        body={'instances': data}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return int(response['predictions'][0])
    
    
PROJECT_ID = 'isb-cab-data-analysis'
MODEL_NAME = 'trip_time_model'

start_lat = 41.158962
start_long = -8.634978
end_lat = 41.140584
end_long = -8.615817
print(predict_trip_time(PROJECT_ID, MODEL_NAME, start_lat, start_long, end_lat, end_long))
#print(predict_trip_time(PROJECT_ID, MODEL_NAME, start_lat, start_long, end_lat, end_long, '2019-11-24'))