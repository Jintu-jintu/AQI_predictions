from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
app=Flask(__name__)
model=pickle.load(open('model1.pkl','rb'))

@app.route('/')
def home():
    return render_template('home1.html')
@app.route('/predict', methods = ['POST'])
def predict():
    so2aqi = float(request.values['so2aqi'])
    no2aqi = float(request.values['no2aqi'])
    o3aqi = float(request.values['o3aqi'])
    coaqi = float(request.values['coaqi'])
    values = [no2aqi, so2aqi, coaqi, o3aqi]
    df = pd.DataFrame(np.array([values]), columns = ['NO2 AQI', 'SO2 AQI','CO AQI', 'O3 AQI'])
    out = model.predict(df)
    output = out.item()

    if output == 0:
        prediction = 'AQI category is : "Good", Air quality is considered satisfactory, and air pollution poses little or no risk.'
    elif output == 1:
        prediction = 'AQI category is : "Moderate", Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.'
    elif output == 2:
        prediction = 'AQI category is : "Unhealthy for Sensitive Groups", Members of sensitive groups may experience health effects. The general public is less likely to be affected.'
    elif output == 3:
        prediction = 'AQI category is : "Unhealthy", Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.'
    elif output == 4:
        prediction = 'AQI category is : "Very Unhealthy", Health alert: The risk of health effects is increased for everyone.'
    elif output == 5:
        prediction = 'AQI category is : "Hazardous", Health warning of en]mergency conditions: everyone is more likely to be affected'
    else:
        prediction = 'Could not find'
    return render_template('result1.html', prediction = prediction)
    
if __name__=='__main__':
    app.run(port=8000)

