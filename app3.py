import pandas as pd
import numpy as np
from flask import Flask, render_template, Response, request
import pickle
from sklearn.preprocessing import LabelEncoder
import pickle



app = Flask(__name__)
model = pickle.load(open('regression.pkl','rb'))
modelresale = pickle.load(open('rf_regression_model.pkl','rb'))
encode = pickle.load(open('encoder.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))
filename = 'resale_model.sav'
model_rand = pickle.load(open(filename, 'rb'))

def performance(mpg):
    label = ""
    if(mpg<=9):
        label = "Worst. Carry extra fuel."
    elif(mpg>9 and mpg<20):
        label = "Low. Don't go to long distance."
    elif(mpg>17.5 and mpg<=29):
        label = "Medium. Go for a ride nearby."
    elif(mpg>29):
        label = "High. You can plan for a tour."
    else:
        label = "Unknown"
    return label


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/car')
def car():
    return render_template('index.html')

@app.route('/predict', methods = ["POST"])
def predict():
    

    features = [[x for x in request.form.values()]]
    print(features)
    features[0][-1] = features[0][-1].split()[0] # extracting company name from car name 
    print(features)
    features[0].pop(1) # removing displacement
    print(features)
    print(features[0][-1])
    features[0][-1] = encode.transform([features[0][-1]])
    print(features)
    features = scaler.transform(features)
    print(features)
    mpg = model.predict(features)
    print(mpg)
    per_label = performance(mpg)
    print(per_label)
    return render_template('index.html', prediction_text= 'Car Performance : {label}  \nMiles Per Galon (MPG) : {}'.format(mpg,label=per_label))






@app.route('/predict1')
def predict1():
    return render_template('resalepredict.html')

@app.route('/y_predict', methods=['GET','POST'])
def y_predict():

        Year = int(request.form['Year'])
        Present_Price=float(request.form['Present_Price'])
        Kms_Driven=int(request.form['Kms_Driven'])
        Kms_Driven2=np.log(Kms_Driven)
        Owner=int(request.form['Owner'])
        Fuel_Type=request.form['Fuel_Type']
        if(Fuel_Type=='Petrol'):
                Fuel_Type=1
        else:
            Fuel_Type=0
        Year=2022-Year
        Seller_Type=request.form['Seller_Type']
        if(Seller_Type=='Individual'):
            Seller_Type=1
        else:
            Seller_Type=0	
        Transmission=request.form['Transmission']
        if(Transmission=='Mannual'):
            Transmission=1
        else:
            Transmission=0
        prediction=modelresale.predict([[Present_Price,Kms_Driven2,Owner,Year,Fuel_Type,Seller_Type,Transmission]])
        output=round(prediction[0],2)
        if output < 0:
            return render_template('resalepredict.html',prediction_texts="Sorry you cannot sell this car !")
        else:
            return render_template('resalepredict.html',prediction_text="You Can Sell The Car at Rs. {} Lakhs".format(output))


if __name__=='__main__':
    app.run(debug=True)
