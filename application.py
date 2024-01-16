from flask import Flask, render_template,request
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

app = Flask(__name__)
model =pickle.load(open('XGBRegressionModel.pkl','rb'))
diamond =pd.read_csv('cleaned.csv')
@app.route('/')
def index():
    cut =sorted(diamond['cut'].unique())
    color =sorted(diamond['color'].unique())
    clarity =sorted(diamond['clarity'].unique())
    
    return  render_template('index.html', cuts=cut, colors=color, clarities=clarity)
@app.route('/predict',methods=['POST'])
def predict():
    cut_encoder = LabelEncoder()
    clarity_encoder = LabelEncoder()
    color_encoder = LabelEncoder()
    # Dummy data for illustration purposes 
    cut_data = ["Ideal" ,"Premium", "Good", "Very Good", "Fair"]
    clarity_data = ["SI2", "SI1", "VS1", "VS2", "VVS2", "VVS1","I1","IF"]
    color_data = ["E", "I", "J", "H", "F", "G", "D"]

# Fit the LabelEncoders with training data
    cut_encoder.fit(cut_data)
    clarity_encoder.fit(clarity_data)
    color_encoder.fit(color_data)
    #fetching the data
    carat=float(request.form.get('carat'))
    cut=request.form.get('cut')
    clarity=request.form.get('clarity')
    color=request.form.get('color')
    table=float(request.form.get('table'))
    depth=float(request.form.get('depth'))
    x=float(request.form.get('x'))
    y=float(request.form.get('y'))
    z=float(request.form.get('z'))
    # Perform label encoding
    cut_encoded = cut_encoder.transform([cut])[0]
    clarity_encoded = clarity_encoder.transform([clarity])[0]
    color_encoded = color_encoder.transform([color])[0]
    prediction=model.predict(pd.DataFrame([[carat,cut_encoded,color_encoded,clarity_encoded,depth,table,x,y,z]],columns=['carat','cut','color','clarity','depth','table','x','y','z']))
    
   
    return str(prediction[0])
if __name__ == '__main__':
    app.run(debug=True)