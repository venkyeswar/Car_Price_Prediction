from flask import Flask,render_template,request,redirect
import pandas as pd
import numpy as np
import os
import joblib
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    
    if request.method == "POST":
        year = request.form.get("year")
        present_price = request.form.get("present_price")
        kms_driven = request.form.get("Kms_Driven")
        fuel_type = request.form.get("fuel_type")
        seller_type = request.form.get("seller")

        fuel= "Petrol" if fuel_type == "1" else "Diesel"
        seller = "Individual" if seller_type == "1" else "Dealer"

        year1 =int(year)
        present_price1 = float(present_price)
        kms_driven1 = float(kms_driven)
        fuel_type1 = int(fuel_type)
        seller_type1 = int(seller_type)


        data = {
             "Year":[year1],
             "Present_Price":[present_price1],
             "Kms_Driven":[kms_driven1],
            "Fuel_Type_Petrol":[fuel_type1],
            "Seller_Type_Individual":[seller_type1]
        }
        data=pd.DataFrame(data)
        scaled_df = scaler.transform(data)
        input_data = pd.DataFrame(scaled_df,columns=data.columns)

        prediction = model.predict(input_data)[0]
        print(prediction)
        prediction =np.round(prediction,2)
        
        return render_template("prediction.html",
                               year=year,
                               present_price=present_price,
                               kms_driven=kms_driven,
                               fuel_type=fuel,
                               seller_type=seller,
                               prediction=prediction)
        


#     return Predicted_Price

if __name__ == "__main__":
     app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    
