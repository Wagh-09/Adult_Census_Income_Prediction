from flask import Flask,render_template,request,jsonify
from src.pipeline.prediction_pipeline import PredictPipeline,CustomData
import numpy as np



application = Flask(__name__)
app = application

@app.route("/",methods = ["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")

    else:
        data = CustomData(
            age = int(request.form.get("age")),
            workclass = int(request.form.get("workclass")),
            education_num = int(request.form.get("education_num")),
            marital_status = int(request.form.get("marital_status")),
            occupation = int(request.form.get("occupation")),
            relationship = int(request.form.get("relationship")),
            race = int(request.form.get("race")),
            sex = int(request.form.get("sex")),
            capital_gain = int(request.form.get("capital_gain")),
            capital_loss = int(request.form.get("capital_loss")),
            hours_per_week = int(request.form.get("hours_per_week")),
            native_country = int(request.form.get("native_country")),
            )

        final_data = data.get_data_as_data_frame()
        predict_pipline = PredictPipeline()
        pred = predict_pipline.predict(final_data=final_data)


        if pred == 0:
            pred= "Your Income is Less Then Equal To 50K"

        elif pred == 1:
            pred= "Your Income is More Then 50K"
            
        return render_template("form.html",final_result =pred)

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)