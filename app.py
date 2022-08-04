from flask import Flask,request,render_template
import pandas as pd
import json

app = Flask(__name__)
app.secret_key = "1/1201085579334815:f^=8*92s49es7n@taynn^zrbh63brmvf3p(%q8#b)p&s4ycpc$"

def getpredictions(text,unique_id):
    import ML_MODEL.pred_class as pc
    df = pd.read_excel('ML_MODEL/Test_data.xlsx', index_col=0)
    pred = pc.prediction()
    data = pred.get_predictions(df,unique_id,text).to_json()
    return json.loads(data)


@app.route("/",methods=["GET"])
def home():
    return render_template("info.html")
    

@app.route("/api/getpredictions",methods=["POST"])
def predict():
    try:
        if request.method == "POST":
            if request.headers.get("Authorization") == "Token "+app.secret_key: 
                # call prediction function
                input_data = request.get_json()
                if input_data:
                    if input_data.get("text") and input_data.get("unique_id"):
                        res = getpredictions(input_data.get("text"),input_data.get("unique_id"))
                        return res,200
                    elif input_data.get("text"):
                        return "Missing or empty required parameter unique_id.",403
                    elif input_data.get("unique_id"):   
                        return "Missing or empty required parameter text.",403 
                else:
                    return "Missing required parameter text and unique_id.",403
            else:
                return "Authentication error. Please pass vaild Token!", 401
        else:
            return "Refused alert Received & Refused!", 400
    except Exception as e:
        return f"[X] ERROR {e}"


if __name__ == "__main__":
 
    app.run()
    # app.run(debug=True,host="0.0.0.0")