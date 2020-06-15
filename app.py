import argparse
from flask import Flask, jsonify, request
from flask import render_template, send_from_directory
import os
import re
import json
import numpy as np
import plotly
import plotly.graph_objects as go
import pickle
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from performance import mape

## import model specific functions and variables
from model import model_train,  model_predict
with open('__version__','r+') as f:
    MODEL_VERSION = f.read()
    f.close

app = Flask(__name__)

@app.route("/")
def landing():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    """
    basic predict function for the API
    """

    ## input checking
    if not request.json:
        print("ERROR: API (predict): did not receive request data")
        return jsonify([])

    if 'query' not in request.json:
        print("ERROR API (predict): received request, but no 'query' found within")
        return jsonify([])

    if 'type' not in request.json:
        print("WARNING API (predict): received request, but no 'type' was found assuming 'numpy'")
        query_type = 'numpy'

    ## set the test flag
    test = False
    if 'mode' in request.json and request.json['mode'] == 'test':
        test = True

    ## extract the query
    query = request.json['query']

    if request.json['type'] == 'dict':
        pass
    else:
        print("ERROR API (predict): only dict data types have been implemented")
        return jsonify([])

    ## make prediction

    _result = model_predict(*query)

    result = {}

    # convert numpy objects to ensure they are serializable
    for key,item in _result.items():
        if isinstance(item,np.ndarray):
            result[key] = item.tolist()
        else:
            result[key] = item

    return(jsonify(result["y_pred"]))

@app.route('/train', methods=['GET','POST'])
def train():
    """
    basic predict function for the API

    the 'mode' flag provides the ability to toggle between a test version and a
    production verion of training
    """

    ## check for request data
    if not request.json:
        print("ERROR: API (train): did not receive request data")
        return jsonify(False)

    ## set the test flag
    test = False
    if 'mode' in request.json and request.json['mode'] == 'test':
        test = True
    query = request.json['query']
    print("... training model")
    model_train(data_dir=query,test=test)
    print("... training complete")

    return(jsonify(True))

@app.route('/logs/<filename>',methods=['GET'])
def logs(filename):
    """
    API endpoint to get logs
    """

    if not re.search(".log",filename):
        print("ERROR: API (log): file requested was not a log file: {}".format(filename))
        return jsonify([])

    log_dir = os.path.join(".","logs")
    if not os.path.isdir(log_dir):
        print("ERROR: API (log): cannot find log dir")
        return jsonify([])

    file_path = os.path.join(log_dir,filename)
    if not os.path.exists(file_path):
        print("ERROR: API (log): file requested could not be found: {}".format(filename))
        return jsonify([])

    return send_from_directory(log_dir, filename, as_attachment=True)

@app.route('/predictgui',methods=['POST'])
def predictgui():
    country,date_ = [x for x in request.form.values()]
    inputs = date_.split('-')
    inputs.insert(0,country)


    output = model_predict(*inputs,from_pickle=True)
    prediction_text = f"Revenue for the next 30 days in {country.replace('_',' ').title()} should be {round(output['y_pred'][0])}"

        # prediction_text = f"This doesn't look right! Try again!"

    return render_template('index.html', prediction_text=prediction_text)

@app.route('/dashboard',methods=['GET','POST'])
def performance():
    # Load all models to prepare for the plot
    inputs = [x for x in request.form.values()]
    version_ = re.sub("\.","_",str(MODEL_VERSION))
    all_data, all_models = pickle.load(open(os.path.join("models",f"all_data_model-{version_}.pickle"), "rb" ))
    country = inputs[0]
    y_true = all_data[country]['y']
    y_pred = all_models[country].predict(all_data[country]['X'])
    all_dates = all_data[country]['dates']
    rmse_ = round(mse(y_true,y_pred,squared=False),2)
    mae_ = round(mae(y_true,y_pred),2)
    mape_ = round(mape(y_true,y_pred),2)
    title = f"{country.replace('_',' ').title()}: RMSE:{rmse_}, MAE:{mae_}, MAPE:{mape_}%"

    trace0 = go.Scatter(x=all_dates, y=y_true, name='Actual Revenue')
    trace1 = go.Scatter(x=all_dates, y=y_pred, name='Predicted Revenue')
    data = [trace0, trace1]
    layout = go.Layout(title=title,yaxis_title="Revenue")
    fig = dict(data=data,layout=layout)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('dashboard.html',
                       graphJSON=graphJSON)

if __name__ == '__main__':

    ## parse arguments for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = vars(ap.parse_args())

    if args["debug"]:
        app.run(debug=True, port=8080)
    else:
        app.run(host='0.0.0.0', threaded=True ,port=8080)
        # apperently the host has to be 0.0.0.0 so docker container can run properly ¯\_(ツ)_/¯
