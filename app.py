import os
import io
import pandas as pd
import numpy as np
import pickle
import sklearn.preprocessing
from sklearn.metrics import r2_score

# import keras
# from keras.preprocessing import image
# from keras.preprocessing.image import img_to_array
# from keras.applications.xception import (
#     Xception, preprocess_input, decode_predictions)
# from keras import backend as K


from flask import Flask, request, redirect, url_for, jsonify
#from flask import send_file

#app = Flask(__name__, static_url_path='')
app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'Uploads'

model = None
graph = None


def load_model():
    global model
    global graph
    #model = Xception(weights="imagenet")
    #graph = K.get_session().graph
    filename="lstm_modela.h5"
    filename="rnn_modela.h5"
    model = pickle.load(open(filename, 'rb'))
    model._make_predict_function()


load_model()


def prepare_image(img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # ##return the processed image
    return img

# normalize the data and shape
def normalize_data(df):
    scaler = sklearn.preprocessing.MinMaxScaler()
    df['CAISO_MW']=scaler.fit_transform(df['CAISO_MW'].values.reshape(-1,1))
    return df

def load_datanew(stock, seq_len):
    X_train = []
    y_train = []
    for i in range(seq_len, len(stock)):
        X_train.append(stock.iloc[i-seq_len : i, 0])
        y_train.append(stock.iloc[i, 0])
    
    #1 last after 2 data as test
    X_test = X_train[2:]             
    y_test = y_train[2:]
    
    #2 ignore first two 
    X_train = X_train[:2]           
    y_train = y_train[:2]
    
    #3 convert to numpy array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    #4 reshape for input into models
    X_train = np.reshape(X_train, (2, seq_len, 1))
    
    X_test = np.reshape(X_test, (X_test.shape[0], seq_len, 1))
    
    return [X_train, y_train, X_test, y_test]


def generateHeader():
    reStr = "<html> <title>Upload new File</title><h1>Result Page</h1><h1>Next 60 days prediction </h1>"
    return reStr

def generateTail():
	
	return ""

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global model
    data = {"success": False}
    if request.method == 'POST':
        if request.files.get('file'):
            # ###read the file
            file = request.files['file']
            # ###read the filename
            filename = file.filename
            # ###create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)
            dfnew = pd.read_csv(filepath, index_col='Datetime', parse_dates=['Datetime'])
            normalize_data(dfnew).shape
            seq_len = 20
            X_train_new, y_train_new, X_test_new, y_test_new = load_datanew(dfnew, seq_len)
            lstm_predictions = model.predict(X_test_new)
            lstm_score_new = r2_score(y_test_new, lstm_predictions)
            print("R^2 Score of LSTM model new = ",lstm_score_new)

            return generateHeader() + "<h1>"+"R^2 Score of LSTM model new = "+str(lstm_score_new) +"</h1>"+ generateTail()

        #return jsonify(data)

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <!-- <h1>Upload new file</h1> -->
   
    <h1>Machine Learning on Electricity Demand</h1>
    <p> Choose "pred_file" from the folder </p>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload> 
         
    </form>


    <div class="container">
      <section class="row">
        <div class="col-md-8">
          <article class="description-content">
            <h1 class="description-header">Summary</h1>
            <hr class="description-hr"/>
           <!-- <img src="LSTM.png" alt="" id="description-image"/> -->
           <!--<img src="{{url_for('static', filename='LSTM.png')}}" />  -->
                        
            <p>For this project we are trying to predict the electricity demand curve for the following month using machine learning. We pulled data from California ISO Open Access Same Time Information System to assemble a dataset for period of January 2016 to July 2019 for training and testing.</p>
            <p>After assembling the dataset, we trained and test the data based on two models: Recurrent Neural Network and Long Short Term Memory. We plot the results  and show the model prediction's R2 score.</p>
          </article>
        </div>
        <div class="col-md-4">


          <!-- Start of Visualizations imageNav Area -->
          <section id="imageNav-area">
            <div class="imageNav-content">
              <h2 class="imageNav-header">Graphs</h2>
              <hr />
              <div id="images">
              
              <img src="static/Demand_bef.png" alt="the demand curve" />
              <hr />
              <img src="static/Demand_aft.png" alt="the demand normalized" />
              <hr />
              <img src="static/simpleRNN.png" alt="the RNN" />
              <hr />
              <img src="static/LSTM.png" alt="the LSTM" />
              <hr />
              <img src="static/Pred_v_Actual.png" alt="the model trained" />
              <hr />
             
              
              </div>
            </div>
          </section>
 


    '''


if __name__ == "__main__":
    app.run(debug=True)
	