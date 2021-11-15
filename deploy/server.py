import pickle
import numpy as np
from flask import Flask, request

model = None
app = Flask(__name__)



def load_model():
    global model
    # model variable refers to the global variable
    with open('final_model.pkl', 'rb') as f:
        model = pickle.load(f)


@app.route('/')
def example_endpoint():
    return 'Endpoint is working!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':

        data = request.get_json()  # Get data posted as a json
        data = np.array(data)[np.newaxis, :]  # converts shapes
        prediction = model.predict_proba(data)

    return str(prediction[0][1])


if __name__ == '__main__':

    load_model()  # load model at the beginning once only

    print('Server launched at port 80')

    app.run(host='0.0.0.0', port=80)
