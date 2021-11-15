import requests

if __name__ == '__main__':
    # Get features from first test sample

    features = [0, 0.0, 0.0, 0.0, 0.00913461538461539, '1.0', '1.0', '-1.0', '1.0',
           '1.0', 20, 6.4, 5.25, 'Youthful Shoes & Clothing',
           'Clothing & Shoes', 1, 7225.0, 7225.0, 'F', 0.0, 0, 0, 0, 5, 0,
           0, 0.0, 0.0, 1, '1', '1', '1', '1', '1', '1', 0, 8815, 0, 27157,
           19.8955555555556, '-1.0']

    response = requests.post("http://0.0.0.0:80/predict", json=features)

    response_json = response.json()

    print('Prediction probability of default for the sample is', round(response_json, 4))