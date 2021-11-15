# Bank Default Prediction

## 1. Problem description
The dataset contains the information regarding bank customers, including their aggregated history of payments, and for each the goal is to predict the probability of default for the data points. The goal is to output a probability (from 0 to 1) of the default occurring for each client. 

## 2. Solution specifications
The solution provided consists of notebooks, later converted into scripts, which are to be executed in order of enumeration. Moreover, the model has been deployed on AWS using a script `server.py` used for the model deployment (below a guide is available to show how to deploy the model on AWS using Flask and Docker). 

The organization of the code is as follows:
  - **/code:** contains the scripts able to preprocess, train, validate and deploy the model 
  - **/notebooks:** contains the notebooks source of the code before being translated into scripts. Moreover, the notebooks contain the data exploration.
  - **/output:** contains the csv file with the test prediction
  - **/deploy:** folder containing the files needed for the model deployment.
Additional files or sections serve as configuration, of for the model deployment stage.

### 2.1 Dataset Exploration and Preprocessing
In this section, I explored the fields of the dataset. I noticed the level of dataset imbalance, typical of an anomaly detection scenario. I distinguished the categorical columns from the numerical and studied their distribution, noticing most of the time a skewness of data. Among the categorical fields, I studied the Merchant Category more in detail, and noticed a large number of distinct categories. Many of the column names presenting missing values seem to be referring to historical values of the user, which might not be available in case the users are newer than the period in scope. Thus, missing values were imputed using an unseen constant-value imputation (-1), as replacing e.g. 0 would result incorrect. Finally, the analysis of simple statistics for each column confirmed the skewness previously observed.

### 2.2 Model Training and Hyperparameter Tuning
The model chosen for the scenario is CatBoost. I took into account that, being a Gradient Boosting tree model, the data do not need to be normalized. Also, Catboost is able to handle categorical variables, we will let it learn the best representation to the categories via cross validation.
For the choice of the Metric, we need to take into account this problem being an anomaly detection problem. As first metric we use AUC, since we are interested on the probability of default as well. As second metric we use Recall (number of correctly classified defaults divided by the real defaults), hence to capture as many real defaults as possible. This means, that a very small improvements of AUC (less than 0.01% inferior to the best AUC), we will prioritize Recall.

### 2.3 Results
The best model so far obtained from the hyperparameter tuning procedure achieves a 5-fold cross validation score of **0.911 AUC** and **0.836 Recall**.

---

## 3. Instructions for Model Deployment

> The deployment has been done using Flask and Docker and using AWS as cloud provider, since it was indicated as preferred solution. I performed the deployment using guides and other online resources.

### 3.1 Hosting the docker container on an AWS ec2 instance

**Note**: We assume that key pairs are already created and that an instance is already running and available on AWS.


1. Initialize docker on AWS instance

**Note**: This step needs to be done only once

```bash
sudo amazon-linux-extras install docker
sudo yum install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
```

2. Move zipped archive within ec2 instance and unzip it
```bash
scp -i /path/my-key-pair.pem code.zip ec2-user@public-dns-name:/home/ec2-user
```

Then log again into the instance and run the following command: 

```bash
aws ec2 unzip code.zip
```


3. Build the docker image and run the server

```bash
docker build -t code .
docker run -p 80:80 code .
```

4. Consume the endpoint

The input should be given as a list, or numpy array

### 3.2 Example using bash 

```bash
curl -X POST \
http://0.0.0.0:80/predict \
-H 'Content-Type: application/json' \
-d '[0, 0, 0, 1, ...]'
```

### 3.3 Example using python request package

```python
# All features ordered as in the dataset, excluded uuid and default fields
features = [0, 0.0, 0.0, 0.0, 0.00913461538461539, '1.0', '1.0', '-1.0', '1.0',
       '1.0', 20, 6.4, 5.25, 'Youthful Shoes & Clothing',
       'Clothing & Shoes', 1, 7225.0, 7225.0, 'F', 0.0, 0, 0, 0, 5, 0,
       0, 0.0, 0.0, 1, '1', '1', '1', '1', '1', '1', 0, 8815, 0, 27157,
       19.8955555555556, '-1.0']
response = requests.post("http://0.0.0.0:80/predict", json=features)
response_json = response.json()
```
