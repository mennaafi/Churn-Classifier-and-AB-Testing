<div align="center">  

## Churn Classifier and AB Testing

 </div>

 Build a robust customer churn prediction pipeline on the Telco dataset, embed an A/B test that helps customer retention, and finally automate the full training→deployment workflow with ZenML.


## Installation

Follow these steps to set up and run the project locally :

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/mennaafi/Churn-Classifier-and-AB-Testing.git
cd churn-classifier-and-AB-testing
```
### **2️⃣ Create a Virtual Environment**  
```bash
python -m venv venv
```
### **Activate the Virtual Environment :**  

#### **Windows:**  
```bash
venv\Scripts\activate
```
#### **Mac/Linux:**  
```bash
source venv/bin/activate
```
### 3️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

## ZenML Setup
-  Register the MLflow model deployer :

```bash 
zenml model-deployer register mlflow --flavor=mlflow
```

 - Update your active stack to include MLflow deployer :
 ```bash
zenml stack update -d mlflow
```

## Usage
**1. Train & Deploy**
```bash
python run_deployment.py deploy
```


This will :

- Ingest & preprocess the data

- Train & log the model in MLflow

- Deploy the model via ZenML’s MLflow deployer

<br><br>


**2. Inspect Experiments & Models**
```bash
mlflow ui --backend-store-uri "file://$(pwd)/mlruns"
```
Open your browser at **http://127.0.0.1:5000** to view runs, metrics, artifacts, and registered models.

**3. Batch Inference**
```bash 
python run_deployment.py inference
```
Runs the inference_pipeline, which fetches the deployed model and scores a test batch.

**4. Ad-Hoc Single Prediction**
 - Edit feature values in sample_predict.py under the record = { … } block.
 - Run :
 ```bash 
 python sample_predict.py
```
- It will print out a churn prediction (0 or 1).


