# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from fastapi import FastAPI
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware



class PredictRequest(BaseModel):
    data: list  # list of feature vectors or a single feature vector

app = FastAPI(title='Iris RF Demo')

# For demo, load a model saved earlier if present; otherwise create a dummy model
try:
    model = joblib.load('artifacts/rf_iris.joblib')
except Exception as e:
    # fallback: a simple scikit-learn model trained quickly
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
# Allow CORS
origins = [
    "http://localhost:5500",  # your frontend URL
    "http://127.0.0.1:5500",  # sometimes needed
    "http://localhost:8000",  # optional
    "*"  # allow all origins (not recommended for production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],        # GET, POST, etc.
    allow_headers=["*"],        # allow all headers
)

@app.post('/predict')
def predict(req: PredictRequest):
    data = np.array(req.data)
    # ensure 2D
    if data.ndim == 1:
        data = data.reshape(1, -1)
    preds = model.predict(data).tolist()
    return {'predictions': preds}

# <!-- 
# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <title>Iris Flower Predictor</title>
#     <style>
#         body {
#             font-family: Arial, sans-serif;
#             margin: 50px;
#             background-color: #f5f5f5;
#         }
#         h1 {
#             color: #333;
#         }
#         input {
#             margin: 5px;
#             padding: 5px;
#             width: 60px;
#         }
#         button {
#             margin: 10px 0;
#             padding: 8px 15px;
#             background-color: #4CAF50;
#             color: white;
#             border: none;
#             cursor: pointer;
#             border-radius: 4px;
#         }
#         button:hover {
#             background-color: #45a049;
#         }
#         #result {
#             margin-top: 20px;
#             font-weight: bold;
#             color: #333;
#         }
#         label {
#             display: inline-block;
#             width: 120px;
#         }
#     </style>
# </head>
# <body>
#     <h1>Iris Flower Predictor</h1>
#     <p>Enter the features of the iris flower:</p>

#     <label>Sepal Length:</label>
#     <input type="number" step="0.1" id="sepalLength"><br>

#     <label>Sepal Width:</label>
#     <input type="number" step="0.1" id="sepalWidth"><br>

#     <label>Petal Length:</label>
#     <input type="number" step="0.1" id="petalLength"><br>

#     <label>Petal Width:</label>
#     <input type="number" step="0.1" id="petalWidth"><br>

#     <button onclick="predictIris()">Predict</button>

#     <div id="result"></div>

#     <script>
#         async function predictIris() {
#             // Read input values
#             const sepalLength = parseFloat(document.getElementById('sepalLength').value);
#             const sepalWidth = parseFloat(document.getElementById('sepalWidth').value);
#             const petalLength = parseFloat(document.getElementById('petalLength').value);
#             const petalWidth = parseFloat(document.getElementById('petalWidth').value);

#             const sample = [sepalLength, sepalWidth, petalLength, petalWidth];

#             // Make POST request to FastAPI endpoint
#             try {
#                 const response = await fetch('/predict', {
#                     method: 'POST',
#                     headers: { 'Content-Type': 'application/json' },
#                     body: JSON.stringify({ data: sample })
#                 });

#                 if (!response.ok) {
#                     throw new Error(`HTTP error! Status: ${response.status}`);
#                 }

#                 const result = await response.json();
#                 document.getElementById('result').innerText = `Prediction: ${result.predictions}`;
#             } catch (error) {
#                 document.getElementById('result').innerText = `Error: ${error}`;
#             }
#         }
#     </script>
# </body>
# </html> -->
