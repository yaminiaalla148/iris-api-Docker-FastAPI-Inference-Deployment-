# iris-api-Docker-FastAPI-Inference-Deployment


A machine learning API for classifying Iris flower species using FastAPI and scikit-learn. The model is trained on the classic Iris dataset and deployed as a containerized application.

##  Features

- **FastAPI Framework**: Modern, fast web framework for building APIs
- **Machine Learning Model**: Random Forest classifier trained on Iris dataset
- **Docker Containerization**: Easy deployment and scaling
- **Automatic Documentation**: Interactive API docs with Swagger UI
- **Health Monitoring**: Endpoint for service health checks
- **Confidence Scores**: Returns prediction confidence along with classification

##  API Endpoints

### Health Check
- **GET** `/health`
- Returns service status and model information

### Iris Classification
- **POST** `/predict`
- Accepts iris measurements and returns predicted species

#### Request Body
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

#### Response
```json
{
  "species": "setosa",
  "species_id": 0,
  "confidence": 0.98
}
```

##  Local Development

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone 
cd iris-api-Docker-FastAPI-Inference-Deployment
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model (optional - model is already trained):
```bash
python train_model.py
```

4. Run the API:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### Using Docker

1. Build the image:
```bash
docker build -t iris-api .
```

2. Run the container:
```bash
docker run -p 8080:8080 iris-api
```

##  Model Details

- **Algorithm**: Random Forest Classifier
- **Training Data**: Iris dataset (150 samples, 3 classes)
- **Features**: Sepal length, sepal width, petal length, petal width
- **Classes**: Setosa (0), Versicolor (1), Virginica (2)
- **Accuracy**: ~97% on test set

##  Deployment

The application is containerized using Docker and can be deployed to any cloud platform supporting containers (Render, Heroku, AWS, etc.).

### Environment Variables
- No environment variables required for basic deployment

### Port Configuration
- Default port: 8080 (configurable via Docker)

##  Usage Examples

### Python
```python
import requests

url = "https://iris-api-9swy.onrender.com/predict"
data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

response = requests.post(url, json=data)
print(response.json())
```

### cURL
```bash
curl -X POST "https://iris-api-9swy.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "sepal_length": 5.1,
       "sepal_width": 3.5,
       "petal_length": 1.4,
       "petal_width": 0.2
     }'
```

##  Project Structure

```
iris-deploy/
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── train_model.py         # Model training script
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI application
│   └── iris_model.pkl     # Trained model (generated)
└── README.md              # This file
```

