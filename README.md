# Machine Learning Pipeline Summative

Links:

Github Repo:  https://github.com/Mikekimm/Machine-Learning-Pipeline-Summative.git

Video Demo Link: 

Live App (Render): https://machine-learning-pipeline-summative.onrender.com

API Docs:  https://machine-learning-pipeline-summative.onrender.com/docs

Health Check: https://machine-learning-pipeline-summative.onrender.com/health

Metrics Endpoint:  https://machine-learning-pipeline-summative.onrender.com/metrics


Project Summary
This project adopts an image classification end-to-end pipeline based on images of scenes. It consists of offline training, evaluation, retraining trigger, API endpoints, a UI dashboard, Docker configuration, and flood request simulation with Locust.

## Directory Structure

Project_name/
- README.md
- notebook/
  - project_name.ipynb
- src/
  - preprocessing.py
  - model.py
  - prediction.py
  - retrain.py
  - api.py
  - dashboard.py
- data/
  - train/
  - test/
  - uploads/
- models/
  - scene_classifier.pkl
- results/
  - metrics.json
  - feature_story.json
- locust/
  - locustfile.py
- Dockerfile
- docker-compose.yml
- requirements.txt

## Setup Steps
1. Create and activate a Python environment.

2. Install dependencies:

   pip install -r requirements.txt

3. Train the high-accuracy EfficientNet model (Python 3.12 environment):

   python3.12 -m venv .venv312
   source .venv312/bin/activate
   pip install -r requirements.txt
   python src/model.py

4. Start the API (default environment is fine):

   uvicorn src.api:app --host 0.0.0.0 --port 8000

5. Start the UI dashboard:

   streamlit run src/dashboard.py

Note:
- If TensorFlow is not available in the active environment, prediction and retraining automatically fall back to the `.venv312` runner when it exists.

Evaluation Results 

The documentation of evaluation of the model is found in projectname.ipynb and metrics.json.
Main test metrics:
- Accuracy: 0.9123
- Precision (weighted): 0.9141
- Recall (weighted): 0.9123
- F1-score (weighted): 0.9118


## Visualizations and Feature Story
The dashboard reports and visualizes at least 3 interpretable features:
- brightness
- blue_ratio
- texture_strength

Story example:
- sea and glacier typically show stronger blue signal,
- forest tends to show stronger green signal,
- street and mountain tend to have stronger texture patterns.

## Prediction and Retraining Flow

- User uploads an image and gets a class prediction.
- User uploads bulk images for future retraining.
- Uploaded files are stored in the data uploads area.
- User presses retrain trigger to rebuild the model.
- New model and metrics are regenerated and served by the API.


## Flood Request Simulation (Locust)
1. Start API.
2. Run Locust:

   locust -f locust/locustfile.py --host http://127.0.0.1:8000

3. Open http://localhost:8089 and run tests with different user loads.
4. Record latency and response time for each run.

### Locust Evidence (Scaling Comparison)
Load tests were executed in headless mode with three scaling levels.
For this local benchmark, scaling levels were represented by API worker count (1, 2, 3) while holding the same user profile.

| Scale Level | Runtime | Users | Avg Response (ms) | P95 (ms) | RPS | Failure Rate |
|---|---:|---:|---:|---:|---:|---:|
| 1 worker | 25s | 40 | 12 | 50 | 25.40 | 0.00% |
| 2 workers | 25s | 40 | 11 | 30 | 25.80 | 0.00% |
| 3 workers | 25s | 40 | 10 | 29 | 25.61 | 0.00% |

Results Summary 
- Increasing workers improved tail latency.
- Throughput remained stable for this test profile.
- Failure rate stayed at 0% after concurrency fixes.


## Docker and Scaling
Build and run with Docker Compose:

   docker compose up --build

To test different container counts:

   docker compose up --build --scale api=1
   docker compose up --build --scale api=2
   docker compose up --build --scale api=3

Run Locust against each scale and compare latency and throughput.

## Notebook Contents
The notebook includes:
- preprocessing steps
- model training
- test and metrics reading
- prediction function demo
- retraining trigger notes

Notebook path: notebook/project_name.ipynb

## Model File
Model artifact is saved to:
- models/scene_classifier.keras
- models/scene_classifier_meta.json

Backward-compatible baseline artifact:
- models/scene_classifier.pkl

## Deployment Guidance
You can deploy on Render, Railway, Azure App Service, or AWS ECS.
- Deploy API container
- Configure persistent volume for models and results
- Set health check to /health
- Expose /predict and /retrain endpoints

## Deployment Proof
- Public URL: [https://machine-learning-pipeline-summative.onrender.com](https://machine-learning-pipeline-summative.onrender.com)
- Health check URL: [https://machine-learning-pipeline-summative.onrender.com/health](https://machine-learning-pipeline-summative.onrender.com/health)
- Metrics URL: [https://machine-learning-pipeline-summative.onrender.com/metrics](https://machine-learning-pipeline-summative.onrender.com/metrics)

