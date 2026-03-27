# Machine Learning Pipeline Summative

Links:

Github Repo:

Video Demo Link:




## Project Description
This project implements an end-to-end image classification pipeline using scene images.
It includes offline training, evaluation, retraining trigger, API endpoints, a UI dashboard,
Docker setup, and flood request simulation with Locust.

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

3. Train the model:

   python src/model.py

4. Start the API:

   uvicorn src.api:app --host 0.0.0.0 --port 8000

5. Start the UI dashboard:

   streamlit run src/dashboard.py

## API Endpoints
- GET /health: model uptime and status
- GET /metrics: training metrics
- POST /predict: predict one uploaded image
- POST /upload-bulk: upload multiple images for retraining data
- POST /retrain: trigger model retraining

## Retraining Pipeline Detail
1. Upload data via `POST /upload-bulk` or dashboard bulk upload.
2. Uploaded files are saved in `data/uploads/`.
3. Retraining trigger runs via `POST /retrain` and executes `src/retrain.py`.
4. Preprocessing runs on training data through `src/preprocessing.py`.
5. Retraining uses the same baseline workflow and regenerates:
   - `models/scene_classifier.pkl`
   - `results/metrics.json`
   - `results/feature_story.json`

## Visualizations and Feature Story
The dashboard reports and visualizes at least 3 interpretable features:
- brightness
- blue_ratio
- texture_strength

Story example:
- sea and glacier typically show stronger blue signal,
- forest tends to show stronger green signal,
- street and mountain tend to have stronger texture patterns.

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

Interpretation:
- 2 and 3 workers improved p95 latency compared with 1 worker.
- Throughput stayed in a similar range because client-side load profile was fixed.
- Failure rate stayed at 0% after fixing concurrent prediction file naming.

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

