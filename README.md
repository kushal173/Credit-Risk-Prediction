# CreditLens — Credit Risk Prediction

A production-styled Streamlit application that serves a trained Home Credit default-risk model. The app preserves the preprocessing objects and decision threshold stored inside `credit1_risk_model.pkl`.

## Project files

- `app.py` — Streamlit UI and inference pipeline
- `credit1_risk_model.pkl` — trained model and preprocessing artifacts (add this file locally)
- `requirements.txt` — Python dependencies
- `Dockerfile` — reproducible application image
- `docker-compose.yml` — local or EC2 container runner

## Run locally

Place `credit1_risk_model.pkl` in this directory, then:

```bash
python -m venv venv

# Windows Command Prompt
venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

## Test with Docker

```bash
docker compose up --build -d
docker compose ps
```

Open `http://localhost:8501`. Check logs with:

```bash
docker compose logs -f
```

## Deploy to AWS EC2

### 1. Prepare the repository

Commit this project to a private GitHub repository. Keep the model file private if its licence or provenance requires that. If the repository is public, confirm that the dataset and trained artifact may be redistributed.

### 2. Create the EC2 instance

In AWS Console:

1. Open **EC2 → Launch instance**.
2. Choose **Ubuntu Server 24.04 LTS**.
3. Choose `t3.small` as a practical starting point. `t3.micro` may run out of memory while building packages or loading the model.
4. Create/download a key pair.
5. In the security group allow:
   - SSH, TCP `22`, source **My IP**
   - Custom TCP, port `8501`, source `0.0.0.0/0` for the initial demo
6. Use at least 16 GiB gp3 storage.

### 3. Connect and install Docker

```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 git
sudo usermod -aG docker ubuntu
exit
```

Reconnect so the Docker group change takes effect.

### 4. Clone and launch

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_DIRECTORY
docker compose up --build -d
docker compose ps
```

Visit `http://YOUR_EC2_PUBLIC_IP:8501`.

### 5. Verify and operate

```bash
docker inspect --format='{{json .State.Health}}' credit-risk-app
docker compose logs --tail=100
docker compose restart
```

To release an update:

```bash
git pull
docker compose up --build -d
docker image prune -f
```

## Production hardening

For a resume demonstration, the EC2 setup above is enough. Before handling real applicant data:

- Put Nginx or an Application Load Balancer in front of Streamlit.
- Configure HTTPS with an AWS Certificate Manager certificate and a domain.
- Restrict inbound port `8501` after the HTTPS proxy is working.
- Add authentication and authorization.
- Do not log sensitive applicant inputs.
- Encrypt storage, enable backups, patch the instance, and apply least-privilege IAM.
- Monitor application health and resource use with CloudWatch.
- Document fairness, drift, explainability, and human-review controls.

## Important compatibility note

The supplied artifact was serialized with scikit-learn `1.6.1` and XGBoost `3.4.1`. These versions are pinned in `requirements.txt` to prevent model-loading incompatibilities.
