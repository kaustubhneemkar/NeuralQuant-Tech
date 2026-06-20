# 🚀 Deploying Alpha-Lens to Google Cloud Run

This guide outlines how to deploy the Alpha-Lens quant agent onto **Google Cloud Platform (GCP)** using **Cloud Run**. This containerized setup leverages serverless hosting, scales down to zero when not in use to save cost, and securely manages API keys using Google Secret Manager.

---

## 📋 Prerequisites
1. A **GCP Project** with billing enabled.
2. The **Google Cloud SDK (gcloud CLI)** installed locally.
3. Your **Gemini API Key** from Google AI Studio.

---

## 🛠️ Step-by-Step Deployment

### Step 1: Configure GCP Environment
Initialize your `gcloud` configuration and select your active project:
```bash
gcloud init
gcloud config set project YOUR_PROJECT_ID
```
Define your configuration variables:
```bash
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
SERVICE_NAME="alpha-lens"
```

### Step 2: Enable Required APIs
Enable the services for container registry, deployment, and secrets:
```bash
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com \
    cloudbuild.googleapis.com
```

### Step 3: Create Google Secret for API Key
Store your `GOOGLE_API_KEY` securely in Google Secret Manager:
```bash
# Create the secret holder
gcloud secrets create GOOGLE_API_KEY --replication-policy="automatic"

# Add the secret value (replace AIza... with your actual API key)
echo -n "AIza...your_key_here..." | gcloud secrets versions add GOOGLE_API_KEY --data-file=-
```

### Step 4: Create Artifact Registry
Create a repository in Google Artifact Registry to house your Docker container images:
```bash
gcloud artifacts repositories create alpha-lens-repo \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for Alpha-Lens Quant Agent"
```

### Step 5: Build and Push Container using Cloud Build
Submit a build request to Google Cloud Build, which compiles your Dockerfile in the cloud and pushes it to your registry:
```bash
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/alpha-lens-repo/alpha-lens-image:latest .
```

### Step 6: Deploy to Cloud Run
Deploy the container image to Cloud Run. We reference the Secret Manager secret created in Step 3, mounting it as the `GOOGLE_API_KEY` environment variable:
```bash
gcloud run deploy $SERVICE_NAME \
    --image=$REGION-docker.pkg.dev/$PROJECT_ID/alpha-lens-repo/alpha-lens-image:latest \
    --region=$REGION \
    --platform=managed \
    --allow-unauthenticated \
    --set-secrets="GOOGLE_API_KEY=GOOGLE_API_KEY:latest" \
    --port=8000
```

Once the deployment completes, the gcloud CLI will output your live URL:
```text
Service [alpha-lens] revision [alpha-lens-00001-abc] has been deployed and is serving 100% of traffic.
Service URL: https://alpha-lens-xxxxxx-uc.a.run.app
```

---

## 🔒 Security Best Practices
- **Never hardcode secrets**: The deployment uses Secret Manager to bind `GOOGLE_API_KEY` at runtime.
- **Service Account**: Cloud Run uses a default compute service account. For production, create a custom service account with minimal IAM roles (e.g., only Secret Manager Secret Accessor role).
