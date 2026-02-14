# GitHub Actions CD Setup Guide

This guide explains how to set up the CI/CD pipeline with your GitHub account.

## Prerequisites

1. **GitHub Account** with the repository
2. **Docker Hub Account** (free tier works)
3. Repository code pushed to GitHub

---

## Step 1: Create Docker Hub Account (if needed)

1. Go to [https://hub.docker.com/](https://hub.docker.com/)
2. Click **Sign Up** and create a free account
3. Note down your **username**

### Create Docker Hub Access Token

1. Log in to Docker Hub
2. Click on your **profile icon** → **Account Settings**
3. Go to **Security** → **Access Tokens**
4. Click **New Access Token**
5. Give it a description: `github-actions`
6. Select scope: **Read, Write, Delete**
7. Click **Generate**
8. **Copy the token immediately** (you can't see it again!)

---

## Step 2: Configure GitHub Repository Secrets

1. Go to your GitHub repository
2. Click **Settings** tab
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**

### Add These Secrets:

| Secret Name | Value |
|-------------|-------|
| `DOCKER_USERNAME` | Your Docker Hub username |
| `DOCKER_PASSWORD` | Your Docker Hub access token (from Step 1) |

**Example:**
```
DOCKER_USERNAME = johndoe
DOCKER_PASSWORD = dckr_pat_xxxxxxxxxxxxxxxxxxxxx
```

---

## Step 3: Set Up GitHub Environments (Optional but Recommended)

1. Go to **Settings** → **Environments**
2. Click **New environment**
3. Name it: `production`
4. Configure protection rules:
   - ✅ Required reviewers (add yourself)
   - ✅ Wait timer: 0 minutes

---

## Step 4: Push Code to GitHub

```bash
# Initialize git (if not already done)
cd /home/kiranahirkar/panasonic/mlops-cats-dogs
git init

# Add remote (replace with your GitHub repo URL)
git remote add origin https://github.com/YOUR_USERNAME/mlops-cats-dogs.git

# Add all files
git add .

# Commit
git commit -m "Initial commit: MLOps Cats vs Dogs project"

# Push to main branch
git branch -M main
git push -u origin main
```

---

## Step 5: Verify Pipeline Execution

1. Go to your GitHub repository
2. Click on the **Actions** tab
3. You should see the "MLOps CI/CD Pipeline" workflow running
4. Click on it to see the details

### Pipeline Jobs:

| Job | Description |
|-----|-------------|
| `lint` | Code quality checks |
| `test` | Unit tests |
| `build` | Build Docker image |
| `deploy` | Deploy with Docker Compose |
| `smoke-tests` | Post-deployment validation |
| `rollback` | Rollback on failure |

---

## Step 6: Manual Triggers

### Trigger Training Manually:

1. Go to **Actions** → **MLOps CI/CD Pipeline**
2. Click **Run workflow**
3. Check **Run model training**
4. Click **Run workflow**

### Deploy to Kubernetes:

1. Go to **Actions** → **MLOps CI/CD Pipeline**
2. Click **Run workflow**
3. Select **Deployment target**: `kubernetes`
4. Click **Run workflow**

---

## How the CD Pipeline Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Lint     │────▶│    Test     │────▶│    Build    │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Rollback   │◀────│Smoke Tests  │◀────│   Deploy    │
│ (on failure)│     └─────────────┘     └─────────────┘
└─────────────┘            │
                           ▼
                    ┌─────────────┐
                    │   Success   │
                    │Notification │
                    └─────────────┘
```

### Flow Explanation:

1. **Lint**: Checks code formatting (Black, isort, Flake8)
2. **Test**: Runs pytest unit tests
3. **Build**: Builds Docker image and pushes to Docker Hub
4. **Deploy**: Deploys using Docker Compose (or Kubernetes)
5. **Smoke Tests**: Validates deployment:
   - Checks `/health` endpoint
   - Tests `/predict` endpoint
   - Tests `/model-info` endpoint
6. **Rollback**: Triggered automatically if smoke tests fail
7. **Notify**: Creates GitHub issue on failure

---

## Smoke Tests Explained

The smoke tests validate that the deployed service is working:

```bash
# Test 1: Health Check
curl http://localhost:8000/health
# Expected: {"status": "healthy", "model_loaded": true/false}

# Test 2: Prediction
curl -X POST http://localhost:8000/predict -F "file=@image.jpg"
# Expected: {"prediction": "cat/dog", "confidence": 0.95}

# Test 3: Model Info
curl http://localhost:8000/model-info
# Expected: {"model_name": "...", "input_shape": [...]}
```

### Running Smoke Tests Locally:

```bash
# Using shell script
./scripts/smoke_test.sh

# Using Python script
python scripts/smoke_test.py --url http://localhost:8000

# With environment variables
API_URL=http://localhost:8000 RETRIES=5 ./scripts/smoke_test.sh
```

---

## Troubleshooting

### Pipeline Fails at Build

**Error:** `unauthorized: authentication required`

**Solution:** Check that `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets are set correctly.

### Smoke Tests Fail

**Error:** `Connection refused`

**Solution:** The container may not have started in time. Increase the wait time in the workflow.

### Deployment Doesn't Happen

**Issue:** Deployment only runs on `main` branch pushes.

**Solution:** Ensure you're pushing to `main`: `git push origin main`

---

## Local Testing Before Push

Test the pipeline locally before pushing:

```bash
# Build Docker image
make docker-build

# Run container
docker run -d -p 8000:8000 --name test-api cats-dogs-classifier:latest

# Wait for startup
sleep 20

# Run smoke tests
./scripts/smoke_test.sh

# Cleanup
docker stop test-api && docker rm test-api
```

---

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Hub Documentation](https://docs.docker.com/docker-hub/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

---

## Questions?

If you encounter issues:
1. Check the **Actions** tab for detailed logs
2. Review the workflow file: `.github/workflows/ci-cd.yaml`
3. Run smoke tests locally first
