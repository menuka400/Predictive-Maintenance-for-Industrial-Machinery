---
title: AeroShield Predictive Maintenance
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker

app_file: app.py
pinned: false
python_version: "3.10"
---
# AeroShield Predictive Maintenance Dashboard

A predictive maintenance dashboard for turbofan engines using a hybrid Random Forest + LLM (LLaMA-3.1-8B) architecture.

## Setup

1. Clone repo:
```bash
git clone <your-repo-url>
cd <repo>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add `.env` file with Groq API key:
```
GROQ_API_KEY=your_key_here
```

4. Run locally:
```bash
python app.py
```

## Cloud Hosting (Render)

This repository includes a `render.yaml` file for easy deployment to Render.com.

1. Push this code to your GitHub account.
2. Log into Render.com and connect your GitHub account.
3. Click "New" -> "Blueprint" and select your repository.
4. Render will automatically detect the `render.yaml` config and set up a Python web service.
5. In your Render dashboard for this web service, go to **Environment** and add the `GROQ_API_KEY` secret.
