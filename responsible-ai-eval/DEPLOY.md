# Deployment Guide — Streamlit Community Cloud

## Why Streamlit Community Cloud?

- Purpose-built for Streamlit apps — zero config
- Forever free
- Connects directly to GitHub — auto-deploys on every push
- Built-in secrets manager for API keys
- Public URL you can share immediately

---

## Step 1 — Sign up

Go to **https://share.streamlit.io** and sign in with your GitHub account.

---

## Step 2 — Create a new app

Click **"Create app"** → **"Deploy a public app from GitHub"**

Fill in the form exactly as follows:

| Field | Value |
|---|---|
| Repository | `gurusin/ai-workshop` |
| Branch | `main` |
| Main file path | `responsible-ai-eval/app.py` |
| App URL (optional) | choose a custom slug e.g. `rai-eval` |

---

## Step 3 — Add API keys as secrets

Before clicking Deploy, open **"Advanced settings"** → **"Secrets"** tab.

Paste the following (replace values with your actual keys):

```toml
GROQ_API_KEY = "gsk_..."
CEREBRAS_API_KEY = "csk_..."
SAMBANOVA_API_KEY = "28ac22a6-..."
```

Streamlit injects these as environment variables at runtime.  
The `.env` file on your local machine is **not** used in the cloud.

---

## Step 4 — Deploy

Click **"Deploy!"**

Streamlit will:
1. Clone your repo
2. Install dependencies from `responsible-ai-eval/requirements.txt`
3. Start the app

This takes 2–3 minutes on first deploy. You'll get a public URL:

```
https://<your-slug>.streamlit.app
```

---

## Updating the app

Every `git push origin main` automatically triggers a redeploy.  
No manual steps needed after the first deployment.

---

## Managing secrets after deployment

Go to your app on share.streamlit.io → **⋮ menu** → **Settings** → **Secrets**  
to add, update or rotate API keys without redeploying.

---

## Fixing IntelliJ IDEA terminal scrolling

While you're here — to fix the terminal scroll in IntelliJ:

1. Open **Settings** (`Cmd+,`)
2. Go to **Tools → Terminal**
3. Increase **"Scrollback lines"** (default is 1000 — set to 5000+)
4. Click **OK** and restart the terminal panel

Alternatively, use **two-finger scroll** on the trackpad directly inside the terminal panel, or click inside the panel first to give it focus before scrolling.
