# Bank Loan Eligibility Prediction — Streamlit

## Run in Google Colab
1. Upload `app.py` and `requirements.txt` to Colab, or unzip the project ZIP.
2. Install packages:
   `!pip -q install -r requirements.txt pyngrok`
3. Set up ngrok once (requires a free ngrok account and authtoken):
   ```python
   from pyngrok import ngrok
   from getpass import getpass
   ngrok.set_auth_token(getpass("Enter your ngrok authtoken: "))
   ```
4. Start Streamlit:
   ```python
   import subprocess, time
   subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"])
   public_url = ngrok.connect(8501)
   print("Open the Streamlit dashboard:", public_url)
   ```
   If port 8501 is already in use, stop the old process or use another port and connect that port.
5. Open the printed URL. Use the **Predict Eligibility** tab and enter applicant details.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features
- Streamlit tabs for overview, descriptive statistics, model comparison, applicant prediction, and data download.
- Models: Logistic Regression, Decision Tree, Random Forest, KNN.
- Larger loan amount input (up to 100,000,000 in dataset units).
- Eligibility output clearly states `LIKELY ELIGIBLE FOR A LOAN` or `NOT LIKELY ELIGIBLE FOR A LOAN`.
- Optional CSV upload; accepted target columns include `Loan_Status` with Y/N or similar approval labels.

Default data is synthetic and only for academic demonstration. Train/test metrics and predictions depend on the dataset.
