import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Bank Loan Eligibility Prediction", page_icon="🏦", layout="wide")
st.title("🏦 Bank Loan Eligibility Prediction")
st.caption("SI Lab project • Statistical analysis, model comparison, and applicant-level prediction")
st.info("Academic demonstration only. Predictions are not actual bank decisions.")

@st.cache_data
def make_demo_data(n=2200, seed=42):
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female"], n, p=[.68, .32])
    married = rng.choice(["Yes", "No"], n, p=[.62, .38])
    dependents = rng.choice(["0", "1", "2", "3+"], n, p=[.55, .18, .17, .10])
    education = rng.choice(["Graduate", "Not Graduate"], n, p=[.78, .22])
    self_employed = rng.choice(["No", "Yes"], n, p=[.86, .14])
    income = np.clip(rng.lognormal(np.log(5200), .55, n), 800, 30000).round()
    co_income = np.where(rng.random(n) < .35, 0, np.clip(rng.lognormal(np.log(1800), .7, n), 100, 15000)).round()
    loan_amount = np.clip(rng.lognormal(np.log(250), .75, n), 20, 2500).round()
    term = rng.choice([120, 180, 240, 300, 360, 480], n, p=[.03, .06, .08, .12, .62, .09])
    credit = rng.choice([0, 1], n, p=[.18, .82])
    area = rng.choice(["Rural", "Semiurban", "Urban"], n, p=[.28, .38, .34])
    total_income = income + co_income
    logit = -1.15 + 2.1*credit + .00008*(total_income-6000) - .004*(loan_amount-140) + .28*(education=="Graduate") + .30*(area=="Semiurban") - .25*(self_employed=="Yes")
    p = 1/(1+np.exp(-logit))
    status = np.where(rng.binomial(1, p) == 1, "Y", "N")
    data = pd.DataFrame({
        "Gender": gender, "Married": married, "Dependents": dependents,
        "Education": education, "Self_Employed": self_employed,
        "ApplicantIncome": income, "CoapplicantIncome": co_income,
        "LoanAmount": loan_amount, "Loan_Amount_Term": term,
        "Credit_History": credit, "Property_Area": area,
        "Loan_Status": status
    })
    for col in ["Gender", "Married", "LoanAmount", "Credit_History"]:
        idx = rng.choice(data.index, size=max(1, int(.02*n)), replace=False)
        data.loc[idx, col] = np.nan
    return data

def map_status(value):
    if pd.isna(value): return np.nan
    value = str(value).strip().lower()
    if value in {"y", "yes", "approved", "approve", "1", "true", "eligible"}: return 1
    if value in {"n", "no", "rejected", "reject", "0", "false", "not eligible", "ineligible"}: return 0
    try:
        number = float(value)
        if number in (0, 1): return int(number)
    except Exception:
        pass
    return np.nan

def prepare_data(raw):
    data = raw.copy()
    data.columns = [str(c).strip().replace(" ", "_") for c in data.columns]
    target_options = ["Loan_Status", "loan_status", "loan_approval", "LoanApproval", "Approved", "approved", "target", "Eligibility"]
    target = next((c for c in target_options if c in data.columns), None)
    if target is None:
        return None, None, None, "Could not find a target column. Use a target such as Loan_Status (Y/N)."
    data["Eligibility"] = data[target].apply(map_status)
    data = data.dropna(subset=["Eligibility"]).copy()
    if data["Eligibility"].nunique() < 2:
        return None, None, None, "Target must contain both eligible/approved and not eligible/rejected examples."
    data["Eligibility"] = data["Eligibility"].astype(int)
    data["Eligibility_Label"] = data["Eligibility"].map({1: "Eligible / Approved", 0: "Not eligible / Rejected"})
    drop_cols = {target, "Eligibility", "Eligibility_Label"}
    drop_cols.update(c for c in data.columns if c.lower() in {"id", "loan_id", "customer_id", "applicant_id"})
    X = data.drop(columns=[c for c in drop_cols if c in data.columns], errors="ignore")
    X = X.dropna(axis=1, how="all")
    y = data["Eligibility"]
    if len(data) < 20:
        return None, None, None, "Please upload at least 20 labelled rows."
    if X.empty:
        return None, None, None, "No usable feature columns found."
    return data, X, y, None

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload loan approval CSV (optional)", type=["csv"])
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            source = "Uploaded CSV"
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            raw, source = make_demo_data(), "Synthetic demo data"
    else:
        raw, source = make_demo_data(), "Synthetic demo data"
    st.caption(f"Data source: {source}")

data, X, y, err = prepare_data(raw)
if err:
    st.error(err)
    st.stop()

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
transformers = []
if numeric_features:
    transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features))
if categorical_features:
    transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features))
preprocessor = ColumnTransformer(transformers, remainder="drop")

try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
except ValueError as exc:
    st.error(f"Unable to create train/test split: {exc}")
    st.stop()

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7)
}
trained, rows = {}, []
for name, classifier in classifiers.items():
    model = Pipeline([("preprocess", preprocessor), ("classifier", classifier)])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    try:
        probabilities = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)
    except Exception:
        auc = np.nan
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": auc
    })
    trained[name] = model
metrics = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)
best_name = metrics.iloc[0]["Model"]
best_model = trained[best_name]

tab_overview, tab_stats, tab_models, tab_predict, tab_data = st.tabs([
    "📊 Overview", "📐 Statistics", "🤖 Model Comparison", "🧮 Predict Eligibility", "🗂️ Data"
])

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applicant records", f"{len(data):,}")
    c2.metric("Eligible / approved", f"{int((y == 1).sum()):,}")
    c3.metric("Not eligible / rejected", f"{int((y == 0).sum()):,}")
    c4.metric("Approval rate", f"{y.mean():.1%}")
    left, right = st.columns(2)
    counts = data["Eligibility_Label"].value_counts().rename_axis("Status").reset_index(name="Applicants")
    left.plotly_chart(px.pie(counts, names="Status", values="Applicants", hole=.45, title="Loan approval distribution"), use_container_width=True)
    right.plotly_chart(px.histogram(data, x="LoanAmount", color="Eligibility_Label" if "Eligibility_Label" in data else None, title="Loan amount distribution"), use_container_width=True)
    if {"ApplicantIncome", "LoanAmount"}.issubset(data.columns):
        st.plotly_chart(px.scatter(data, x="ApplicantIncome", y="LoanAmount", color="Eligibility_Label", title="Applicant income vs loan amount"), use_container_width=True)

with tab_stats:
    st.subheader("Descriptive statistics")
    numeric = data.select_dtypes(include=np.number)
    if not numeric.empty:
        summary = numeric.describe().T
        summary["median"] = numeric.median(numeric_only=True)
        summary["mode"] = numeric.mode().iloc[0]
        summary["variance"] = numeric.var(numeric_only=True)
        summary["Q1"] = numeric.quantile(.25)
        summary["Q3"] = numeric.quantile(.75)
        summary["IQR"] = summary["Q3"] - summary["Q1"]
        st.dataframe(summary.round(3), use_container_width=True)
        numeric_cols = [c for c in numeric.columns if c != "Eligibility"]
        if numeric_cols:
            col = st.selectbox("Choose numeric feature", numeric_cols)
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.histogram(data, x=col, marginal="box", title=f"Distribution of {col}"), use_container_width=True)
            c2.plotly_chart(px.box(data, y=col, title=f"Box plot of {col}"), use_container_width=True)

with tab_models:
    st.subheader("Held-out test-set model comparison")
    st.dataframe(metrics.round(3), use_container_width=True)
    selected = st.selectbox("Confusion matrix for model", list(trained.keys()), index=list(trained.keys()).index(best_name))
    pred = trained[selected].predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    st.plotly_chart(px.imshow(cm, text_auto=True, x=["Predicted not eligible", "Predicted eligible"], y=["Actual not eligible", "Actual eligible"], title=f"Confusion matrix — {selected}"), use_container_width=True)
    st.caption(f"Best model selected by F1-score: {best_name}")

with tab_predict:
    st.subheader("Check whether an applicant is likely to be eligible")
    st.write("Enter the applicant's details. Loan amount accepts large values; the form is not limited to the training data's observed maximum.")
    values = {}
    with st.form("eligibility_form"):
        cols = st.columns(3)
        for i, feature in enumerate(X.columns):
            container = cols[i % 3]
            if feature in numeric_features:
                vals = pd.to_numeric(X[feature], errors="coerce").dropna()
                default = float(vals.median()) if len(vals) else 0.0
                if feature.lower() in {"loanamount", "loan_amount", "loan_amount_thousands"}:
                    values[feature] = container.number_input(feature.replace("_", " ").title(), min_value=0.0, max_value=100000000.0, value=max(0.0, default), step=10.0, help="Enter a loan amount up to 100,000,000 in your dataset's units.")
                else:
                    values[feature] = container.number_input(feature.replace("_", " ").title(), value=default, step=1.0)
            else:
                options = sorted(X[feature].dropna().astype(str).unique().tolist())
                if not options:
                    options = [""]
                default = str(X[feature].dropna().astype(str).mode().iloc[0]) if X[feature].notna().any() else options[0]
                if default not in options:
                    default = options[0]
                values[feature] = container.selectbox(feature.replace("_", " ").title(), options, index=options.index(default))
        submitted = st.form_submit_button("Predict loan eligibility", type="primary", use_container_width=True)
    if submitted:
        applicant = pd.DataFrame([values], columns=X.columns)
        prediction = int(best_model.predict(applicant)[0])
        try:
            probability = float(best_model.predict_proba(applicant)[0, 1])
        except Exception:
            probability = None
        if prediction == 1:
            st.success("✅ RESULT: LIKELY ELIGIBLE FOR A LOAN")
        else:
            st.error("❌ RESULT: NOT LIKELY ELIGIBLE FOR A LOAN")
        if probability is not None:
            st.metric("Estimated eligibility probability", f"{probability:.1%}")
        st.write(f"Model used: **{best_name}**")
        st.caption("This is a model estimate based on historical labels, not a guarantee of actual bank approval.")

with tab_data:
    st.subheader("Cleaned dataset")
    st.dataframe(data.head(500), use_container_width=True)
    st.download_button("Download cleaned CSV", data.to_csv(index=False).encode("utf-8"), "cleaned_loan_eligibility.csv", "text/csv")
    st.download_button("Download model comparison CSV", metrics.to_csv(index=False).encode("utf-8"), "loan_model_metrics.csv", "text/csv")
