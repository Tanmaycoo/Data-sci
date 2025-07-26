import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

st.title("📊 Train & Predict ML Model from Your CSV")

# -------------------------------
# SECTION 1: Upload CSV & Train
# -------------------------------
st.header("📁 Upload CSV and Train Model")
model_name = st.text_input("💾 Enter a name for your model (used for saving)", value="my_model")
uploaded_file = st.file_uploader("Upload CSV file for training", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Normalize column names
    st.write("🔍 Preview of uploaded data:")
    st.dataframe(df.head())

    target_column = st.selectbox("🎯 Select the target column (what to predict)", df.columns)

    if target_column:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Drop rows with missing values
        data = pd.concat([X, y], axis=1).dropna()
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Detect categorical and numerical features
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # Pipeline
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42))
        ])

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)
        accuracy = pipeline.score(X_test, y_test)

        st.success(f"✅ Model trained with accuracy: {accuracy:.2f}")

        report = classification_report(y_test, pipeline.predict(X_test), output_dict=True)
        st.write("📄 Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())

        # Save model & feature names
        pickle.dump(pipeline, open(f"{model_name}_model.pkl", "wb"))
        pickle.dump(X.columns.tolist(), open(f"{model_name}_features.pkl", "wb"))

        st.info("💾 Model and feature names saved successfully!")

# -------------------------------
# SECTION 2: Prediction
# -------------------------------
st.header("🔮 Predict with Trained Model")
load_model_name = st.text_input("📂 Enter the model name to load for prediction", value="")

try:
    model = pickle.load(open(f"{load_model_name}_model.pkl", "rb"))
    feature_names = pickle.load(open(f"{load_model_name}_features.pkl", "rb"))
except:
    st.warning("⚠️ Could not find the specified model files.")
    st.stop()

# Validate uploaded file again for prediction
if uploaded_file:
    df.columns = df.columns.str.strip()  # Normalize again
    actual_columns = set(df.columns)
    required_columns = set(feature_names)

    missing = required_columns - actual_columns
    if missing:
        st.error(f"❗ Prediction failed: columns are missing: {missing}")
        st.stop()

    df_sample = df[feature_names]
else:
    df_sample = None

# Manual input for prediction
input_data = {}
for col in feature_names:
    if df_sample is not None:
        if df_sample[col].dtype == 'object' or df_sample[col].nunique() < 10:
            options = df_sample[col].dropna().unique().tolist()
            input_data[col] = st.selectbox(f"{col}", options)
        else:
            min_val = float(df_sample[col].min())
            max_val = float(df_sample[col].max())
            input_data[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=min_val)
    else:
        input_data[col] = st.text_input(f"{col}")

# Predict
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([input_data])

        # Convert numeric columns safely
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except:
                pass

        # Predict
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        # Display result
        st.write(f"🟠 Probability of Churn: {proba[1]*100:.2f}%")
        st.write(f"🟢 Probability of No Churn: {proba[0]*100:.2f}%")

    except Exception as e:
        st.error(f"❗ Prediction failed: {e}")

