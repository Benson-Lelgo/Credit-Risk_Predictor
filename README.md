
# Credit Risk Prediction App

This project is a **Streamlit-based web application** for predicting credit risk using a trained Random Forest model. It allows users to input loan and borrower features and receive a risk classification with an explanation using SHAP (SHapley Additive exPlanations).

## Features

- Predict credit risk probability
- Classify loans as High Risk or Low Risk
- Interactive sliders and dropdowns for user inputs
- SHAP-based local and global feature explanations
- Dynamic charts showing feature importance

## Project Structure

```
├── app.py                    # Streamlit app script
├── random_forest_model.joblib  # Trained model
├── background_sample.csv     # Background dataset for SHAP
├── shap_beeswarm_class1.png  # Optional SHAP image
├── requirements.txt          # Python dependencies
```

## Try it Yourself

You can deploy and run the app live on **Streamlit Cloud** or locally:

### Option 1: Deploy on Streamlit Cloud
1. Push all files to a GitHub repo
2. Go to https://streamlit.io/cloud
3. Click “New App” and connect your GitHub repo
4. Set `app.py` as the main file and deploy

### Option 2: Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model

- **Type**: Random Forest Classifier
- **Framework**: Scikit-learn
- **Explainability**: SHAP

## 📫 Contact

For questions or collaboration, feel free to connect!

---
© 2025 Benson Lelgo
