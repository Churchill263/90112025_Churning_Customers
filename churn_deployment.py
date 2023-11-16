import streamlit as st
import pickle
import pandas as pd

# Load the model
with open('churn_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler
with open('Churn_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

st.title('Churn Prediction App')

st.sidebar.header('User Input Features')

st.title('Churn Prediction App')

st.sidebar.header('User Input Features')

# Sample input features (you can customize this based on your needs)
senior_citizen = st.sidebar.selectbox('Senior Citizen', [0, 1])
tenure = st.sidebar.slider('Tenure', 0, 72, 1)
monthly_charges = st.sidebar.slider('Monthly Charges', 0.0, 200.0, 50.0)
total_charges = st.sidebar.slider('Total Charges', 0.0, 8000.0, 2000.0)
partner = st.sidebar.selectbox('Partner', ['Yes', 'No'])
dependents = st.sidebar.selectbox('Dependents', ['Yes', 'No'])
online_security = st.sidebar.selectbox('Online Security', ['Yes', 'No'])
online_backup = st.sidebar.selectbox('Online Backup', ['Yes', 'No'])
device_protection = st.sidebar.selectbox('Device Protection', ['Yes', 'No'])
tech_support = st.sidebar.selectbox('Tech Support', ['Yes', 'No'])
contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.sidebar.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

# Add more input features as needed

input_data = pd.DataFrame({
    'SeniorCitizen': [senior_citizen],
    'Tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Partner': [partner],
    'Dependents': [dependents],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    
})


encoded_input_data = input_data.copy()
encoded_input_data['Partner'] = encoded_input_data['Partner'].apply(lambda x: 1 if x == 'Yes' else 0)
encoded_input_data['Dependents'] = encoded_input_data['Dependents'].apply(lambda x: 1 if x == 'Yes' else 0)



scaled_input_data = scaler.transform(encoded_input_data)

if st.button('Predict'):
    prediction = model.predict(scaled_input_data)

    st.subheader('Prediction')
    st.write('Churn: Yes' if prediction[0] == 1 else 'Churn: No')



