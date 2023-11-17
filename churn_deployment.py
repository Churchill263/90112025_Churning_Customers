import streamlit as st
import pickle
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the model
model =tf.keras.models.load_model("mlp_model.h5")

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
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
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
})

input_data_categorical = pd.DataFrame({
    'SeniorCitizen': [senior_citizen],
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


input_data_categorical['Partner'] = input_data_categorical['Partner'].apply(lambda x: 1 if x == 'Yes' else 0)
input_data_categorical['Dependents'] = input_data_categorical['Dependents'].apply(lambda x: 1 if x == 'Yes' else 0)
input_data_categorical['OnlineSecurity'] = input_data_categorical['OnlineSecurity'].apply(lambda x: 1 if x == 'Yes' else 0)
input_data_categorical['OnlineBackup'] = input_data_categorical['OnlineBackup'].apply(lambda x: 1 if x == 'Yes' else 0)
input_data_categorical['DeviceProtection'] = input_data_categorical['DeviceProtection'].apply(lambda x: 1 if x == 'Yes' else 0)
input_data_categorical['TechSupport'] = input_data_categorical['TechSupport'].apply(lambda x: 1 if x == 'Yes' else 0)
input_data_categorical['Contract'] = input_data_categorical['Contract'].apply(lambda x: 0 if 'Month-to-month' == 'One year' else 1)
input_data_categorical['PaperlessBilling'] = input_data_categorical['PaperlessBilling'].apply(lambda x: 1 if x == 'Yes' else 0)
input_data_categorical['PaymentMethod'] = input_data_categorical['PaymentMethod'].apply(lambda x: 0 if 'Bank transfer (automatic)' == 'Credit card (automatic)' else 1)

scaler_1 = StandardScaler()
numerical_scaled = scaler_1.transform(numerical)
numerical = pd.DataFrame(numerical_scaled, columns=numerical.columns)

# Now, scale the input_data using the same scaler
scaled_input_data = scaler_1.transform(input_data[numerical.columns])

scaled_input_data = scaler.transform(input_data)
scaled = pd.DataFrame(scaled_input_data, input_data.columns)

sc = input_data_categorical["SeniorCitizen"]
input_data_categorical.drop("SeniorCitizen", axis =1, inplace = True)

final_data = pd.concat([sc,scaled_input_data, input_data_categorical])

if st.button('Predict'):
    prediction = model.predict(final_data)

    st.subheader('Prediction')
    st.write('Churn: Yes' if prediction[0] == 1 else 'Churn: No')



