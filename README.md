# 90112025_Churning_Customers

## Project Structure

- `Jeffrey_Adei_Assignment 3.ipynb`: Jupyter Notebook containing the code for data preprocessing, exploratory data analysis, model training, and evaluation.
- `mlp_model.h5`: Saved model file for the trained neural network.
- `scaler.pkl`: Pickle file containing the StandardScaler used for numerical feature scaling.
- `encoder.pkl`: Pickle file containing the LabelEncoder used for categorical feature encoding.
- `requirements.txt`: List of required Python packages for the project.

## Data

The dataset (`CustomerChurn_dataset.csv`) contains information about customers, including various features such as tenure, monthly charges, and customer churn status.

## Data Preprocessing

- Removed irrelevant columns (`customerID`, `gender`).
- Handled missing values and converted the `TotalCharges` column to a numeric format.
- Scaled numerical features using `StandardScaler`.
- Encoded categorical features using `LabelEncoder`.

## Exploratory Data Analysis (EDA)

- Explored the correlation matrix to understand relationships between features.
- Visualized feature importance using a Random Forest Classifier.

## Model Training

- Utilized a neural network with multiple hidden layers for customer churn prediction.
- Performed hyperparameter tuning using GridSearchCV.
- Oversampled the minority class to address class imbalance.
- Evaluated model performance using accuracy, AUC, and other metrics.

## Model Deployment

- Saved the trained model, scaler, and encoder for deployment.
- Generated a `requirements.txt` file for easy environment setup.

## Instructions for Use

1. Install dependencies by running `pip install -r requirements.txt`.
2. Run the Jupyter Notebook `Jeffrey_Adei_Assignment 3.ipynb` to explore the data, train the model, and save the necessary files.
3. Use the saved model, scaler, and encoder for customer churn prediction in a production environment.

Feel free to explore the Jupyter Notebook for a detailed walkthrough of the project.

**Note:** Ensure you have Python and the required packages installed before running the code.
