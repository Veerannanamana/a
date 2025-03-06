import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Streamlit Page Config
st.set_page_config(page_title="EAMCET Seat Prediction", page_icon="🎓", layout="centered")

# Custom CSS Styles
st.markdown("""
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        .stButton>button {
            background-color: #008CBA;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            font-size: 18px;
        }
        .stButton>button:hover {
            background-color: #005f73;
        }
        .stSelectbox, .stNumberInput {
            border-radius: 8px;
            padding: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("data.csv")

# Encode categorical columns
le_college = LabelEncoder()
le_branch = LabelEncoder()
le_caste = LabelEncoder()
le_gender = LabelEncoder()

df['college_encoded'] = le_college.fit_transform(df['college_name'])
df['branch_encoded'] = le_branch.fit_transform(df['branch'])
df['caste_encoded'] = le_caste.fit_transform(df['caste'])
df['gender_encoded'] = le_gender.fit_transform(df['gender'])

X = df[['branch_encoded', 'caste_encoded', 'gender_encoded', 'start_rank', 'close_rank']]
y = df['college_encoded']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Logistic Regression Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Streamlit UI
st.title("🎓 EAMCET Seat Prediction")
st.write("### Predict your top 5 possible colleges based on your rank.")

# UI Inputs
branch = st.selectbox("🔹 Select Branch", df['branch'].unique())
caste = st.selectbox("🔹 Select Caste", df['caste'].unique())
gender = st.selectbox("🔹 Select Gender", df['gender'].unique())
rank = st.number_input("🔢 Enter Your Rank", min_value=1, step=1)

# Prediction
if st.button("🚀 Predict Top 5 Colleges"):
    with st.spinner("Predicting... Please wait ⏳"):
        # Encode inputs
        branch_encoded = le_branch.transform([branch])[0]
        caste_encoded = le_caste.transform([caste])[0]
        gender_encoded = le_gender.transform([gender])[0]
        
        # Prepare input data
        input_data = pd.DataFrame([[branch_encoded, caste_encoded, gender_encoded, rank, rank]],
                                  columns=['branch_encoded', 'caste_encoded', 'gender_encoded', 'start_rank', 'close_rank'])
        
        # Scale input
        scaled_input = scaler.transform(input_data)
        
        # Predict probabilities for all colleges
        probabilities = model.predict_proba(scaled_input)[0]
        
        # Get the top 5 college indices
        top_5_indices = probabilities.argsort()[-5:][::-1]
        top_5_colleges = le_college.inverse_transform(top_5_indices)
    
    # Display Results
    st.success("✅ Prediction Complete!")
    st.write("### 🎯 Predicted Top 5 Colleges:")
    for i, college in enumerate(top_5_colleges, 1):
        st.markdown(f"**{i}. {college}** 🎓")

st.markdown("##### Made with ❤️ using Streamlit")
