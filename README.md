# Virtual Reality in Education: Analysis and Prediction Model

## 📌 Project Overview
This project is a **practical assignment** from a **Machine Learning course**, focusing on analyzing the impact of Virtual Reality (VR) in education. The goal is to develop a **predictive model** that estimates learning outcomes based on VR engagement levels, stress factors, and other key variables.

## 🎯 Objectives
- Apply **machine learning techniques** to real-world educational data.
- Analyze the effect of VR usage on learning performance.
- Develop a **prediction model** to estimate educational improvements based on VR interaction.
- Deploy the model using **FastAPI** to provide real-time predictions and insights.

## 📊 Dataset Overview
- **Total Entries:** 5,000 participants
- **Features:** 10 key variables, including:
  - **Engagement_Level** (Most influential predictor)
  - **Hours_of_VR_Usage_Per_Week**
  - **Instructor_VR_Proficiency**
  - **Access_to_VR_Equipment**
  - **Age**

## 🔍 Data Analysis & Preprocessing
**Tools Used:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
- Handled **missing values** appropriately.
- Removed **duplicates and outliers** using **Z-score & IQR** methods.
- **Normalized numerical data** (e.g., Age, Hours of VR Usage).
- Encoded categorical variables for model readiness.

### Key Insights:
✔ High engagement correlated with increased learning outcomes.  
✔ Stress levels varied significantly based on **instructor proficiency**.  
✔ Consistent access to VR equipment played a crucial role in learning performance.

## 🛠 Model Development
**Tools Used:** `Scikit-learn`
- Data split: **80% training, 20% testing**
- Tested multiple models:
  - **Logistic Regression**
  - **Random Forest**
  - **DecisionTreeRegressor**
- **Best Model Selection:** Based on Accuracy & F1 Score

### 🔹 Final Model Performance:
- **Mean Squared Error (MSE):** 0.0674
- **Root Mean Squared Error (RMSE):** 0.26
- **R-squared (R²):** 0.97 (High predictive accuracy)

## 🚀 API Development with FastAPI
To make the model easily accessible, an **API was built using FastAPI**.

### 🔹 API Features:
✔ Input validation for user data  
✔ Real-time predictions based on the trained model  
✔ Endpoints for querying **data insights**

### 🔹 Example Endpoint:
```python
POST /predict
{
    "Engagement_Level": 8.5,
    "Hours_of_VR_Usage_Per_Week": 10,
    "Instructor_VR_Proficiency": 7,
    "Access_to_VR_Equipment": 1,
    "Age": 22
}
```
#### ✅ Expected Response:
```json
{
    "Predicted_Learning_Outcome": 0.89
}
```

## 📌 Future Work
- Incorporate **emotional and cognitive metrics** from VR interactions.
- Expand the dataset to include **global educational contexts**.
- Develop a **web and mobile interface** for institutions to utilize predictive insights.



