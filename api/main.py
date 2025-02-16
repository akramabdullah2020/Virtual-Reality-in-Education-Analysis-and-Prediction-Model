import uvicorn
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from typing import Literal,List,Dict
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import traceback



class VREducationModel(BaseModel):
    Age: int = None
    Grade_Level: Literal["Postgraduate", "Undergraduate", "High School"] = Field(..., description="Educational grade level")
    Usage_of_VR_in_Education: Literal["Yes", "No"] = Field(..., description="Whether VR is used in education")
    Hours_of_VR_Usage_Per_Week: int = Field(..., description="Number of hours per week VR is used", ge=0)
    # Engagement_Level: int = Field(..., description="Engagement level with VR", ge=0)
    Improvement_in_Learning_Outcomes: Literal["Yes", "No"] = Field(
        ..., description="Impact of VR on learning outcomes"
    )
    Instructor_VR_Proficiency: Literal["Beginner", "Intermediate", "Advanced"] = Field(
        ..., description="Proficiency level of the instructor in VR"
    )
    Access_to_VR_Equipment: Literal["Yes", "No"] = Field(..., description="Access to VR equipment")
    Stress_Level_with_VR_Usage: Literal["High", "Medium", "Low"] = Field(
        ..., description="Stress level experienced when using VR"
    )
    Collaboration_with_Peers_via_VR: Literal["Yes", "No"] = Field(
        ..., description="Whether collaboration with peers via VR is present"
    )



app =FastAPI()
items:List[VREducationModel] =[]

 # Load saved encoders and scaler and model
label_encoders = joblib.load('./models/label_encoders.pkl')
scaler = joblib.load('./models/scaler.pkl')
model = joblib.load('./models/rfr_model.pkl')
        


@app.get('/')
def root():
    example = VREducationModel(
    Age=25,
    Grade_Level="Undergraduate",
    Usage_of_VR_in_Education="Yes",
    Hours_of_VR_Usage_Per_Week=10,
    # Engagement_Level=8,
    Improvement_in_Learning_Outcomes="Yes",
    Instructor_VR_Proficiency="Intermediate",
    Access_to_VR_Equipment="Yes",
    Stress_Level_with_VR_Usage="Medium",
    Collaboration_with_Peers_via_VR="Yes")
    return example



@app.post("/items")
def create_item(item: VREducationModel):
    items.append(item)
    return items



@app.post("/predict/")
def predict(VREducation_Data: VREducationModel):
    try:
        # Convert input to DataFrame
        data = pd.DataFrame([VREducation_Data.dict()])
        print("Raw Input Data:")
        print(data.head(1))
        
        # Transform categorical columns
        categorical_columns = ['Grade_Level', 'Usage_of_VR_in_Education', 
                               'Improvement_in_Learning_Outcomes', 'Instructor_VR_Proficiency',
                               'Access_to_VR_Equipment', 'Stress_Level_with_VR_Usage',
                               'Collaboration_with_Peers_via_VR']
        for col in categorical_columns:
            data[col] = label_encoders[col].transform(data[col])
        
        
        # Scale numerical columns
        numerical_columns = ['Age', 'Hours_of_VR_Usage_Per_Week']
        data[numerical_columns] = scaler.transform(data[numerical_columns])
        
        print("Scaled Data:")
        print(data.head(1))
        
        # Make prediction
        prediction = model.predict(data)
        
        return {"prediction": prediction[0]}
    
    except Exception as e:
        print("An error occurred during prediction:")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)