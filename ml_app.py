import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd

# Dictionary mapping
dep = {'Sales & Marketing':1, 'Operations':2, 'Technology':3, 'Analytics':4,
       'R&D':5, 'Procurement':6, 'Finance':7, 'HR':8, 'Legal':9}
edu = {'Below Secondary':1, "Bachelor's":2, "Master's & above":3}
rec = {'referred':1, 'sourcing':2, 'others':3}
gen = {'m':1, 'f':2}
reg = {f"region_{i}":i for i in range(1,35)}

def get_value(val, my_dict):
    return my_dict.get(val)

def load_model(model_file):
    return joblib.load(open(os.path.join(model_file), 'rb'))

def run_ml_app():
    st.title("📊 Employee Promotion Prediction")
    st.markdown("Masukkan data karyawan untuk melihat peluang promosi.")

    with st.expander("ℹ️ Attribute Info"):
        st.markdown("""
        - Department: Sales & Marketing, Operations, Technology, Analytics, R&D, Procurement, Finance, HR, Legal  
        - Region: region 1 - region 34  
        - Education: Below Secondary, Bachelor's, Master's & above  
        - Gender: Male and Female  
        - Recruitment Channel: Referred, Sourcing, Others  
        - No of Training: 1-10  
        - Age: 10-60  
        - Previous Year Rating: 1-5  
        - Length of Service: 1-37 Month  
        - Awards Won: 1. Yes, 0. No  
        - Avg Training Score: 0-100  
        """)

    # Input form
    st.subheader("📝 Input Your Data")
    department = st.selectbox('Department', list(dep.keys()))
    region = st.selectbox('Region', list(reg.keys()))
    education = st.selectbox('Education', list(edu.keys()))
    gender = st.radio('Gender', list(gen.keys()))
    recruitment = st.selectbox("Recruitment Channel", list(rec.keys()))
    training = st.slider("No of Training", 1, 10, 3)
    age = st.slider("Age",10,60,25)
    rating = st.slider("Previous Year Rating",1,5,3)
    service = st.slider("Length of Service (Month)",1,37,12)
    awards = st.radio("Awards Won", [0,1])
    avg_training = st.slider("Average Training Score",0,100,50)

    result = {
        'Department':department,
        'Region':region,
        'education':education,
        'gender':gender,
        'recruitment_channel':recruitment,
        'no_of_trainings':training,
        'age':age,
        'previous_year_rating':rating,
        'length_of_service':service,
        'awards_won':awards,
        'avg_training_score':avg_training,
    }

    with st.expander("🔍 Your Selected Options"):
        st.json(result)

    # Encoding
    encoded_result = [
        get_value(department, dep),
        get_value(region, reg),
        get_value(education, edu),
        get_value(gender, gen),
        get_value(recruitment, rec),
        training,
        age,
        rating,
        service,
        awards,
        avg_training
    ]

    # Prediction
    st.subheader('📈 Prediction Result')
    single_array = np.array(encoded_result).reshape(1, -1)
    model = load_model("model_grad.pkl")
    prediction = model.predict(single_array)
    pred_proba = model.predict_proba(single_array)

    pred_probability_score = {
        'Promoted': round(pred_proba[0][1]*100, 1),
        'Not Promoted': round(pred_proba[0][0]*100, 1)
    }

    st.bar_chart(pred_probability_score)
    st.metric("Promoted Probability", f"{pred_probability_score['Promoted']}%")
    st.progress(int(pred_probability_score['Promoted']))

    if prediction == 1:
        st.success("🎉 Congratulation, you get promotion!")
    else:
        st.error("⚠️ Need to improve, keep learning!")

    # Personalized tips
    if avg_training < 50:
        st.info("💡 Coba tingkatkan skor training dengan mengikuti lebih banyak pelatihan.")
    if rating < 3:
        st.info("📌 Perbaiki kinerja tahunan agar rating lebih tinggi.")

    # Download result
    df_result = pd.DataFrame([result])
    st.download_button("⬇️ Download Result", df_result.to_csv(index=False), "result.csv")

if __name__ == '__main__':
    run_ml_app()
