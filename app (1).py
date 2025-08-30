import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open(f'{working_dir}/saved_model/diabetes_model.sav', 'rb'))
scaler = pickle.load(open(f'{working_dir}/saved_model/diabetes_scaler.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_model/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open(f'{working_dir}/saved_model/parkinsons_model.sav', 'rb'))
brain_model = pickle.load(open(f'{working_dir}/saved_model/headbrain_model.sav', 'rb'))  # or .sav if you used joblib

# sidebar for navigation

# with st.sidebar:
#     selected = option_menu('Multiple Disease Prediction System',
#                            ['Home','Diabetes Prediction',
#                             'Heart Disease Prediction',
#                             'Brain Size Normalcy Prediction',
#                             'Parkinsons Prediction'],  # New option added
#                            menu_icon='hospital-fill',
#                            icons=[ 'activity',     
#             'heart-pulse',  
#             'cpu-fill',
#             'person-fill'],
#                            default_index=0)

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Home',
         'Diabetes Prediction',
         'Heart Disease Prediction',
         'Brain Size Normalcy Prediction',
         'Parkinsons Prediction'],
        icons=['house-heart-fill', 'activity', 'heart-pulse', 'cpu-fill', 'person-fill'],
        menu_icon='hospital-fill',
        default_index=0
    )

# Home Page
if selected == 'Home':
    st.title("🏥 Welcome to the Multiple Disease Prediction System")
    st.markdown("""
    This application uses machine learning models to predict your likelihood of certain health conditions based on your input data.

    ---
    ### 🔍 Available Predictions:
    - **Diabetes**
    - **Heart Disease**
    - **Brain Size Normalcy**
    - **Parkinson’s Disease**

    Use the sidebar to choose a disease, enter your details, and click Predict.

    > ⚠️ **Disclaimer**: This app is for informational purposes only and should not replace medical advice from professionals.
    """)


if selected == 'Diabetes Prediction':

    st.title('Diabetes Prediction using ML')
    st.write("Please provide your health details below. We'll help assess your risk for diabetes disease.")

    # Normal Range Chart
    with st.expander("ℹ️ View Normal Ranges for Input Features"):
        st.markdown("""
        | Feature | Normal Range |
        |--------|---------------|
        | **Pregnancies** | 0 – 10 |
        | **Glucose** | 70 – 99 mg/dL (fasting) |
        | **Blood Pressure** | 90 – 120 mmHg |
        | **Skin Thickness** | 10 – 50 mm |
        | **Insulin** | 16 – 166 µU/mL |
        | **BMI** | 18.5 – 24.9 |
        | **Diabetes Pedigree Function** | 0.0 – 2.5 |
        | **Age** | Any (risk increases > 45) |
        """, unsafe_allow_html=True)

    # Input Fields
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0, step=1)

    with col2:
        Glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=300, value=100, step=1)

    with col3:
        BloodPressure = st.number_input('Blood Pressure (mmHg)', min_value=0, max_value=200, value=80, step=1)

    with col1:
        SkinThickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20, step=1)

    with col2:
        Insulin = st.number_input('Insulin Level (µU/mL)', min_value=0, max_value=900, value=80, step=1)

    with col3:
        BMI = st.number_input('BMI', min_value=0.0, max_value=60.0, value=25.0, step=0.1, format="%.1f")

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=5.0, value=0.5, step=0.01, format="%.2f")

    with col2:
        Age = st.number_input('Age of the Person', min_value=1, max_value=120, value=30, step=1)

    # Optional: Show warnings if values are outside healthy ranges
    if Glucose > 125:
        st.warning("⚠️ High glucose level – indicates risk of diabetes.")
    if BloodPressure > 130:
        st.warning("⚠️ High blood pressure – could be hypertensive.")
    if BMI > 30:
        st.warning("⚠️ High BMI – considered obese.")
    if DiabetesPedigreeFunction > 1.5:
        st.warning("⚠️ High diabetes pedigree function – family risk.")

    # Predict
    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        try:
            user_input = [
                int(Pregnancies), int(Glucose), int(BloodPressure),
                int(SkinThickness), int(Insulin),
                float(BMI), float(DiabetesPedigreeFunction), int(Age)
            ]

            # Custom validation: check for unrealistic low inputs
            if Glucose < 50 or BloodPressure < 40 or BMI < 10 or Insulin < 10 or SkinThickness < 5:
                st.error("❌ Some input values are too low to be considered healthy. Please consult a doctor.")
            else:
                input_array = np.array(user_input).reshape(1, -1)
                input_scaled = scaler.transform(input_array)
                diab_prediction = diabetes_model.predict(input_scaled)
                diab_prob = diabetes_model.predict_proba(input_scaled)[0][1]

                if diab_prediction[0] == 1:
                    diab_diagnosis = '⚠️ The person is  **diabetic**.'
                else:
                    diab_diagnosis = '✅ The person is  **not diabetic**.'

                st.success(diab_diagnosis)
               
        except Exception as e:
            st.error(f"Invalid input detected: {e}")



# Heart Disease Prediction Page


if selected == 'Heart Disease Prediction':

    st.title(" Heart Disease Risk Checker")

    st.write("Please provide your health details below. We'll help assess your risk for heart disease.")

    # Collecting user inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Your Age", min_value=1, max_value=120)

    with col2:
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

    with col3:
        cp = st.selectbox("Type of Chest Pain",
                          options=[0, 1, 2, 3],
                          format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"][x])

    with col1:
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=60, max_value=200)

    with col2:
        chol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=600)

    with col3:
        fbs = st.selectbox("Fasting Blood Sugar Over 120 mg/dL?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    with col1:
        restecg = st.selectbox("Resting ECG Results",
                               options=[0, 1, 2],
                               format_func=lambda x: ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"][x])

    with col2:
        thalach = st.number_input("Maximum Heart Rate Achieved (bpm)", min_value=60, max_value=220)

    with col3:
        exang = st.selectbox("Do you experience chest pain during exercise?",
                             options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    with col1:
        oldpeak = st.number_input("ST Depression due to Exercise (Oldpeak)", min_value=0.0, max_value=6.0, step=0.1)

    with col2:
        slope = st.selectbox("Slope of ST Segment After Exercise",
                             options=[0, 1, 2],
                             format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])

    with col3:
        ca = st.selectbox("Number of Major Vessels (0–3) Colored by X-ray", options=[0, 1, 2, 3, 4])

    with col1:
        thal = st.selectbox("Thalassemia Test Result",
                            options=[1, 2, 3],
                            format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x - 1])

    # Show a helpful guide to values
    with st.expander("ℹ️ What are the normal ranges?"):
        st.markdown("""
        | Health Metric | Normal Range |
        |---------------|--------------|
        | **Age** | Under 65 is lower risk |
        | **Sex** | Male or Female |
        | **Chest Pain Type** | Typical/Non-Anginal are lower risk |
        | **Blood Pressure** | 90 – 120 mm Hg |
        | **Cholesterol** | Under 200 mg/dL |
        | **Fasting Sugar** | Should be under 120 mg/dL |
        | **ECG** | Normal or minor abnormalities |
        | **Heart Rate** | 100 – 200 bpm (depends on age) |
        | **Exercise Angina** | No pain is better |
        | **Oldpeak** | 0.0 – 2.0 |
        | **Slope** | Upsloping is normal |
        | **Vessels Colored** | 0 – 3 |
        | **Thalassemia** | Normal is best |
        """)

    # Simple user guidance based on inputs
    if chol > 240:
        st.warning("⚠️ Your cholesterol is higher than recommended (above 240 mg/dL).")
    if trestbps > 140:
        st.warning("⚠️ Your blood pressure is on the higher side (above 140 mm Hg).")
    if oldpeak > 2.0:
        st.warning("⚠️ Elevated ST depression – could indicate heart stress.")
    if thalach < 100:
        st.warning("⚠️ Your max heart rate is quite low – might need medical attention.")

    # Button and prediction
    if st.button("🔍 Check My Heart Disease Risk"):
    # Feature names used during training
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                     'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    # Convert user input to floats
        user_input = [age, sex, cp, trestbps, chol, fbs,
                  restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]

        import pandas as pd    # Create a DataFrame with column names
        input_df = pd.DataFrame([user_input], columns=feature_names)

    # Predict
        prediction = heart_disease_model.predict(input_df)[0]

    # Output result
        if prediction == 1:
            st.error("⚠️ There's a high chance of heart disease. Please consult a doctor.")
        else:
            st.success("✅ You're unlikely to have heart disease based on these inputs.")


# Brain Health Prediction Page
if selected == 'Brain Size Normalcy Prediction':
    st.title("Brain Size Normalcy Prediction using ML")
    st.write("Please provide your health details below. We'll help assess your brain size normalcy.")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])

    with col2:
        age_range = st.selectbox("Age Range (1 = 20–40, 2 = >40)", [1, 2])

    with col3:
        head_size = st.text_input("Head Size (cm³)", help="Normal: 2700 – 4500 cm³")

    with col1:
        brain_weight = st.text_input("Brain Weight (grams)", help="Normal: 1000 – 1600 grams")

    with st.expander("ℹ️ Reference: Normal Ranges for Brain Features"):
        st.markdown("""
        - **Head Size**: 2700 – 4500 cm³  
        - **Brain Weight**: 1000 – 1600 grams  
        - **Gender**: 0 = Female, 1 = Male  
        - **Age Range**: 1 = 20–40 years, 2 = >40 years
        """)

    brain_diagnosis = ""

    if st.button("Brain Health Test Result"):
        try:
            head_size_val = float(head_size)
            brain_weight_val = float(brain_weight)

    # Reject unrealistic input values before prediction
            if head_size_val < 1000 or head_size_val > 6000 or brain_weight_val < 500 or brain_weight_val > 2500:
                st.error("❌ Enter realistic head size (1000–6000 cm³) and brain weight (500–2500 grams).")
            else:
                import pandas as pd
                feature_names = ['Gender', 'Age Range', 'Head Size(cm^3)', 'Brain Weight(grams)']
                input_df = pd.DataFrame([[gender, age_range, head_size_val, brain_weight_val]], columns=feature_names)

                brain_prediction = brain_model.predict(input_df)[0]

                if brain_prediction == 1:
                    brain_diagnosis = "⚠️ The person may have **abnormal brain metrics**."
                else:
                    brain_diagnosis = "✅ The person's brain size appears **normal**."

                st.success(brain_diagnosis)

        except ValueError:
            st.error("❌ Please enter valid numerical values.")




        # try:
        #     head_size_val = float(head_size)
        #     brain_weight_val = float(brain_weight)

        #     import pandas as pd
        #     feature_names = ['Gender', 'Age Range', 'Head Size(cm^3)', 'Brain Weight(grams)']
        #     input_df = pd.DataFrame([[gender, age_range, head_size_val, brain_weight_val]], columns=feature_names)

        #     brain_prediction = brain_model.predict(input_df)[0]

        #     if brain_prediction == 1:
        #         brain_diagnosis = "⚠️ The person may have **abnormal brain metrics**."
        #     else:
        #         brain_diagnosis = "✅ The person's brain size appears **normal**."

        #     st.success(brain_diagnosis)

        # except ValueError:
        #     st.error("❌ Please enter valid numerical values.")


# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")
    st.write("Please provide your health details below. We'll help assess your risk for parkinson's disease.")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    with st.expander("ℹ️ View Normal Ranges for Parkinson's Input Features"):
     st.markdown("""
    | Feature | Normal Range (Approximate) | Description |
    |---------|----------------------------|-------------|
    | **MDVP:Fo(Hz)** | 100 – 200 Hz | Average vocal fundamental frequency |
    | **MDVP:Fhi(Hz)** | 150 – 250 Hz | Maximum vocal frequency |
    | **MDVP:Flo(Hz)** | 80 – 160 Hz | Minimum vocal frequency |
    | **MDVP:Jitter(%)** | < 1% | Frequency variation – should be low |
    | **MDVP:Jitter(Abs)** | < 0.001 | Absolute jitter value |
    | **MDVP:RAP** | < 0.005 | Relative average perturbation |
    | **MDVP:PPQ** | < 0.005 | Pitch perturbation quotient |
    | **Jitter:DDP** | < 0.015 | Three-point jitter measurement |
    | **MDVP:Shimmer** | < 0.01 | Amplitude variation – should be low |
    | **MDVP:Shimmer(dB)** | < 0.3 dB | Logarithmic shimmer value |
    | **Shimmer:APQ3** | < 0.005 | Amplitude perturbation quotient (3-point) |
    | **Shimmer:APQ5** | < 0.006 | APQ over 5 cycles |
    | **MDVP:APQ** | < 0.01 | Average amplitude perturbation |
    | **Shimmer:DDA** | < 0.015 | Difference of differences of amplitude |
    | **NHR** | < 0.03 | Noise-to-harmonics ratio (lower is better) |
    | **HNR** | > 20 dB | Harmonics-to-noise ratio (higher is better) |
    | **RPDE** | ~0.4 – 0.6 | Nonlinear dynamic complexity measure |
    | **DFA** | ~0.5 – 0.7 | Signal self-similarity measure |
    | **Spread1** | -7 to -4 | Linear projection spread (voice signal) |
    | **Spread2** | 0 to 0.5 | Orthogonal projection spread |
    | **D2** | 2 – 3 | Correlation dimension (chaos metric) |
    | **PPE** | < 0.2 | Pitch period entropy – voice irregularity |
    """, unsafe_allow_html=True)


    # code for Prediction
    parkinsons_diagnosis = ''

 

    if st.button("Parkinson's Test Result"):
        raw_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                    RAP, PPQ, DDP, Shimmer, Shimmer_dB,
                    APQ3, APQ5, APQ, DDA, NHR, HNR,
                    RPDE, DFA, spread1, spread2, D2, PPE]

        # Step 1: Check for any empty fields
        if any(val.strip() == '' for val in raw_input):
            st.warning("🚫 Please fill in **all** the fields before submitting.")
        else:
            try:
                # Step 2: Convert inputs to float
                user_input = [float(x) for x in raw_input]

                # Step 3: Define validation ranges
                out_of_range = []

                def check_range(value, name, min_val=None, max_val=None):
                    if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                        out_of_range.append(f"{name}: {value}")

                check_range(user_input[0], "MDVP:Fo(Hz)", 100, 200)
                check_range(user_input[1], "MDVP:Fhi(Hz)", 150, 250)
                check_range(user_input[2], "MDVP:Flo(Hz)", 80, 160)
                check_range(user_input[3], "MDVP:Jitter(%)", 0, 1)
                check_range(user_input[4], "MDVP:Jitter(Abs)", 0, 0.001)
                check_range(user_input[5], "MDVP:RAP", 0, 0.005)
                check_range(user_input[6], "MDVP:PPQ", 0, 0.005)
                check_range(user_input[7], "Jitter:DDP", 0, 0.015)
                check_range(user_input[8], "MDVP:Shimmer", 0, 0.01)
                check_range(user_input[9], "MDVP:Shimmer(dB)", 0, 0.3)
                check_range(user_input[10], "Shimmer:APQ3", 0, 0.005)
                check_range(user_input[11], "Shimmer:APQ5", 0, 0.006)
                check_range(user_input[12], "MDVP:APQ", 0, 0.01)
                check_range(user_input[13], "Shimmer:DDA", 0, 0.015)
                check_range(user_input[14], "NHR", 0, 0.03)
                check_range(user_input[15], "HNR", 20, None)  # > 20 dB
                check_range(user_input[16], "RPDE", 0.4, 0.6)
                check_range(user_input[17], "DFA", 0.5, 0.7)
                check_range(user_input[18], "Spread1", -7, -4)
                check_range(user_input[19], "Spread2", 0, 0.5)
                check_range(user_input[20], "D2", 2, 3)
                check_range(user_input[21], "PPE", 0, 0.2)

                # Step 4: Warn if any values are out of range
                if out_of_range:
                    st.warning("⚠️ Some of your values are outside the normal expected range:")
                    for item in out_of_range:
                        st.markdown(f"- **{item}**")

                # Step 5: Make prediction
                parkinsons_prediction = parkinsons_model.predict([user_input])

                if parkinsons_prediction[0] == 1:
                    parkinsons_diagnosis = "⚠️ The person **may have Parkinson's disease**"
                else:
                    parkinsons_diagnosis = "✅ The person is **unlikely to have Parkinson's disease**"

                st.success(parkinsons_diagnosis)

            except ValueError:
                st.error("❌ Please enter valid **numeric values only**.")
