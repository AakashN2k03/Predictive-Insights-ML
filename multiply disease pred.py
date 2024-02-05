# -*- coding: utf-8 -*-  "C:\Users\AAKASH\Desktop\2diseases\multiply disease pred.py"

"""
Created on Mon Sep 11 09:50:57 2023

@author: AAKASH
"""

import pickle
import streamlit as st
import numpy as np

from streamlit_option_menu import option_menu

jamboree_mod=pickle.load(open('C:/Users/AAKASH/Desktop/2diseases/models/jamboree.sav','rb'))
sugar_mod=pickle.load(open('C:/Users/AAKASH/Desktop/2diseases/models/sugar.sav','rb'))

with st.sidebar:
    selected=option_menu('LINEAR and CLASSIFICATION PROJECT WITH ML',
                         ['jamboree probability prediction',
                          'diagnosis prediction for diabetes'],
                         icons=['book-half','h-circle'],
                         default_index=0)

if selected == 'jamboree probability prediction':
    st.title('Jamboree probability prediction using ml')
    scaler = pickle.load(open('C:/Users/AAKASH/Desktop/2diseases/models/scaler.pkl', 'rb'))
    
    
    st.markdown("<b>Jamboree is an organization used to select the university in abroad by the marks obtained by the students and provide consultancy", unsafe_allow_html=True)

    serial = st.text_input("Serial No")
    gre = st.text_input('GRE Score')
    toefl = st.text_input('TOEFL Score')
    university_rating = st.text_input('University Rating')
    sop = st.text_input('Statement of Purpose (SOP)')
    lor = st.text_input('Letter of Recommendation (LOR)')
    cgpa = st.text_input('CGPA')
    Research = st.text_input('Research')
    
    if st.button('Predict Admission Probability'):
        try:
            # Convert input values to the appropriate data types
            serial = float(serial)
            gre = float(gre)
            toefl = float(toefl)
            university_rating = float(university_rating)
            sop = float(sop)
            lor = float(lor)
            cgpa = float(cgpa)
            Research = float(Research)
    
            # Standardize the input data using the same scaler from training
            input_data = np.array([[serial, gre, toefl, university_rating, sop, lor, cgpa, Research]])
            input_data = scaler.transform(input_data)  # Use the pre-fitted scaler
    
            probability = jamboree_mod.predict(input_data)[0]
    
            st.success(f'Predicted Admission Probability: {probability:.2f}')
        except ValueError as e:
            st.error(f"Error: {e}. Please check your input values.")
            
    st.markdown("<b>Note:</b> <i>Result below 50% may have some deviation (range:1 -10)</i>", unsafe_allow_html=True)

            
            
            

if selected == 'diagnosis prediction for diabetes':
    st.title('DIAGNOSIS PREDICTION FOR DIABETES USING ML')
    age = st.text_input('Age', key='age')
    gender = st.text_input("Gender(male = 1 or female = 0)", key='gender')
   
    BMI = st.text_input('BMI', key='BMI')
    Blood_Pressure = st.text_input('Blood Pressure  (high=0 or low=1 or normal=2)', key='blood_pressure')
    FBS = st.text_input('FBS', key='FBS')
    HbA1c = st.text_input('HbA1c', key='HbA1c')
    history = st.text_input('History of diabetes ( yes=1 or no=0)', key='history')
    smoking = st.text_input('Smoking ( yes=1 and no=0)', key='smoking')
    diet = st.text_input('Diet ( healthy=0 and poor=1)', key='diet')
    exercise = st.text_input('Exercise (regular=1 or no=0)', key='exercise')
    
    pred = ''
    if st.button("Result"):
        # Convert input values to the appropriate data types (as mentioned in the previous response)
        # Make sure you handle exceptions and errors properly
        try:
            age = float(age)
            gender = int(gender)
            BMI = float(BMI)
            Blood_Pressure = float(Blood_Pressure)
            FBS = float(FBS)
            HbA1c = float(HbA1c)
            history = int(history)
            smoking = int(smoking)
            diet = int(diet)
            exercise = int(exercise)

            # Make a prediction using your loaded model (sugar_mod)
            # Ensure your model expects a specific format of input features
            result = sugar_mod.predict([[age, gender, BMI, Blood_Pressure, FBS, HbA1c, history, smoking, diet, exercise]])

            if result[0] == 0:
                pred = 'Diagnosis not needed'
            else:
                pred = 'Diagnosis is needed'
        except ValueError as e:
            st.error(f"Error: {e}. Please check your input values.")
    
    st.success(pred)
            












    
    
