# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 21:17:46 2024

@author: bhavy
"""

import numpy as np
import pickle 
import streamlit as st

loaded_model = pickle.load(open("D:/SEM 5 INTERNSHIP/api_model/iris_model.sav",'rb'))

def iris_pred(input_data):

    input_data_np = np.asarray(input_data)
    
    input_reshape = input_data_np.reshape(1,-1)
    
    pred = loaded_model.predict(input_reshape)
    
    if pred[0] == 0:
        return 'Species : Iris-setosa'
    elif pred[0] == 1:
        return 'Species : Iris-versicolor'
    else:
        return 'Species : Iris-virginica'
  
def main():
    st.title('Species Prediction Web App')
    
    Id = st.text_input('Enter Id:')
    SepalLengthCm = st.text_input('Enter Sepal Length:')
    SepalWidthCm = st.text_input('Enter Sepal Width:')
    PetalLengthCm = st.text_input('Enter Petal Length:')
    PetalWidthCm = st.text_input('Enter Petal Width:')
    
    result = ''
    
    if st.button('Type of Species'):
        result = iris_pred([int(Id),float(SepalLengthCm),float(SepalWidthCm),float(PetalLengthCm),float(PetalWidthCm)])
    st.success(result)
    
if __name__ == '__main__':
    main()