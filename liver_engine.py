# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

#Engine Begins

liver_df= pd.read_csv("indian_liver_patient_preprocessed.csv")

X = liver_df.iloc[:,0:5]  #independent columns
y = liver_df.iloc[:,-1]    #target column i.e Dataset

# Importing modules for machine learning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier


def begin_scan(data):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
    print (X_train.shape)
    print (y_train.shape)
    print (X_test.shape)
    print (y_test.shape)

    # create random_forest object

    random_forest = RandomForestClassifier(max_depth=3,n_estimators=56,criterion='entropy')
    random_forest.fit(X_train, y_train)

    #Predict Output

    rf_predicted = random_forest.predict(X_test)

    random_forest_score = round(random_forest.score(X_train, y_train) * 100, 2)
    random_forest_score_test = round(random_forest.score(X_test, y_test) * 100, 2)
    print('Random Forest Score: \n', random_forest_score)
    print('Random Forest Test Score: \n', random_forest_score_test)
    print('Accuracy: \n', accuracy_score(y_test,rf_predicted))
    print(confusion_matrix(y_test,rf_predicted))
    print(classification_report(y_test,rf_predicted))
    
    predicted = random_forest.predict(data)
    return predicted

import streamlit as st
import time

st.title("Liver Disease Detection System")
st.header("Where hope arrives!")

with st.sidebar:
    st.write("Please Input Patient Parameters")
    add_age = st.text_input('Patient Age')
    add_Total_Bilirubin = st.text_input('Total Bilirubin')
    add_Alkaline_Phosphotase = st.text_input('Alkaline Phosphotase')
    add_Alamine_Aminotransferase = st.text_input('Alamine Aminotransferase')
    Asparate_Aminotransferase = st.text_input('Asparate Aminotransferase')
    clicked = st.button('BEGIN')

input_features_list = [[add_age, add_Total_Bilirubin, add_Alkaline_Phosphotase, add_Alamine_Aminotransferase, Asparate_Aminotransferase]]
#input_features = np.array(input_features_list)
#Convert to df
#input_features = pd.DataFrame(input_features, columns = ['Age', 'Total_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Asparate_Aminotransferase'])

if clicked:
    prediction = begin_scan(input_features_list)
    with st.spinner('Analyzing, Please wait...'):
        time.sleep(5)
    st.success('Analysis Completed')

    #with st.expander("Confidence Metrics"):
     #    st.subheader("Analysis Results:")
      #   st.metric(label="Random Forest Training Score", value=random_forest_score)
       #  st.metric(label="Random Forest Test Score", value=random_forest_score_test)
        # st.metric(label="Accuracy", value=accuracy_score(y_test,rf_predicted))

    st.subheader("Prediction:")
    if prediction[0] == 0:
        st.caption("Positive")
        st.write("Disease Deteced! Further evaluation is advised")
    else:
        st.caption("Negative")
        st.write("No disease detected.")







