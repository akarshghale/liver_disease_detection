# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

"""#Engine Begins"""

liver_df= pd.read_csv("liver_dataset.csv")

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

    return rf_predicted

import streamlit as st

st.title("Liver Disease Detection System")
st.write("##")

with st.sidebar:
    st.write("##Please Input Patient Parameters")
    add_age = st.text_input('Patient Age')
    add_Total_Bilibubin =st.text_input('Total Bilibubin')
    add_Alkaline_Photophase = st.text_input('Alkaline Photophase')
    Alamine_Aminotransferase = st.slider('Alamine Aminotransferase', 0, 100)
    Asparate_Aminotransferase = st.slider('Asparate Aminotransferase', 0, 100)
    clicked = st.button('BEGIN')

input_features_list = [add_age, add_Total_Bilibubin, add_Alkaline_Photophase, Alamine_Aminotransferase, Asparate_Aminotransferase]
input_features = numpy.array(input_features_list)

if clicked:
    prediction = begin_scan(input_features)





