'''
INSURANCE PREDICTOR WEB APPP
'''

## -----------
# Libraries
## -----------

import matplotlib.pyplot as plt
import missingno as ms
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import streamlit as st

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model

from src.constants import (TITLE,
                            OBJECTIVE,
                            IMAGE,
                            NOTE,
                            MISSING_VALUES,
                            SUMMARY_STATISTICS_TITLE,
                            GRAPH,
                            DIS_VAR_DISTR,
                            BOX,
                            CORR,
                            DATA_TRANS,
                            MODEL)


# -------------------
# Page configuration
# -------------------

st.set_page_config(
    page_title='Insurance Predictor',
    page_icon=':bar_chart:',
    layout='wide',
    initial_sidebar_state='expanded')


# -------------------
# UI
# -------------------

st.title(TITLE)
st.write(OBJECTIVE)      
st.image(IMAGE, caption='Insurance')
st.write(NOTE)

# READ DATA-
insurance_df = pd.read_csv("data/insurance.csv")

# First 5 observations
st.write(insurance_df.head())

# Number of observations
rows = insurance_df.shape[0]

# Number of variables
columns = insurance_df.shape[1]

st.write("""

#### Data dimensions
The data source has """ + str(rows) + """ observations and """ + str(columns) + """ variables. 

#### Data description
- ***age***: The age of the client {""" + str(min(insurance_df["age"])) + """, """ + str(max(insurance_df["age"])) + """}. 

- ***sex***: The gender of the client. { There are """ + str(insurance_df[insurance_df["sex"] == "male"]["age"].count()) + """ males and """ + str(insurance_df[insurance_df["sex"] == "female"]["age"].count()) + """ females}. 

- ***bmi***: The bmi of the client {""" + str(min(insurance_df["bmi"])) + """, """ + str(max(insurance_df["bmi"])) + """}. 

- ***children***: The number of children a client has {""" + str(min(insurance_df["children"])) + """, """ + str(max(insurance_df["bmi"])) + """}. 

- ***smoker***: Whether the client smokes or not. { There are """ + str(insurance_df[insurance_df["smoker"] == "yes"]["smoker"].count()) + """ smokers and """ + str(insurance_df[insurance_df["smoker"] == "no"]["smoker"].count()) + """ non-smokers}. 

- ***region***: The region where the client lives at. { There are """ + str(insurance_df[insurance_df["region"] == "southwest"]["region"].count()) + """ from the southwest, """ + str(insurance_df[insurance_df["region"] == "southeast"]["region"].count()) + """ from the southeast, """ + str(insurance_df[insurance_df["region"] == "northwest"]["region"].count()) + """ from the northwest, and """ + str(insurance_df[insurance_df["region"] == "northeast"]["region"].count()) + """ from the northeast}.

- ***charges***: The cost for the insurance {""" + str(round(min(insurance_df["charges"]),0)) + """, """ + str(round(max(insurance_df["charges"]),0)) + """}. 

#### Data exploration

Before creating the model, it is important to fully understand the data that we are working with. One way to do that is by looking into missing values.
""")

# Check if there are any Null values
missing_values_df = insurance_df.isnull().sum().to_frame(name="Missing Values")
st.write(missing_values_df) 
st.write(MISSING_VALUES)

# Display summary statistics
st.write(SUMMARY_STATISTICS_TITLE)
st.dataframe(insurance_df.describe())
st.write(GRAPH)

# Create a grid
col1, col2, col3 = st.columns(3)

# Age distribution
with col1:
    st.write("Age Distribution")
    fig_age = plt.figure()
    plt.hist(insurance_df["age"], bins=20)
    st.pyplot(fig_age)

# BMI distribution
with col2:
    st.write("BMI Distribution")
    fig_bmi = plt.figure()
    plt.hist(insurance_df["bmi"], bins=20)
    st.pyplot(fig_bmi)

# Insurance charges distribution
with col3:
  st.write("Insurance Charges")
  fig_charges = plt.figure()
  plt.hist(insurance_df["charges"], bins=20)
  st.pyplot(fig_charges)

st.write(DIS_VAR_DISTR)

# Create a grid
col1, col2, col3, col4 = st.columns(4)

# Gender distribution
with col1:
    st.write("Gender Distribution")
    gender_counts = insurance_df["sex"].value_counts()
    st.bar_chart(gender_counts)

# Number of children distribution
with col2:
    st.write("Number of Children Distribution")
    children_counts = insurance_df["children"].value_counts()
    st.bar_chart(children_counts)

# Smoking status
with col3:
    st.write("Smoking Status")
    smoker_counts = insurance_df["smoker"].value_counts()
    st.bar_chart(smoker_counts)

# Region distribution
with col4:
    st.write("Region Distribution")
    region_counts = insurance_df["region"].value_counts()
    st.bar_chart(region_counts)

st.write(BOX)

# Select the categorical variables
categorical_columns = ["sex", "smoker", "region"]

# Create a grid for the box plots
col1, col2, col3 = st.columns(3)

# Create box plots for each categorical variable against charges in a grid
for i, column in enumerate(categorical_columns):
    with col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3:
        st.write(f"Plot: {column.capitalize()} vs Charges")
        fig, ax = plt.subplots()
        sns.boxplot(data=insurance_df, x=column, y="charges")
        st.pyplot(fig)

## -----------
## DATA TRANSFORMATION
## -----------

st.write(DATA_TRANS)

# Binary transformation
insurance_df['sex'] = insurance_df['sex'].apply(lambda x: 1 if x == "male" else 0)
insurance_df['smoker'] = insurance_df['smoker'].apply(lambda x: 1 if x == "yes" else 0)

# Multiclass transformation
region_dummies = pd.get_dummies(insurance_df['region'], drop_first = True)
insurance_df=pd.concat([insurance_df, region_dummies], axis=1)
insurance_df.drop(["region"], axis=1, inplace = True)

st.write(insurance_df.head(5))
st.write(CORR)


# -----------------------
# Linear correlation 
# -----------------------

# Calculate the correlation matrix
corr_matrix = insurance_df.corr()

# Create a heatmap of the correlation matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)

# Set the heatmap title
ax.set_title("Correlation Heatmap")

# Display the heatmap
st.pyplot(fig)


#---------------------
# MODEL
#---------------------

st.write(MODEL)


## -----------
## USER INPUT FEATURES
## -----------

with open("models/lasso_model.pkl", 'rb') as file:
    reg = pickle.load(file)

st.sidebar.write("**SELECT USER PARAMETERS**")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 60, 30)
    sex = st.sidebar.selectbox("Sex", ("male", "female"))
    bmi = st.sidebar.slider("BMI", 15, 58, 30)
    children = st.sidebar.slider("Children", 0, 4, 2)
    smoker = st.sidebar.selectbox("Smoker", ("yes", "no"))
    region = st.sidebar.selectbox("Region", ("northwest", "southeast", "southwest", "northeast"))

    data = {"age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children, 
            "smoker": smoker,
            "region": region}

    features = pd.DataFrame(data, index=[0])
                                           
    return features


df = user_input_features()


df['sex'] = df['sex'].apply(lambda x: 0 if x == 'female' else 1)
df['smoker'] = df['smoker'].apply(lambda x: 0 if x == 'no' else 1)

df['northwest'] = df['region'].apply(lambda x: 1 if x == 'northwest' else 0)
df['southeast'] = df['region'].apply(lambda x: 1 if x == 'southeast' else 0)
df['southwest'] = df['region'].apply(lambda x: 1 if x == 'southwest' else 0)

df.drop(["region"], axis=1, inplace = True)

st.sidebar.write("**PREDICTION**")
st.sidebar.write("Given the inputed parameter, the insuracnce charge would be: $" + str(round(reg.predict(df)[0],2)))