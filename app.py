import  numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
import streamlit as st


#load titanic dataset 
@st.cache_data
def load_data():
    data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

    return data

data  = load_data()
#Title and descrpitions 
st.title('Exploratory Data analysis  of Titanic dataset')
st.write('This is an EDA on the titanic dataset')
st.markdown("<h2 style='color:green;'>Exploring the Data</h2>", unsafe_allow_html=True)
st.write('First few lines of dataset')
st.dataframe(data.head())


# data Cleaning  Sections 
st.subheader('Missing Value')
missing_data = data.isnull().sum()
st.write(missing_data)


if st.checkbox('Fill missing age with medain'):
    data['Age'].fillna(data['Age'].median(),inplace=True)

if st.checkbox('Filling missing value using mode '):
    data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)

if st.checkbox('Drop Duplicate'):
    data.drop_duplicates(inplace=True)
st.subheader("Cleaned data")
st.dataframe(data.head())


#EDA Sections 
st.subheader('Statistical Summary of the Data ')
st.write(data.describe())
st.write(data.columns)

# Age Distributions 
st.subheader("Age Distributions")
fig , ax = plt.subplots()
sns.histplot(data['Age'],kde=True,ax=ax)
ax.set_title("Age Distributions ")

st.pyplot(fig)


# Gender Distributions 
st.subheader('Gender Distributions')
fig, ax = plt.subplots()
sns.countplot(x='Sex',data=data,ax=ax)
ax.set_title('Gender Distributions')
st.pyplot(fig)

#Pclass VS Survived 

st.subheader('Places Vs Survived ')
fig , ax=plt.subplots()
sns.countplot(x='Pclass',hue='Survived',data=data,ax=ax)
ax.set_title('Pclass Vs Srvived')
st.pyplot(fig)


# Feature Engineerings 
st.subheader('Feature Engineering : Family Size ')
data['FamilySize'] =data['SibSp'] + data['Parch']
fig , ax=plt.subplots()
sns.histplot(data['FamilySize'],kde=True,ax=ax)
ax.set_title('Family Size Distributions ')
st.pyplot(fig)



#Conculsions 
st.subheader('Key Insights')
insights = """
- Females have a higher survival rate than males.
- Passengers in 1st class had the highest survival rate.
- The majority of passengers are in Pclass 3.
- Younger passengers tended to survive more often.
"""
st.write(insights)
