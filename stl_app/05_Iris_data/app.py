import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



st.title('Iris Classifier')
st.write("This app uses 6 inputs to predict the Variety of Iris using "
         "a model built on the Palmer's Iris's dataset. Use the form below"
         " to get started!")

iris_file = st.file_uploader('Upload your own Iris data')

if iris_file is None:
    rf_pickle = open('stl_app/05_Iris_data/output_iris.pickle', 'rb')
    map_pickle = open('stl_app/05_Iris_data/random_forest_iris.pickle', 'rb')
    iris_df = pd.read_csv('stl_app/05_Iris_data/iris.csv')

    rfc = pickle.load(rf_pickle)
    unique_penguin_mapping = pickle.load(map_pickle)


else:
    iris_df = pd.read_csv(iris_file)
    iris_df = iris_df.dropna()
st.title("Iris ")
st.markdown('สร้าง `scatter plot` แสดงผลข้อมูล **Palmer\'s Penguins** กัน แบบเดียวกับ **Iris dataset**')

choices = ['sepal.length',
           'sepal.width',
           'petal.length',
           'petal.width']

selected_x_var = st.selectbox('เลือก แกน x', (choices))
selected_y_var = st.selectbox('เลือก แกน y', (choices))


st.subheader('ข้อมูลตัวอย่าง')
st.write(iris_df)

st.subheader('แสดงผลข้อมูล')
sns.set_style('darkgrid')
markers = {"Setosa": "v", "Versicolor": "s", "Virginica": 'o'}

fig, ax = plt.subplots()
ax = sns.scatterplot(data=iris_df,
                     x=selected_x_var, y=selected_y_var,
                     hue='variety', markers=markers, style='variety')
plt.xlabel(selected_x_var)
plt.ylabel(selected_y_var)
plt.title("Iris Data")
st.pyplot(fig)