import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("<h1 style='text-align: center; color: grey;'>مدل تشخیص گل زنبق</h1>", unsafe_allow_html=True)
st.sidebar.header('پارامتر هارا وارد کنید')

def user_input_features():
    sepal_length = st.sidebar.slider('طول کاسبرگ', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('عرض کاسبرگ', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('طول گلبرگ', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('عرض گلبرگ', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index = [0])
    return features

df = user_input_features()

st.subheader("<h3 style='text-align: center; color: grey;'>\پارامتر های وارد شده</h3>", unsafe_allow_html=True)
st.write(df)

iris = datasets.load_iris()
x = iris.data
y = iris.target

model = RandomForestClassifier()
model.fit(x, y)

prediction = model.predict(df)
prediction_proba = model.predict_proba(df)

st.subheader("<h3 style='text-align: center; color: grey;'>\نوع گل قابل تشخیص</h3>", unsafe_allow_html=True)
st.write(iris.target_names)

st.subheader("<h3 style='text-align: center; color: grey;'>\تشخیص نهایی</h3>", unsafe_allow_html=True)
st.write(iris.target_names[prediction])

st.subheader("<h3 style='text-align: center; color: grey;'>\احتمال پیش بینی هرکدام، بر اساس این پارامترها</h3>", unsafe_allow_html=True)
st.write(prediction_proba)


