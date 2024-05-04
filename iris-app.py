import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import time

def show_page():
    st.write("<h1 style='text-align: center; color: blue;'>تشخیص نوع گل با هوش مصنوعی</h1>", unsafe_allow_html=True)
    st.write("<h4 style='text-align: center; color: gray;'>Robo-Ai.ir طراحی شده توسط</h4>", unsafe_allow_html=True)
    st.link_button("Robo-Ai بازگشت به", "https://robo-ai.ir")
    
    with st.sidebar:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')
        with col2:
            st.image('img.png')
        with col3:
            st.write(' ')
        
        st.divider()
        st.write("<h4 style='text-align: right; color: gray;'>تشخیص نوع گل بر اساس ابعاد با دقت 98 درصد</h>", unsafe_allow_html=True)
        st.write("<h4 style='text-align: right; color: gray;'>ساخته شده با مختصات 150 شاخه گل زنبق</h>", unsafe_allow_html=True)
        st.divider()
        st.write('Developed & Designed by')
        st.write('Hamidreza Bahrami')

    def user_input_features():
        sepal_length = st.slider('طول کاسبرگ', 4.3, 7.9, 5.4)
        sepal_width = st.slider('عرض کاسبرگ', 2.0, 4.4, 3.4)
        petal_length = st.slider('طول گلبرگ', 1.0, 6.9, 1.3)
        petal_width = st.slider('عرض گلبرگ', 0.1, 2.5, 0.2)
        data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
        features = pd.DataFrame(data, index = [0])
        return features

    df = user_input_features()

    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    model = RandomForestClassifier()
    model.fit(x, y)

    ibutton = st.button('شناسایی')
    if ibutton:
        with st.chat_message("assistant"):
            with st.spinner('''درحال شناسایی'''):
                time.sleep(2)
                st.success(u'\u2713''شناسایی انجام شد')
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)
        if prediction == 0:
            st.write("<h4 style='text-align: right; color: gray;'>بر اساس تحلیل من نوع گل زنبق وارد شده ، ستوسا است</h4>", unsafe_allow_html=True)
            st.write("<h4 style='text-align: left; color: gray;'>Based on my analysis, this dimension belong to Setosa</h4>", unsafe_allow_html=True)
            st.image('setosa.jpg')

        elif prediction == 1:
            st.write("<h4 style='text-align: right; color: gray;'>بر اساس تحلیل من نوع گل زنبق وارد شده ، گل ورسیکالر است</h4>", unsafe_allow_html=True)
            st.write("<h4 style='text-align: left; color: gray;'>Based on my analysis, this dimension belong to Versicolor</h4>", unsafe_allow_html=True)
            st.image('versicolor.jpeg')

        else:
            st.write("<h4 style='text-align: right; color: gray;'>بر اساس تحلیل من نوع گل زنبق وارد شده ، گل ویرجینیکا است</h4>", unsafe_allow_html=True)
            st.write("<h4 style='text-align: left; color: gray;'>Based on my analysis, this dimension belong to Virginica</h4>", unsafe_allow_html=True)
            st.image('virginica.jpg')

show_page()
