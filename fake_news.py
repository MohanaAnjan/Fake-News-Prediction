import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd 
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



df=pd.read_csv(r"D:\My Projects\fake-news\train.csv")

# -------------------------------This is the configuration page for our Streamlit Application---------------------------
st.set_page_config(
    page_title="Fake News Prediction ",
    layout="wide"
)

# -------------------------------This is the sidebar in a Streamlit application, helps in navigation--------------------
with st.sidebar:
    selected = option_menu("Main Menu", ["About Project", "Predictions"],
                           icons=["house", "gear"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "centre"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "#0072b1"},
                                   "icon": {"font-size": "20px"}
                                   }
                           )
# -----------------------------------------------About Project Section--------------------------------------------------
if selected == "About Project":
    st.markdown("# :blue[Fake News Prediction]")
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("### :blue[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, EDA")
    st.markdown("### :blue[Overview :] The primary goal is to develop a tool that can predict whether a news article is"
                "likely to be fake or misleading. This tool helps users differentiate between credible and"
                  "non-credible news sources, contributing to the fight against misinformation.")
    st.markdown("### :blue[BY :] Mohana Anjan A V ")

if selected == "Predictions":
    st.markdown("# :blue[Predicting Results based on Trained Models]")
    with st.form("form1"):
        authors = df['author'].dropna().astype(str).unique()
        sorted_authors = sorted(authors)

# Use the sorted list in the selectbox
        author = st.selectbox('Select your Author', sorted_authors)
        text=st.text_input('Write Your text here ')
        content=author+" "+text
        # -----Submit Button for PREDICT RESALE PRICE-----
        submit_button = st.form_submit_button(label="Check News State ")
        if submit_button is not None:
            port_stem = PorterStemmer()
            def stemming(contents):
                stemmed_content = re.sub('[^a-zA-Z]',' ',content)
                stemmed_content = stemmed_content.lower()
                stemmed_content = stemmed_content.split()
                stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
                stemmed_content = ' '.join(stemmed_content)
                return stemmed_content
            content=stemming(content)
            with open(r'TfidfVectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            with open(r'model.pkl', 'rb') as f:
                model_loaded  = pickle.load(f)
            new_sample=np.array([content])  
            new_sample = vectorizer.transform(new_sample)
            new_pred = model_loaded.predict(new_sample)[0]
            st.write('## :green[Predicted Fake News OR  Not:] ', new_pred)
        



       
    