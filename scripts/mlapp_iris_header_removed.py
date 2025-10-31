from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd
from PIL import Image
import os

# Load this iris dataset as your dataframe 
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Save as csv in data folder only if the file doesn't already exist
csv_path = 'data/iris.csv'
if not os.path.exists(csv_path):
    df.to_csv(csv_path, index=False)

st.markdown("""
    <style>
        /* Remove top padding and menu space */
        div.block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# 🌸 App Title and Header
# ----------------------------
st.write("""
## Iris Flower Type Prediction App
This app predicts the **Iris Flower** type using a Machine Learning model trained on a dataset.
""")

# ----------------------------
# Sidebar - User Input
# ----------------------------
st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.4)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 1.3)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

    # ✅ Match column names exactly to those in the CSV
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    return pd.DataFrame(data, index=[0])

DF2 = user_input_features()

# ----------------------------
# Load Dataset and Train Model
# ----------------------------
# ✅ Load your dataset (use the exact file and column names)
df = pd.read_csv('data/iris.csv')

# ✅ Match column names from dataset
feature_columns = [
    'sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)'
]

X = df[feature_columns].copy()
y = df['target'].copy()

# ✅ Train model explicitly with feature names
model = RandomForestClassifier(random_state=42)
model.fit(pd.DataFrame(X, columns=feature_columns), y)

# ----------------------------
# Prediction
# ----------------------------
# Ensure DF2 uses same column order
DF2 = DF2[feature_columns]

prediction = model.predict(DF2)
prediction_proba = model.predict_proba(DF2)

# ----------------------------
# Results
# ----------------------------
st.subheader('Prediction')
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
predicted_flower = species_map[prediction[0]]
st.write(f"###### Predicted Flower: **{predicted_flower.title()}**")

# Display the corresponding predicted flower image
flower_image_path = f"images/{predicted_flower}.jpg"
st.image(flower_image_path, width=250)

# ----------------------------
# Prediction Probability Table
# ----------------------------
st.subheader('Prediction Probability')
prob_df = pd.DataFrame(prediction_proba, columns=['setosa', 'versicolor', 'virginica'])
# st.write(prob_df) # Option1:index is shown by default in st.dataframe
# Option 2: Remove index using to_html
st.markdown(prob_df.to_html(index=False), unsafe_allow_html=True)

# ----------------------------
# User Input Parameters
# ----------------------------
st.subheader('User Input Parameters')
# st.write(DF2) 
st.markdown(DF2.to_html(index=False), unsafe_allow_html=True)

# ----------------------------
# Description Section
# ----------------------------

st.subheader('How This App Works')
st.markdown("""
            Welcome to my *Iris flower detective lab*! Just slide the sepal and petal measurements on the left sidebar, and watch my **Random Forest Classifier** spring into action. It’s a small creation, it works, and I am excited to share it with you!
                       
          You’ll get the **predicted flower species**, the **probabilities for each class**, and even the **matching flower image**. Every click sparks a little bit of magic, and this is only the beginning — there’s so much more I’m eager to build. Jump in, have fun, and celebrate these tiny victories with me! 🚀
            
            """)

st.subheader('Iris Flower Types From this Dataset')
# Display iris flower types image
st.image('images/iris_flowers.jpg', width=425)