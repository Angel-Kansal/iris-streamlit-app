import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC

# Load the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
target_names = iris.target_names

# Train the model
X = df.drop('target', axis=1)
y = df['target']
model = SVC(kernel='linear')
model.fit(X, y)

# Streamlit UI
st.title("🌸 Iris Flower Classifier")
st.write("Select flower features and predict the species:")

# User input sliders
sepal_length = st.slider("Sepal Length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.slider("Sepal Width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.slider("Petal Length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.slider("Petal Width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

# Predict button
if st.button("Predict"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.success(f"The predicted Iris species is: **{target_names[prediction[0]]}**")
