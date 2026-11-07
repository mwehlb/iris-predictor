import os
import streamlit as st
import pandas as pd
from joblib import load

# folder = the folder where this file lives, so the paths work anywhere
folder = os.path.dirname(os.path.abspath(__file__))

model = load(os.path.join(folder, "data", "iris_model.joblib"))

st.title("IRIS FLOWER PREDICTOR APP")
st.write("Enter the flower measurements in centimeters, then click Predict.")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=15.0, value=3.0, step=0.1)

with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=15.0, value=1.5, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=15.0, value=0.2, step=0.1)

# Average size of each flower type (from the Iris dataset)
# Order: sepal length, sepal width, petal length, petal width
average_setosa = [5.01, 3.43, 1.46, 0.25]
average_versicolor = [5.94, 2.77, 4.26, 1.33]
average_virginica = [6.59, 2.97, 5.55, 2.03]

c1, c2, c3 = st.columns(3)
with c2:
    predict_button = st.button("Predict")

if predict_button:
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

    # 0: Iris Setosa, 1: Iris Versicolour, 2: Iris Virginica
    if prediction[0] == 0:
        flower = "Setosa"
        image_file = os.path.join(folder, "images", "iris_setosa.png")
    elif prediction[0] == 1:
        flower = "Versicolour"
        image_file = os.path.join(folder, "images", "iris_versicolor.png")
    else:
        flower = "Virginica"
        image_file = os.path.join(folder, "images", "iris_virginica.png")

    st.subheader("Flower predicted: " + flower)

    d1, d2, d3 = st.columns(3)
    with d2:
        st.image(image_file)

    # Visualization: your flower compared with the average of each type
    st.subheader("Your flower compared with the average of each type")
    chart_data = pd.DataFrame(
        {
            "Your flower": [sepal_length, sepal_width, petal_length, petal_width],
            "Setosa": average_setosa,
            "Versicolour": average_versicolor,
            "Virginica": average_virginica,
        },
        index=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
    )
    st.bar_chart(chart_data)
