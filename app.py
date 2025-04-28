import joblib
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from datetime import datetime

st.title("🔮 Predicción de Categoría de Crimen en SF")
m = folium.Map(location=[37.77, -122.42], zoom_start=12)
m.add_child(folium.LatLngPopup())
st.subheader("Haz clic en el mapa para seleccionar un lugar:")
map_data = st_folium(m, height=500, width=800)
st.subheader("Configura los datos del incidente:")


#model = joblib.load("crime_model.pkl")
col1, col2 = st.columns(2)
with col1:
    fecha = st.date_input("Fecha del incidente:", value=datetime.today())
with col2:
    hora = st.slider("Hora del día:", min_value=0, max_value=23, value=12)


st.subheader("Selecciona el Modelo de Predicción:")
opcion_modelo = st.selectbox(
    "Modelos disponibles:",
    ("Árbol de Decisión", "Regresión Logística", "Comparar ambos modelos")
)


if st.button("🚨 Predecir categoría de crimen"):
    modelo_arbol = joblib.load("modelo_arbol_2.pkl")
    model_distrito = joblib.load("modelo_distrito.pkl")
    modelo_rl=joblib.load("modelo_rl.pkl")

    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    day = fecha.day
    month = fecha.month
    year = fecha.year
    day_of_week = fecha.strftime('%A')  # Ej: 'Friday'
    
    data=   pd.DataFrame({
        'X': [lon],
        'Y': [lat]})
    distrito=model_distrito.predict(data)
    st.subheader("Distrito seleccionado: ",distrito[0])
    
    X = pd.DataFrame({
        'X': [lon],
        'Y': [lat],
        'Hour': [hora],
        'Day': [day],
        'Month': [month],
        'Year': [year],
        'DayOfWeek': [day_of_week],
        'PdDistrict': [distrito[0]]})
    
    if opcion_modelo == "Árbol de Decisión":
        y_proba = modelo_arbol.predict_proba(X)
        clases = modelo_arbol.classes_

        resultados = pd.DataFrame({
            "Categoría": clases,
            "Probabilidad": y_proba[0]
        }).sort_values("Probabilidad", ascending=False).reset_index(drop=True)

        st.subheader("🌳 Resultados de Árbol de Decisión")
        st.dataframe(resultados.style.format({"Probabilidad": "{:.2%}"}))

    elif opcion_modelo == "Regresión Logística":
        y_proba = modelo_rl.predict_proba(X)
        clases = modelo_rl.classes_

        resultados = pd.DataFrame({
            "Categoría": clases,
            "Probabilidad": y_proba[0]
        }).sort_values("Probabilidad", ascending=False).reset_index(drop=True)

        st.subheader("📈 Resultados de Regresión Logística")
        st.dataframe(resultados.style.format({"Probabilidad": "{:.2%}"}))
    
    elif opcion_modelo == "Comparar ambos modelos":
        # Predicción Árbol
        y_proba_arbol = modelo_arbol.predict_proba(X)
        resultados_arbol = pd.DataFrame({
            "Categoría": modelo_arbol.classes_,
            "Probabilidad": y_proba_arbol[0]
        }).sort_values("Probabilidad", ascending=False).reset_index(drop=True)

        # Predicción Regresión Logística
        y_proba_rl = modelo_rl.predict_proba(X)
        resultados_rl = pd.DataFrame({
            "Categoría": modelo_rl.classes_,
            "Probabilidad": y_proba_rl[0]
        }).sort_values("Probabilidad", ascending=False).reset_index(drop=True)

        # Mostrar en columnas
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🌳 Árbol de Decisión")
            st.dataframe(resultados_arbol.style.format({"Probabilidad": "{:.2%}"}))

        with col2:
            st.subheader("📈 Regresión Logística")
            st.dataframe(resultados_rl.style.format({"Probabilidad": "{:.2%}"}))


