import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import tensorflow as tf

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("datasets\\prepared\\cars.csv")

tree = pickle.load(open('models\\TreeRegr.sav', 'rb'))
treeGB = pickle.load(open('models\\TreeRegrGBoost.sav', 'rb'))
neural = tf.keras.models.load_model('models\\NeuralRegr')

st.sidebar.title("Меню")

def get_empty_object():
    return dict(zip(df.columns, np.zeros(df.shape[1])))

def set_object_params(car):
    manufacturers = ["-"] + get_categories("manufacturer_name")
    manufacturer = st.selectbox("Производитель", tuple(manufacturers))
    for i in manufacturers[1:]:
        car['manufacturer_name_' + i] = 0
    if manufacturer != "manufacturer_name_-":
        car['manufacturer_name_' + manufacturer] = 1

    car['year_produced'] = st.number_input("Год выпуска", min_value = 0, step=1)
    car['odometer_value'] = st.number_input("Пробег (в км)", min_value=0, step=1)
    
    engines = ["-", "Бензиновый", "Дизельный", "Электрический"]
    translation_engines = {
        "-" : "-",
        "Бензиновый" : "gasoline",
        "Дизельный" : "diesel",
        "Электрический" : "electric"
    }
    engine = st.selectbox("Тип двигателя", tuple(engines))
    for i in engines[1:]:
        car['engine_type_' + translation_engines[i]] = 0
    if engine != "engine_type_-":
        car["engine_type_" + translation_engines[engine]] = 1
    car['engine_capacity'] = st.number_input("Объём двигателя (в литрах)", min_value=float(0), step=0.1)
    car['engine_has_gas'] = int(st.checkbox("ГБО"))

    drives = ["-", "Передний", "Задний"]
    translation_drives = {
        "-" : "-",
        "Передний" : "front",
        "Задний" : "rear",
    }
    drive = st.selectbox("Привод", tuple(drives))
    for i in drives[1:]:
        car['drivetrain_' + translation_drives[i]] = 0
    if engine != "drivetrain_-":
        car["drivetrain_" + translation_drives[drive]] = 1


    gbs = ["-", "Механическая", "Автоматическая"]
    translation_gbs = {
        "-" : "-",
        "Механическая" : "mechanical",
        "Автоматическая" : "automatic"
    }
    gb = st.selectbox("Тип трансмиссии", tuple(gbs))
    for i in gbs[1:]:
        car['transmission_' + translation_gbs[i]] = 0
    if gb != "transmission_-":
        car["transmission_" + translation_gbs[gb]] = 1

    car['has_warranty'] = int(st.checkbox("Авто на гарантии"))
    car['is_exchangeable'] = int(st.checkbox("Рассматриваю обмен на другое авто"))

    states = ["-", "Новый", "Подержаный", "Битый"]
    translation_st = {
        "-" : "-",
        "Новый" : "new",
        "Подержаный" : "owned",
        "Битый" : "emergency"
    }
    state = st.selectbox("Состояние авто", tuple(states))
    for i in states[1:]:
        car['state_' + translation_st[i]] = 0
    if state != "state_-":
        car["state_" + translation_st[state]] = 1

    was_listed = st.checkbox("Объявение ранее выкладывалось")
    if was_listed:
        car['duration_listed'] = st.number_input("Сколько дней провисело объявление", min_value=0, step=1)
    else:
        car['duration_listed'] = 0

    car['number_of_photos'] = int(df['number_of_photos'].mean())

    bodies = ["-", "Кабриолет", "Купе", "Хэтчбек", "Лифтбэк", "Лимузин", "Микроавтобус", "Минивэн", "Пикап", "Седан", "SUV", "Универсал", "Фургон"]
    translation_bodies = {
        "-" : "-",
        "Кабриолет" : "cabriolet",
        "Купе" : "coupe",
        "Хэтчбек" : "hatchback",
        "Лифтбэк" : "liftback", 
        "Лимузин" : "limousine",
        "Микроавтобус" : "minibus",
        "Минивэн" : "minivan",
        "Пикап" : "pickup",
        "Седан" : "sedan",
        "SUV" : "suv",
        "Универсал" : "universal",
        "Фургон" : "van"
    }
    body = st.selectbox("Тип кузова", tuple(bodies))
    for i in bodies[1:]:
        car['body_type_' + translation_bodies[i]] = 0
    if body != "body_type_-":
        car["body_type_" + translation_bodies[body]] = 1


def display_prediction(car, model):
    if st.button("Показать цену"):
        if len(car) != df.shape[1]:
            st.write("Корректно укажите все параметры авто")
        else:
            st.write(model.predict([list(car.values())[1:]]))

def get_categories(cat):
    result = []
    for i in df.columns:
        if cat in i:
            result.append(i.replace(cat + "_", ""))
    return result
    

selectedModel = st.selectbox(
    "Выберите модель",
    ("-", "Дерево решений", "Дерево решений с градиентным бустингом", "Нейронная сеть")
)

if selectedModel == "-":
    st.write()
elif selectedModel == "Дерево решений":
    st.write("Дерево решений")
    current_car = get_empty_object()
    set_object_params(current_car)
    display_prediction(current_car, tree)
elif selectedModel == "Дерево решений с градиентным бустингом":
    st.write("Дерево решений с градиентным бустингом")
    current_car = get_empty_object()
    set_object_params(current_car)
    display_prediction(current_car, treeGB)
else:
    st.write("Нейронная сеть")
    current_car = get_empty_object()
    set_object_params(current_car)
    display_prediction(current_car, neural)
