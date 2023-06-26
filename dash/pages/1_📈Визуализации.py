import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv("datasets\\prepared\\cars_full.csv")

st.sidebar.title("Меню")


st.markdown(
    """
    # Визуализация зависимостей в наборе данных📈
    - Представление исходного набора данных в виде графиков и диаграмм помогает увидеть, как одни параметры зависят от других 👀
    """
)
st.markdown(
    """
    ### Год и стоимость + пробег авто📅
    - Непосредственно главный фактор формирования стоимости автомобиля -- это его пробег. Влиять может также и год выпуска автомобиля, но в общем случае год и пробег довольно сильно коррелируют. Попробуем в этом убедиться
    """
)

# ВИЗУАЛИЗАЦИЯ №1
fig, axes = plt.subplots(2, 1, figsize=(7, 8))

fig.tight_layout(pad=1.6)
axes[0].plot(df.groupby('year_produced').mean()['price_usd'])
axes[0].set_xlabel("Год")
axes[0].set_ylabel("Средняя стоимость в USD")
axes[0].grid(color=(0, 0, 0), alpha=0.25)


axes[1].plot(df.groupby('year_produced').mean()['odometer_value'])
axes[1].set_xlabel("Год")
axes[1].set_ylabel("Средний пробег в тыс.км.")
axes[1].grid(color=(0, 0, 0), alpha=0.2)
st.pyplot(fig)

df_mini = df.sample(frac=1).head(200)
x = df_mini['price_usd']
y = df_mini['odometer_value']

fig, axes = plt.subplots(1, 1)

plt.xlabel("Цена в USD")
plt.ylabel("Пробег")
axes.grid(color=(0, 0, 0), alpha=0.2)

axes.scatter(x, y)
st.pyplot(fig)

st.markdown(
    """
    ### Год и ширина ассортимента🛒
    - Посмотрим, как много автомобилей определённого года представлено на рынке
    """
)

fig, axes = plt.subplots(1, 1)

plt.xlabel("Год")
plt.ylabel("Число авто на рынке")

axes.plot(df.groupby('year_produced').count()['odometer_value'])
axes.grid(color=(0, 0, 0), alpha=0.2)
st.pyplot(fig)


# ВИЗУАЛИЗАЦИЯ №2


st.markdown(
    """
    ### Ценовой сегмент и тип трансмиссии⚙️
    - На визуализациях ниже можно увидеть распределение автомобилей на рынке по типу трансмиссии в ценовых сегментах выше и ниже 12.000$
    """
)

fig, axes = plt.subplots(1, 1)

plt.ylabel("Число предложений на рынке")
plt.xlabel("Тип трансмиссии")

print()
print([df[df['transmission_mechanical'] == 1].shape[0], df[df['transmission_automatic'] == 1].shape[0]])

axes.bar(["Механическая", "Автоматическая"], [df[df['transmission_mechanical'] == 1].shape[0], 
                                              df[df['transmission_automatic'] == 1].shape[0]],
         color=["green", "orange"])
st.pyplot(fig)

# ВИЗУАЛИЗАЦИЯ 2.1 (3)

fig, axes = plt.subplots(1, 2)


df1 = df[(df['price_usd'].astype(int) > 12000) & df['transmission_mechanical'] == 1]
df2 = df[(df['price_usd'].astype(int) > 12000) & df['transmission_automatic'] == 1]

counts = [df1.shape[0], df2.shape[0]]
axes[0].pie(counts, labels=['Механическая', 'Автоматическая'])
axes[0].set_title("Дороже 12.000 USD")

df1 = df[(df['price_usd'].astype(int) <= 12000) & df['transmission_mechanical'] == 1]
df2 = df[(df['price_usd'].astype(int) <= 12000) & df['transmission_automatic'] == 1]

axes[1].set_title("Дешевле 12.000 USD")
counts = [df1.shape[0], df2.shape[0]]
axes[1].pie(counts, labels=['Механическая', 'Автоматическая'], startangle=90)

st.pyplot(fig)