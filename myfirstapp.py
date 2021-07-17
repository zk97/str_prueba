import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv("in/income.csv")
st.write(df.columns)

st.image("pig.jpg")

siteHeader = st.beta_container()
with siteHeader:
    st.title("Modelo de Evaluación de Ingresos")
    st.markdown("El blabla *eis* **negritas**")
    st.subheader("subheader")
    st.header("header")
    st.text("\n\n texto\naj")
    st.dataframe(df.iloc[:5])
    
dataViz = st.beta_container()
with dataViz:
    st.subheader("grafica")
    st.text("dist sexo")
    st.area_chart(df.sex.value_counts())
    st.text("texto")
    st.bar_chart(df.age.value_counts())
    st.plotly_chart(px.area(df.sex.value_counts()))

newFeat = st.beta_container()
with newFeat:
    st.subheader("otro")
    st.markdown("otra cosa")
    st.text("texto")
    
optional_cols = ['education-num','marital-status','occupation','relationship']
options = st.multiselect('Variables que se añadirán al modelo:',
     optional_cols)

principal_columns = ['race','sex','workclass','education']
drop_columns = ['income','fnlwgt','capital-gain','capital-loss','native-country','income_bi']

if len(options):
    principal_columns = principal_columns + options
    drop_columns = drop_columns + [x for x in optional_cols if not x in options]
else:
    drop_columns = drop_columns + optional_cols
    
modelTrain = st.beta_container()
with modelTrain:
    st.subheader("otro sub")
    st.text("mas texto")
    
y= df.income_bi
df.drop(drop_columns, axis = 1, inplace = True)
X = pd.get_dummies(df, columns = principal_columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 15)

max_depth = st.slider("Valor maximos", min_value = 1, max_value = 10, value = 7, step = 1)

t = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = max_depth)
model = t.fit(X_train, y_train)

score_train = model.score(X_train, y_train)
score_test = model.score(X_test, y_test)

Perfor = st.beta_container()
with Perfor:
    st.subheader('otro')
    col1, col2 = st.beta_columns(2)
    with col1:
        st.text(f'train:\n\n{round(score_train*100, 2)}')
    with col2:
        st.text('test:')
        st.text(round(score_test*100, 2))