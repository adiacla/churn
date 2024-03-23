#pip install -U scikit-learn
# pip install joblib
# pip install matplotlib
# pip install seaborn
# pip freeze


import streamlit as st

#importar las bibliotecas tradicionales de numpy y pandas
import numpy as np
import pandas as pd

#importar las biliotecas graficas e imágenescd 
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sn

#importar libreria de paralelizacion de modelos
import joblib as jb

#importar la libreria del modelo seleccionada
#from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import ExtraTreesClassifier


imagen_inicial = Image.open("churn.JPG") 



## Iniciar barra lateral en la página web y título e ícono

st.set_page_config(
  page_title="Predcción de deserción de clientes de la compañía Alfredo's Retail",
  page_icon="cliente.ico",
  initial_sidebar_state='auto'
  )



@st.cache_resource
def load_models():
  modeloNB=jb.load('modeloNB.bin')
  modeloArbol=jb.load('ModeloArbol.bin')
  modeloBosque=jb.load('ModeloBosque.bin')

  return modeloNB,modeloArbol,modeloBosque


modeloNB,modeloArbol,modeloBosque= load_models()


#Primer contenedor

with st.container():
  st.subheader("Modelo Machine Learning para predecir la deserción de clientes")
  st.title("Aplicación de predicción")
  st.write("Realizado por Alfredo Díaz Claros:wave:")
  st.write("""

**Introducción** 
cliente de nuestros sueños es el que permanece fiel a la empresa, comprando siempre sus productos o servicios. Sin embargo, en la realidad, 
los clientes a veces deciden alejarse de la empresa para probar o empezar a comprar otros productos o servicios y esto puede ocurrir en 
cualquier fase del customer journey. Sin embargo, existen varias medidas para prevenir o gestionar mejor esta circunstancia. Por eso lo mejor
es tener una herramienta predictiva que nos indique el estado futuro de dichos clientes usando inteligencia artificial, tomar las acciones 
de retenció necesaria. Constituye pues esta aplicación una herramienta importante para la gestión del marketing.

Los datos fueron tomados con la Información de la base de datos CRM de la empresa ubicada en Bucaramanfa,donde se
preparó 3 modelos de machine Learnig para predecir la deserció de clientes, tanto actuales como nuevos.

Datos Actualizados en la fuente: 20 de Marzo del 2024


Se utilizó modelos supervidados de clasificacion  tanto Naive Bayes, Arboles de decisión y Bosques Aleatorios 
entendiendo que hay otras técnicas, es el resultado de la aplicacion practico del curso de inteligencia artificial en estos modelos
revisado en clase. Aunqe la aplicación final sería un solo modelo, aqui se muestran los tres modelos para 
comparar los resultados.

 """)
  
  

#Otro contenedor y lo partimos en derecha e izquierda

with st.container():
  st.write("Nuevo contenedor")
  left_column, right_column = st.columns(2)
  with left_column:
    st.subheader("Las librerias usadas para entrenar")
    st.write(
      """
      El objetivo de este trabajo acadeémico es construir una herramienta en código Python para predecir la la deserción y requiere de las
      siguientes caracteristicas parra predecir:
      
      'COMP', 'PROM', 'COMINT', 'COMPPRES', 'RATE', 'DIASSINQ', 'TASARET', 'NUMQ', 'RETRE'.
      El modelo elegido sería Naive Bayes, pero vamos a predecir en los tres modelos solo para efectos de comparación.
      
Dentro del universo de la analítica predictiva, existen numerosos modelos que, basados en Inteligencia Artificial, ayudan a las organizaciones a dar un paso más allá en su Data Journey y resolver sus problemas de negocio de manera efectiva.
Los modelos predictivos más conocidos (y potentes) son los de regresión y clasificación:

Los modelos de regresión nos permiten predecir un valor. Por ejemplo, cuál es el beneficio estimado que obtendremos de un determinado cliente (o segmento) en los próximos meses o nos ayudan a estimar el forecast de ventas.
Los modelos de clasificación en cambio nos permiten predecir la pertenencia a una clase.Por ejemplo, clasificar entre nuestros clientes quiénes son más propensos a una compra, a un abandono o a un fraude.
Y dentro de estos últimos, encontramos el modelo predictivo tipo churn: aquel que te ofrece información sobre qué clientes tienen más probabilidad de abandonarte. ¿Cómo funciona? Este modelo combina una serie de variables con datos históricos de tus clientes junto con datos de la situación actual. Los resultados son binarios: obtendremos un sí o un no (en forma de 0 y 1) en base a su grado de probabilidad de abandono.

Como todos los modelos predictivos, será indispensable ir reentrenándolo con nuevos datos conforme vaya pasando el tiempo para que no pierda fiabilidad y evitar que quede desactualizado.

Pese a que el modelo churn en sí mismo es valioso, en Keyrus trabajamos combinando diferentes casos de uso que nos ayuden a crear esa visión 360º del cliente tan buscada y deseada por todas las compañías, como podrían ser la propensión a la compra o el análisis del carrito de la compra, entre otros.

Este tipo de modelos para predecir la propensión al abandono te aportarán beneficios como:

Activar acciones de marketing más efectivas al conocer qué grupo de clientes es susceptible de dejar de comprarte.

Aumentar el CLTV de tus clientes, lo que se traduce en una reducción el CAC y una mayor rentabilidad al contar con esos clientes durante más tiempo.

Potenciar el branding de tu compañía al conseguir tener clientes más fieles, e incluso, transformarlos de manera natural en embajadores de tu marca.

Conocer más y mejor a tus clientes, lo que se traducirá en iterar la estrategia de cara a ser cada vez más customer-centric.

Tomar decisiones más estratégicas de cara a optimizar procesos y campañas.

      """
    )

  with right_column:
      st.subheader("Librerías usadas")
      code = '''
      # 
      import pandas as pd 
      import numpy as np 
      mport csv
      import matplotlib.pyplot as plt
      import seaborn as sns
      
      import joblib as jb
      
      import warnings, requests, zipfile, io
      warnings.simplefilter('ignore')
      from scipy.io import arff
      
      #importo los modelosfrom sklearn.naive_bayes import GaussianNB
      from sklearn.model_selection import train_test_split
      
      #importo las métricas
      
      from sklearn.metrics import confusion_matrix
      from sklearn.metrics import ConfusionMatrixDisplay
      from sklearn.metrics import classification_report
      from sklearn.metrics import roc_curve,auc
      
      #Librerias de validacion cruzada
      from sklearn.model_selection import KFold
      from sklearn.model_selection import cross_val_score
      from sklearn.model_selection import cross_val_predict
      
      import ipywidgets as widgets
      #arboles
      from sklearn import tree
      from sklearn.tree import plot_tree

      #bosque
      from sklearn.ensemble import RandomForestClassifier

     '''
      st.code(code, language="python", line_numbers=True)



modeloA=['Naive Bayes', 'Arbol de Decisión', 'Bosque Aleatorio']

churn = {1 : 'Se retira', 0 : 'No se Retira' }



logo=Image.open("churn.JPG")
st.sidebar.write('')
st.sidebar.image(logo, width=100)
st.sidebar.header('Seleccione los datos de entrada')


def seleccionar(modeloL):

    #Filtrar por el modelo

  st.sidebar.subheader('Selector de Modelo')
  modeloS=st.sidebar.selectbox("Modelo",modeloL)

  #Filtrar por COMP
  st.sidebar.subheader('Seleccione la compRa')
  COMPS=st.sidebar.slider("Seleccion",4000,12000,100)
  
  #Filtrar por PROM
  st.sidebar.subheader('Selector del PROM')
  PROMS=st.sidebar.slider("Seleccion",   0.7, 9.0,.5)
  
  #Filtrar por COMINT
  st.sidebar.subheader('Selector de COMINT')
  COMINTS=st.sidebar.slider("Seleccione",8000,24000,100)
  
  #Filtrar por COMPPRES
  st.sidebar.subheader('Selector de COMPPRES') 
  COMPPRESS=st.sidebar.slider('Seleccione', 13000,57000,100)
  
  #Filtrar por RATE
  st.sidebar.subheader('Selector de RATE')
  RATES=st.sidebar.slider("Seleccione",0.5,4.0,0.1)

  #Filtrar por DIASSINQ
  st.sidebar.subheader('Selector de DIASSINQ')
  DIASSINQS=st.sidebar.slider("Seleccione", 270,1800,10)
  
    #Filtrar por TASARET
  st.sidebar.subheader('Selector de TASARET')
  TASARETS=st.sidebar.slider("Seleccione",0.3,1.9,.5)
  
    #Filtrar por NUMQ
  st.sidebar.subheader('Selector de NUMQ')
  NUMQS=st.sidebar.slider("Seleccione",3.0,10.0,0.5)
  
    #Filtrar por departamento
  st.sidebar.subheader('Selector de RETRE')
  RETRES=st.sidebar.slider("Seleccione",3.3,35.0,.5)

  
  return modeloS,COMPS, PROMS, COMINTS ,COMPPRESS, RATES, DIASSINQS,TASARETS, NUMQS, RETRES


modelo,COMP, PROM, COMINT ,COMPPRES, RATE, DIASSINQ,TASARET, NUMQ, RETRE=seleccionar(modeloA)


with st.container():
  st.subheader("Predición")
  st.title("Predicción de Churn")
  st.write("""
           El siguiente es el pronóstico de la deserción usanDo el modelo
           """)

  st.write(modelo)
  st.write("Se han seleccionado los siguientes parámetros:")
  lista=[[COMP, PROM, COMINT ,COMPPRES, RATE, DIASSINQ,TASARET, NUMQ, RETRE]]
  X_predecir=pd.DataFrame(lista,columns=['COMP', 'PROM', 'COMINT', 'COMPPRES', 'RATE', 'DIASSINQ','TASARET', 'NUMQ', 'RETRE'])
  st.write(X_predecir)
  if modelo=='Naive Bayes':
      y_predict=modeloNB.predict(X_predecir)
      probabilidad=modeloNB.predict_proba(X_predecir)
  elif modelo=='Arbol de Decisión':
      y_predict=modeloArbol.predict(X_predecir)
      probabilidad=modeloNB.predict_proba(X_predecir)
  else :
      y_predict=modeloBosque.predict(X_predecir)
      probabilidad=modeloNB.predict_proba(X_predecir)
    
  
  prediccion= '<p style="font-family:sans-serif; color:Green; font-size: 42px;">La predicción es</p>'
  st.markdown(prediccion, unsafe_allow_html=True)
  prediccion='Resultado: '+ str(y_predict[0])+ "    - en conclusion el cliente  "+churn[y_predict[0]]
  st.header(prediccion+':sunglasses:')