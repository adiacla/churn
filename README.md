# churn
Deserción de clientes
#INTRODICCION

"""Este es un ejemplo de cómo realizar una pagina Web usando streamlite
para desplegar un modelo de Machine Learning después de haber sido entrenado 


#Crear proyecto en Visual Studio Code

Cree una carpeta y descargue todos los assests o recursos que va usar en la aplicacion
imagenes, modelo.bin, archivos de datos, entre otros

./nombre_proyecto
                 /modelo.bin
                 /proyecto.py
                 /imagen.jpg
                 /datos.csv
                 
 Los datos pueden estar en remoto y llamarlo en python con pandas.               
 
#Bibliotecas requeridas para ejecutar el Deployment (Despiliegue a producción)

En este archivo no requieres tener instalado en python todas lasbibliotecas utilizadas
en el entrenamiento, solo las básicas para la página web

#Caso especial scikit-learn

Nota: antes de ejecutar cualquier código verifique con qué versión scikit-learn entrenó el modelo
y con qué versión va a desplegar el mismo para el usuario final.

Recomiento ir al enviroment (ambiente) donde lo entrenó. Si lo hizo con Google Colab cree un nuevo notebook
y ejecute el siguiente código:
!pip show scikit-learn

Y haga lo mismo en el enviroment de su máquina local, es decir abra el terminal del VSC, <en menú View/Terminal> o con CTRL ñ
y ejecute 
pip show scikit-learn

Una vez verificado que son la misma Versión, podrá ejecutar el desplieqgue, de lo contrario no podrá usar el modelo en producción.

Si requiere actualizar recomiendo actualice Google Colab y vuelva a entrenar, y descargue los modelos salvados con joblib.
!pip install --upgrade scikit-learn

Si requiere instalar la última versión (cuando no esté instalada en su pc)
# pip install -U scikit-learn


#Otras bliotecas básicas

Estas son algunas bibliotecas básicas, las instalas desde el Terminal- CTRL ñ en VSC (Visual Studio Code) 
pip install joblib
pip install matplotlib
pip install seaborn
pip freeze


# Crear un Repositorio en Github 

Antes de iniciar a escribir puede hacer la gestión de versiones con Github el proyecto, o siga los pasos para crear el
Repositorio básico.

Vaya a la opcion Create a new repository,
Digiete el nombre Repository name, por ejemplo churn
y haga un check a la casilla de Add a README file

En esta carpeta vas a copiar todos los archivos de la carpeta del proyecto


#Ejecutar el nombre_proyecto.py 


Cuando ingrese a Terminal de VSC, verifique que está ubicado en la carpeta del proyeco de lo contrario cambie
el directorio como se muestra en el ejemplo:

prompt> cd c:/Users/adiaz/Documents/churn/Churn.py

Si está ubicado en la carpeta del proyecto podrá ejecutar el servidor web de streamlit.
Nota: El RUN del VSC NO lanza el servidor web, si usan el RUN puede ir validando la sintaxis y el debbug pero no lanza el servidor.

Use el siguiente comando en cualquier shell o en Terminal de VSC para ejecutar y lanzar la página.

prompt>streamlit run nombre_proyecto.py

Tan pronto como ejecute el script como se muestra arriba, un servidor web Streamlit local se lanza y 
la aplicación se abrirá en una nueva pestaña en tu navegador web predeterminado. 

"""
