# Clasificador Alzheimer TFG

Te explicaré a continuación todo lo que podrás encontrar en este repositorio y lo que necesitarás para el TFG.

## Contexto y carperta papers

Encontrarás dos papers sobre el algoritmo wav2vec2 y el modelo preentrenado XLSR-wav2vec2, y un tercer paper que explica lo que el algoritmo aprende en cada capa durante el entrenamiento. Nos basaremos en este último para extraer información de alguna capa intermedia y probar diferentes clasificadores. También hay un paper sobre lo que aprende el algoritmo en cada una de sus capas.

No pretendo que leas los papers al detalle, ya que contienen mucha información técnica. La idea es que a partir de ellos busques información y preguntes cualquier duda que te surja. Por ejemplo, puedes preguntar qué significa que un algoritmo sea semi-supervisado, qué es un modelo preentrenado, qué son las transformers, etc. Tampoco hace falta profundizar mucho en la parte técnica, ya que el enfoque del estudio son los clasificadores.

Utilizaremos el modelo preentrenado XLSR-wav2vec2, que es un modelo de lenguaje preentrenado con una gran cantidad de datos de habla multilingüe. Es capaz de aprender patrones lingüísticos entre diferentes idiomas y mejorar la calidad de la transcripción. En concreto, usaremos un modelo subido a https://huggingface.co/ (Hugging Face es una comunidad en la que las personas pueden compartir modelos de lenguaje preentrenados y también proporcionan APIs para trabajar con ciertos algoritmos o redes neuronales). Dado que los audios de Alzheimer están en inglés, utilizaremos esta inferencia: https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english.

## Dataset 

El dataset se ha extraído de [este challenge](https://luzs.gitlab.io/adresso-2021/) que se llevó a cabo en 2021 para la conferencia internacional Interspeech. Los audios se han obtenido de [DementiaBank](https://dementia.talkbank.org/) una base de datos compartida que contiene videos y audios de pacientes con demencia (Alzheimer). En concreto, los datos se encuentran en la página del challenge: https://dementia.talkbank.org/ADReSS-2021/ . Para descargar los audios/videos de esta web, es necesario solicitar un usuario y contraseña.

En este repositorio, encontrarás un archivo CSV que es el dataframe con el que trabajarás en los clasificadores. Debido a que los audios son confidenciales y de carácter sensible, los compartiré contigo por email.

## Pipeline

1) Procesamiento de los archivos de audio y csv
2) Fine-tuning con los datos de Alzheimer
3) Se selecciona previamente de que capa intermedia vamos a extraer los tensores para luego aplicarlos al clasificador. 
4) **Aplicar diferentes algoritmos de clasificacion para pacientes con Alzheimer y control**
5) **Optimizadores de hiperparametros para estos clasificadores**

Puntos 4) y 5) son los tuyos 😉 

## Scripts de python y archivo csv

El más importante para ti es el script clasificadores.py. El resto los he subido por si quieres tener más contexto de donde saco los tensores que te pasaré para los clasificadores. Te explico brevemente a continuación: 

-  inferencia.py : hago una llamada al modelo subido a Hugging Face que comenté anteriormente y extraigo los tensores de una capa intermedia específica.
-  **clasificador.py: código con cómo leer tensores, dividir el conjunto de datos en entrenamiento y prueba, clasificador XGBoost y métricas de puntuación**
-  bayesianopt.py : pequeño intento de un optimizador bayesiano sin usar la biblioteca Optuna porque me daba errores en el servidor. En este caso, te recomendaría usar la biblioteca Optuna porque no me funcionó correctamente. Te lo dejo por si te sirviera de algo.
-  gridsearch.py : XGBoost con un GridSearch para buscar mejores hiperparámetros. Tampoco fue muy eficaz por cómo es el método en sí de lento.

- df_filter.csv: DataFrame donde viene toda la información.


## Propuestas de clasificadores

Esto es orientativo, para nada hay que hacerlos todos y se pueden proponer otros que veamos mejores o más interesantes. Seguramente, para una clasificación binaria, muchos no sean los mejores, iremos analizándolo.

- XGBoost
- LightBoost
- CatBoost
- AdaBoost
- Logistic Regression
- SVM
- Red neuronal de 2 salidas
- Random forest 
