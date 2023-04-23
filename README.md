# Clasificador Alzheimer TFG

Te explico a continuacion todo lo que vas a poder encontrar en este repositorio y lo que hara falta para el TFG. 

## Contexto y carperta papers

Vas a encontrar dos papers sobre el algoritmo wav2vec2 y modelo preentrenado XLSR-wav2vec2 y un tercer paper sobre que es lo que el algoritmo aprende en cada capa durante el entrenamiento. En este ultimo es en el que nos vamos a basar para extraer la informacion de alguna capa intermedia y probar los diferentes clasificadores y otro sobre que aprende el algoritmo en cada una de sus capas. 

No pretendo que te leas los papers al detalle porque contiene mucha informacion sobre todo tecnica la idea es que a partir de estos busques informacion y vayas preguntando dudas que te surjan, como por ejemplo que es un algoritmo semi-supervisado, que quiere decir que esta preentrenado, que es son las transformers... lo dicho tampoco hace falta profundizar tecnicamente mucho en esto ya que el foco del estudio que son los clasificadores. 

Vamos a usar el modelo preentrenado XLSR-wav2vec2 que son modelos de lenguaje preentrenados con gran cantidad de datos de habla multilingüe por lo que es capaz de aprender patrones en el habla, es decir, lingüisticos entre diferentes idiomas y con ellos mejorar la calidad de la transcripcion. En concreto, vamos a usar un modelo subido a https://huggingface.co/ (Hugging Face es una comunidad donde la gente pueden compartir modelos del lenguaje preentreados y tambien proporcionan apis para poder trabajar con cierto algoritmos o redes neuronales). Como los audios de Alzheimer son en ingles estoy usando esta inferencia: https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english

## Dataset 

El dataset se ha extraido de [este challenge](https://luzs.gitlab.io/adresso-2021/) que se hizo en 2021 para la conferencia internacional Interspeech. Estos audios se han obtenido de [DementiaBank](https://dementia.talkbank.org/) esta es una base de datos compartida donde hay videos y audios de pacientes con dementia (Alzheimer), en concreto, la pagina del challenge donde se encuentran los datos es: https://dementia.talkbank.org/ADReSS-2021/ . Para poder descargar de esta web los audios/videos hay que pedir un usuario y contraseña. 

En este repositorio vas a encontrar un archivo csv que es el dataframe con que vas a trabajar los clasificadores. Los audios al ser confidenciales y de caracter sensible los comparto contigo por email. 

## Pipeline

1) Procesamiento de los archivos de audio y csv
2) Fine tuning con los datos de Alzheimer
3) Se selecciona previamente de que capa intermedia vamos a extraer los tensores para luego aplicarlos al clasificador. 
4) **Aplicar diferentes algoritmos de clasificacion para pacientes con Alzheimer y control**
5) **Optimizadores de hiperparametros para estos clasificadores**

Puntos 4) y 5) son los tuyos 😉 

## Scripts de python y archivos csv

El mas importante para ti y los clasificadores seria clasificadores.py. El resto los he subido por si quieres tener mas contexto de donde saco los tensores que te pasare para los clasificadores. Te explico brevemente a continuacion: 

-  inferencia.py : hago una llamda al modelo subido a huggingface que comente anteriormente y extraigo los tensores de una capa intermedia especifica. 
-  **clasificador.py: codigo con como leer tensores, splitear el dataset en train y test, clasificador XGBoost y metricas de scoring.**
-  bayesianopt.py : pequeño intento de un optimizador bayesiano sin usar la libreria optuna porque me daba errores en el servidor. En este caso te recomendaria usar la libreria optuna porque no me funciono correctamente. Te lo dejo por si te sirviera de algo. 
-  gridsearch.py : XGboost con un gridsearch para buscar mejores hiperparametros. Tampoco fue muy eficaz por como es el método en si de lento. 


