# Clasificador Alzheimer TFG

Te explico a continuacion todo lo que vas a poder encontrar en este repositorio y lo que hara falta para el TFG. 

## Dataset 

El dataset lo vamos a extraer de [este challenge](https://luzs.gitlab.io/adresso-2021/) que se hizo en 2021 para una conferencia internacional llamada Interspeech. Estos audios se han obtenido de [DementiaBank](https://dementia.talkbank.org/) esta es una base de datos compartida donde hay videos y audios de pacientes con dementia (Alzheimer), en concreto, la pagina del challenge donde se encuentran los datos es: https://dementia.talkbank.org/ADReSS-2021/ . Para poder descargar de esta web los audios/videos hay que pedir un usuario y contraseña que te pasare al email. 

## Carperta Papers + explicaciones varias sobre los modelos de ASR

Vas a encontrar dos papers sobre el algoritmo wav2vec2 y modelo preentrenado XLSR-wav2vec2 y un tercer paper sobre que es lo que el algoritmo aprende en cada capa durante el entrenamiento. En este ultimo es en el que nos vamos a basar para extraer la informacion de alguna capa intermedia y probar los diferentes clasificadores y otro sobre que aprende el algoritmo en cada una de sus capas. 

No pretendo que te leas los papers al detalle porque contiene mucha informacion sobre todo tecnica la idea es que a partir de estos busques informacion y vayas preguntando dudas que te surjan, como por ejemplo que es un algoritmo semi-supervisado, que quiere decir que esta preentrenado, que es son las transformers... lo dicho tampoco hace falta profundizar tecnicamente mucho en esto ya que el foco del estudio que son los clasificadores. 

En este caso no vamos a "finetunear" el modelo, ya que tardariamos mucho y no es el fin ahora mismo, por lo que vamos a conger una inferencia subida a https://huggingface.co/ (Hugging Face es una comunidad donde la gente pueden compartir modelos del lenguaje preentreados y tambien proporcionan apis para poder trabajar con cierto algoritmos o redes neuronales). Vamos a trabajar con el modelo preentrenado XLSR-wav2vec2 que son modelos de lenguaje preentrenados con gran cantidad de datos de habla multilingüe por lo que es capaz de aprender patrones en el habla, es decir, lingüisticos entre diferentes idiomas y con ellos mejorar la calidad de la transcripcion. 

Como los audios de Alzheimer son en ingles estoy usando esta inferencia: https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english

## Carperta Papers
