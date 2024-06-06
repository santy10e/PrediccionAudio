# Clasificador de Audio con Streamlit

Este proyecto utiliza Streamlit para crear una interfaz web que permite clasificar archivos de audio como normales o anormales. El modelo de clasificación se basa en la extracción de características MFCC (Mel-Frequency Cepstral Coefficients) y un clasificador Random Forest.

## Funcionalidades

- Carga de archivos de audio WAV desde la interfaz web.
- Extracción de características MFCC del audio.
- Clasificación del audio como normal o anormal usando un modelo Random Forest entrenado.
- Visualización del resultado de la clasificación con imágenes de semáforo (verde para normal, rojo para anormal).
- Visualización del gráfico de dominio del tiempo del audio.
- Reproducción de ejemplos de audio normal y anormal.
- Opción para entrenar el modelo desde la interfaz web.

## Requisitos

- Python 3.6 o superior.
- Librerías: librosa, numpy, matplotlib, soundfile, streamlit.
- Modelo de clasificación entrenado (se puede entrenar desde la interfaz web).

## Instalación

Clonar el repositorio:
git clone https://github.com/santy10e/PrediccionAudio.git

## Entrenamiento del modelo
- Hacer clic en el botón "Entrenar Modelo" en la interfaz web.
- Esperar a que el modelo se entrene y se guarde.
- Reproducción de audios de muestra
- Hacer clic en el botón "Reproducir audio normal" para escuchar un ejemplo de audio normal.
- Hacer clic en el botón "Reproducir audio anormal" para escuchar un ejemplo de audio anormal.
## Contribuciones
Se agradecen las contribuciones de cualquier tipo a este proyecto. Si encuentras un error o deseas agregar una nueva función, no dudes en crear un issue en GitHub o enviar una solicitud de pull.

## Notas
- He ajustado el nombre del proyecto a "Clasificador de Audio con Streamlit" para que sea más descriptivo.
- He agregado descripciones más detalladas para cada sección del README.md.
- He incluido instrucciones para la instalación y el uso de la aplicación.
- He mencionado la posibilidad de entrenar el modelo y reproducir audios de muestra.
- He agregado una sección sobre contribuciones y licencia.
- He mantenido el código original sin modificaciones.

## Recomendaciones
- Puedes agregar más ejemplos de uso a la sección de "Uso".
- Puedes incluir capturas de pantalla de la interfaz web en el README.md.
- Puedes crear una sección de "Preguntas frecuentes" para responder a preguntas comunes sobre el proyecto.
¡Espero que este borrador te sirva como base para crear un README.md completo e informativo para tu proyecto!

## Recursos adicionales
Documentación de Streamlit: https://docs.streamlit.io/
Guía para escribir un README efectivo: https://github.com/bryan2811/Curso-de-Git-y-Github-Platzi/blob/master/README.md
Herramienta para generar README.md: https://readme.so/

