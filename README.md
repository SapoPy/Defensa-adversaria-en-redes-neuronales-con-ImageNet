# Defensa adversaria en redes neuronales con ImageNet

Proyecto semestral del curso **EL4106 – Inteligencia Computacional**, realizado por **Sebastián Cáceres** y **Benjamín Wulf**.

Este repositorio contiene implementaciones de distintos ataques adversarios y defensas aplicados sobre modelos entrenados en **ImageNet**.  

## Ataques implementados

En los archivos `ataqueFGSM.py`, `ataqueRFGSM.py`, `ataquePGD.py`, `ataqueRPGD.py` y `ataquesGaussianos.py` se encuentran las funciones que generan imágenes adversarias.  
Estas funciones siguen el formato:

- `generar_imagen_fgsm(...)`
- `generar_imagen_fgsm_dirigido(...)`
- `generar_imagen_pgd(...)`  
- `generar_imagen_rfgsm(...)`
- `generar_imagen_rpgd(...)`
- `generar_imagen_gaussian(...)`

Para usarlas, se debe entregar:

- El **modelo** a evaluar  
- Las **imágenes** a atacar  
- Los **labels** de dichas imágenes  
- Los **parámetros del ataque** en forma de lista (por ejemplo: `eps`, número de iteraciones, etc.)

## Defensas implementadas

Las defensas se aplican mediante funciones que reciben:

- Las **imágenes** ya cargadas  
- El **parámetro de defensa** correspondiente (por ejemplo, nivel de reducción de bits o calidad JPEG)

Entre las defensas consideradas están:

- **Entrenamiento Adversal**
- **Reducción de profundidad de bits**  
- **Compresión JPEG** con calidad ajustable  

## Notebooks

El repositorio también incluye algunos archivos `.ipynb` utilizados para ejecutar y validar los ataques y defensas en Google Colab, facilitando la reproducción de los experimentos, tambien esta `utils.py` que contiene funciones auxiliares para que los otros archivos
funciones.

