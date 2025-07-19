# 🕹️ Super Mario Bros AI con PPO (Proximal Policy Optimization)


## 🎮 Super Mario RL Demo

| 🟥 Sin Reward shaping | 🟩 Con Reward shaping |
|---------------|----------------|
| <img src="videos/demo_no_reward.gif" width="300"/> | <img src="videos/demo_reward.gif" width="300"/> |

Este proyecto entrena un agente de IA a traves de RL (Reinforcement Learning) para jugar al nivel 1-1 de *Super Mario Bros* usando el algoritmo de **Proximal Policy Optimization (PPO)**.

He utilizado las siguientes librerias:
  - gym
  - gym_super_mario_bros
  - nes_py
  - stable_baselines3

---

## Descripción

Para este proyecto, decidi utilizar una CNN, mi objetivo era que la IA pudiese aprender a jugar el primer nivel de Mario, aprendiese las mecanicas, y mas adelante pudiese intentar otros niveles.

Para el preprocesado de mi Env realicé lo siguiente:
  - Introduje un espacio de acciones restringido, los modelos están entrenados con RIGHT_ONLY, pero se podría cambiar por SIMPLE_MOVEMENT y entrenar un nuevo modelo.
  - Un Skip de 4 frames, esto limita la cantidad de acciones que puede realizar, mejorando su ratio de aprendizaje.
  - Rewards Wrapper que puede ser desactivado. Con este activado, el modelo llega a la meta de manera estable en 1 millón de timesteps, si se desactiva, comienza a estabilizase a los 4-5 millones.
  - Posteriormente, convierto la imagen a escala de grises, esto permite ahorrarnos los datos de los canales de color, ya que el color no proporciona información en este caso.
  - Reduzco las dimensiones de la imagen de 240x256 a 84x84. Esto reduce los datos a procesar, la medida es la usada de manera estandar en proyectos con Stable_Baselines3.
  - Aplico un VecMonitor para poder obtener estadísticas de como progresa mi entrenamiento en el tensorboard.
  - Para poder dar un contexto de movimiento a mi modelo, stackeo 4 frames.
  - Por ultimo aplico un transpose para el cambio de HWC -> CHW, necesario en redes CNN.


Gracias a este preprocesamiento, el tamaño de los datos a procesar se reduce considerablemente, teniendo mi modelo final unicamente 22MB de peso.

## Reward Wrapper

Este wrapper proporciona recompensas continuas al moverse hacia la derecha dando un mayor feedback a nuestro agente. Además aplico un reward de 100 puntos por alcanzar la meta.
La penalizacion es un -50 por morir.
Utilizo un max_steps que por defecto es de 20.000 y no debería entrar en acción, no obstante, he observado que el agente se queda atascado en varios puntos, esto acabaría la simulación de manera temprana cambiando el valor a unos 1000 timesteps en caso de ser necesario.

La recompensa intrínseca esta delimitada de -15, 15 y solo incluye una penalización de -15 por morir y un bonus por moverse hacia la derecha.

## Callbacks

Utilizo varios callbacks que posteriormente unifico.

EvalCallBack -> Este es mi callback de evaluación, evalua el rendimiento de mi modelo de manera determinista y guarda los datos en tensorboard. En este mismo callback
utilizo otro callback -> StopTrainingOnNoModelImprovement que finaliza el entrenamiento en caso de que mi modelo no continúe mejorando tras 15 evaluaciones.

CheckpointCallback -> Este callback guarda el estado de mi modelo cada 100.000 timesteps.


## Hyperparametros

Respecto a los hyperparametros, tras muchas pruebas, lo que me ha funcionado ha sido reducir el minibatch, utilizando:
  - n_steps = 512
  - batch_size = 16
  - n_epochs = 16

La formula es: n_minibatches = num_envs x n_steps / batch_size.

Esto da como resultado que cada epoch procesa 512 minibatches de  16 muestras, dando un total de 5120 actualizaciones de gradiente.

Anteriormente estuve usando minibatches de mucho mayor tamaño, lo que provocaba que mi modelo no tuviese suficiente muestreo de datos para conseguir un entrenamiento efectivo.

Otro dato importante ha sido reducir el gamma a 0.9. Esto da mayor importancia al reward por moverse hacia la derecha, ya que no tengo un reward final por lograr la bandera en el reward base.
Anteriormente estuve probando con 0.99 y 0.995, sin conseguir ningún resultado.



# Uso del modelo
Dentro de la carpeta models/best, se pueden encontrar los dos modelos, para inicializarlos se puede utilizar el script test.py cambiando la ruta de cargado del modelo.

En el caso de querer entrenar tu propio modelo, puedes usar train.py
