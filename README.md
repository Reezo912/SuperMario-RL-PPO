# 🕹️ Super Mario Bros AI con PPO (Proximal Policy Optimization)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-1.6.0-green.svg)](https://stable-baselines3.readthedocs.io/)
[![Gym](https://img.shields.io/badge/Gym-0.21.0-orange.svg)](https://gym.openai.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

## 📋 Tabla de Contenidos

- [🎮 Super Mario RL Demo](#-super-mario-rl-demo)
- [🚀 Instalación](#-instalación)
- [⚡ Uso Rápido](#-uso-rápido)
- [📖 Descripción Técnica](#-descripción-técnica)
- [🏆 Reward Wrapper](#-reward-wrapper)
- [📊 Callbacks](#-callbacks)
- [⚙️ Hyperparámetros](#️-hyperparámetros)
- [📚 Referencias](#-referencias)

---

## 🎮 Super Mario RL Demo

Este proyecto entrena un agente de IA a través de RL (Reinforcement Learning) para jugar al nivel 1-1 de *Super Mario Bros* usando el algoritmo de **Proximal Policy Optimization (PPO)**.

He utilizado las siguientes librerías:
  - gym
  - gym_super_mario_bros
  - nes_py
  - stable_baselines3

---

## 🚀 Instalación

### 📋 Requisitos Previos

- **Python 3.8+**
- **CUDA 11.8+** (opcional, para aceleración GPU)
- **8GB+ RAM** (recomendado)
- **Git**

### 🔧 Pasos de Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/SuperMarioIA.git
   cd SuperMarioIA
   ```

2. **Crear entorno virtual (recomendado):**
   ```bash
   python -m venv mario_env
   source mario_env/bin/activate  # En Windows: mario_env\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verificar instalación:**
   ```bash
   python -c "import gym_super_mario_bros; print('✅ Instalación exitosa')"
   ```

### 🐛 Solución de Problemas

**Error con nes-py:**
```bash
pip install nes-py==8.2.1 --no-deps
pip install -r requirements.txt
```

**Error con CUDA:**
```bash
# Instalar versión CPU de PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## ⚡ Uso Rápido

### 🎮 Probar Modelo Pre-entrenado

1. **Ejecutar demo:**
   ```bash
   cd src
   python Test.py
   ```

2. **Usar Jupyter Notebook:**
   ```bash
   jupyter notebook Test.ipynb
   ```

### 🏋️ Entrenar Nuevo Modelo

1. **Entrenamiento básico:**
   ```bash
   cd src
   python Train.py
   ```

2. **Continuar entrenamiento existente:**
   ```bash
   python Train.py --resume ../models/checkpoints/mario_ppo_4700000_steps.zip
   ```

### 📊 Monitorear Entrenamiento

```bash
tensorboard --logdir ../logs
```

---

## 📖 Descripción Técnica

Para este proyecto, decidí utilizar una CNN, mi objetivo era que la IA pudiese aprender a jugar el primer nivel de Mario, aprendiese las mecánicas, y más adelante pudiese intentar otros niveles.

Para el preprocesado de mi Env realicé lo siguiente:
  - Introduje un espacio de acciones restringido, los modelos están entrenados con RIGHT_ONLY, pero se podría cambiar por SIMPLE_MOVEMENT y entrenar un nuevo modelo.
  - Un Skip de 4 frames, esto limita la cantidad de acciones que puede realizar, mejorando su ratio de aprendizaje.
  - Rewards Wrapper que puede ser desactivado. Con este activado, el modelo llega a la meta de manera estable en 1 millón de timesteps, si se desactiva, comienza a estabilizarse a los 4-5 millones.
  - Posteriormente, convierto la imagen a escala de grises, esto permite ahorrarnos los datos de los canales de color, ya que el color no proporciona información en este caso.
  - Reduzco las dimensiones de la imagen de 240x256 a 84x84. Esto reduce los datos a procesar, la medida es la usada de manera estándar en proyectos con Stable_Baselines3.
  - Aplico un VecMonitor para poder obtener estadísticas de cómo progresa mi entrenamiento en el tensorboard.
  - Para poder dar un contexto de movimiento a mi modelo, stackeo 4 frames.
  - Por último aplico un transpose para el cambio de HWC -> CHW, necesario en redes CNN.

Gracias a este preprocesamiento, el tamaño de los datos a procesar se reduce considerablemente, teniendo mi modelo final únicamente 22MB de peso.

---

## 🏆 Reward Wrapper

Este wrapper proporciona recompensas continuas al moverse hacia la derecha dando un mayor feedback a nuestro agente. Además aplico un reward de 100 puntos por alcanzar la meta.
La penalización es un -50 por morir.
Utilizo un max_steps que por defecto es de 20.000 y no debería entrar en acción, no obstante, he observado que el agente se queda atascado en varios puntos, esto acabaría la simulación de manera temprana cambiando el valor a unos 1000 timesteps en caso de ser necesario.

La recompensa intrínseca está delimitada de -15, 15 y solo incluye una penalización de -15 por morir y un bonus por moverse hacia la derecha.

| 🟥 Sin Reward shaping | 🟩 Con Reward shaping |
|---------------|----------------|
| <img src="videos/demo_no_reward.gif" width="300"/> | <img src="videos/demo_reward.gif" width="300"/> |

---

## 📊 Callbacks

Utilizo varios callbacks que posteriormente unifico.

**EvalCallBack** -> Este es mi callback de evaluación, evalúa el rendimiento de mi modelo de manera determinista y guarda los datos en tensorboard. En este mismo callback
utilizo otro callback -> **StopTrainingOnNoModelImprovement** que finaliza el entrenamiento en caso de que mi modelo no continúe mejorando tras 15 evaluaciones.

**CheckpointCallback** -> Este callback guarda el estado de mi modelo cada 100.000 timesteps.

---

## ⚙️ Hyperparámetros

Respecto a los hyperparámetros, tras muchas pruebas, lo que me ha funcionado ha sido reducir el minibatch, utilizando:
  - n_steps = 512
  - batch_size = 16
  - n_epochs = 16

La fórmula es: n_minibatches = num_envs x n_steps / batch_size.

Esto da como resultado que cada epoch procesa 512 minibatches de 16 muestras, dando un total de 5120 actualizaciones de gradiente.

Anteriormente estuve usando minibatches de mucho mayor tamaño, lo que provocaba que mi modelo no tuviese suficiente muestreo de datos para conseguir un entrenamiento efectivo.

Otro dato importante ha sido reducir el gamma a 0.9. Esto da mayor importancia al reward por moverse hacia la derecha, ya que no tengo un reward final por lograr la bandera en el reward base.
Anteriormente estuve probando con 0.99 y 0.995, sin conseguir ningún resultado.

---

## 🎯 Uso del Modelo

Dentro de la carpeta `models/best`, se pueden encontrar los dos modelos, para inicializarlos se puede utilizar el script `test.py` cambiando la ruta de cargado del modelo.

En el caso de querer entrenar tu propio modelo, puedes usar `train.py`.

---

## 📚 Referencias

### 📖 Papers
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization
- [Atari DQN](https://arxiv.org/abs/1312.5602) - Deep Q-Networks

### 🔗 Librerías
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gym Super Mario Bros](https://github.com/Kautenja/gym-super-mario-bros)
- [NES-Py](https://github.com/Kautenja/nes-py)

### 🎮 Recursos Adicionales
- [OpenAI Gym](https://gym.openai.com/)
- [PyTorch](https://pytorch.org/)
- [TensorBoard](https://www.tensorflow.org/tensorboard)

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

---

## 🙏 Agradecimientos

- OpenAI por Gym
- Kautenja por gym-super-mario-bros
- Stable-Baselines3 por la implementación de PPO
- La comunidad de RL por el conocimiento compartido

---

**⭐ Si este proyecto te ha sido útil, ¡dale una estrella al repositorio!**
