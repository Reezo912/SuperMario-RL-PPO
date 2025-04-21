from __future__ import annotations

import os
from typing import Callable, Optional

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY

from gym.wrappers import GrayScaleObservation, ResizeObservation, TimeLimit
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
    VecMonitor,
)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    CheckpointCallback,
    CallbackList,
    )
from stable_baselines3.common.utils import constant_fn


# Clase heredada de los Wrapper de la libreria gym
class MarioRewardWrapper(gym.Wrapper):
    '''Clase para hacer tweaks a mi funcion de recompensa,
    esto facilita el aprendizaje de mi modelo

    .. factor:: reward por movimiento hacia la derecha

    .. goal_bonus:: recompensa por bandera
    
    .. death_penalty:: penalizacion por muerte
    
    .. max_steps:: numero de pasos maximo antes de cortar la simulacion.

    '''
    
# -----------------------------------------------------------------------------
# 1. Clase Rewards Wrapper
# -----------------------------------------------------------------------------

    def __init__(self, env, factor, goal_bonus=100, death_penalty=-50, max_steps=20_000):
        super().__init__(TimeLimit(env, max_episode_steps=max_steps)) # llamo a constructor de la clase padre e introduzco el limite de pasos
        self.factor = factor
        self.goal_bonus = goal_bonus
        self.death_penalty = death_penalty
        self._prev_x = 0    # marco la posicion inicial por cada frame, para posteriormente calcular si ha avanzado y dar recompensa.
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._prev_x = 0
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        x = info.get("x_pos", 0)
        # shaped reward por distancia
        shaped = self.factor * (x - self._prev_x)
        self._prev_x = x
        reward += shaped
        # bonus por bandera
        if info.get("flag_get", False):
            reward += self.goal_bonus
        # penalizacion por muerte sin llegar a bandera
        if done and not info.get("flag_get", False):
            reward += self.death_penalty
        return obs, reward, done, info
    
# ----------------------------------------------------------------------------
# 2. Funcion de creacion de entorno.
# ----------------------------------------------------------------------------

ENV_ID = "SuperMarioBros-1-1-v0"

# Esta función devuelve otra función (_init), que al ejecutarse crea un entorno desde cero.
# Es necesario hacerlo así porque SubprocVecEnv espera una lista de funciones, no entornos ya creados.

def make_env(rank: int = 0) -> Callable[[], gym.Env]:  # devuelve una funcion sin argumentos[], rank por defecto en 0.
    ''' Esta función devuelve otra función (_init), que al ejecutarse crea un entorno desde cero.
    
    Es necesario hacerlo así porque SubprocVecEnv espera una lista de funciones, no entornos ya creados

    .. clases_internas:: gym_super_mario_bros.make, JoypadSpace, MaxAndSkipEnv, MarioRewardWrapper, GrayScaleObservation,ResizeObservation

    .. seed_base:: 42

    .. rank::  se suma a la seed base para crear variaciones de entornos
    '''
    
    def _init() -> gym.Env: # esta funcion devuelve una funcion de entorno.
        env = gym_super_mario_bros.make(ENV_ID)
        env = JoypadSpace(env, RIGHT_ONLY)      # se puede cambiar a SIMPLE_MOVEMENT para un espacio de accion mas amplio.
        env = MaxAndSkipEnv(env, skip=4)
        env = MarioRewardWrapper(env, factor=0.03, goal_bonus=100, death_penalty=-50, max_steps=10_000)     # reward wrapper
        env = GrayScaleObservation(env, keep_dim=True)      # keep_dim=true para conservar dimensiones y evitar errores
        env = ResizeObservation(env, (84, 84))
        env.seed(42 + rank)
        env.action_space.seed(42 + rank)
        return env
    return _init


# ----------------------------------------------------------------------------
# 3. Settings entrenamiento y Callbacks 
# ----------------------------------------------------------------------------

LOG_DIR        = "../logs/"                  # save tensorboard
BEST_DIR       = "../models/best/"           
CHECKPOINT_DIR = "../models/checkpoints/"

TOTAL_STEPS    = 10_000_000
PHASE_1        = 8_000_000           # hasta aqui entrenamiento principal TOTAL_STEPS - PHASE_1 = refinado del modelo
NUM_ENVS       = 16                  # numero de procesos en paralelo. Debe ser menor que el numero de procesadores logicos del pc

# frecuencias en pasos *de entorno*, se dividen entre NUM_ENVS para que corresponda con la frecuencia de paso deseada
EVAL_FREQ      = 100_000 // NUM_ENVS
CHECKPOINT_FREQ = 100_000 // NUM_ENVS   # 12_500

def train(resume_path: Optional[str] = None) -> None:
    '''
    Funcion de entrenamiento, puede recibir un string que le indique la ruta donde se encuentra un modelo
    esto reanudará el entrenamiento.
    
    En caso de recibir None. Iniciara un nuevo entrenamiento

    Recordar cambiar reset_num_timesteps de estado en model.learn dependiendo de lo que se desee.
    '''
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)]) # donde i es rank, +1 por cada NUM_ENVS, esto hace que tengan diferente semilla cada uno de los procesos, util para exploracion
    env = VecMonitor(env) # crea estadisticas de eval y rollout para tensorboard
    env = VecFrameStack(env, 4, channels_order='last') # stack de 4 frames para contexto de movimiento
    env = VecTransposeImage(env) # cambio del orden de HWC → CHW, necesario para CNN

    eval_env = DummyVecEnv([make_env(999)]) 
    eval_env = VecMonitor(eval_env)
    eval_env = VecFrameStack(eval_env, 4, channels_order="last")
    eval_env = VecTransposeImage(eval_env)

    # evaluacion + early stopping
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=BEST_DIR,
        log_path=LOG_DIR,
        n_eval_episodes=20,
        eval_freq=EVAL_FREQ,
        deterministic=True, # sin ruido aleatorio
        callback_after_eval=StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=15,
            min_evals=10
        ),
    )

    # callback guardado periodico cada CHECKPOINT_FREQ
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix='mario_ppo',
        verbose=1
    )

    # uno mis callbacks
    callbacks = CallbackList([eval_cb, checkpoint_cb])

    # -----------------------------------------------------------------------
    # Entrenamiento del modelo (nuevo o reanudado)
    # -----------------------------------------------------------------------

    if resume_path and os.path.isfile(resume_path):    # si recibe una ruta y contiene un archivo
        model = PPO.load(resume_path, env=env, device='cuda')
        print(f'Reanudando desde {resume_path}'
            f'({model.num_timesteps:,} pasos).')
    else:
        model = PPO(
        "CnnPolicy",
        env,
        learning_rate=0.0001,
        n_steps=512,
        batch_size=16,
        n_epochs=10,
        gamma=0.9,          # parametro importante, este da recompensa a corto plazo, mario se centra en recoger la recompensa por moverse a la derecha
        gae_lambda=1,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cuda",
        verbose=1,
        tensorboard_log=LOG_DIR,
    )
    print('Comenzando entrenamiento desde cero.')

    # -----------------------------------------------------------------------
    # Fase 1 entrenamiento -> Exploracion
    # Fase 2 entrenamiento -> Refinado
    # -----------------------------------------------------------------------

    done_steps = model.num_timesteps
    if done_steps < PHASE_1:
        model.learn(
        total_timesteps=PHASE_1 - done_steps,
        callback=callbacks,
        reset_num_timesteps=True,       # modificar para mantener el tensor log o reiniciarlo
    )
        # Deben ser funciones, se puede usar una funcion que lo reduzca en el tiempo
    if model.learning_rate != constant_fn(3e-5):
        model.learning_rate = constant_fn(3e-5)
        model.clip_range = constant_fn(0.15)
        model.ent_coef = 0.0              # se reduce la exploracion
        print('Comenzando fase 2 de refinado')

    model.learn(
        total_timesteps=max(0, TOTAL_STEPS - model.num_timesteps), # continua desde donde estaba
        callback=callbacks,
        reset_num_timesteps=False,
    )

    # Guardado del modelo final
    model.save("../models/mario_ppo_final")
    print("✅ Entrenamiento completo → ../models/mario_ppo_final")

# ---------------------------------------------------------------------------
# 4 · Main guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()       # necesario en Windows

    # None para entrenar de 0
    # introducir una ruta para continuar el modelo, hay que cambiar reset_num_timesteps a False para que no reinicie tensorboard
    train(resume_path='../models/checkpoints/mario_ppo_4700000_steps.zip')