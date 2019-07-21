import gym
import flappy_bird

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: gym.make('flappy_bird-v0')])
model = PPO2.load('ppo2_flappybird')
obs = env.reset()
while True:
	action, _states = model.predict(obs)
	obs, rewards, dones, info = env.step(action)
	val = env.render()	
env.close()