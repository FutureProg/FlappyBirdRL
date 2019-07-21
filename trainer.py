import gym
import flappy_bird

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

n_cpu = 2
env = SubprocVecEnv([(lambda: gym.make('flappy_bird-v0')) for i in range(n_cpu)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=250000)
model.save('ppo2_flappybird')
del model
model = PPO2.load('ppo2_flappybird')
obs = env.reset()
while True:
	action, _states = model.predict(obs)
	obs, rewards, dones, info = env.step(action)
	val = env.render()	
env.close()