import gym
import sys
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random
import pygame

class FlappyBird(gym.Env):
	"""
	Description:
		An environment based on the mobile flappy bird game where you have to maneuvery between pipes

	Source:
		The environment corresponds to the flappy bird iOS and Android game

	Observation:
		Type: Box(2)
		Num			Observation		Min					Max
		0			Bird y			0					screen height
		1			Pipe height		50					screen height - 60
		2			Pipe x			0					screen width + 20

	Actions:
		Type: Discrete(2)
		Num			Action
		0			Allow bird to fall
		1			Fly up
	
	Reward:
		Reward is 1 for every step taken, including the termination step
	
	Starting State:
		Num 	Observation			Value
		0		Bird y				40
		1		Pipe height			Random seeded number within bounds
		2		Pipe x				screeen width + 20 as it is off screen
	
	Episode Termination:
		Bird falls to the ground
		Bird flies up and off the screen
		Bird collides with pipe
		Episode length > 400	
	"""

	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 50
	}

	def __init__(self, screen_size = (400, 400)):
		super(FlappyBird, self).__init__()
		self.screen_size = screen_size
		self.reset_pipe()
		self.bird = [70, self.screen_size[1]//2, 10, 10]
		
		high = np.array([
			self.screen_size[1],
			self.screen_size[1] - 60,
			self.screen_size[0] + 20
		])
		low = np.array([
			0, 50, 0
		])

		self.action_space = spaces.Discrete(2)
		self.observation_space = spaces.Box(low, high, dtype=np.int32)
		
		self.seed()
		self.viewer = None
		self.state = None
		self.steps_beyond_done = None

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
	
	def reset_pipe(self):
		height = np.random.randint(50, self.screen_size[1]-60)
		self.pipe = [self.screen_size[0] + 20, height/2, 20, height]


	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		state = self.state
		self.bird[1] = self.bird[1] - 10
		self.pipe[0] -= 18
		if action == 1:
			self.bird[1] = self.bird[1] + 20

		self.state = (self.bird[1], self.pipe[3], self.pipe[0])		
		done = self.bird[1] < 0 or self.bird[1] > self.screen_size[1]
		done = done or (self.bird[0] < self.pipe[0] + self.pipe[2] and
			self.bird[0] + self.bird[2] > self.pipe[0] and
			self.bird[1] < self.pipe[1] + self.pipe[3]/2)		

		if self.pipe[0] + self.pipe[2]*2 <= 0:
			self.reset_pipe()

		if not done:			
			reward = 1.0
		elif self.steps_beyond_done is None:			
			self.steps_beyond_done = 0
			reward = 0.0
		else:
			if self.steps_beyond_done == 0:
				print("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
			self.steps_beyond_done += 1
			reward = 0.0
				
		return np.array(self.state), reward, done, {}
	
	def reset(self):
		self.state = self.np_random.uniform(low=0.05, high=0.05, size=(3,))
		self.steps_beyond_done = None
		self.bird[1] = self.screen_size[1]//2
		self.reset_pipe()
		return np.array(self.state)
	
	def render(self, mode='human'):
		scale = 1
		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(self.screen_size[0], self.screen_size[1])
			l,r,t,b = -self.pipe[2]/2, self.pipe[2]/2, -self.pipe[3]/2, self.pipe[3]/2
			pipe = rendering.FilledPolygon([(l,b), (l, t), (r, t), (r, b)])
			self.pipetrans = rendering.Transform()
			pipe.add_attr(self.pipetrans)
			self.viewer.add_geom(pipe)

			l,r,t,b = -self.bird[2]/2, self.bird[2]/2, -self.bird[3]/2, self.bird[3]/2
			bird = rendering.FilledPolygon([(l,b), (l, t), (r, t), (r, b)])
			self.birdtrans = rendering.Transform()
			bird.add_attr(self.birdtrans)
			bird.set_color(.8, .6, .4)
			self.viewer.add_geom(bird)
			
			self._bird_geom = bird
			self._pipe_geom = pipe
		
		if self.state is None: 
			print("Return NONE")
			return None
		
		l,r,t,b = -self.pipe[2]/2, self.pipe[2]/2, -self.pipe[3]/2, self.pipe[3]/2
		self.pipetrans.set_translation(self.pipe[0], self.pipe[1])
		self._pipe_geom.v = [(l,b), (l,t), (r,t), (r,b)]

		self.birdtrans.set_translation(self.bird[0], self.bird[1])

		return self.viewer.render(return_rgb_array=mode == 'rgb_array')
	
	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None

