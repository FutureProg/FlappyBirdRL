from gym.envs.registration import register

register(
	id='flappy_bird-v0',
	entry_point='flappy_bird.envs:FlappyBird'
)