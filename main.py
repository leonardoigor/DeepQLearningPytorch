import gym
from DeepQNetwork import Agent
import numpy as np


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[8],
                  lr=0.001)
    scores, eps_history = [], []
    n_games = 50000

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            # env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action,
                                   reward, observation_, done)
            observation = observation_
            agent.learn()
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        print('Game: ', i+1, 'score: ', score, 'average score %.1f' % avg_score,
              'epsilon %.2f' % agent.epsilon)
