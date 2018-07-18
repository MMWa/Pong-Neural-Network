import random
import pygame

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import TFOptimizer
import tensorflow as tf
from tqdm import tqdm

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


class PongGame:
    WIDTH = 600
    HEIGHT = 400
    BALL_RADIUS = 5
    PAD_WIDTH = 8
    PAD_HEIGHT = 80
    HALF_PAD_WIDTH = PAD_WIDTH / 2
    HALF_PAD_HEIGHT = PAD_HEIGHT / 2

    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('Pong using NN')

        self.reset()

    def reset(self):
        self.game_end = 1
        self.ball_pos = [0, 0]
        self.ball_vel = [0, 0]
        self.paddle1_pos = [self.HALF_PAD_WIDTH - 1, self.HEIGHT / 2]
        self.paddle2_pos = [self.WIDTH + 1 - self.HALF_PAD_WIDTH, self.HEIGHT / 2]

        self.l_score = 0
        self.r_score = 0

        self.paddle1_vel = 0
        self.paddle2_vel = 0

        self.agent1_catch = 0
        self.agent2_catch = 0

        if random.randrange(0, 1) == 0:
            self.ball_init(False)
        else:
            self.ball_init(True)

    def ball_init(self, ri):
        h_vel = random.randrange(2, 4)
        if ri:
            h_vel *= -1
        v_vel = random.randrange(-4, 4)
        self.ball_pos = [int(self.WIDTH / 2), int(self.HEIGHT / 2)]
        self.ball_vel = [h_vel, -v_vel]

    def tick(self, agent_action_1, agent_action_2):
        # update paddle's vertical position, keep paddle on the screen

        self.paddle1_vel = agent_action_1
        self.paddle2_vel = agent_action_2

        if self.HALF_PAD_HEIGHT < self.paddle1_pos[1] < self.HEIGHT - self.HALF_PAD_HEIGHT:
            self.paddle1_pos[1] += self.paddle1_vel
        elif self.paddle1_pos[1] == self.HALF_PAD_HEIGHT and self.paddle1_vel > 0:
            self.paddle1_pos[1] += self.paddle1_vel
        elif self.paddle1_pos[1] == self.HEIGHT - self.HALF_PAD_HEIGHT and self.paddle1_vel < 0:
            self.paddle1_pos[1] += self.paddle1_vel

        if self.HALF_PAD_HEIGHT < self.paddle2_pos[1] < self.HEIGHT - self.HALF_PAD_HEIGHT:
            self.paddle2_pos[1] += self.paddle2_vel
        elif self.paddle2_pos[1] == self.HALF_PAD_HEIGHT and self.paddle2_vel > 0:
            self.paddle2_pos[1] += self.paddle2_vel
        elif self.paddle2_pos[1] == self.HEIGHT - self.HALF_PAD_HEIGHT and self.paddle2_vel < 0:
            self.paddle2_pos[1] += self.paddle2_vel

        # update ball
        self.ball_pos[0] += int(self.ball_vel[0])
        self.ball_pos[1] += int(self.ball_vel[1])

        # ball collision check on top and bottom walls
        if int(self.ball_pos[1]) <= self.BALL_RADIUS:
            self.ball_vel[1] = - self.ball_vel[1]
        if int(self.ball_pos[1]) >= self.HEIGHT + 1 - self.BALL_RADIUS:
            self.ball_vel[1] = -self.ball_vel[1]

        # ball collison check on gutters or paddles
        if int(self.ball_pos[0]) <= self.BALL_RADIUS + self.PAD_WIDTH and int(self.ball_pos[1]) in range(
                int(self.paddle1_pos[1] - self.HALF_PAD_HEIGHT), int(self.paddle1_pos[1] + self.HALF_PAD_HEIGHT), 1):
            self.ball_vel[0] = -self.ball_vel[0]
            self.agent1_catch += 1
            self.ball_vel[0] *= 1.1
            self.ball_vel[1] *= 1.1
        elif int(self.ball_pos[0]) <= self.BALL_RADIUS + self.PAD_WIDTH:
            self.r_score += 1
            self.reset()
            self.ball_init(True)

        if int(self.ball_pos[0]) >= self.WIDTH + 1 - self.BALL_RADIUS - self.PAD_WIDTH and int(
                self.ball_pos[1]) in range(
            int(self.paddle2_pos[1] - self.HALF_PAD_HEIGHT), int(self.paddle2_pos[1] + self.HALF_PAD_HEIGHT), 1):
            self.ball_vel[0] = -self.ball_vel[0]
            self.agent2_catch += 1
            self.ball_vel[0] *= 1.1
            self.ball_vel[1] *= 1.1
        elif int(self.ball_pos[0]) >= self.WIDTH + 1 - self.BALL_RADIUS - self.PAD_WIDTH:
            self.l_score += 1
            self.reset()
            self.ball_init(False)

        return (self.ball_pos[0]-self.BALL_RADIUS)/self.WIDTH, (self.ball_pos[1]-self.BALL_RADIUS)/self.HEIGHT, self.ball_vel[0]/self.WIDTH, self.ball_vel[1]/self.HEIGHT

    def draw(self):
        event = pygame.event.get()
        self.window.fill(BLACK)

        # draw paddles and ball
        pygame.draw.circle(self.window, RED, self.ball_pos, self.BALL_RADIUS, 0)
        pygame.draw.polygon(self.window, GREEN,
                            [[self.paddle1_pos[0] - self.HALF_PAD_WIDTH, self.paddle1_pos[1] - self.HALF_PAD_HEIGHT],
                             [self.paddle1_pos[0] - self.HALF_PAD_WIDTH, self.paddle1_pos[1] + self.HALF_PAD_HEIGHT],
                             [self.paddle1_pos[0] + self.HALF_PAD_WIDTH, self.paddle1_pos[1] + self.HALF_PAD_HEIGHT],
                             [self.paddle1_pos[0] + self.HALF_PAD_WIDTH, self.paddle1_pos[1] - self.HALF_PAD_HEIGHT]],
                            0)
        pygame.draw.polygon(self.window, GREEN,
                            [[self.paddle2_pos[0] - self.HALF_PAD_WIDTH, self.paddle2_pos[1] - self.HALF_PAD_HEIGHT],
                             [self.paddle2_pos[0] - self.HALF_PAD_WIDTH, self.paddle2_pos[1] + self.HALF_PAD_HEIGHT],
                             [self.paddle2_pos[0] + self.HALF_PAD_WIDTH, self.paddle2_pos[1] + self.HALF_PAD_HEIGHT],
                             [self.paddle2_pos[0] + self.HALF_PAD_WIDTH, self.paddle2_pos[1] - self.HALF_PAD_HEIGHT]],
                            0)

        # update scores
        myfont1 = pygame.font.SysFont("Comic Sans MS", 20)
        label1 = myfont1.render("Score " + str(self.l_score), 1, (255, 255, 0))
        self.window.blit(label1, (50, 20))

        myfont2 = pygame.font.SysFont("Comic Sans MS", 20)
        label2 = myfont2.render("Score " + str(self.r_score), 1, (255, 255, 0))
        self.window.blit(label2, (470, 20))
        pygame.display.flip()


class Agent:
    def __init__(self, press_mul):
        self.mul = press_mul
        self.action_state = np.empty([1, 3])

    def action(self):
        value = np.argmax(self.action_state) - 1
        return value * self.mul


class Network:
    # TODO: add training function
    def __init__(self, filename):
        self.learning_rate = 1e-4
        self.filename = filename

        self.__define_network()
        try:
            self.model.load_weights(self.filename)
        except:
            print("error loading " + filename)
            pass

        self.__compile_network()
        self.model.summary()
        self.run_count = 0

    def __define_network(self):
        self.model = Sequential()
        self.model.add(Dense(32, input_shape=(5,)))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(3, activation='softmax'))

    def __compile_network(self):
        optimizer = TFOptimizer(tf.train.AdamOptimizer(0.001))
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer)

    def action(self, state):
        value = self.model.predict(state)
        return value

    def train(self, state, reward):
        self.model.train_on_batch(state, reward)
        self.run_count += 1
        if self.run_count % 1000 == 1:
            self.model.save_weights(self.filename)


if __name__ == "__main__":
    # the actual game
    # TODO: clean
    game = PongGame()
    agent1 = Agent(8)
    agent2 = Agent(8)

    model_net_1 = Network("m1.h5")
    model_net_2 = Network("m2.h5")

    i = 0
    for i in tqdm(range(20000)):
        # TODO: implement mirroring for shared network
        new_state = np.array(game.tick(agent_action_1=agent1.action(), agent_action_2=agent2.action()))

        hit_state = [game.agent1_catch, game.agent2_catch]
        game.agent1_catch = 0
        game.agent2_catch = 0

        hit_sum = np.sum(hit_state)

        reward_1 = agent1.action_state
        reward_2 = agent2.action_state

        norm_ball_h = game.ball_pos[1]/game.HEIGHT

        # delta ball to paddle
        paddle1_s = 1 - abs(norm_ball_h-(game.paddle1_pos[1]/game.HEIGHT))
        paddle2_s = 1 - abs(norm_ball_h-(game.paddle2_pos[1]/game.HEIGHT))

        reward_1 = np.multiply(reward_1, paddle1_s)
        reward_2 = np.multiply(reward_2, paddle2_s)

        new_state_1 = np.append(new_state, np.array([(game.paddle1_pos[1]-game.HALF_PAD_HEIGHT)/game.HEIGHT]))
        new_state_1.flatten()
        new_state_1 = np.reshape(new_state_1, (-1, 1))

        new_state_2 = np.append(new_state, np.array([(game.paddle2_pos[1]-game.HALF_PAD_HEIGHT)/game.HEIGHT]))
        new_state_2.flatten()
        new_state_2 = np.reshape(new_state_2, (-1, 1))

        model_net_1.train(new_state_1.T, reward_1)
        model_net_2.train(new_state_2.T, reward_2)

        agent1.action_state = model_net_1.action(new_state_1.T)
        agent2.action_state = model_net_2.action(new_state_2.T)

        game.draw()

