import random
from time import sleep

import pygame
from pygame.locals import *
from keras.models import Sequential
from keras.layers import *
from keras.backend import argmax
from keras.optimizers import SGD

# colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)


class PongGame:
    WIDTH = 600
    HEIGHT = 400
    BALL_RADIUS = 20
    PAD_WIDTH = 8
    PAD_HEIGHT = 80
    HALF_PAD_WIDTH = PAD_WIDTH / 2
    HALF_PAD_HEIGHT = PAD_HEIGHT / 2

    def __init__(self):

        self.ball_pos = [0, 0]
        self.ball_vel = [0, 0]
        self.paddle1_pos = [self.HALF_PAD_WIDTH - 1, self.HEIGHT / 2]
        self.paddle2_pos = [self.WIDTH + 1 - self.HALF_PAD_WIDTH, self.HEIGHT / 2]

        self.l_score = 0
        self.r_score = 0

        self.paddle1_vel = 0
        self.paddle2_vel = 0

        if random.randrange(0, 1) == 0:
            self.ball_init(False)
        else:
            self.ball_init(True)

    def ball_init(self, ri):
        h_vel = random.randrange(2, 4)
        if ri:
            h_vel *= -1
        v_vel = random.randrange(1, 3)
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
            self.ball_vel[0] *= 1.1
            self.ball_vel[1] *= 1.1
        elif int(self.ball_pos[0]) <= self.BALL_RADIUS + self.PAD_WIDTH:
            self.r_score += 1
            self.ball_init(True)

        if int(self.ball_pos[0]) >= self.WIDTH + 1 - self.BALL_RADIUS - self.PAD_WIDTH and int(
                self.ball_pos[1]) in range(
            int(self.paddle2_pos[1] - self.HALF_PAD_HEIGHT), int(self.paddle2_pos[1] + self.HALF_PAD_HEIGHT), 1):
            self.ball_vel[0] = -self.ball_vel[0]
            self.ball_vel[0] *= 1.1
            self.ball_vel[1] *= 1.1
        elif int(self.ball_pos[0]) >= self.WIDTH + 1 - self.BALL_RADIUS - self.PAD_WIDTH:
            self.l_score += 1
            self.ball_init(False)

        return self.ball_pos, self.ball_vel

    def draw(self, canvas):
        canvas.fill(RED)

        # draw paddles and ball
        pygame.draw.circle(canvas, RED, self.ball_pos, 20, 0)
        pygame.draw.polygon(canvas, GREEN,
                            [[self.paddle1_pos[0] - self.HALF_PAD_WIDTH, self.paddle1_pos[1] - self.HALF_PAD_HEIGHT],
                             [self.paddle1_pos[0] - self.HALF_PAD_WIDTH, self.paddle1_pos[1] + self.HALF_PAD_HEIGHT],
                             [self.paddle1_pos[0] + self.HALF_PAD_WIDTH, self.paddle1_pos[1] + self.HALF_PAD_HEIGHT],
                             [self.paddle1_pos[0] + self.HALF_PAD_WIDTH, self.paddle1_pos[1] - self.HALF_PAD_HEIGHT]],
                            0)
        pygame.draw.polygon(canvas, GREEN,
                            [[self.paddle2_pos[0] - self.HALF_PAD_WIDTH, self.paddle2_pos[1] - self.HALF_PAD_HEIGHT],
                             [self.paddle2_pos[0] - self.HALF_PAD_WIDTH, self.paddle2_pos[1] + self.HALF_PAD_HEIGHT],
                             [self.paddle2_pos[0] + self.HALF_PAD_WIDTH, self.paddle2_pos[1] + self.HALF_PAD_HEIGHT],
                             [self.paddle2_pos[0] + self.HALF_PAD_WIDTH, self.paddle2_pos[1] - self.HALF_PAD_HEIGHT]],
                            0)

        # update scores
        myfont1 = pygame.font.SysFont("Comic Sans MS", 20)
        label1 = myfont1.render("Score " + str(self.l_score), 1, (255, 255, 0))
        canvas.blit(label1, (50, 20))

        myfont2 = pygame.font.SysFont("Comic Sans MS", 20)
        label2 = myfont2.render("Score " + str(self.r_score), 1, (255, 255, 0))
        canvas.blit(label2, (470, 20))



class Agent:
    def __init__(self, press_mul):
        self.mul = press_mul
        self.action_state = 0

    def action(self):
        return self.action_state * self.mul


class Network:
    #TODO: add training function
    def __init__(self):
        self.__define_network()
        self.__compile_network()
        self.model.summary()

    def __define_network(self):
        self.model = Sequential()
        self.model.add(Dense(32,input_shape=(8,)))
        self.model.add(Dense(255, activation='relu'))
        self.model.add(GaussianDropout(.1))
        self.model.add(Dense(255, activation='relu'))
        self.model.add(Dense(3, activation='softmax'))

    def __compile_network(self):
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    def action(self, state):
        value = self.model.predict(state)
        print(str(value))
        return value


if __name__ == "__main__":
    # the actual game
    # TODO: clean
    game = PongGame()
    agent1 = Agent(8)
    agent2 = Agent(8)

    model_net = Network()
    pygame.init()
    fps = pygame.time.Clock()
    window = pygame.display.set_mode((game.WIDTH, game.HEIGHT), 0, 32)
    pygame.display.set_caption('Hello World')
    while True:
        print(str(game.l_score) + " - " + str(game.r_score))
        new_state = game.tick(agent_action_1=agent1.action(), agent_action_2=agent2.action())

        new_state_1 = np.array(new_state)
        tmp_ss = np.array([game.paddle1_pos, game.paddle2_pos])
        new_state_1 = np.append(new_state_1,tmp_ss)
        new_state_1.flatten()

        #TODO: fix input shape

        agent1.action_state = model_net.action(new_state_1)


        game.draw(window)
        pygame.display.update()

