import numpy as np
import pygame
import random
import time
from Graphics import *
from Environment import *
from Constant import *
from HumanAgent import HumanAgent
from AgentDQN import AgentDQN

class Environment ():

    def __init__(self):
        self.graphics = Graphics()
        self.state = Environment()
        self.action = None
        self.agent = HumanAgent()
        # self.agent = AgentDQN()
    
    def restart(self):
        self.state = Environment()
        self.action = None

    def play(self):
        #self.graphics.show_start_screen(self.state)

        # Main game loop
        running = True
        clock = pygame.time.Clock()
        start_time = None  # Timer will start on first move
        
        while running:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return False
            
            action = self.agent.get_action(events=events, state=self.state)
            if action is not None:
                self.action = action

            # Process full game tick inside State
            if self.action and start_time is None:  # Start timer on first move
                start_time = time.time()
            if not self.state.step(self.action, time.time()):
                running = False

            # Redraw the board
            self.graphics.draw(self.state)
            clock.tick(FPS)

        # Game ended - automatically restart without showing exit screen
        return True

    def move(self, action):
        return self.state.move(action)

if __name__ == "__main__":
    env = Environment()
    play_again = env.play()
    # running = True
    # while running:  # Keep restarting the game when "Play Again" is pressed
    #     env = Environment()
    #     play_again = env.play()
    #     if not play_again:
    #         running = False