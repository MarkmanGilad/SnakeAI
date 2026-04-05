import numpy as np
import pygame
from Constant import *

class HumanAgent:
    
    def get_action(self, events, state):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return ACTION_UP
                elif event.key == pygame.K_DOWN:
                    return ACTION_DOWN
                elif event.key == pygame.K_LEFT:
                    return ACTION_LEFT
                elif event.key == pygame.K_RIGHT:
                    return ACTION_RIGHT

                return None