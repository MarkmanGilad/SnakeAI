import numpy as np
import pygame

class HumanAgent:
    
    def get_action(self, events, state):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return 1
                elif event.key == pygame.K_DOWN:
                    return 2
                elif event.key == pygame.K_LEFT:
                    return 3
                elif event.key == pygame.K_RIGHT:
                    return 4

                return None