import numpy as np
import pygame
import time
from Fonts import *
from Images import *

BOARD_SIZE = 17
SQUARE_SIZE = 50

BLUE = (137, 168, 178)
LIGHTBLUE = (179, 200, 207)

DARK_BROWN = (175, 143, 111)
LIGHT_BROWN = (209, 187, 158)

PINK = (255, 205, 201)
LIGHT_PINK = (253, 172, 172)



class Graphics:
    def __init__ (self):
        pygame.init()
        self.COLOR1 = DARK_BROWN
        self.COLOR2 = LIGHT_BROWN
        self.size = BOARD_SIZE  # Board size (NxN)
        self.square_size = SQUARE_SIZE
        self.font = pygame.font.Font("Fonts/FunnyKids.otf", 36)

        # Create the screen dimensions
        self.width = self.size * self.square_size
        self.height = self.size * self.square_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake")
        
        # Load and scale the snake head
        self.head_icon = pygame.image.load("Images/SnakeHead.png")
        self.head_icon = pygame.transform.scale(self.head_icon, (self.square_size, self.square_size))

        # Load and scale the snake body
        self.body_icon = pygame.image.load("Images/SnakeBody.png")
        self.body_icon = pygame.transform.scale(self.body_icon, (self.square_size, self.square_size))

        # Load and scale the mouse
        self.mouse_icon = pygame.image.load("Images/Mouse.png")
        self.mouse_icon = pygame.transform.scale(self.mouse_icon, (self.square_size, self.square_size))

        self.button_surface = pygame.Surface((200, 50), pygame.SRCALPHA)

    def draw (self, state):
        if(state.score < 10):
            self.draw_checkered_board1(state.board)
        elif(state.score >= 10 and state.score):
            self.draw_checkered_board2(state.board)
        self.draw_snake(state)
        self.draw_mouse(state)
        self.draw_bomb(state)
        self.draw_score(state)

        pygame.display.flip()

    def draw_score(self, state):
        text = self.font.render(f"Score: {state.score}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
    

    def draw_checkered_board1(self, board):
        # Draw the 1st board
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                if(row + col) % 2 == 0: color = self.COLOR1
                else: color = self.COLOR2
                pygame.draw.rect(self.screen, color, (row * self.square_size, col * self.square_size, self.square_size, self.square_size))

    def draw_checkered_board2(self, board):
        color1 = BLUE
        color2 = LIGHTBLUE
        # Draw the 2nd board
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                if(row + col) % 2 == 0: color = color1
                else: color = color2
                pygame.draw.rect(self.screen, color, (row * self.square_size, col * self.square_size, self.square_size, self.square_size))

    def draw_snake(self, state):

        # Draw head
        head = state.get_head()
        self.screen.blit(self.head_icon, (head[1] * self.square_size, head[0] * self.square_size))

        # Draw body
        for body in state.snake[1:]:
            self.screen.blit(self.body_icon, (body[1] * self.square_size, body[0] * self.square_size))

    def draw_mouse(self, state):
        for mouse in state.mouse[:]:
            self.screen.blit(self.mouse_icon, (mouse[1] * self.square_size, mouse[0] * self.square_size))

    def draw_bomb(self, state):
        # Draw the bomb if it exists
        if state.bomb is None:
            return
        bomb = state.bomb
        r, c = bomb["pos"]

        # Draw explosion preview (3x3 area) as a translucent red overlay
        overlay = pygame.Surface((self.square_size * 3, self.square_size * 3), pygame.SRCALPHA)
        overlay.fill((255, 0, 0, 60))
        top_left_x = (c - 1) * self.square_size
        top_left_y = (r - 1) * self.square_size
        self.screen.blit(overlay, (top_left_x, top_left_y))

        # Draw bomb as a red circle at the center of its square
        center_x = c * self.square_size + self.square_size // 2
        center_y = r * self.square_size + self.square_size // 2
        pygame.draw.circle(self.screen, (180, 0, 0), (center_x, center_y), self.square_size // 2 - 6)
        pygame.draw.circle(self.screen, (255, 200, 200), (center_x, center_y), self.square_size // 2 - 10)

        # Draw countdown timer on top
        remaining = max(0, round(bomb["explode_time"] - time.time(), 1))
        small_font = pygame.font.Font("Fonts/FunnyKids.otf", 18)
        text = small_font.render(str(remaining), True, (255, 255, 255))
        text_rect = text.get_rect(center=(center_x, center_y))
        self.screen.blit(text, text_rect)

    def show_start_screen(self, state):
        font = pygame.font.Font("Fonts/FunnyKids.otf", 50)
        screen_width, screen_height = self.width, self.height

        # Step 1: Draw the game board before applying the overlay
        self.draw(state)
        pygame.display.flip()

        # Step 2: Create a semi-transparent overlay
        overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Dark transparent layer (opacity 180 out of 255)
        self.screen.blit(overlay, (0, 0))  # Render the overlay on top of the game screen

        # Step 3: Add centered welcome text
        text_surface = font.render("Welcome to Snake!", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(screen_width / 2, screen_height / 2 - 50))
        self.screen.blit(text_surface, text_rect)

        pygame.display.flip()  # Refresh the screen again to ensure all elements are drawn

        waiting = True
        while waiting:
            button_x = screen_width / 2 - 100
            button_y = screen_height / 2 - 25

            # Step 4: Draw the "Start Game" button
            if self.draw_button("Start Game", button_x, button_y, 200, 50, LIGHT_BROWN, DARK_BROWN, (255, 255, 255)):
                waiting = False  # Button clicked, start the game

            # Step 5: Handle user events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            pygame.display.flip()  # Refresh display to update button appearance

    def show_exit_screen(self, state, start_time):
        # Displays the exit screen with final score, playtime, and exit/play again buttons
        font = pygame.font.Font("Fonts/FunnyKids.otf", 50)
        screen_width, screen_height = self.width, self.height

        # Calculate total playtime
        play_time = round(time.time() - start_time)

        # Step 1: Draw the final game state before overlaying the exit screen
        self.draw(state)
        pygame.display.flip()

        # Step 2: Create a semi-transparent overlay
        overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Dark transparent background
        self.screen.blit(overlay, (0, 0))

        # Step 3: Display final score and playtime
        score_text = font.render(f"Final Score: {state.score}", True, (255, 255, 255))
        score_rect = score_text.get_rect(center=(screen_width / 2, screen_height / 2 - 35))
        self.screen.blit(score_text, score_rect)

        time_text = font.render(f"Play Time: {play_time} sec", True, (255, 255, 255))
        time_rect = time_text.get_rect(center=(screen_width / 2, screen_height / 2))
        self.screen.blit(time_text, time_rect)

        pygame.display.flip()

        # Step 4: Add "Play Again" and "Exit" buttons
        waiting = True
        while waiting:
            play_x, play_y = screen_width / 2 - 150, screen_height / 2 + 25
            exit_x, exit_y = screen_width / 2 + 50, screen_height / 2 + 25

            if self.draw_button("Play Again", play_x - 50, play_y, 200, 50, LIGHT_BROWN, DARK_BROWN, (255, 255, 255)):
                return True  # Restart the game

            if self.draw_button("Exit", exit_x, exit_y, 200, 50, (218, 108, 108), (205, 86, 86), (255, 255, 255)):
                pygame.quit()  # Close the game window
                exit()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            pygame.display.flip()

    def draw_button(self, text, x, y, width, height, color, hover_color, border_color):
        # Draws a button with a border, shadow effect, and transparency
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()

        self.button_surface.fill((0, 0, 0, 0))  # Make button background fully transparent
        pygame.draw.rect(self.button_surface, color, (0, 0, width, height), border_radius=10)  # Normal button color
        pygame.draw.rect(self.button_surface, border_color, (0, 0, width, height), 3, border_radius=10)  # Border color

        # Change button color when hovered
        if x < mouse[0] < x + width and y < mouse[1] < y + height:
            pygame.draw.rect(self.button_surface, hover_color, (0, 0, width, height), border_radius=10)
            if click[0] == 1:  # If clicked, return True
                return True

        self.screen.blit(self.button_surface, (x, y))  # Draw the button onto the screen

        # Render text in the center of the button
        text_surface = self.font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(x + width/2, y + height/2))
        self.screen.blit(text_surface, text_rect)

        return False
