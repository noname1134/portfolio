import pygame
import sys
import random
import time

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
ROWS, COLS = 4, 4
NUM_PAIRS = (ROWS * COLS) // 2
SQUARE_SIZE = WIDTH // COLS

COLOR_GRID = (0, 0, 0)
IMAGES = {}

HIDDEN = 0

# Setup Screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Memory Card Game")

class MemoryGame:
    def __init__(self):
        self.board = self.generate_random_board()
        self.visible_cards = [[0 for col in range(COLS)] for row in range(ROWS)]
        self.matched_cards = ()
        self.hide_time = 0
        self.selected_cards = []

    def check_match(self):
        first_card = self.board[self.selected_cards[0][0]][self.selected_cards[0][1]]
        second_card = self.board[self.selected_cards[1][0]][self.selected_cards[1][1]]
        return first_card == second_card

    def check_win(self):
        for row in self.visible_cards:
            if HIDDEN in row:
                return False
        return True

    @staticmethod
    def generate_random_board():
        card_values = list(range(1, NUM_PAIRS + 1)) * 2
        random.shuffle(card_values)
        return [
            card_values[row * COLS:(row + 1) * COLS] for row in range(ROWS)
        ]

def load_game_assets():
    asset_files = {
        0: "memory_game_img/card_0.png",
        1: "memory_game_img/card_1.png",
        2: "memory_game_img/card_2.png",
        3: "memory_game_img/card_3.png",
        4: "memory_game_img/card_4.png",
        5: "memory_game_img/card_5.png",
        6: "memory_game_img/card_6.png",
        7: "memory_game_img/card_7.png",
        8: "memory_game_img/card_8.png",
    }
    for card_id, filename in asset_files.items():
        try:
            raw_image = pygame.image.load(filename).convert_alpha()
            IMAGES[card_id] = pygame.transform.scale(raw_image, (SQUARE_SIZE, SQUARE_SIZE))
        except pygame.error:
            print(f"Could not find '{filename}'. Creating a fallback colored square.")
            fallback_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
            color = (255, 0, 0) if card_id == 1 else (0, 0, 255)
            fallback_surface.fill(color)
            IMAGES[card_id] = fallback_surface


def draw_board(board):
    for row in range(ROWS):
        for col in range(COLS):
            x_pos = col * SQUARE_SIZE
            y_pos = row * SQUARE_SIZE
            card_val = board[row][col]
            if card_val in IMAGES:
                screen.blit(IMAGES[card_val], (x_pos, y_pos))
            rect = pygame.Rect(x_pos, y_pos, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, COLOR_GRID, rect, 1)


def update_board(game):
    if game.hide_time != 0:
        if pygame.time.get_ticks() >= game.hide_time:
            game.visible_cards[game.selected_cards[0][0]][game.selected_cards[0][1]] = 0
            game.visible_cards[game.selected_cards[1][0]][game.selected_cards[1][1]] = 0
            game.hide_time = 0
            game.selected_cards = []


def handle_click(event, game):
    if game.hide_time != 0:
        return

    mouse_x, mouse_y = event.pos
    clicked_col = mouse_x // SQUARE_SIZE
    clicked_row = mouse_y // SQUARE_SIZE

    current_card = game.visible_cards[clicked_row][clicked_col]
    if current_card == 0:
        open_card = game.board[clicked_row][clicked_col]
        game.visible_cards[clicked_row][clicked_col] = open_card
        game.selected_cards.append([clicked_row, clicked_col])

        if len(game.selected_cards) >= 2:
            if game.check_match():
                if game.check_win():
                    print("you win")
                game.selected_cards = []
            else:
                game.hide_time += pygame.time.get_ticks() + 1000


def handle_event(event, game):
    if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()

    elif event.type == pygame.MOUSEBUTTONDOWN:
        if event.button == 1:
            handle_click(event, game)


def main():
    clock = pygame.time.Clock()
    game = MemoryGame()
    while True:
        for event in pygame.event.get():
            handle_event(event, game)

        update_board(game)

        draw_board(game.visible_cards)
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    load_game_assets()
    main()
