import sys
import pygame
import numpy as np
import torch

# Import file chứa định nghĩa model của bạn
from models import * 

# Khởi tạo Pygame
pygame.init()

# Cấu hình cửa sổ và pixel
GRID_SIZE = 28
PIXEL_SIZE = 14  # Kích thước 1 ô vuông (28 * 14 = 392px)
PADDING = 4      # Viền cho bảng vẽ
WIDTH, HEIGHT = (GRID_SIZE * PIXEL_SIZE + PADDING*2) * 2, (GRID_SIZE * PIXEL_SIZE + PADDING*2)

# Màu sắc
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
GREEN = (0, 255, 0)
RED = (255, 50, 50)
BLUE_BG = (10, 10, 30)

# Cài đặt font
try:
    font = pygame.font.Font(pygame.font.match_font('courier'), 18)
except:
    font = pygame.font.SysFont(None, 24)

# ==============================================================
# 1. LOAD MODEL VÀ CHUYỂN SANG CHẾ ĐỘ EVAL (RẤT QUAN TRỌNG)
# ==============================================================
model = CNN.load(yaml_path='models/cnn_model.yaml', 
                 pth_path='models/cnn_model.pth')
model.eval() # Bắt buộc phải có để tắt Dropout và cố định BatchNorm
print("Model loaded successfully in EVAL mode!")

def draw_grid(screen, grid):
    """Vẽ bảng lưới 28x28 ở nửa bên trái"""
    pygame.draw.rect(screen, GRAY, (PADDING-2, PADDING-2, GRID_SIZE*PIXEL_SIZE+4, GRID_SIZE*PIXEL_SIZE+4), 2)
    
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            color_val = int(grid[y][x] * 255)
            color = (color_val, color_val, color_val)
            
            rect = pygame.Rect(
                PADDING + x * PIXEL_SIZE, 
                PADDING + y * PIXEL_SIZE, 
                PIXEL_SIZE, PIXEL_SIZE
            )
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (30, 30, 30), rect, 1)

def draw_logits(screen, logits):
    """Vẽ thanh Bar biểu diễn Logits"""
    start_x = WIDTH // 2 + PADDING + 20
    start_y = 20
    max_bar_len = 150  
    
    title = font.render("RAW LOGITS (0 -> 9)", True, WHITE) # Thường class MNIST là 0-9
    screen.blit(title, (start_x, start_y))
    
    start_y += 40
    
    zero_line_x = start_x + 80 + max_bar_len // 2
    pygame.draw.line(screen, GRAY, (zero_line_x, start_y), (zero_line_x, start_y + 10 * 30), 2)
    
    for i, logit in enumerate(logits):
        label = font.render(f"Class {i}:", True, WHITE)
        screen.blit(label, (start_x, start_y + i * 30))
        
        val = logit.item()
        bar_len = int((val / 10.0) * (max_bar_len // 2)) 
        
        if val >= 0:
            color = GREEN
            rect = pygame.Rect(zero_line_x, start_y + i * 30 + 2, min(bar_len, max_bar_len//2), 16)
        else:
            color = RED
            bar_len = abs(max(bar_len, -max_bar_len//2))
            rect = pygame.Rect(zero_line_x - bar_len, start_y + i * 30 + 2, bar_len, 16)
            
        pygame.draw.rect(screen, color, rect)
        
        val_text = font.render(f"{val:>6.2f}", True, WHITE)
        screen.blit(val_text, (zero_line_x + (max_bar_len//2) + 15, start_y + i * 30))

def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pixel Art Model Tester")
    clock = pygame.time.Clock()

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    logits = torch.zeros(10)

    drawing = False
    running = True

    while running:
        screen.fill(BLUE_BG)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: drawing = True
                elif event.button == 3: drawing = False
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1: drawing = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE or event.key == pygame.K_c:
                    grid.fill(0) 
                    logits = torch.zeros(10)

        # Xử lý logic vẽ / xóa
        if drawing or pygame.mouse.get_pressed()[2]:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            
            if PADDING <= mouse_x < PADDING + GRID_SIZE * PIXEL_SIZE and \
               PADDING <= mouse_y < PADDING + GRID_SIZE * PIXEL_SIZE:
                
                grid_x = (mouse_x - PADDING) // PIXEL_SIZE
                grid_y = (mouse_y - PADDING) // PIXEL_SIZE
                
                if pygame.mouse.get_pressed()[0]: # Chuột trái -> Vẽ trắng
                    grid[grid_y][grid_x] = 1.0
                    if grid_y > 0: grid[grid_y-1][grid_x] = max(grid[grid_y-1][grid_x], 0.5)
                    if grid_y < GRID_SIZE-1: grid[grid_y+1][grid_x] = max(grid[grid_y+1][grid_x], 0.5)
                    if grid_x > 0: grid[grid_y][grid_x-1] = max(grid[grid_y][grid_x-1], 0.5)
                    if grid_x < GRID_SIZE-1: grid[grid_y][grid_x+1] = max(grid[grid_y][grid_x+1], 0.5)
                
                elif pygame.mouse.get_pressed()[2]: # Chuột phải -> Tẩy (Xóa sạch sẽ cả ô lân cận)
                    grid[grid_y][grid_x] = 0.0
                    if grid_y > 0: grid[grid_y-1][grid_x] = 0.0
                    if grid_y < GRID_SIZE-1: grid[grid_y+1][grid_x] = 0.0
                    if grid_x > 0: grid[grid_y][grid_x-1] = 0.0
                    if grid_x < GRID_SIZE-1: grid[grid_y][grid_x+1] = 0.0

        # Chạy qua Model (Inference)
        if np.max(grid) > 0:
            with torch.no_grad():
                input_tensor = torch.tensor(grid).unsqueeze(0).unsqueeze(0)
                # Đảm bảo device của tensor giống với model (trường hợp load trên GPU)
                device = next(model.parameters()).device
                input_tensor = input_tensor.to(device)
                
                output = model(input_tensor)
                logits = output[0]

        # Render đồ họa
        draw_grid(screen, grid)
        draw_logits(screen, logits)
        
        help_text = font.render("Left Click: Draw | Right Click: Erase | Space: Clear", True, GRAY)
        screen.blit(help_text, (PADDING, HEIGHT - 25))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()