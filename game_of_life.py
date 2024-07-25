import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import pygame

# CUDA kernel for Game of Life
cuda_code = """
__global__ void game_of_life(unsigned char *grid, unsigned char *next_grid, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int alive_neighbors = 0;

    #pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = (x + dx + width) % width;
            int ny = (y + dy + height) % height;
            alive_neighbors += grid[ny * width + nx];
        }
    }
    
    unsigned char cell = grid[idx];
    next_grid[idx] = (cell == 1 && (alive_neighbors == 2 || alive_neighbors == 3)) || 
                     (cell == 0 && alive_neighbors == 3);
}

__global__ void update_texture(unsigned char *grid, unsigned int *texture, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    unsigned char cell = grid[idx];
    texture[idx] = cell ? 0xFFFFFFFF : 0xFF000000;
}
"""

# Initialize Pygame
pygame.init()
screen_info = pygame.display.Info()
width, height = screen_info.current_w, screen_info.current_h
screen = pygame.display.set_mode(
    (width, height), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
)
pygame.display.set_caption("Game of Life - CUDA (Fullscreen)")

# Compile CUDA kernel
mod = SourceModule(cuda_code)
game_of_life_kernel = mod.get_function("game_of_life")
update_texture_kernel = mod.get_function("update_texture")

# Initialize grids
grid = np.random.choice([0, 1], size=(height, width), p=[0.85, 0.15]).astype(np.uint8)
next_grid = np.zeros((height, width), dtype=np.uint8)

# Allocate memory on GPU
grid_gpu = cuda.mem_alloc(grid.nbytes)
next_grid_gpu = cuda.mem_alloc(next_grid.nbytes)

# Create texture for rendering
texture = np.zeros((height, width), dtype=np.uint32)
texture_gpu = cuda.mem_alloc(texture.nbytes)

# Set up block and grid sizes for CUDA
block_size = (32, 32, 1)
grid_size = (
    (width + block_size[0] - 1) // block_size[0],
    (height + block_size[1] - 1) // block_size[1],
)

# Main loop
running = True
paused = False
clock = pygame.time.Clock()

# Copy initial grid to GPU
cuda.memcpy_htod(grid_gpu, grid)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                grid = np.random.choice([0, 1], size=(height, width), p=[0.85, 0.15]).astype(
                    np.uint8
                )
                cuda.memcpy_htod(grid_gpu, grid)
            elif event.key == pygame.K_ESCAPE:
                running = False

    if not paused:
        # Run CUDA kernels
        game_of_life_kernel(
            grid_gpu,
            next_grid_gpu,
            np.int32(width),
            np.int32(height),
            block=block_size,
            grid=grid_size,
        )

        # Swap grid pointers
        grid_gpu, next_grid_gpu = next_grid_gpu, grid_gpu

    # Update texture
    update_texture_kernel(
        grid_gpu, texture_gpu, np.int32(width), np.int32(height), block=block_size, grid=grid_size
    )

    # Copy texture from GPU to CPU
    cuda.memcpy_dtoh(texture, texture_gpu)

    # Update Pygame surface and display
    surface = pygame.surfarray.make_surface(texture.transpose(1, 0))
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    # Calculate and display FPS
    fps = clock.get_fps()
    pygame.display.set_caption(f"Game of Life - CUDA Fullscreen (FPS: {fps:.2f})")
    clock.tick()

pygame.quit()
