import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import pygame

# CUDA kernel for Game of Life with adjusted light effects
cuda_code = """
#include <cuda_runtime.h>

__device__ float3 hsv2rgb(float h, float s, float v) {
    float c = v * s;
    float x = c * (1 - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    float3 rgb;
    
    if (h < 60) rgb = make_float3(c, x, 0);
    else if (h < 120) rgb = make_float3(x, c, 0);
    else if (h < 180) rgb = make_float3(0, c, x);
    else if (h < 240) rgb = make_float3(0, x, c);
    else if (h < 300) rgb = make_float3(x, 0, c);
    else rgb = make_float3(c, 0, x);
    
    return make_float3(rgb.x + m, rgb.y + m, rgb.z + m);
}

__global__ void game_of_life(unsigned char *grid, unsigned char *next_grid, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int alive_neighbors = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = (x + dx + width) % width;
            int ny = (y + dy + height) % height;
            alive_neighbors += (grid[ny * width + nx] > 0);
        }
    }
    
    unsigned char cell = grid[idx];
    if (cell > 0) {
        if (alive_neighbors == 2 || alive_neighbors == 3) {
            next_grid[idx] = min(255, cell + 1);  // Increase brightness for sustained life
        } else {
            next_grid[idx] = max(0, cell - 10);  // Start fading if dying
        }
    } else {
        if (alive_neighbors == 3) {
            next_grid[idx] = 255;  // New cell born at full brightness
        } else {
            next_grid[idx] = max(0, cell - 10);  // Continue fading
        }
    }
}

__global__ void apply_glow(unsigned char *grid, float *glow_grid, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float glow = 0.0f;

    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int nx = (x + dx + width) % width;
            int ny = (y + dy + height) % height;
            int nidx = ny * width + nx;
            float distance = sqrtf(dx*dx + dy*dy);
            glow += grid[nidx] / (1.0f + distance * 5.0f);  // Increased falloff
        }
    }

    glow_grid[idx] = fminf(glow / 1000.0f, 1.0f);  // Reduced glow intensity
}

__global__ void update_texture(unsigned char *grid, float *glow_grid, unsigned int *texture, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float cell_value = grid[idx] / 255.0f;
    float glow = glow_grid[idx] * 0.3f;  // Reduced glow influence
    
    // Combine cell value and glow
    float combined = fminf(cell_value + glow * 0.5f, 1.0f);
    
    // Create color based on cell value and glow
    float3 rgb;
    if (cell_value > 0) {
        // Live cells: white to yellow based on age
        rgb = hsv2rgb(60.0f * (1.0f - cell_value), 0.5f * cell_value, 1.0f);
    } else if (glow > 0) {
        // Glow effect: blue tint
        rgb = make_float3(glow * 0.3f, glow * 0.5f, glow * 0.7f);  // Reduced glow colors
    } else {
        // Dead cells: black
        rgb = make_float3(0, 0, 0);
    }
    
    unsigned char r = (unsigned char)(fminf(rgb.x * 255.0f, 255.0f));
    unsigned char g = (unsigned char)(fminf(rgb.y * 255.0f, 255.0f));
    unsigned char b = (unsigned char)(fminf(rgb.z * 255.0f, 255.0f));
    
    texture[idx] = (0xFF << 24) | (b << 16) | (g << 8) | r;
}
"""

# Initialize Pygame
pygame.init()
screen_info = pygame.display.Info()
width, height = screen_info.current_w, screen_info.current_h
screen = pygame.display.set_mode(
    (width, height), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
)
pygame.display.set_caption("Game of Life - Light Effects")

# Compile CUDA kernel
mod = SourceModule(cuda_code)
game_of_life_kernel = mod.get_function("game_of_life")
apply_glow_kernel = mod.get_function("apply_glow")
update_texture_kernel = mod.get_function("update_texture")

# Initialize grids
grid = np.random.choice([0, 255], size=(height, width), p=[0.8, 0.2]).astype(np.uint8)
next_grid = np.zeros((height, width), dtype=np.uint8)
glow_grid = np.zeros((height, width), dtype=np.float32)

# Allocate memory on GPU
grid_gpu = cuda.mem_alloc(grid.nbytes)
next_grid_gpu = cuda.mem_alloc(next_grid.nbytes)
glow_grid_gpu = cuda.mem_alloc(glow_grid.nbytes)

# Create texture for rendering
texture = np.zeros((height, width), dtype=np.uint32)
texture_gpu = cuda.mem_alloc(texture.nbytes)

# Set up block and grid sizes for CUDA
block_size = (16, 16, 1)
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
                grid = np.random.choice([0, 255], size=(height, width), p=[0.8, 0.2]).astype(
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

        # Apply glow effect
        apply_glow_kernel(
            grid_gpu,
            glow_grid_gpu,
            np.int32(width),
            np.int32(height),
            block=block_size,
            grid=grid_size,
        )

    # Update texture
    update_texture_kernel(
        grid_gpu,
        glow_grid_gpu,
        texture_gpu,
        np.int32(width),
        np.int32(height),
        block=block_size,
        grid=grid_size,
    )

    # Copy texture from GPU to CPU
    cuda.memcpy_dtoh(texture, texture_gpu)

    # Update Pygame surface and display
    surface = pygame.surfarray.make_surface(texture.transpose(1, 0))
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    # Calculate and display FPS
    fps = clock.get_fps()
    pygame.display.set_caption(f"Game of Life - Light Effects (FPS: {fps:.2f})")
    clock.tick(60)

pygame.quit()
