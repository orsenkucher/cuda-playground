import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import pygame
import math

# CUDA kernel for Game of Life with invisible ray lighting
cuda_code = """
#define MAX_RAY_LENGTH 2000  // Will be set dynamically in Python

__device__ float2 directions[] = {
    {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    {0.707f, 0.707f}, {-0.707f, 0.707f}, {0.707f, -0.707f}, {-0.707f, -0.707f}
};

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
            alive_neighbors += grid[ny * width + nx];
        }
    }
    
    unsigned char cell = grid[idx];
    next_grid[idx] = (cell == 1 && (alive_neighbors == 2 || alive_neighbors == 3)) || 
                     (cell == 0 && alive_neighbors == 3);
}

__global__ void raycast(unsigned char *grid, float *light_intensity, int width, int height, int max_ray_length)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Only process border cells
    if (x > 0 && x < width-1 && y > 0 && y < height-1) return;
    
    for (int d = 0; d < 8; d++) {
        float2 dir = directions[d];
        float px = x + 0.5f, py = y + 0.5f;
        
        for (int step = 0; step < max_ray_length; step++) {
            px += dir.x;
            py += dir.y;
            
            int nx = (int)px;
            int ny = (int)py;
            
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) break;
            
            int idx = ny * width + nx;
            float distance = sqrtf((px - x)*(px - x) + (py - y)*(py - y));
            float intensity = 1.0f / (1.0f + 0.005f * distance);  // Adjusted for slower falloff
            
            atomicAdd(&light_intensity[idx], intensity);
            
            if (grid[idx] == 1) break;  // Ray hits a live cell and stops
        }
    }
}

__global__ void update_texture(unsigned char *grid, float *light_intensity, unsigned int *texture, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    unsigned char cell = grid[idx];
    float intensity = min(light_intensity[idx], 1.0f);
    
    if (cell && intensity > 0.05f) {  // Only render live cells with significant light
        unsigned char color_value = (unsigned char)(255 * intensity);
        texture[idx] = 0xFF000000 | (color_value << 16) | (color_value << 8) | color_value;
    } else {
        // Everything else is black
        texture[idx] = 0xFF000000;
    }
    
    // Reset light intensity for next frame
    light_intensity[idx] = 0.0f;
}
"""

# Initialize Pygame
pygame.init()
screen_info = pygame.display.Info()
width, height = screen_info.current_w, screen_info.current_h
screen = pygame.display.set_mode(
    (width, height), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
)
pygame.display.set_caption("Game of Life - Invisible Ray Border Lighting")

# Calculate max ray length
max_ray_length = int(math.sqrt(width**2 + height**2))

# Compile CUDA kernel
mod = SourceModule(cuda_code.replace('MAX_RAY_LENGTH 2000', f'MAX_RAY_LENGTH {max_ray_length}'))
game_of_life_kernel = mod.get_function("game_of_life")
raycast_kernel = mod.get_function("raycast")
update_texture_kernel = mod.get_function("update_texture")

# Initialize grids
grid = np.random.choice([0, 1], size=(height, width), p=[0.85, 0.15]).astype(np.uint8)
next_grid = np.zeros((height, width), dtype=np.uint8)
light_intensity = np.zeros((height, width), dtype=np.float32)

# Allocate memory on GPU
grid_gpu = cuda.mem_alloc(grid.nbytes)
next_grid_gpu = cuda.mem_alloc(next_grid.nbytes)
light_intensity_gpu = cuda.mem_alloc(light_intensity.nbytes)

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
                prob = np.random.uniform(0.0, 1.0)
                grid = np.random.choice([0, 1], size=(height, width), p=[1 - prob, prob]).astype(
                    np.uint8
                )
                cuda.memcpy_htod(grid_gpu, grid)
            elif event.key == pygame.K_ESCAPE:
                running = False

    if not paused:
        # Run Game of Life update
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

    # Perform raycasting
    raycast_kernel(
        grid_gpu,
        light_intensity_gpu,
        np.int32(width),
        np.int32(height),
        np.int32(max_ray_length),
        block=block_size,
        grid=grid_size,
    )

    # Update texture
    update_texture_kernel(
        grid_gpu,
        light_intensity_gpu,
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
    pygame.display.set_caption(f"Game of Life - Invisible Ray Border Lighting (FPS: {fps:.2f})")
    clock.tick()

pygame.quit()
