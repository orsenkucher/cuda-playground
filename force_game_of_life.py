import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import pygame

# CUDA kernel for Game of Life with brightness, attractors, and repellers
cuda_code = """
#define MAX_FORCES 20

struct Force {
    float x;
    float y;
    float strength;
};

__device__ float2 calculate_force(int x, int y, Force* forces, int num_forces, float max_distance)
{
    float2 total_force = make_float2(0.0f, 0.0f);
    
    for (int i = 0; i < num_forces; i++) {
        float dx = forces[i].x - x;
        float dy = forces[i].y - y;
        float distance = sqrtf(dx*dx + dy*dy);
        
        if (distance < max_distance) {
            float force_magnitude = forces[i].strength * (1.0f - distance / max_distance);
            total_force.x += force_magnitude * dx / distance;
            total_force.y += force_magnitude * dy / distance;
        }
    }
    
    return total_force;
}

__global__ void game_of_life(unsigned char *grid, unsigned char *next_grid, unsigned char *brightness,
                             Force* forces, int num_forces,
                             int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int alive_neighbors = 0;

    // Calculate forces
    float max_distance = 150.0f;
    float2 total_force = calculate_force(x, y, forces, num_forces, max_distance);
    
    // Apply forces to neighbor calculation
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = (int)(x + dx + total_force.x + width) % width;
            int ny = (int)(y + dy + total_force.y + height) % height;
            alive_neighbors += grid[ny * width + nx];
        }
    }
    
    unsigned char cell = grid[idx];
    next_grid[idx] = (cell == 1 && (alive_neighbors == 2 || alive_neighbors == 3)) || 
                     (cell == 0 && alive_neighbors == 3);
    
    // Calculate brightness based on number of live neighbors and force magnitude
    float force_magnitude = sqrtf(total_force.x * total_force.x + total_force.y * total_force.y);
    brightness[idx] = (unsigned char)(alive_neighbors * 28 + force_magnitude * 50);
}

__global__ void update_texture(unsigned char *grid, unsigned char *brightness, unsigned int *texture, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    unsigned char cell = grid[idx];
    unsigned char cell_brightness = brightness[idx];
    
    if (cell) {
        // Live cell: use calculated brightness
        texture[idx] = 0xFF000000 | (cell_brightness << 16) | (cell_brightness << 8) | cell_brightness;
    } else {
        // Dead cell: dim glow based on neighbor activity
        unsigned char glow = cell_brightness >> 2;  // Divide by 4 for a subtle glow
        texture[idx] = 0xFF000000 | (glow << 16) | (glow << 8) | glow;
    }
}
"""

# Initialize Pygame
pygame.init()
screen_info = pygame.display.Info()
width, height = screen_info.current_w, screen_info.current_h
screen = pygame.display.set_mode(
    (width, height), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
)
pygame.display.set_caption("Game of Life - With Attractors and Repellers")

# Compile CUDA kernel
mod = SourceModule(cuda_code)
game_of_life_kernel = mod.get_function("game_of_life")
update_texture_kernel = mod.get_function("update_texture")

# Initialize grids
grid = np.random.choice([0, 1], size=(height, width), p=[0.85, 0.15]).astype(np.uint8)
next_grid = np.zeros((height, width), dtype=np.uint8)
brightness = np.zeros((height, width), dtype=np.uint8)

# Allocate memory on GPU
grid_gpu = cuda.mem_alloc(grid.nbytes)
next_grid_gpu = cuda.mem_alloc(next_grid.nbytes)
brightness_gpu = cuda.mem_alloc(brightness.nbytes)

# Create texture for rendering
texture = np.zeros((height, width), dtype=np.uint32)
texture_gpu = cuda.mem_alloc(texture.nbytes)

# Set up block and grid sizes for CUDA
block_size = (32, 32, 1)
grid_size = (
    (width + block_size[0] - 1) // block_size[0],
    (height + block_size[1] - 1) // block_size[1],
)

# Initialize forces (attractors and repellers)
forces = []

# Allocate memory for forces on GPU
max_forces = 20
force_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('strength', np.float32)])
forces_gpu = cuda.mem_alloc(max_forces * force_dtype.itemsize)

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
                prob = np.random.uniform(0.01, 0.35)
                grid = np.random.choice([0, 1], size=(height, width), p=[1 - prob, prob]).astype(
                    np.uint8
                )
                cuda.memcpy_htod(grid_gpu, grid)
            elif event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            if event.button == 1:  # Left click
                forces.append((x, y, 1.0))
            elif event.button == 3:  # Right click
                forces.append((x, y, -1.0))
            forces = forces[-max_forces:]  # Keep only the last max_forces

    if not paused:
        # Update forces on GPU
        forces_array = np.array(forces, dtype=force_dtype)
        if len(forces_array) > 0:
            cuda.memcpy_htod(forces_gpu, forces_array)

        # Run CUDA kernels
        game_of_life_kernel(
            grid_gpu,
            next_grid_gpu,
            brightness_gpu,
            forces_gpu,
            np.int32(len(forces)),
            np.int32(width),
            np.int32(height),
            block=block_size,
            grid=grid_size,
        )

        # Swap grid pointers
        grid_gpu, next_grid_gpu = next_grid_gpu, grid_gpu

    # Update texture
    update_texture_kernel(
        grid_gpu,
        brightness_gpu,
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

    # Draw forces (attractors and repellers)
    for force in forces:
        color = (0, 255, 0) if force[2] > 0 else (255, 0, 0)
        pygame.draw.circle(screen, color, (int(force[0]), int(force[1])), 5)

    pygame.display.flip()

    # Calculate and display FPS
    fps = clock.get_fps()
    pygame.display.set_caption(f"Game of Life - With Attractors and Repellers (FPS: {fps:.2f})")
    clock.tick()

pygame.quit()
