import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import pygame
import time

# CUDA kernel with improved coloring and fixed vector operations
cuda_code = """
#include <cuComplex.h>

__device__ float3 hsv2rgb(float h, float s, float v) {
    float c = v * s;
    float x = c * (1 - abs(fmodf(h / 60.0f, 2) - 1));
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

__device__ float3 palette(float t, float zoom_factor) {
    // Adjust base colors based on zoom level
    float3 c0 = make_float3(0.5f + 0.3f * zoom_factor, 0.2f, 0.4f);
    float3 c1 = make_float3(0.9f, 0.6f + 0.2f * zoom_factor, 0.5f);
    float3 c2 = make_float3(0.3f, 0.8f + 0.1f * zoom_factor, 0.9f);
    float3 c3 = make_float3(1.0f, 0.9f, 0.3f + 0.4f * zoom_factor);

    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float freq = 6.28318f * (1.0f + zoom_factor * 0.1f);  // Increase frequency with zoom
    float cos1 = cosf(freq * (c1.x * t + 0.00f));
    float cos2 = cosf(freq * (c1.y * t + 0.10f));
    float cos3 = cosf(freq * (c1.z * t + 0.20f));
    float cos4 = cosf(freq * (c2.x * t + 0.30f));

    color.x += c0.x * cos1 + c1.x * cos2 + c2.x * cos3 + c3.x * cos4;
    color.y += c0.y * cos1 + c1.y * cos2 + c2.y * cos3 + c3.y * cos4;
    color.z += c0.z * cos1 + c1.z * cos2 + c2.z * cos3 + c3.z * cos4;

    float gamma = 0.6f - 0.2f * zoom_factor;  // Adjust gamma based on zoom
    color.x = powf(fmaxf(color.x, 0.0f), gamma);
    color.y = powf(fmaxf(color.y, 0.0f), gamma);
    color.z = powf(fmaxf(color.z, 0.0f), gamma);

    return color;
}

__global__ void mandelbrot(unsigned char *output, double xmin, double xmax, double ymin, double ymax, int width, int height, int max_iter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    double cx = xmin + (xmax - xmin) * x / width;
    double cy = ymin + (ymax - ymin) * y / height;

    cuDoubleComplex z = make_cuDoubleComplex(0, 0);
    cuDoubleComplex c = make_cuDoubleComplex(cx, cy);

    int i;
    double smooth_i = 0;
    for (i = 0; i < max_iter; i++) {
        z = cuCadd(cuCmul(z, z), c);
        if (cuCabs(z) > 2) {
            smooth_i = i + 1 - log2(log2(cuCabs(z)));
            break;
        }
    }

    float zoom_factor = log10f(2.0f / (xmax - xmin));  // Calculate zoom factor
    zoom_factor = fmaxf(0.0f, fminf(zoom_factor, 1.0f));  // Clamp between 0 and 1

    float3 color;
    if (i == max_iter) {
        color = make_float3(0, 0, 0);  // Black for points inside the set
    } else {
        float t = smooth_i / max_iter;
        color = palette(t, zoom_factor);
        
        // Add hue shift based on zoom level
        float hue_shift = 360.0f * zoom_factor;
        float h, s, v;
        h = fmodf(t * 360.0f + hue_shift, 360.0f);
        s = 0.8f + 0.2f * zoom_factor;
        v = color.x * 0.299f + color.y * 0.587f + color.z * 0.114f;  // Luminance
        v = v * (1.0f - zoom_factor) + zoom_factor;  // Increase brightness with zoom
        
        color = hsv2rgb(h, s, v);
    }

    int offset = (y * width + x) * 3;
    output[offset] = (unsigned char)(fminf(fmaxf(color.x, 0.0f), 1.0f) * 255);
    output[offset + 1] = (unsigned char)(fminf(fmaxf(color.y, 0.0f), 1.0f) * 255);
    output[offset + 2] = (unsigned char)(fminf(fmaxf(color.z, 0.0f), 1.0f) * 255);
}
"""

# The rest of your code remains the same, starting from:
# Compile the CUDA kernel
mod = SourceModule(cuda_code)
mandelbrot_kernel = mod.get_function("mandelbrot")

# Set up Pygame
pygame.init()
width, height = 1920, 1080
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Mandelbrot Explorer")

# Set up the computation
xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
max_iter = 256

# Allocate memory on the GPU
output = np.zeros((height, width, 3), dtype=np.uint8)
output_gpu = cuda.mem_alloc(output.nbytes)

# Set up the grid and block dimensions
block_size = (16, 16, 1)
grid_size = (
    (width + block_size[0] - 1) // block_size[0],
    (height + block_size[1] - 1) // block_size[1],
)

# Main loop
running = True
clock = pygame.time.Clock()
move_speed = 0.01
zoom_speed = 0.03
drastic_zoom_factor = 0.1

try:
    while running:
        start_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Drastic zoom in
                    center_x = (xmin + xmax) / 2
                    center_y = (ymin + ymax) / 2
                    xmin = center_x - drastic_zoom_factor * (center_x - xmin)
                    xmax = center_x + drastic_zoom_factor * (xmax - center_x)
                    ymin = center_y - drastic_zoom_factor * (center_y - ymin)
                    ymax = center_y + drastic_zoom_factor * (ymax - center_y)
                    max_iter = min(max_iter + 50, 1000)  # Increase detail
                elif event.key == pygame.K_BACKSPACE:
                    # Drastic zoom out
                    center_x = (xmin + xmax) / 2
                    center_y = (ymin + ymax) / 2
                    xmin = center_x - (center_x - xmin) / drastic_zoom_factor
                    xmax = center_x + (xmax - center_x) / drastic_zoom_factor
                    ymin = center_y - (center_y - ymin) / drastic_zoom_factor
                    ymax = center_y + (ymax - center_y) / drastic_zoom_factor
                    max_iter = max(max_iter - 50, 100)  # Decrease detail

        # Handle keyboard input for movement and zoom
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            xmin -= move_speed * (xmax - xmin)
            xmax -= move_speed * (xmax - xmin)
        if keys[pygame.K_d]:
            xmin += move_speed * (xmax - xmin)
            xmax += move_speed * (xmax - xmin)
        if keys[pygame.K_w]:
            ymin -= move_speed * (ymax - ymin)
            ymax -= move_speed * (ymax - ymin)
        if keys[pygame.K_s]:
            ymin += move_speed * (ymax - ymin)
            ymax += move_speed * (ymax - ymin)
        if keys[pygame.K_z]:  # Zoom in
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            xmin = center_x - (1 - zoom_speed) * (center_x - xmin)
            xmax = center_x + (1 - zoom_speed) * (xmax - center_x)
            ymin = center_y - (1 - zoom_speed) * (center_y - ymin)
            ymax = center_y + (1 - zoom_speed) * (ymax - center_y)
        if keys[pygame.K_x]:  # Zoom out
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            xmin = center_x - (1 + zoom_speed) * (center_x - xmin)
            xmax = center_x + (1 + zoom_speed) * (xmax - center_x)
            ymin = center_y - (1 + zoom_speed) * (center_y - ymin)
            ymax = center_y + (1 + zoom_speed) * (ymax - center_y)

        # Launch the kernel
        mandelbrot_kernel(
            output_gpu,
            np.float64(xmin),
            np.float64(xmax),
            np.float64(ymin),
            np.float64(ymax),
            np.int32(width),
            np.int32(height),
            np.int32(max_iter),
            block=block_size,
            grid=grid_size,
        )

        # Copy the result back to the CPU
        cuda.memcpy_dtoh(output, output_gpu)

        # Transpose the output array to match Pygame's format
        output_transposed = np.transpose(output, (1, 0, 2))

        # Update the display
        pygame.surfarray.blit_array(screen, output_transposed)
        pygame.display.flip()

        # Calculate and display FPS
        end_time = time.time()
        frame_time = end_time - start_time
        fps = 1 / frame_time
        pygame.display.set_caption(
            f"Mandelbrot Explorer - FPS: {fps:.2f} - Zoom: {1/(xmax-xmin):.2e}"
        )

        # Cap the frame rate
        clock.tick(60)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    pygame.quit()
