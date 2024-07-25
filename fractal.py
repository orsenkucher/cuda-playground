import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import matplotlib.pyplot as plt

# CUDA kernel
cuda_code = """
__global__ void mandelbrot(unsigned char *output, float xmin, float xmax, float ymin, float ymax, int width, int height, int max_iter)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float cx = xmin + (xmax - xmin) * x / width;
    float cy = ymin + (ymax - ymin) * y / height;

    float zx = 0, zy = 0;
    int i;
    for (i = 0; i < max_iter; i++) {
        float zx2 = zx * zx, zy2 = zy * zy;
        if (zx2 + zy2 > 4) break;
        float tmp = zx2 - zy2 + cx;
        zy = 2 * zx * zy + cy;
        zx = tmp;
    }

    int offset = (y * width + x) * 3;
    output[offset] = i % 256;
    output[offset + 1] = (i * 7) % 256;
    output[offset + 2] = (i * 13) % 256;
}
"""

# Compile the CUDA kernel
mod = SourceModule(cuda_code)
mandelbrot_kernel = mod.get_function("mandelbrot")

# Set up the computation
width, height = 800, 600
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

# Launch the kernel
mandelbrot_kernel(
    output_gpu,
    np.float32(xmin),
    np.float32(xmax),
    np.float32(ymin),
    np.float32(ymax),
    np.int32(width),
    np.int32(height),
    np.int32(max_iter),
    block=block_size,
    grid=grid_size,
)

# Copy the result back to the CPU
cuda.memcpy_dtoh(output, output_gpu)

# Display the result
plt.imshow(output)
plt.axis('off')
plt.show()
