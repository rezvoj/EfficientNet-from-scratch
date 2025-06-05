import torch
import torch.nn as nn
import time

# --- 1. Define Standard Convolution Parameters ---
N_in = 16    # Batch size
C_in = 32    # Input Channels
H_in = 224   # Input Height
W_in = 224   # Input Width

C_out = 32   # Output Channels (Number of filters)
R = 3        # Filter Height
S = 3        # Filter Width


PADDING = 1
STRIDE = 1
DILATION = 1

print(f"--- Standard Convolution Configuration ---")
print(f"  Input Tensor: (N={N_in}, C_in={C_in}, H_in={H_in}, W_in={W_in})")
print(f"  Filter: (C_out={C_out}, C_in={C_in}, R={R}, S={S})")
print(f"  Padding: {PADDING}, Stride: {STRIDE}, Dilation: {DILATION}")

# --- 2. Check for CUDA availability ---
if not torch.cuda.is_available():
    print("CUDA is not available. Please ensure PyTorch is installed with CUDA support and a compatible GPU is present.")
    exit()

device = torch.device("cuda")
print(f"  Running on device: {torch.cuda.get_device_name(0)}")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA version: {torch.version.cuda}")
print(f"  cuDNN enabled: {torch.backends.cudnn.enabled}")
print(f"  cuDNN version: {torch.backends.cudnn.version()}")


# --- 3. Create the Convolution Layer ---
# nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
conv_layer = nn.Conv2d(
    in_channels=C_in,
    out_channels=C_out,
    kernel_size=(R, S),
    stride=STRIDE,
    padding=PADDING,
    dilation=DILATION,
    bias=False # Typically bias is separated or not used for pure benchmarking of conv op
).to(device)

# Create a dummy input tensor and move it to the GPU
input_tensor = torch.randn(N_in, C_in, H_in, W_in, device=device)

# --- Calculate output dimensions for verification ---
# PyTorch calculates this automatically, but for cross-checking:
H_out = (H_in + 2 * PADDING - DILATION * (R - 1) - 1) // STRIDE + 1
W_out = (W_in + 2 * PADDING - DILATION * (S - 1) - 1) // STRIDE + 1

print(f"\n--- Output Tensor Dimensions ---")
print(f"  Expected Output: (N={N_in}, C_out={C_out}, H_out={H_out}, W_out={W_out})")


# --- 4. Benchmark the Forward Pass ---
num_warmup_runs = 10
num_benchmark_runs = 100

print(f"\n--- Benchmarking Convolution Forward Pass ({num_benchmark_runs} runs) ---")

# Warm-up runs: essential for accurate GPU timing as initial runs can be slower
print("  Performing warm-up runs...")
for _ in range(num_warmup_runs):
    _ = conv_layer(input_tensor)
    # No need to synchronize after each warm-up if many runs,
    # just ensure the last warm-up is done before starting timing.
torch.cuda.synchronize() # Ensure all warm-up operations are complete

# Create CUDA events for precise timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Start measuring
start_event.record()

for _ in range(num_benchmark_runs):
    output_tensor = conv_layer(input_tensor)

# Stop measuring
end_event.record()

# Wait for all GPU operations to complete
torch.cuda.synchronize()

# Calculate elapsed time in milliseconds
elapsed_time_ms = start_event.elapsed_time(end_event)
average_time_ms = elapsed_time_ms / num_benchmark_runs

print(f"  Average Convolution Forward Pass time: {average_time_ms:.4f} ms")
print(f"  Actual Output Tensor shape: {output_tensor.shape}")























# # Calculate Output Dimensions
# H_out = (H_in + 2 * PADDING - DILATION * (R - 1) - 1) // STRIDE + 1
# W_out = (W_in + 2 * PADDING - DILATION * (S - 1) - 1) // STRIDE + 1

# print(f"--- Standard Convolution Configuration ---")
# print(f"  Input Tensor: (N={N_in}, C_in={C_in}, H_in={H_in}, W_in={W_in})")
# print(f"  Filter: (C_out={C_out}, C_in={C_in}, R={R}, S={S})")
# print(f"  Padding: {PADDING}, Stride: {STRIDE}, Dilation: {DILATION}")
# print(f"  Output Spatial: (H_out={H_out}, W_out={W_out})")

# # --- 2. Derive GEMM Dimensions (using im2col concept) ---
# # Convolution (N, C_in, H_in, W_in) * (C_out, C_in, R, S) -> (N, C_out, H_out, W_out)
# # Mapped to a GEMM: (Matrix A @ Matrix B = Matrix C)
# # Matrix A (Filters, reshaped): (C_out) x (C_in * R * S)
# # Matrix B (Im2col-transformed Input): (C_in * R * S) x (N * H_out * W_out)
# # Result C (Output, reshaped): (C_out) x (N * H_out * W_out)

# m_gemm = C_out                           # Rows of A and C (Output Channels)
# k_gemm = C_in * R * S                    # Columns of A, Rows of B (Input Channels * Filter Area)
# n_gemm = N_in * H_out * W_out            # Columns of B and C (Batch Size * Output Spatial Area)

# print(f"\n--- Equivalent PyTorch torch.matmul Dimensions (A @ B = C) ---")
# print(f"  Matrix A (Filters): ({m_gemm}, {k_gemm})")
# print(f"  Matrix B (Im2col Input): ({k_gemm}, {n_gemm})")
# print(f"  Result C (Output): ({m_gemm}, {n_gemm})")


# # --- 3. Check for CUDA availability ---
# if not torch.cuda.is_available():
#     print("CUDA is not available. This benchmark requires a GPU.")
#     exit()

# device = torch.device("cuda")
# print(f"\n  Running on device: {torch.cuda.get_device_name(0)}")
# print(f"  PyTorch version: {torch.__version__}")
# print(f"  CUDA version: {torch.version.cuda}")
# print(f"  cuDNN enabled: {torch.backends.cudnn.enabled}")


# # --- 4. Create Tensors on GPU and Perform GEMM ---

# # Create dummy data for A and B on GPU
# # Use `requires_grad=False` as we are only interested in forward pass performance
# matrix_A = torch.randn(m_gemm, k_gemm, device=device, requires_grad=False)
# matrix_B = torch.randn(k_gemm, n_gemm, device=device, requires_grad=False)

# # Optional: Print tensor sizes in MB for comparison
# print(f"\n  Matrix A size: {matrix_A.numel() * matrix_A.element_size() / (1024*1024):.2f} MB")
# print(f"  Matrix B size: {matrix_B.numel() * matrix_B.element_size() / (1024*1024):.2f} MB")
# print(f"  Expected Output C size: {m_gemm * n_gemm * matrix_A.element_size() / (1024*1024):.2f} MB")


# # --- 5. Benchmark the GEMM (torch.matmul) ---
# num_warmup_runs = 10
# num_benchmark_runs = 100

# print(f"\n--- Benchmarking torch.matmul ({num_benchmark_runs} runs) ---")

# # Warm-up runs: essential for accurate GPU timing
# print("  Performing warm-up runs...")
# for _ in range(num_warmup_runs):
#     _ = torch.matmul(matrix_A, matrix_B)
# torch.cuda.synchronize() # Ensure all warm-up operations are complete

# # Create CUDA events for precise timing
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)

# # Start measuring
# start_event.record()

# for _ in range(num_benchmark_runs):
#     result_C = torch.matmul(matrix_A, matrix_B)

# # Stop measuring
# end_event.record()

# # Wait for all GPU operations to complete
# torch.cuda.synchronize()

# # Calculate elapsed time in milliseconds
# elapsed_time_ms = start_event.elapsed_time(end_event)
# average_time_ms = elapsed_time_ms / num_benchmark_runs

# print(f"  Average torch.matmul time: {average_time_ms:.4f} ms")
# print(f"  Actual Output Tensor C shape: {result_C.shape}")