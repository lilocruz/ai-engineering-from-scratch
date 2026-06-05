import time
import sys

def check_gpu():
    try:
        import torch
    except ImportError:
        print("PyTorch not installed. Run: uv pip install torch")
        return

    print("=== GPU Check ===\n")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available:  {torch.backends.mps.is_available()}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Memory: {props.total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"GPU: Apple Silicon (Metal/MPS)")
        print("Memory: Shared with system RAM")
    else:
        print("\nNo GPU detected. That's fine for most lessons.")
        print("For GPU-heavy lessons, use Google Colab (free).")
        return

    print(f"\nUsing device: {device}")
    print("\n=== CPU vs GPU Benchmark ===\n")

    size = 4000
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    start = time.time()
    _ = a @ b
    cpu_time = time.time() - start
    print(f"CPU matrix multiply ({size}x{size}): {cpu_time:.3f}s")

    a_gpu = a.to(device)
    b_gpu = b.to(device)

    # Warm up
    _ = a_gpu @ b_gpu
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    _ = a_gpu @ b_gpu
    if device.type == "cuda":
        torch.cuda.synchronize()
    gpu_time = time.time() - start

    print(f"GPU matrix multiply ({size}x{size}): {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")

if __name__ == "__main__":
    check_gpu()
