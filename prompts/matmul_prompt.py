from kernelllm import KernelLLM
# Initialize the model
model = KernelLLM()
# Define your PyTorch module
pytorch_code = '''
import torch
import torch.nn as nn
class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a large K dimension
    """
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, A, B):
        """
            Performs matrix multiplication of A and B.
            Args:
                A: Input tensor of shape (M, K)
                B: Input tensor of shape (K, N)
            Returns:
                Output tensor of shape (M, N)
        """
        return torch.matmul(A, B)
M, K, N = 256, 32, 256
def get_inputs():
    # Return BF16 tensors for KernelLLM
    A = torch.randn(M, K, dtype=torch.bfloat16)
    B = torch.randn(K, N, dtype=torch.bfloat16)
    return [A, B]
def get_init_inputs():
    return [] # No special initialization inputs needed
'''
# Generate Triton kernel code
optimized_code = model.generate_triton(pytorch_code, max_new_tokens=2048)
print("=== Generated Triton Kernel (BF16) ===")
print(optimized_code)
with open("matmul_response.py", "w") as f:
    f.write(optimized_code)
