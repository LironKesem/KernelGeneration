
from kernelllm import KernelLLM
import torch
# Initialize the model
model = KernelLLM()

# Define your PyTorch module
pytorch_code = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.gelu = nn.GELU()  # GELU module

    def forward(self, a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
        out = torch.matmul(a, b)
        out = out + bias
        out = self.gelu(out)  # use module
        return out
# This is how functions below will be used during
# validation and benchmarking
# model = Model(*get_init_inputs())
# output = model(*get_inputs())

def get_inputs():
    M, K, N = 512, 2048, 4096
    A = torch.rand((M, K), dtype=torch.bfloat16)
    B = torch.rand((K, N), dtype=torch.bfloat16)
    bias = torch.rand((N,), dtype=torch.bfloat16)
    return [A,B,bias]

def get_init_inputs():
    return []
'''

optimized_code = model.generate_triton(pytorch_code, max_new_tokens=1024*10)
print("=== Generated Triton Kernel (BF16) ===")
print(optimized_code)
with open("gelu_response.py", "w") as f:
    f.write(optimized_code)