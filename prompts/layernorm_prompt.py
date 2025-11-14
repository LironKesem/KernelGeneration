from kernelllm import KernelLLM
import torch
# Initialize the model
model = KernelLLM()

# Define your PyTorch module
pytorch_code = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Layer Normalization with learnable weight and bias.
    """
    def __init__(self, normalized_shape: tuple):
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape, dtype=torch.float16)
    
    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        # Return normalized output using weight and bias
        return nn.functional.layer_norm(
            x,
            normalized_shape=x.shape[-1:],
            weight=weight,
            bias=bias,
            eps=eps
        )

# Provide both x, weight, and bias as inputs
def get_inputs():
    N = 1024
    normalized_shape= 8192
    x = torch.rand((N, normalized_shape), dtype=torch.bfloat16, device='cuda')
    weight = torch.ones(normalized_shape, dtype=torch.bfloat16, device='cuda')
    bias = torch.zeros(normalized_shape, dtype=torch.bfloat16, device='cuda')
    eps = 1e-5
    return [x, weight, bias, eps]

def get_init_inputs():
    return [(8192)]
'''

# Generate optimized Triton code

optimized_code = model.generate_triton(pytorch_code, max_new_tokens=1024*10)
print("=== Generated Triton Kernel (BF16) ===")
print(optimized_code)
with open("layernorm_response.py", "w") as f:
    f.write(optimized_code)