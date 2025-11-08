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
    Simple model that performs Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return self.ln(x)


sizes = [
    (32, 32)
]

def get_inputs():
    for M, N in sizes:
        x = torch.rand(M, N, dtype=torch.bfloat16)
        yield [x]

def get_init_inputs():
    return [(M, N)]
    '''

# Generate optimized Triton code

optimized_code = model.generate_triton(pytorch_code, max_new_tokens=64)
print("=== Generated Triton Kernel (BF16) ===")
print(optimized_code)
with open("layernorm_response.py", "w") as f:
    f.write(optimized_code)