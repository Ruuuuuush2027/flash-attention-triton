import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1 - block id
    pid = tl.program_id(axis = 0) # block id

    # 2 - compute range ids for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE) 
    # it is always a multiple of BLOCK_SIZE, so those exceeding n_elements must be masked out
    mask = offsets < n_elements

    # 3 - load to SRAM, distribute threads automatically; triton enables this SRAM movement
    x = tl.load(x_ptr + offsets, mask = mask)
    y = tl.load(y_ptr + offsets, mask = mask)

    # 4 - compute
    output = x + y

    # 5 - get results
    tl.store(output_ptr + offsets, output, mask = mask)

# wrapper for add_kernel, need to produce tensors to store results
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()

    # use lambda to compute grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE = 1024)

    return output


torch.manual_seed(42)
size = 98_432
x = torch.rand(size, device = 'cuda')
y = torch.rand(size, device = 'cuda')
output_torch = x + y
output_triton = add(x, y)

print(f'Max diff: {torch.max(torch.abs(output_torch - output_triton))}')
