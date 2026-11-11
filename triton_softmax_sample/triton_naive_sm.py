# eager softmax in PyTorch

"""
TRITON PATTERN: You manipulate tensors/vectors directly at the block scope. 
You define a tile (BLOCK_SIZE), and operations act on that whole chunk at once:

    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
        # 1. Identify which block/tile this program instance handles
        pid = tl.program_id(0)

        # 2. Generate a vector/block of offsets [0, 1, ..., BLOCK_SIZE-1]
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        # 3. Vectorized mask and bulk operations
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)

        # 4. Elementwise math on the entire block
        output = x + y

        # 5. Write the entire block back to global memory
        tl.store(output_ptr + offsets, output, mask=mask)

If you need to reduce, you write:
    row_max = tl.max(x, axis=0)
    row_sum = tl.sum(x, axis=0)

So we only need to define: warps (*32 num threads per block) and grid (num blocks, how blocks allocated),
and writing code for block level tl.load will auto distribute the tasks across warps.
"""

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """eager mode softmax"""
    x_max = x.max(dim = 1)[0] # x.max(dim = 1) returns a tuple of (max_values, index)
    safe_x = x - x_max[:, None]
    numer = torch.exp(safe_x)
    denom = numer.sum(dim = 1)
    sm_out = numer / denom[:, None]
    return sm_out

# as opposed to thread level C++, triton is block_level, so in this case each block is one row
# so one fetches one entire block at a time
@triton.jit
def _softmax_fwd_kernel(output_ptr, stride_output_row, input_ptr,
                        stride_input_row, num_cols, block_size: tl.constexpr):

    row_index = tl.program_id(0) # return id of the block
    row_start_ptr = input_ptr + (row_index * stride_input_row) # use block id to get this block's start loc

    col_offsets = tl.arange(0, block_size)
    input_pointers = row_start_ptr + col_offsets # thus gives entire row index
    row_mask = col_offsets < num_cols

    # move to sram, .load automatically distributes block data across warps
    row = tl.load(input_pointers, mask = row_mask, other = float("-inf")).to(tl.float32)

    # softmax
    safe_row = row - tl.max(row, axis = 0)
    numer = tl.exp(safe_row)
    denom = tl.sum(numer, axis = 0)
    sm_out = numer / denom

    # move sm_out from sram to hbm
    output_row_ptr = output_ptr + (row_index * stride_output_row)
    output_pointers = output_row_ptr + col_offsets
    tl.store(output_pointers, sm_out.to(output_ptr.dtype.element_ty), mask = row_mask)

def softmax(x:torch.Tensor) -> torch.Tensor:
    """triton softmax, a wrapper for kernel"""
    x = x.contiguous()
    assert x.dim() == 2, f"currently we only support 2d tensors"
    rows, cols = x.shape

    # let each row be parallelized in one kernel
    block_size = triton.next_power_of_2(cols)

    num_warps = 4 # number of warps (* 32 threads) per block
    if block_size > 4095: # always make sure each thread hold at most 16 elements
        num_warps = 16
    elif block_size > 2047:
        num_warps = 8

    grid = (rows, ) # grid specifies number of blocks

    # allocate out output buffer
    sm_out = torch.empty_like(x)

    _softmax_fwd_kernel[grid](
        sm_out,
        sm_out.stride(0),
        x,
        x.stride(0),
        cols,
        block_size = block_size,
        num_warps = num_warps
    )

    return sm_out


# ref
sample = torch.tensor([[1,2,3,4,5], [5,4,3,2,1]], dtype = torch.float32, device = 'cuda')
ref_out = F.softmax(sample, dim = 1)
print(f'{ref_out=}')

# eager
eager_out = naive_softmax(sample)
print(f'{eager_out=}')

# triton
triton_out = softmax(sample)
print(f'{triton_out=}')