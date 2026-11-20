import torch

import triton
import triton.language as tl

@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q,
    offs_kv,
    SEQ_LEN: tl.constexpr,
):
    if STAGE == 1: # blocks where K index < Q index; P left of diagonal
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2: # blocks where Q index == K index; P diagonal
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q) # hint to tell compiler optimization
    else: # all blocks
        lo, hi = 0, SEQ_LEN

    K_block_ptr = tl.advance(K_block_ptr, (0, lo)) # axis 1 all + lo
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0)) # axis 0 all + lo

    # K, V inner loop using advance
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # QK
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        if STAGE == 2: # for Q index == K index blocks, get mask
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6) # 0 true, else -1e6
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1)) # m_ij means current iter's rolling max, same shape as m_i which is previous all rolling max
            QK_block -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        P_block = tl.math.exp(QK_block)
        
        alpha = tl.math.exp(m_i - m_ij)
        l_ij = tl.sum(P_block, 1)
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)

        # get correction factor
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block) # third one accumulator, P@V+O

        m_i = m_ij # update m_i

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))

    return O_block, l_i, m_i


@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN; log sum exp to store
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch, stride_Q_head, stride_Q_seq, stride_Q_dim,
    stride_K_batch, stride_K_head, stride_K_seq, stride_K_dim,
    stride_V_batch, stride_V_head, stride_V_seq, stride_V_dim,
    stride_O_batch, stride_O_head, stride_O_seq, stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr, # if causal 3 or not 1
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # identify my id
    block_index_q = tl.program_id(0) # index of block
    index_batch_head = tl.program_id(1)
    index_batch = index_batch_head // NUM_HEADS # of the block, which batch
    index_head = index_batch_head % NUM_HEADS # of the block, which head

    # get (SEQ_LEN, HEAD_DIM) block of Q,K,V
    qvk_offset = index_batch.to(tl.int64) * stride_Q_batch + index_head.to(tl.int64) * stride_Q_head

    """
    1. conceptually a 2D arrangement of pointers, breaking (seq_len, head_dim) into block_size_q * head_dim per block
    2. triton we should always write block pointers not thread pointers, each pointer is a block
    3. Every CUDA block holds its own fixed slice of Q, while its K and V pointers act as sliding 
        windows that advance step-by-step through the sequence inside the inner loop until they've 
        scanned all the relevant keys and values.
    """

    # Q is small square block Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q, :]
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0), # jump direct to the block
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # V is whole row, V[index_batch, index_head, :, :], increment to cover entire V rows
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0), # not skipping anything on V, slides downward from 0 by self
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    # K is whole col, K[index_batch, index_head, :, :]
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(
            stride_K_dim,
            stride_K_seq,
        ),  # We invert the strides w.r.t Q, so we transpose the matrix
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    # O is also block square like Q
    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q) # get specific q in block
    offs_kv = tl.arange(0, BLOCK_SIZE_KV) # 0, 1, 2, 3

    # running max of a block
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype = tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype = tl.float32) + 1.0
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype = tl.float32)

    Q_block = tl.load(Q_block_ptr) # load to SRAM the block

    # get inner loop
    if STAGE == 1 or STAGE == 3: # runs for non-causal attention or the left of diagonal of causal attention
        O_block, l_i, m_i = _attn_fwd_inner( # inner loop, no new parallel just regular sequential
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN
        )

    if STAGE == 3: # run blocks for the diagonal blocks
        O_block, l_i, m_i = _attn_fwd_inner( # inner loop
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN
        )

    m_i += tl.math.log(l_i)
    





class TritonAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale): 
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        O = torch.empty_like(Q)
        stage = 3 if causal else 1 # causal or not

        # grid defines number of blocks, in our case, each block handles BLOCK_SIZE_Q * HEAD_DIM number of elements
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args['BLOCK_SIZE_Q']), # which query group is this?
            BATCH_SIZE * NUM_HEADS, # which heads of which batch is this thread?
            1,
        )

        # number of parallel programs: (BATCH_SIZE * NUM_HEADS * NUM_BLOCKS_Q)
        # logsumexp for backward pass, one for each query, max of each Q*K row for backward
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device = Q.device, dtype = torch.float32
        )

        # launch kernel
        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            # provide stride to calculate tensor index
            stride_Q_batch=Q.stride(0), stride_Q_head=Q.stride(1), stride_Q_seq=Q.stride(2), stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0), stride_K_head=K.stride(1), stride_K_seq=K.stride(2), stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0), stride_V_head=V.stride(1), stride_V_seq=V.stride(2), stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0), stride_O_head=O.stride(1), stride_O_seq=O.stride(2), stride_O_dim=O.stride(3),
            # other params
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )

        # ctx, save info for backward pass
        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return O
        


def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    # requires grad because want to test gradient flow also later
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1 / (HEAD_DIM ** 0.5)
    d0 = torch.randn_like(Q) # needed for backward pass testing, gradient passed from future layers

    # regular softmax compute
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), device = 'cuda'))
    P = torch.matmul(Q, K.transpose(2, 3)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float('-inf')
    P = torch.softmax(P.float(), dim = -1).half()
    ref_0 = torch.matmul(P, V)

    # backward gradient
    ref_0.backward(d0) # d0 is incoming gradient from the back
    ref_dV, V.grad = V.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None

    # triton implementation
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(d0)
    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    # compare
    rtol = 0.0
    atol = 1e-2 # absolute distance
    assert torch.allclose(ref_0, tri_out, atol = atol, rtol = rtol)
    assert torch.allclose(ref_dK, tri_dK, atol = atol, rtol = rtol)
    assert torch.allclose(ref_dV, tri_dV, atol = atol, rtol = rtol)
    assert torch.allclose(ref_dQ, tri_dQ, atol = atol, rtol = rtol)


