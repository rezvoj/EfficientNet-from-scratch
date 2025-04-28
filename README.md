DONE:

Activations:
    ✅ ReLU
    ✅ Swish / SiLU
    ✅ GELU
    ✅ Sigmoid

Elementwise Operations:
    ✅ Add / Multiply / Subtract / Divide
    ✅ Matrix-to-Matrix Elementwise (Residual connections, scaling, etc.)

Tensor Manipulation:
    ✅ Transposes, Reordering dims
    ✅ Flatten / Reshape / View (unpatching / repatching)


✅ [JUST UP DOWN OPERATION] 
4. Global Average Pooling (GAP) ? 
    up and down


[LRNABLE] 2. BatchNorm2D
    Normalize across batch for each channel.
    Needs:
        Running mean / var.
        Learnable gamma and beta.
    In inference mode, just a per-channel affine transform.


value initialization (random?)



✅ [LRNABLE] 5. Linear (Fully Connected Layer) 
    Used in:
        SE blocks (squeeze → excitation).
        Final classification head.
    // parallel reduction for sum
    // Assembled


✅ 7. Softmax
    Final classification layer over logits.
    You’ll want numerically stable version.

// Standard Conv:
(c_out, batch * h_out * w_out) = (c_out, c_in * h_ker * w_ker) @ (c_in * h_ker * w_ker, batch * h_out * w_out)
Backwards Conv to calc kernel grad:
(c_out, c_in * h_ker * w_ker) = (c_out, batch * h_out * w_out) @ (batch * h_out * w_out, c_in * h_ker * w_ker)
Backwards Conv to calc dX from dY
(C_in, C_out * K_180 * K_180) @ (C_out * K * K, N * H_in * W_in) = (C_in, N * H_in * W_in)
Depthwise Conv - to calc kernel grad:
(c, 1, h_ker * w_ker) = (c, 1, batch * h_out * w_out) @ (c, batch * h_out * w_out, h_ker * w_ker)




---



TODO:
OPTIMIZERS ...
