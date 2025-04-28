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


------

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


TODO:
OPTIMIZERS ...
