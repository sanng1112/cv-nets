import torch
from layers.attention.multihead_attention import MultiHeadAttention
from layers.attention.fixed_multihead_attention import FixedMultiHeadAttention
from layers.attention.linear_attention import LinearAttention
from layers.attention.inla_layer import INLALayer
from blocks.transformer_block import TransformerBlock

def test_attention():
    print("Testing MultiHeadAttention...")
    mha = MultiHeadAttention(embed_dim=256, num_heads=8)
    x = torch.randn(2, 64, 256)
    out = mha(x)
    assert out.shape == (2, 64, 256), f"MHA failed, got {out.shape}"
    print("MHA OK.")

    print("Testing FixedMultiHeadAttention...")
    fmha = FixedMultiHeadAttention(embed_dim=256, num_heads=8, head_dim=128)
    out = fmha(x)
    assert out.shape == (2, 64, 256), f"FixedMHA failed, got {out.shape}"
    print("FixedMHA OK.")

    print("Testing LinearAttention (Bidirectional)...")
    lin = LinearAttention(embed_dim=256, num_heads=8, causal=False)
    out = lin(x)
    assert out.shape == (2, 64, 256), f"Linear (bi) failed, got {out.shape}"
    print("Linear Attention (Bi) OK.")

    print("Testing LinearAttention (Causal)...")
    lin_c = LinearAttention(embed_dim=256, num_heads=8, causal=True)
    out = lin_c(x)
    assert out.shape == (2, 64, 256), f"Linear (causal) failed, got {out.shape}"
    print("Linear Attention (Causal) OK.")

    print("Testing INLALayer...")
    inla = INLALayer(embed_dim=256, num_heads=8, bottleneck_dim=16, expansion_dim=128)
    out = inla(x)
    assert out.shape == (2, 64, 256), f"INLALayer failed, got {out.shape}"
    print("INLA Layer OK.")

    print("Testing TransformerBlock...")
    block = TransformerBlock(embed_dim=256, num_heads=8, attention_type="multihead")
    out = block(x)
    assert out.shape == (2, 64, 256), f"TransformerBlock failed, got {out.shape}"
    print("TransformerBlock OK.")

if __name__ == "__main__":
    test_attention()
