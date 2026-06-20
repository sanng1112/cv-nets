import torch
from layers.etf_classifier import ETFClassifier
from layers.information_bottleneck import InformationBottleneck
from layers.gcn_layer import SpectralScaledGCN

def test_proposals():
    print("Testing ETFClassifier...")
    etf = ETFClassifier(in_features=128, num_classes=10)
    x = torch.randn(32, 128)
    out = etf(x)
    assert out.shape == (32, 10), f"ETFClassifier failed, got {out.shape}"
    print("ETFClassifier OK.")

    print("Testing InformationBottleneck...")
    ib = InformationBottleneck(in_features=256, bottleneck_dim=64)
    ib.train()
    x2 = torch.randn(32, 256)
    out2 = ib(x2)
    assert out2.shape == (32, 64), f"InformationBottleneck failed, got {out2.shape}"
    assert hasattr(ib, 'kl_loss'), "InformationBottleneck missing kl_loss"
    print("InformationBottleneck OK.")

    print("Testing SpectralScaledGCN...")
    gcn = SpectralScaledGCN(in_channels=64, out_channels=32, target_singular_value=1.5)
    x3 = torch.randn(50, 64) # 50 nodes
    adj = torch.rand(50, 50) # Adjacency matrix
    out3 = gcn(x3, adj)
    assert out3.shape == (50, 32), f"SpectralScaledGCN failed, got {out3.shape}"
    
    # Check max singular value
    U, S, Vh = torch.linalg.svd(gcn.weight.data, full_matrices=False)
    assert S.max() <= 1.5 + 1e-5, f"GCN max singular value {S.max()} > target 1.5"
    print("SpectralScaledGCN OK.")

if __name__ == "__main__":
    test_proposals()
