import torch
from models.heads import (
    ClassificationHead, 
    DecoupledDetectionHead, 
    FCNHead,
    ArcFaceHead,
    HeatmapKeypointHead,
    ProtoMaskHead,
    PixelShuffleHead,
    DepthEstimationHead,
    OpticalFlowHead
)

def test_heads():
    B, C, H, W = 2, 256, 14, 14
    x = torch.randn(B, C, H, W)
    x_1d = torch.randn(B, C)
    
    # 1. Classification
    cls_head = ClassificationHead(C, num_classes=10)
    out_cls = cls_head(x)
    assert out_cls.shape == (B, 10), "Classification Head Failed"
    
    # 2. Segmentation
    seg_head = FCNHead(C, num_classes=21)
    out_seg = seg_head(x, target_size=(28, 28))
    assert out_seg.shape == (B, 21, 28, 28), "Segmentation Head Failed"
    
    # 3. Detection
    det_head = DecoupledDetectionHead(C, num_classes=80)
    out_det = det_head(x)
    assert out_det['cls'].shape == (B, 80, 14, 14), "Detection Cls Branch Failed"
    assert out_det['reg'].shape == (B, 4, 14, 14), "Detection Reg Branch Failed"
    assert out_det['obj'].shape == (B, 1, 14, 14), "Detection Obj Branch Failed"
    
    # 4. Metric Learning (ArcFace)
    labels = torch.randint(0, 100, (B,))
    arc_head = ArcFaceHead(C, num_classes=100)
    out_arc = arc_head(x_1d, labels)
    assert out_arc.shape == (B, 100), "ArcFace Head Failed"
    
    # 5. Keypoint Heatmap
    kp_head = HeatmapKeypointHead(C, num_keypoints=17, num_deconv_layers=3, num_filters=64)
    out_kp = kp_head(x)
    assert out_kp.shape == (B, 17, 112, 112), f"Keypoint Head Failed: {out_kp.shape}"
    
    # 6. Instance Segmentation ProtoNet
    proto_head = ProtoMaskHead(C, num_prototypes=32)
    out_proto = proto_head(x)
    assert out_proto.shape == (B, 32, 28, 28), "ProtoMask Head Failed"
    
    # 7. Super Resolution (PixelShuffle)
    sr_head = PixelShuffleHead(C, out_channels=3, upscale_factor=2)
    out_sr = sr_head(x)
    assert out_sr.shape == (B, 3, 28, 28), "PixelShuffle Head Failed"
    
    # 8. Depth Estimation
    depth_head = DepthEstimationHead(C, use_sigmoid=True)
    out_depth = depth_head(x)
    assert out_depth.shape == (B, 1, 14, 14), "Depth Estimation Head Failed"
    
    # 9. Optical Flow
    flow_head = OpticalFlowHead(C)
    out_flow = flow_head(x)
    assert out_flow.shape == (B, 2, 14, 14), "Optical Flow Head Failed"
    
    print("All 9 Heads OK!")

if __name__ == "__main__":
    test_heads()
