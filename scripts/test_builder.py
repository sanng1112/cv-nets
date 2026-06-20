import torch
from models.builder import CVNetModel
from engine.ema import ModelEMA

def test_builder_and_ema():
    # Khởi tạo mô hình (Tắt pretrained để chạy nhanh trong CI test)
    print("Testing Model Builder (ResNet18 + Classification Head)...")
    model = CVNetModel(
        backbone_name="resnet18", 
        head_type="classification", 
        head_kwargs={"num_classes": 100}, 
        pretrained=False
    )
    
    # Giả lập dữ liệu đầu vào
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 100), f"Builder Failed, Output Shape: {out.shape}"
    print("Model Builder OK.")
    
    # Kiểm thử kỹ thuật EMA
    print("Testing Model EMA...")
    ema = ModelEMA(model, decay=0.999)
    
    # Giả lập một bước huấn luyện làm thay đổi trọng số gốc
    with torch.no_grad():
        for p in model.parameters():
            p.data.add_(0.5)
            
    # Cập nhật EMA
    ema.update(model)
    print("Model EMA OK.")

if __name__ == "__main__":
    test_builder_and_ema()
