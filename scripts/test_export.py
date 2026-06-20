import torch
from engine.quantization import Quantizer
from engine.exporter import Exporter
from models.heads import ClassificationHead
import os

def test_export_and_quantize():
    print("Testing Inference Mechanisms (Quantization & Export)...")
    model = ClassificationHead(256, 10).eval()
    dummy_input = torch.randn(2, 256, 14, 14)
    
    # 1. Test FP16
    fp16_model = Quantizer.to_fp16(model)
    out_fp16 = fp16_model(dummy_input.half())
    assert out_fp16.dtype == torch.float16, "FP16 Quantization Failed"
    print("[OK] FP16 Conversion.")
    
    # 1.5 Test BF16
    bf16_model = Quantizer.to_bf16(model)
    out_bf16 = bf16_model(dummy_input.to(torch.bfloat16))
    assert out_bf16.dtype == torch.bfloat16, "BF16 Quantization Failed"
    assert Quantizer.check_quantization_health(bf16_model, dummy_input.to(torch.bfloat16))
    print("[OK] BF16 Conversion & Sanity Check.")
    
    # 2. Test INT8 Dynamic Quantization
    int8_model = Quantizer.to_int8_dynamic(model)
    out_int8 = int8_model(dummy_input)
    assert out_int8.shape == (2, 10), "INT8 Dynamic Quantization Failed"
    print("[OK] INT8 Dynamic Quantization (PTQ).")
    
    # 2.5 Test Quantization-Aware Training (QAT)
    qat_model = Quantizer.prepare_qat(model)
    # Giả lập 1 bước Forward Pass mô phỏng QAT trong lúc Train
    out_qat = qat_model(dummy_input)
    assert out_qat.shape == (2, 10), "QAT Forward Failed"
    # Convert sang Hard INT8 sau khi train xong
    hard_int8_model = Quantizer.convert_qat(qat_model)
    assert hard_int8_model is not None, "QAT Convert Failed"
    print("[OK] Quantization-Aware Training (QAT).")
    
    # 3. Test ONNX Export
    os.makedirs("temp_checkpoints", exist_ok=True)
    onnx_path = "temp_checkpoints/model.onnx"
    Exporter.export_onnx(model, dummy_input, onnx_path)
    assert os.path.exists(onnx_path), "ONNX Export Failed"
    print("[OK] ONNX Exporter.")
    
    # 4. Test TorchScript Export
    ts_path = "temp_checkpoints/model.pt"
    Exporter.export_torchscript(model, dummy_input, ts_path)
    assert os.path.exists(ts_path), "TorchScript Export Failed"
    print("[OK] TorchScript Exporter.")
    
    print("Export & Quantization Full Pipeline OK!")

if __name__ == "__main__":
    test_export_and_quantize()
