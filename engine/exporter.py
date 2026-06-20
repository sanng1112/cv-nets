import torch
import torch.nn as nn
import os

class Exporter:
    """
    [Chi tiết hàm]: Trình xuất định dạng (Model Exporter)
    Đóng gói mô hình từ PyTorch sang các định dạng triển khai công nghiệp (Deployment)
    để tích hợp vào C++, Java, Node.js, TensorRT, OpenVINO, hoặc Mobile.
    """
    @staticmethod
    def export_onnx(model: nn.Module, dummy_input: torch.Tensor, save_path: str, opset_version: int = 14):
        """Xuất mô hình chuẩn ONNX"""
        model.eval()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Định nghĩa dynamic batch size (Cho phép truyền vào batch ảnh linh hoạt)
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        
        print(f"[Export] Đang xuất mô hình ONNX ra file: {save_path} ...")
        # Quá trình này sẽ trace (theo vết) tensor đi xuyên suốt qua graph của mạng
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True, # Tối ưu hóa hằng số
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        print("[Export] Xuất ONNX thành công!")

    @staticmethod
    def export_torchscript(model: nn.Module, dummy_input: torch.Tensor, save_path: str):
        """Xuất mô hình chuẩn TorchScript (Thực thi độc lập không cần thư viện Python)"""
        model.eval()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"[Export] Đang rà soát (Trace) TorchScript ra file: {save_path} ...")
        traced_script_module = torch.jit.trace(model, dummy_input)
        traced_script_module.save(save_path)
        print("[Export] Xuất TorchScript thành công!")
