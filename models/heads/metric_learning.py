import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceHead(nn.Module):
    """
    [Chi tiết hàm]: Đầu Phân loại Metric Learning (ArcFace)
    Chuyên dụng cho bài toán nhận diện khuôn mặt hoặc Zero-shot Retrieval.
    Ép các vector đặc trưng phân tách theo góc (Angular Margin) trên không gian Hypersphere thay vì không gian Euclidean.
    """
    def __init__(self, in_features: int, num_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels=None):
        # Normalize features và weights (Chiếu lên hình cầu)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        
        # Nếu đang Inference, chỉ trả về cosine similarity đã scale
        if labels is None:
            return cosine * self.s
            
        # Nếu đang Training, áp dụng Angular Margin Penalty
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Xử lý các góc viền an toàn
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Tạo One-hot mask để chỉ cộng Margin vào class mục tiêu (Ground truth)
        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Gộp Margin
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
