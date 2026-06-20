# Phân Tích Paper: Low-Rank Bottleneck in Multi-Head Attention Models

**Thông tin:**  
- **arXiv:** [2002.07028](https://arxiv.org/abs/2002.07028) (Feb 2020)  
- **Tác giả:** Srinadh Bhojanapalli, Chulhee Yun, Ankit Singh Rawat, Sashank J. Reddi, Sanjiv Kumar (Google Research + MIT)  
- **File PDF:** `2002.07028_Bhojanapalli_Low_Rank_Bottleneck.pdf`  

---

## 1. Bối Cảnh và Động Cơ

Các Transformer models thành công nhờ **embedding dimension lớn** (BERT: 1024, GPT-2: 1600). Tuy nhiên, điều này dẫn đến models quá lớn.

**Câu hỏi:** Tại sao cần embedding dimension lớn? Có ràng buộc gì từ multi-head attention?

**Phát hiện:** Có **low-rank bottleneck** trong multi-head attention — rank của attention head bị giới hạn bởi head size.

---

## 2. Toán Học Cốt Lõi

### 2.1 Multi-Head Attention

Standard multi-head attention với $h$ heads:

$$d_v = d_{model} / h \quad \text{(head size)}$$

Mỗi head:
1. Project input xuống $d_v$-dim subspace
2. Compute self-attention trong subspace đó
3. Output: $SA_h(X) \in \mathbb{R}^{n \times d_v}$

### 2.2 Low-Rank Bottleneck

**Theorem (informal):** Với $d_v$ heads, mỗi head có kích thước $d_v$, tổng rank bị giới hạn:

$$\text{rank}(\text{multi-head output}) \leq \sum_{i=1}^h \min(d_v, n)$$

Với $d_v = d_{model}/h$:

$$\text{rank}(\text{output}) \leq h \cdot \min(d_{model}/h, n)$$

**Khi $d_{model}/h < n$:** Rank ≤ $d_{model}$ (bottleneck xuất hiện).  
**Khi $n$ rất lớn (long sequences):** Bottleneck càng nghiêm trọng.

### 2.3 Giải Pháp Đề Xuất

Đặt **head size** = sequence length $n$:

$$d_v = n$$

Khi đó:
$$\text{rank}(\text{output}) \leq \min(h \cdot n, d_{model})$$

**Expressive power tăng lên đáng kể** vì rank có thể lên đến $d_{model}$.

---

## 3. Phân Tích Expressive Power

### 3.1 So Sánh Expressive Power

| Configuration | Standard ($d_v = d_m/h$) | Proposed ($d_v = n$) |
|--------------|--------------------------|---------------------|
| Single head | rank ≤ min($d_m$, $n$) | rank ≤ min($n$, $d_m$) |
| Multi-head ($h$) | rank ≤ min($d_m$, $h \cdot n$) | rank ≤ min($d_m$, $h \cdot n$) |
| **Khi $n$ lớn** | rank bị chặn bởi $d_m$ | rank bị chặn bởi $h \cdot n$ |
| **Ưu điểm** | — | Có thể đạt rank cao hơn |

### 3.2 Khi Nào Bottleneck Xảy Ra?

Bottleneck xảy ra khi:
$$d_v = d_{model}/h < n$$

Hay:
$$d_{model} < h \cdot n$$

Trong thực tế, $n$ thường = 512 (BERT) và $d_{model}$ = 1024, $h$ = 16:
- Standard: $d_v = 1024/16 = 64$ → rank ≤ 64 (với $n = 512$)
- Proposed: $d_v = 512$ → rank có thể lên đến 1024

**Cải thiện 16x về rank capacity!**

---

## 4. Thí Nghiệm

### 4.1 Thiết Lập

- Model: Transformer với BERT-style architecture
- Tasks: SQuAD, MNLI
- So sánh standard head size vs proposed head size

### 4.2 Kết Quả

- **Standard (head size = 64):** Cần embedding dimension lớn để đạt performance tốt
- **Proposed (head size = sequence length):** Đạt performance tương đương với embedding dimension nhỏ hơn
- Trên SQuAD: Proposed method cho F1 cao hơn với cùng số parameters
- Trên MNLI: Cải thiện accuracy

---

## 5. Liên Quan Đến INLA

### 5.1 Tầm Quan Trọng

Đây là **paper nền tảng cho động cơ của INLA**. Nó chỉ ra rằng:

1. **Multi-head attention có inherent rank limitation**
2. **Low-rank bottleneck** giới hạn expressive power
3. **Cần thiết kế lại attention** để vượt qua bottleneck này

### 5.2 Áp Dụng Cho INLA

| Khía cạnh | Liên quan |
|-----------|-----------|
| **Linear attention** | Cũng bị low-rank bottleneck tương tự (hoặc tệ hơn) |
| **Feature maps** | Có thể dùng feature maps để tăng effective rank |
| **Normalization** | INLA normalization có thể giúp bảo toàn rank |
| **Kết nối** | INLA cần chứng minh rank preservation |

### 5.3 Mở Rộng

- Nghiên cứu low-rank bottleneck trong linear attention cụ thể
- So sánh effective rank của INLA vs softmax attention
- Phân tích trade-off giữa rank và computational efficiency

---

## 6. Thách Thức và Hạn Chế

1. **Scalability:** Head size = sequence length không scalable cho $n$ rất lớn ($>10K$)
2. **Computational cost:** Chưa phân tích computational overhead của proposed method
3. **Empirical validation:** Chỉ thử nghiệm trên BERT-scale models
4. **Linear attention chưa được xét:** Paper tập trung vào softmax attention
5. **Cần lý thuyết mạnh hơn:** Chưa có lower bound cho expressive power gain

---

## 7. Tóm Tắt

| Khía cạnh | Giá trị |
|-----------|---------|
| **Vấn đề** | Low-rank bottleneck: $d_v \ll n$ → rank giới hạn |
| **Giải pháp** | Head size = sequence length |
| **Cải thiện** | Rank capacity tăng từ $d_m$ lên $h \cdot n$ |
| **Kết quả** | Performance tốt hơn với cùng số parameters |
| **Với INLA** | Nền tảng cho động cơ: cần vượt qua rank bottleneck |

---

*Phân tích cho dự án INLA — 06/2026*
