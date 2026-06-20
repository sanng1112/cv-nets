# Phân Tích Paper: CosFormer: Rethinking Softmax in Attention

**Thông tin:**  
- **arXiv:** [2202.08791](https://arxiv.org/abs/2202.08791) (ICLR 2022)  
- **Tác giả:** Zhen Qin, Weixuan Sun, Hui Deng, Dongxu Li, Yunshen Wei, Baohong Lv, Junjie Yan, Lingpeng Kong, Yiran Zhong (SenseTime, ANU)  
- **File PDF:** `2202.08791_CosFormer.pdf`  

---

## 1. Bối Cảnh và Động Cơ

Linear attention methods (Linear Transformers, Performer) có performance gap với softmax attention do approximation errors.

**Câu hỏi:** Thuộc tính nào của softmax là quan trọng? Làm sao để giữ chúng trong linear attention?

**Phát hiện:** Hai thuộc tính chính của softmax:
1. **Non-negativeness:** Attention weights $\geq 0$
2. **Non-linear re-weighting:** Tập trung phân bố attention

---

## 2. Toán Học Cốt Lõi

### 2.1 Phân Tích Softmax

Softmax attention:
$$A_{ij} = \frac{\exp(Q_i K_j^\top / \sqrt{d})}{\sum_k \exp(Q_i K_k^\top / \sqrt{d})}$$

Hai thuộc tính:
1. $A_{ij} \geq 0$ (non-negative entries)
2. $\|A_{i:}\|_p \gg \frac{1}{N}\|A_{i:}\|_1$ (concentrated distribution)

### 2.2 CosFormer Attention

$$V_i' = \frac{\sum_j [\text{ReLU}(Q_i)\text{ReLU}(K_j)^\top + \cos(Q_i - K_j)] V_j}{\sum_j [\text{ReLU}(Q_i)\text{ReLU}(K_j)^\top + \cos(Q_i - K_j)]}$$

**Phân tích từng phần:**

1. **ReLU part:** $\phi(Q_i)^\top \phi(K_j)$ với $\phi(x) = \text{ReLU}(x)$
   - Đảm bảo non-negativeness
   - Linear attention formulation

2. **Cosine part:** $\cos(Q_i - K_j)$
   - Re-weighting dựa trên khoảng cách
   - Concentrates attention distribution
   - Translation-invariant: $\cos(q - k) = \cos(q)\cos(k) + \sin(q)\sin(k)$

### 2.3 Linear Form

$$\text{CosFormer}(Q, K, V) = D^{-1}[(\phi(Q)\phi(K)^\top) \odot \cos(Q - K)] \cdot V$$

Có thể viết thành linear form:

$$\text{CosFormer}(Q, K, V) = D^{-1}[(\phi(Q) \otimes \cos(Q)) \cdot ((\phi(K) \otimes \cos(K))^\top V) + (\phi(Q) \otimes \sin(Q)) \cdot ((\phi(K) \otimes \sin(K))^\top V)]$$

**Complexity:** $O(N \cdot m)$ với $m = 2 \cdot d$ (ReLU + cos + sin).

---

## 3. Thí Nghiệm

### 3.1 Tasks

- **Language modeling:** WikiText-103
- **Text understanding:** GLUE benchmark
- **Long-Range Arena (LRA)** — 6 tasks with sequences từ 1K-4K tokens

### 3.2 Kết Quả

| Model | LRA Avg | WikiText-103 PPL | GLUE Avg |
|-------|---------|-----------------|----------|
| Transformer (softmax) | 57.5 | 20.5 | 83.2 |
| Linear Transformer | 48.2 | 24.1 | 80.1 |
| Performer | 49.8 | 22.8 | 81.0 |
| **CosFormer** | **58.3** | **20.8** | **82.8** |

**CosFormer beats softmax trên LRA!** Là linear attention method đầu tiên làm được điều này.

### 3.3 Ablation Studies

- **ReLU only:** 54.2 LRA (thiếu re-weighting)
- **Cosine only:** 55.1 LRA (thiếu non-negativeness)
- **ReLU + Cosine:** 58.3 LRA (cả hai)

→ **Cả hai properties đều quan trọng.**

---

## 4. Liên Quan Đến INLA

### 4.1 Giao Điểm Lớn Nhất

CosFormer là paper **có giao điểm lớn nhất với INLA**:

| Khía cạnh | CosFormer | INLA |
|-----------|-----------|------|
| **Feature map** | ReLU | Learned/information-theoretic |
| **Re-weighting** | Cosine distance | Information-theoretic |
| **Non-negativeness** | ✅ ReLU | ✅ Feature map design |
| **Concentration** | ✅ Cosine | ✅ Normalization |
| **Complexity** | $O(N)$ | $O(N)$ |

### 4.2 Học Hỏi Cho INLA

1. **Non-negativeness là quan trọng:**
   - Feature maps cần sinh ra positive values
   - ReLU, ELU, softplus là candidates

2. **Re-weighting cải thiện quality:**
   - CosFormer dùng cosine distance
   - INLA có thể dùng information-theoretic distance (KL, JS)

3. **Linear formulation là khả thi:**
   - CosFormer chứng minh linear attention có thể beats softmax
   - INLA có thể đạt được điều tương tự với normalization

### 4.3 Mở Rộng

- Kết hợp cosine re-weighting + information-theoretic normalization
- Thử nghiệm các re-weighting schemes khác nhau
- Theoretical analysis: tại sao re-weighting giúp?

---

## 5. Thách Thức

1. **Cosine re-weighting:** Có thể không optimal cho mọi tasks
2. **Rank preservation:** Chưa được phân tích lý thuyết
3. **Generalization:** Chưa rõ tại sao CosFormer beats softmax trên LRA
4. **Hyperparameters:** Cần tune weight giữa ReLU và cosine parts
5. **Transformer equivalence:** Chưa có RNN-equivalent form

---

## 6. Tóm Tắt

| Khía cạnh | Giá trị |
|-----------|---------|
| **Thuộc tính** | Non-negativeness + re-weighting |
| **Kết quả** | Linear attention beats softmax trên LRA |
| **Toán học** | Cosine re-weighting có thể linear hóa |
| **Giao điểm INLA** | **Lớn nhất** trong tất cả papers |

---

*Phân tích cho dự án INLA — 06/2026*
