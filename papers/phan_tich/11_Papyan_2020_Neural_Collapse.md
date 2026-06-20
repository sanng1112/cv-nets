# Phân Tích Paper: Prevalence of Neural Collapse

## During the Terminal Phase of Deep Learning Training

**Thông tin:**  
- **arXiv:** [2008.08186](https://arxiv.org/abs/2008.08186) (PNAS 2020)  
- **Tác giả:** Vardan Papyan, X. Y. Han, David L. Donoho (Stanford)  
- **File PDF:** `2008.08186_Papyan_Neural_Collapse.pdf`  

---

## 1. Bối Cảnh và Động Cơ

Trong training classification deepnets, sau khi training error = 0 (Terminal Phase of Training — TPT), loss tiếp tục giảm về 0.

**Phát hiện:** Trong TPT, xuất hiện **Neural Collapse** — 4 hiện tượng hình học đẹp đẽ.

---

## 2. Bốn Hiện Tượng Neural Collapse

### 2.1 NC1: Variability Collapse

**Within-class covariance → 0:**

$$\Sigma_W = \frac{1}{K} \sum_{k=1}^K \frac{1}{n_k} \sum_{i: y_i = k} (h_i - \mu_k)(h_i - \mu_k)^\top \to 0$$

Mọi activations trong cùng class collapse về class mean.

**Metric:** $\Phi_1 = \text{Tr}(\Sigma_W \Sigma_B^\dagger)/K$, với $\Sigma_B$ là between-class covariance. $\Phi_1 \to 0$.

### 2.2 NC2: Convergence to Simplex ETF

**Simplex Equiangular Tight Frame (ETF):**

$$M = [\mu_1 - \mu_G, \mu_2 - \mu_G, \ldots, \mu_K - \mu_G] \in \mathbb{R}^{p \times K}$$

$$M^\top M = \frac{K}{K-1} \left(I_K - \frac{1}{K}\mathbf{1}\mathbf{1}^\top\right)$$

**Tính chất:**
- Tất cả vectors có cùng độ dài
- Tất cả góc giữa các vectors đều bằng nhau ($\arccos(-1/(K-1))$)
- Maximal separation trên sphere

### 2.3 NC3: Self-duality

Classifiers $W \in \mathbb{R}^{K \times p}$ collapse về class means:

$$W_k \propto \mu_k - \mu_G$$

**Metric:** $\Phi_3 = \|W^\top - M\|$.

### 2.4 NC4: Nearest Class Center

Decision rule → Nearest Class Center (NCC):

$$\arg\max_k \langle W_k, h \rangle \to \arg\min_k \|h - \mu_k\|$$

---

## 3. Thí Nghiệm

### 3.1 Thiết Lập

- **Architectures:** VGG, ResNet, DenseNet
- **Datasets:** CIFAR-10/100, ImageNet, MNIST, etc.
- **Measurements:** $\Phi_1, \Phi_2, \Phi_3, \Phi_4$ qua training

### 3.2 Kết Quả

- **NC1 ($\Phi_1$):** Giảm exponentially sau epoch zero-error
- **NC2 ($\Phi_2$):** Class means hội tụ về simplex ETF
- **NC3 ($\Phi_3$):** Self-duality xuất hiện
- **NC4 ($\Phi_4$):** NCC trở nên chính xác

**Generalization và robustness tăng cùng với NC.**

---

## 4. Liên Quan Đến INLA

### 4.1 Connection với Rank Collapse

| Khía cạnh | Neural Collapse | Rank Collapse |
|-----------|----------------|---------------|
| **Hiện tượng** | Last-layer activations → ETF | Attention output → rank-1 |
| **Nguyên nhân** | TPT optimization | Attention degeneracy |
| **Kết quả** | Structured geometry | Loss of expressivity |
| **Liên quan** | Cùng là collapse phenomena | Có thể tương quan |

### 4.2 Kỹ Thuật Cho INLA

1. **Geometric analysis:**
   - Phân tích attention representations dưới góc nhìn NC
   - INLA có dẫn đến NC khác? Hay chống lại?

2. **Spectrum evolution:**
   - NC: class means → simplex ETF
   - INLA: attention output → cần bảo toàn spectrum

3. **Metrics:**
   - Similar metrics cho attention collapse
   - Effective rank thay vì $\Phi_1$

---

## 5. Thách Thức

1. **Chỉ last-layer:** NC chỉ quan sát ở layer cuối
2. **Supervised classification:** Chưa rõ cho self-supervised
3. **Attention chưa được xét:** NC chưa nghiên cứu cho transformers
4. **Cần mở rộng:** Cho attention representations và rank dynamics

---

## 6. Tóm Tắt

| Khía cạnh | Giá trị |
|-----------|---------|
| **Phát hiện** | 4 hiện tượng NC trong TPT |
| **Toán học** | Simplex ETF, variability collapse |
| **Với INLA** | Framework cho collapse phenomena analysis |

---

*Phân tích cho dự án INLA — 06/2026*
