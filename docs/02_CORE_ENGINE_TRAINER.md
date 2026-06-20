# 02. ĐỘNG CƠ CỐT LÕI (CORE ENGINE & TRAINER)

Bên trong thư mục `engine/`, file vĩ đại nhất và phức tạp nhất chính là `trainer.py`. File này chứa class `BaseTrainer`. Nó bao bọc xung quanh thuật toán Backpropagation truyền thống của PyTorch bằng vô số lớp khiên bảo vệ và bộ kích tốc phần cứng.

Dưới đây là chi tiết mã nguồn và cách dùng của TẤT CẢ các module trong Engine.

---

## 1. DDP (Distributed Data Parallel) - Phân Phối Đa GPU
Khi chạy trên cụm Server (Cluster), `BaseTrainer` tự động dò tìm cấu hình phân tán:
```python
self.is_ddp = dist.is_available() and dist.is_initialized()
if self.is_ddp:
    self.local_rank = dist.get_rank()
    self.device = torch.device(f"cuda:{self.local_rank}")
    self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank])
```
**Chống rác log (Rank-aware Logging):** Trong DDP, nếu có 8 GPU thì vòng lặp chạy 8 lần song song. Nếu đặt lệnh `print` bình thường, màn hình sẽ nổ tung với 8 dòng log in ra cùng lúc. CV-Nets xử lý bằng cờ `self.is_main_process = (self.local_rank == 0)`. Chỉ GPU số 0 mới được quyền in ra Terminal và lưu Checkpoint!

---

## 2. AMP (Automatic Mixed Precision)
```python
self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

with torch.amp.autocast('cuda', enabled=self.use_amp):
    outputs = self.model(inputs)
    loss = self.criterion(outputs, targets)
self.scaler.scale(loss).backward()
```
Thay vì dùng `loss.backward()`, việc sử dụng `GradScaler` là bắt buộc khi bật AMP. Do Loss bị ép về Float16, số sẽ quá nhỏ (Underflow) và biến thành 0. `scaler.scale` sẽ nhân Loss lên một tỷ lệ khổng lồ trước khi tính đạo hàm, sau đó chia ngược lại lúc gọi `scaler.step()`.

---

## 3. Gradient Accumulation
Được điều khiển qua tham số YAML `accumulation_steps: N`.
```python
loss = loss / self.accum_steps # Chia trung bình Loss
self.scaler.scale(loss).backward()

if (batch_idx + 1) % self.accum_steps == 0:
    self.scaler.step(self.optimizer)
    self.scaler.update()
    self.optimizer.zero_grad()
```
Code này cho phép bạn nhân bản kích thước Batch Size lý thuyết lên `N` lần mà không tốn thêm 1 Byte VRAM nào.

---

## 4. History Plotter (`utils/plotter.py`) & Checkpoint Saving
Sau mỗi Epoch, GPU số 0 sẽ lưu tiến trình:
1. `history.json` / `.csv`: Dành cho phân tích Pandas sau này.
2. `training_history.png`: Biểu đồ tự động.
3. Checkpoint: Cập nhật `checkpoint_best.pt` nếu độ chính xác (Val Acc / F1) cao hơn epoch trước.

---

## 5. Seed Everything (`utils/seed.py`)
Tại sao AI lại "chạy mỗi lúc ra một kết quả khác nhau"? Là do 4 hàm sinh ngẫu nhiên sau. CV-Nets khóa chặt toàn bộ:
```python
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Tắt thuật toán tự tối ưu Convolution của Cudnn
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
```

---

## 6. EMA (Exponential Moving Average) - `engine/ema.py`
EMA lưu giữ một phiên bản "Bóng ma" (Shadow Model) của mạng hiện tại. Bóng ma này không học từ Gradient, mà học bằng cách lấy Trung bình cộng có trọng số của các Epoch trước đó.
```python
# Cập nhật Shadow Model:
new_weight = decay * shadow_weight + (1 - decay) * current_weight
```
EMA giúp độ chính xác của Model cực kỳ ổn định, không bị răng cưa nhảy vọt khi gặp Batch nhiễu.
