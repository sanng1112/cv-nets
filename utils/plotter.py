import matplotlib.pyplot as plt
import os

class HistoryPlotter:
    """
    [Chi tiết hàm]: Trình vẽ biểu đồ lịch sử huấn luyện (Training History Plotter).
    Hỗ trợ xuất biểu đồ Loss và Metric ra file ảnh (Offline), cực kỳ hữu ích khi 
    viết báo cáo khoa học hoặc không có mạng để dùng WandB.
    """
    @staticmethod
    def plot_and_save(history: dict, save_dir: str = "temp_checkpoints"):
        """
        history = {
            'train_loss': [0.9, 0.5, 0.2],
            'val_loss': [1.0, 0.6, 0.3],
            'train_metric': [50, 75, 90],
            'val_metric': [45, 70, 85]
        }
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Tìm số epoch thực tế
        epochs = []
        for key in history:
            if isinstance(history[key], list) and len(history[key]) > 0:
                epochs = range(1, len(history[key]) + 1)
                break
                
        if not epochs:
            print("[Plotter] Dữ liệu history trống, không thể vẽ biểu đồ.")
            return

        # Tự động tìm khóa (key) của Metric nếu người dùng không truyền đúng 'train_metric'
        if 'train_metric' not in history:
            for k in history.keys():
                if k.startswith('train_') and k not in ['train_loss', 'train_epoch', 'train_lr']:
                    history['train_metric'] = history[k]
                    break
                    
        if 'val_metric' not in history:
            for k in history.keys():
                if k.startswith('val_') and k not in ['val_loss', 'val_epoch', 'val_lr']:
                    history['val_metric'] = history[k]
                    break

        plt.figure(figsize=(12, 5))
        
        # 1. Vẽ đồ thị Loss
        plt.subplot(1, 2, 1)
        if 'train_loss' in history: plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', linewidth=2)
        if 'val_loss' in history: plt.plot(epochs, history['val_loss'], label='Val Loss', marker='s', linewidth=2)
        plt.title('Hành trình của Loss (Càng thấp càng tốt)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 2. Vẽ đồ thị Độ chính xác (Metric)
        plt.subplot(1, 2, 2)
        if 'train_metric' in history: plt.plot(epochs, history['train_metric'], label='Train Metric', marker='o', linewidth=2, color='green')
        if 'val_metric' in history: plt.plot(epochs, history['val_metric'], label='Val Metric', marker='s', linewidth=2, color='orange')
        plt.title('Hành trình của Metric (Càng cao càng tốt)')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        save_path = os.path.join(save_dir, "training_history.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Plotter] Đã xuất bản biểu đồ huấn luyện đẹp mắt tại: {save_path}")
