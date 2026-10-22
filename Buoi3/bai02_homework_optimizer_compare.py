"""Thực hành 2 — so sánh optimizer; Early Stopping riêng, vẫn tối đa 5 epoch.

8000 train, 1500 validation; MLP 784-128-10; batch128; LR đúng script mẫu.
Chạy: python Buoi3/bai02_homework_optimizer_compare.py
"""
# %% [markdown]
# # Thực hành 2 — So sánh phương pháp tối ưu
# **Lê Sơn Trường**. Nguồn: `baitap.txt`, slide trang 27 và mã mẫu trong `ma_nguon_goc/`.
# Cùng 8000/1500 mẫu, MLP 784 → 128 → 10, ReLU, batch128 và **5 epoch**.
# Giữ đúng cấu hình mẫu: SGD LR=0.05, Momentum LR=0.03, momentum=0.9;
# Adam LR=0.001. Cùng khởi tạo và thứ tự batch trong cả ba lần chạy.
# Vì LR khác nhau, kết luận chỉ áp dụng cho ba cấu hình này.

# %%
from common import BASE, Config, save_experiment
from data import load_subset
from bai02_class_training_loop import train_configuration

CONFIGS = [Config("sgd", "sgd", .05, 128),
           Config("momentum", "sgd", .03, 128, momentum=.9),
           Config("adam", "adam", .001, 128)]
EPOCHS = 5

# %% [markdown]
# ## Cơ bản và khá: cùng kiến trúc, khác optimizer
# Dùng cùng vòng lặp đã viết rõ ở bài 1: zero_grad → forward → loss → backward → step.
# Mỗi epoch lưu train/validation loss, accuracy và Macro-F1, vẽ learning curve.
# Bảng chính lấy epoch cuối, không dùng Early Stopping để đổi ngân sách so sánh.

# %%
def run_comparison(train_set, val_set, *, early_stopping=False):
    runs = [train_configuration(c, train_set, val_set, batch_size=128, epochs=EPOCHS,
                                early_stopping=early_stopping) for c in CONFIGS]
    folder = "bai2_early_stopping" if early_stopping else "bai2_optimizer"
    title = ("Thực hành 2 — Early Stopping (patience=2, tối đa 5 epoch)" if early_stopping
             else "Thực hành 2 — SGD / Momentum / Adam (5 epoch)")
    return save_experiment(runs, BASE / "results" / folder, val_set, title, CONFIGS)

# %% [markdown]
# ## Nâng cao: Early Stopping
# Theo dõi validation loss. Nếu 2 epoch liên tiếp không giảm (min_delta=0), dừng.
# Lưu bản sao trọng số và khôi phục epoch có validation loss thấp nhất.
# Chạy riêng với giới hạn vẫn là 5 epoch; nếu loss liên tục cải thiện, sẽ không dừng sớm.
# Không tự tăng epoch chỉ để tạo ra một ví dụ dừng.

# %%
def main():
    train_set, val_set = load_subset(8000, 1500)
    baseline = run_comparison(train_set, val_set)
    advanced = run_comparison(train_set, val_set, early_stopping=True)
    return baseline, advanced


if __name__ == "__main__":
    main()
