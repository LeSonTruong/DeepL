"""Bài 2: khảo sát độ sâu, độ rộng, activation và L2 trên digits."""

from sklearn.datasets import load_digits

from common import Config, parse_args, prepare_data, run_experiments


def main():
    args = parse_args("Bài 2 - Khảo sát MLP trên digits", "bai2")
    X, y = load_digits(return_X_y=True)
    data = prepare_data(X, y)
    # Mỗi cấu hình chỉ đổi một yếu tố so với baseline để dễ giải thích.
    configs = [
        Config("baseline", (32, 32)),
        Config("depth_1", (32,)),
        Config("depth_3", (32, 32, 32)),
        Config("width_8", (8, 8)),
        Config("width_64", (64, 64)),
        Config("width_128", (128, 128)),
        Config("tanh", (32, 32), activation="tanh"),
        Config("sigmoid", (32, 32), activation="logistic"),
        Config("l2_0.01", (32, 32), alpha=0.01),
        Config("l2_0.1", (32, 32), alpha=0.1),
    ]
    run_experiments(
        configs, data, args, title="Bài 2 – Khảo sát kiến trúc và regularization",
        dataset="digits (1.797 mẫu, 64 đặc trưng, 10 lớp)",
    )


if __name__ == "__main__":
    main()
