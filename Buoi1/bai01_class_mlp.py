"""Bài 1: so sánh MLP có 8, 32, 64 và 128 neuron mỗi tầng ẩn."""

from sklearn.datasets import make_moons

from common import Config, parse_args, prepare_data, run_experiments


def main():
    args = parse_args("Bài 1 - MLP trên make_moons", "bai1")
    X, y = make_moons(n_samples=1200, noise=0.25, random_state=42)
    data = prepare_data(X, y)
    configs = [Config(f"width_{width}", (width, width)) for width in (8, 32, 64, 128)]
    run_experiments(
        configs, data, args, title="Bài 1 – Ảnh hưởng của số neuron",
        dataset="make_moons (1.200 mẫu, noise=0.25)", boundaries=True,
    )


if __name__ == "__main__":
    main()
