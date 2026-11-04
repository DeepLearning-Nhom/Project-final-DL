"""
Tạo demo/demo_model.pt từ các checkpoint đã huấn luyện, KHÔNG cần mở notebook.

Cách chạy (đặt file này cạnh thư mục checkpoints/ và results/, hoặc cạnh app.py):
    python make_demo_model.py            # tự chọn cấu hình + fold tốt nhất theo results/runs.csv
    python make_demo_model.py E1 3       # chọn tay: cấu hình E1, fold 3
    python make_demo_model.py --ckpt-dir "D:/đường/dẫn/checkpoints"

Kết quả: demo/demo_model.pt cạnh file này (app.py sẽ tự tìm thấy).
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent

VI_LIST = [
    "Chó sủa", "Gà trống gáy", "Lợn kêu", "Bò rống", "Ếch kêu", "Mèo kêu", "Gà mái",
    "Côn trùng", "Cừu kêu", "Quạ kêu",
    "Mưa", "Sóng biển", "Lửa cháy tí tách", "Dế kêu", "Chim hót", "Nước nhỏ giọt",
    "Gió", "Nước chảy", "Xả bồn cầu", "Sấm sét",
    "Em bé khóc", "Hắt hơi", "Vỗ tay", "Thở", "Ho", "Bước chân", "Cười", "Đánh răng",
    "Ngáy", "Uống nước",
    "Gõ cửa", "Click chuột", "Gõ bàn phím", "Cửa gỗ kẽo kẹt", "Mở lon", "Máy giặt",
    "Máy hút bụi", "Chuông báo thức", "Đồng hồ tích tắc", "Kính vỡ",
    "Trực thăng", "Cưa máy", "Còi hú", "Còi xe", "Động cơ", "Tàu hỏa", "Chuông nhà thờ",
    "Máy bay", "Pháo hoa", "Cưa tay",
]
GROUP_NAMES = ["Động vật", "Thiên nhiên và nước", "Âm thanh con người",
               "Âm thanh trong nhà", "Tiếng ồn đô thị"]


def find_ckpt_dir(custom):
    candidates = [Path(custom)] if custom else []
    candidates += [HERE / "checkpoints", Path.cwd() / "checkpoints",
                   HERE.parent / "checkpoints"]
    for p in candidates:
        if p.is_dir() and any(p.glob("*.pt")):
            return p
    sys.exit("Không tìm thấy thư mục checkpoints (chứa các file E1_fold1.pt...). "
             "Dùng: python make_demo_model.py --ckpt-dir \"đường/dẫn/checkpoints\"")


def pick_best(ckpt_dir):
    # Chọn cấu hình có Accuracy trung bình cao nhất, rồi fold tốt nhất của cấu hình đó
    for runs_csv in (ckpt_dir.parent / "results" / "runs.csv", HERE / "results" / "runs.csv"):
        if runs_csv.exists():
            break
    else:
        sys.exit("Không thấy results/runs.csv. Hãy chọn tay: python make_demo_model.py E1 1")
    by_exp = defaultdict(list)
    with open(runs_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_exp[row["exp"]].append((float(row["acc"]), int(row["fold"])))
    means = {e: sum(a for a, _ in v) / len(v) for e, v in by_exp.items()}
    for e in sorted(means):
        print(f"  {e}: Accuracy trung bình {means[e] * 100:.2f}% ({len(by_exp[e])} fold)")
    exp = max(means, key=means.get)
    acc, fold = max(by_exp[exp])
    return exp, fold


def main():
    args = sys.argv[1:]
    custom = None
    if "--ckpt-dir" in args:
        i = args.index("--ckpt-dir")
        custom = args[i + 1]
        del args[i:i + 2]
    ckpt_dir = find_ckpt_dir(custom)
    print("Thư mục checkpoint:", ckpt_dir)

    if len(args) >= 2:
        exp, fold = args[0], int(args[1])
    else:
        print("Tự chọn mô hình tốt nhất:")
        exp, fold = pick_best(ckpt_dir)
    path = ckpt_dir / f"{exp}_fold{fold}.pt"
    if not path.exists():
        sys.exit(f"Không thấy {path}")
    print(f"Dùng {exp}, fold {fold}: {path.name}")

    ck = torch.load(path, map_location="cpu", weights_only=False)
    out_dir = HERE / "demo"
    out_dir.mkdir(exist_ok=True)
    torch.save(dict(
        model=ck["model"], n_mels=ck["n_mels"], conv_per_block=ck.get("conv_per_block", 1),
        state={k: v.cpu() for k, v in ck["state"].items()},
        mean=float(ck["mean"]), std=float(ck["std"]),
        class_names=VI_LIST, group_names=GROUP_NAMES,
        sample_rate=ck.get("sample_rate", 22050), n_fft=1024, hop_length=512,
        duration=5, threshold=0.5, experiment=exp,
    ), out_dir / "demo_model.pt")
    print("Đã lưu:", out_dir / "demo_model.pt")


if __name__ == "__main__":
    main()
