"""Ứng dụng demo phân loại âm thanh môi trường ESC-50 (Streamlit).

Chạy:  streamlit run app.py
Cần một trong hai file do notebook 15 (best model) xuất ra, đặt trong thư mục demo/ cạnh app.py:
  - demo/demo_ensemble.pt : ensemble nhiều mô hình + TTA (độ chính xác cao hơn)
  - demo/demo_model.pt    : một mô hình đơn
File app.py này tự chứa định nghĩa mô hình, không cần esc_lib.py và không cần Google Colab.
"""
import io
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DURATION = 5                                      # giây
MAX_SEGMENTS = 24                                 # tối đa số đoạn 5 giây cho mỗi file dài
DEFAULT_SHIFTS = (0.0, 0.2, 0.4, 0.6, 0.8)        # dịch vòng cho TTA (giống notebook 14)
APP_DIR = Path(__file__).resolve().parent
SEARCH_DIRS = ([Path(os.environ["ESC_DEMO_DIR"])] if os.environ.get("ESC_DEMO_DIR") else []) + [APP_DIR / "demo", APP_DIR]


# ===================== Mô hình (giống esc_lib.py) =====================
class LogMel(nn.Module):
    def __init__(self, n_mels, sr=22050, n_fft=1024, hop=512):
        super().__init__()
        self.mel = T.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
                                    f_min=0.0, f_max=sr / 2, power=2.0)
        self.to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, wav):
        return self.to_db(self.mel(wav).unsqueeze(1))


def conv_block(c_in, c_out, pool, n_conv):
    layers = []
    for k in range(n_conv):
        layers += [nn.Conv2d(c_in if k == 0 else c_out, c_out, kernel_size=3, padding=1, bias=False),
                   nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    layers.append(nn.MaxPool2d(pool))
    return nn.Sequential(*layers)


class CNN(nn.Module):
    def __init__(self, n_classes=50, n_conv=2):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32, (2, 2), n_conv), conv_block(32, 64, (2, 2), n_conv),
            conv_block(64, 128, (2, 2), n_conv), conv_block(128, 256, (2, 2), n_conv))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=2)
        x = x.max(dim=2).values + x.mean(dim=2)
        return self.fc(self.drop(x))


class CRNN(nn.Module):
    def __init__(self, n_mels, n_classes=50, n_conv=2, hidden=128):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32, (2, 2), n_conv), conv_block(32, 64, (2, 2), n_conv),
            conv_block(64, 128, (2, 1), n_conv), conv_block(128, 256, (2, 1), n_conv))
        self.gru = nn.GRU(256 * (n_mels // 16), hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        x = self.features(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x, _ = self.gru(x)
        return self.fc(self.drop(x.mean(dim=1)))


class PannsFront(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = T.MelSpectrogram(sample_rate=32000, n_fft=1024, win_length=1024, hop_length=320,
                                    f_min=50.0, f_max=14000.0, n_mels=64, power=2.0, norm="slaney",
                                    mel_scale="slaney", center=True, pad_mode="reflect")
        self.to_db = T.AmplitudeToDB(stype="power", top_db=None)

    def forward(self, wav):
        return self.to_db(self.mel(wav).unsqueeze(1))


class PannsConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.bn2 = nn.BatchNorm2d(c_out)

    def forward(self, x, pool):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, kernel_size=pool)


class PannsNet(nn.Module):
    def __init__(self, n_classes=50, specaug=None):
        super().__init__()
        self.specaug = specaug
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = PannsConvBlock(1, 64)
        self.conv_block2 = PannsConvBlock(64, 128)
        self.conv_block3 = PannsConvBlock(128, 256)
        self.conv_block4 = PannsConvBlock(256, 512)
        self.conv_block5 = PannsConvBlock(512, 1024)
        self.conv_block6 = PannsConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_out = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = self.bn0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.training and self.specaug is not None:
            x = self.specaug(x)
        x = x.transpose(2, 3)
        for block in (self.conv_block1, self.conv_block2, self.conv_block3,
                      self.conv_block4, self.conv_block5):
            x = F.dropout(block(x, (2, 2)), p=0.2, training=self.training)
        x = F.dropout(self.conv_block6(x, (1, 1)), p=0.2, training=self.training)
        x = x.mean(dim=3)
        x = x.max(dim=2).values + x.mean(dim=2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.fc_out(x)


# ===================== Nạp mô hình =====================
def find_file(name):
    for d in SEARCH_DIRS:
        p = d / name
        if p.exists():
            return p
    return None


def build_member(b):
    n_classes = len(b.get("class_names", range(50)))
    if b["model"] == "panns":
        model, front = PannsNet(n_classes), PannsFront()
    else:
        n_conv = b.get("conv_per_block", 1)
        model = CNN(n_classes, n_conv) if b["model"] == "cnn" else CRNN(b["n_mels"], n_classes, n_conv)
        front = LogMel(b["n_mels"], b["sample_rate"], b.get("n_fft", 1024), b.get("hop_length", 512))
    state = {k: (v.float() if v.is_floating_point() else v) for k, v in b["state"].items()}
    model.load_state_dict(state)
    return dict(model=model.to(DEVICE).eval(), front=front.to(DEVICE).eval(), sr=int(b["sample_rate"]),
                mean=float(b["mean"]), std=float(b["std"]), name=str(b.get("experiment", b["model"])))


@st.cache_resource(show_spinner="Đang nạp mô hình...")
def load_bundle(path_str, mtime):
    obj = torch.load(path_str, map_location="cpu", weights_only=False)
    bundles = obj["members"] if "members" in obj else [obj]
    first = bundles[0]
    return dict(
        members=[build_member(b) for b in bundles],
        class_names=list(first.get("class_names", [str(i) for i in range(50)])),
        group_names=list(first.get("group_names", [])),
        threshold=float(first.get("threshold", 0.5)),
        shifts=tuple(obj.get("tta_shifts", DEFAULT_SHIFTS)),
        is_ensemble="members" in obj,
    )


# ===================== Xử lý âm thanh và dự đoán =====================
def read_audio(data, filename="audio.wav"):
    """Đọc bytes âm thanh thành tín hiệu mono float32 và tần số lấy mẫu."""
    try:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return torch.from_numpy(wav.mean(axis=1)), sr
    except Exception:
        import torchaudio
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix or ".wav", delete=False) as f:
            f.write(data)
            tmp = f.name
        try:
            w, sr = torchaudio.load(tmp)
        finally:
            os.remove(tmp)
        return w.mean(dim=0), sr


def prepare(wav, orig_sr, target_sr, start_sec=0.0):
    """Lấy đoạn 5 giây từ start_sec, đổi tần số lấy mẫu, đệm số 0 nếu ngắn hơn 5 giây."""
    start = int(start_sec * orig_sr)
    seg = wav[start:start + DURATION * orig_sr]
    if orig_sr != target_sr:
        seg = AF.resample(seg, orig_sr, target_sr)
    n = DURATION * target_sr
    return F.pad(seg, (0, max(0, n - seg.numel())))[:n]


@torch.no_grad()
def predict(members, wav, orig_sr, start_sec, shifts):
    total = 0
    for m in members:
        w = prepare(wav, orig_sr, m["sr"], start_sec).unsqueeze(0).to(DEVICE)
        p = 0
        for s in shifts:
            ws = torch.roll(w, int(s * w.shape[1]), dims=1) if s else w
            x = (m["front"](ws) - m["mean"]) / m["std"]
            p = p + torch.softmax(m["model"](x).float(), dim=1)
        total = total + p / len(shifts)
    return (total / len(members))[0].cpu().numpy()


@torch.no_grad()
def show_features(member, wav, orig_sr, start_sec):
    w = prepare(wav, orig_sr, member["sr"], start_sec).unsqueeze(0).to(DEVICE)
    spec = member["front"](w)[0, 0].cpu().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    t = np.arange(w.shape[1]) / member["sr"]
    ax[0].plot(t, w[0].cpu().numpy(), linewidth=0.5)
    ax[0].set_title("Waveform (5 giây)")
    ax[0].set_xlabel("Thời gian (s)")
    im = ax[1].imshow(spec, origin="lower", aspect="auto", extent=[0, DURATION, 0, spec.shape[0]])
    ax[1].set_title(f"Log-Mel Spectrogram ({spec.shape[0]} Mel bin)")
    ax[1].set_xlabel("Thời gian (s)")
    ax[1].set_ylabel("Mel bin")
    fig.colorbar(im, ax=ax[1], label="dB")
    fig.tight_layout()
    return fig


# ===================== Giao diện =====================
@st.cache_data(show_spinner=False)
def read_audio_np(data, filename):
    wav, sr = read_audio(data, filename)
    return wav.numpy(), int(sr)


def segment_starts(dur, split, start):
    """Danh sách thời điểm bắt đầu (giây) của các đoạn 5 giây cần phân loại."""
    if split and dur > DURATION:
        n = int(np.ceil(dur / DURATION))
        starts = [float(i * DURATION) for i in range(n)]
        if n > 1 and dur - starts[-1] < 1.0:      # bỏ đoạn cuối quá ngắn
            starts = starts[:-1]
        return starts[:MAX_SEGMENTS]
    return [float(start)]


def unique_names(names):
    seen, out = {}, []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{Path(n).stem} ({seen[n]}){Path(n).suffix}")
    return out


st.set_page_config(page_title="Phân loại âm thanh môi trường ESC-50", page_icon="🔊", layout="centered")
st.title("🔊 Phân loại âm thanh môi trường ESC-50")
st.caption("Môn Học sâu · CNN, CRNN và PANNs trên 50 loại âm thanh. Tải một hoặc nhiều file âm thanh, ứng dụng sẽ đoán từng file là âm thanh gì.")

ens_path, one_path = find_file("demo_ensemble.pt"), find_file("demo_model.pt")
if ens_path is None and one_path is None:
    st.error("Chưa tìm thấy file mô hình.")
    st.markdown("Chép `demo_ensemble.pt` và/hoặc `demo_model.pt` từ Google Drive (`MyDrive/ESC50/demo/`) vào thư mục "
                "`demo/` cạnh file `.py` này, rồi chạy lại. Hai file này do notebook `15-best-model` tạo ra. "
                "Có thể đặt ở thư mục khác bằng biến môi trường `ESC_DEMO_DIR`.")
    st.stop()

with st.sidebar:
    st.header("Cài đặt")
    options = {}
    if ens_path is not None:
        options["Ensemble (nhiều mô hình + TTA)"] = ens_path
    if one_path is not None:
        options["Mô hình đơn"] = one_path
    choice = st.radio("Mô hình", list(options), help="Ensemble thường chính xác hơn nhưng chạy chậm hơn.")
    path = options[choice]

bundle = load_bundle(str(path), path.stat().st_mtime)
members, class_names, group_names = bundle["members"], bundle["class_names"], bundle["group_names"]

with st.sidebar:
    use_tta = st.checkbox("Dùng TTA (dự đoán trên 5 phiên bản dịch vòng)", value=True)
    split = st.radio("File dài hơn 5 giây", ["Lấy một đoạn 5 giây", "Chia nhiều đoạn 5 giây, lấy trung bình"],
                     help=f"Chia đoạn tối đa {MAX_SEGMENTS} đoạn mỗi file. Nếu chỉ tải một file, "
                          "chế độ đầu cho phép chọn vị trí đoạn.") == "Chia nhiều đoạn 5 giây, lấy trung bình"
    thr = st.slider("Ngưỡng “Không xác định”", 0.05, 0.95, float(np.clip(bundle["threshold"], 0.05, 0.95)), 0.05,
                    help="Nếu xác suất cao nhất thấp hơn ngưỡng này, ứng dụng trả lời “Không xác định”. "
                         "Giá trị mặc định lấy từ validation khi xuất mô hình. Đổi ngưỡng không cần chạy lại.")
    st.markdown("**Mô hình đang dùng**")
    for m in members:
        st.write(f"- {m['name']} ({m['sr']} Hz)")
    st.caption(f"Chạy trên: {DEVICE.type.upper()}")

# ---- Bước 1: tải lên ----
st.subheader("1. Tải lên âm thanh")
files = []
tab_up, tab_mic = st.tabs(["Tải file", "Ghi âm"])
with tab_up:
    ups = st.file_uploader("Chọn một hoặc nhiều file âm thanh (wav, mp3, ogg, flac)",
                           type=["wav", "mp3", "ogg", "flac"], accept_multiple_files=True)
    for u in ups or []:
        files.append((u.name, u.getvalue()))
with tab_mic:
    if hasattr(st, "audio_input"):
        rec = st.audio_input("Bấm để ghi âm")
        if rec is not None:
            files.append(("mic.wav", rec.getvalue()))
    else:
        st.info("Phiên bản Streamlit hiện tại chưa hỗ trợ ghi âm. Hãy cập nhật Streamlit hoặc dùng tab Tải file.")

if not files:
    st.info("Hãy chọn một hoặc nhiều file âm thanh để bắt đầu. Có thể thử các clip trong bộ ESC-50.")
    st.stop()

names = unique_names([n for n, _ in files])
items = []
for name, (_, data) in zip(names, files):
    try:
        wav, sr = read_audio_np(data, name)
        items.append(dict(name=name, data=data, wav=torch.from_numpy(wav), sr=sr, dur=len(wav) / sr))
    except Exception as e:
        st.error(f"Không đọc được file {name}: {e}")
if not items:
    st.stop()

st.write(f"Đã nhận **{len(items)}** file, tổng {sum(it['dur'] for it in items):.0f} giây.")
start = 0.0
if len(items) == 1:
    it = items[0]
    st.audio(it["data"])
    if it["dur"] > DURATION and not split:
        start = st.slider("File dài hơn 5 giây. Chọn đoạn 5 giây bắt đầu từ giây thứ", 0.0, float(it["dur"] - DURATION), 0.0, 0.5)
elif not split and any(it["dur"] > DURATION for it in items):
    st.caption("File dài hơn 5 giây chỉ được lấy 5 giây đầu. Đổi ở thanh bên nếu muốn chia nhiều đoạn.")
if any(it["dur"] < DURATION for it in items):
    st.caption("File ngắn hơn 5 giây được đệm khoảng lặng ở cuối, giống lúc huấn luyện.")
silent = [it["name"] for it in items if float(it["wav"].abs().max()) < 1e-4]
if silent:
    st.warning("File gần như im lặng, kết quả có thể không đáng tin: " + ", ".join(silent))

sig = (tuple((it["name"], len(it["data"])) for it in items), str(path), use_tta, split, start)

# ---- Bước 2: phân loại ----
if st.button("Phân loại", type="primary"):
    shifts = bundle["shifts"] if use_tta else (0.0,)
    results, bar = [], st.progress(0.0, text="Đang xử lý...")
    for i, it in enumerate(items):
        starts = segment_starts(it["dur"], split, start)
        probs = np.mean([predict(members, it["wav"], it["sr"], s, shifts) for s in starts], axis=0)
        results.append(dict(name=it["name"], probs=probs, starts=starts))
        bar.progress((i + 1) / len(items), text=f"Đã xử lý {i + 1}/{len(items)}: {it['name']}")
    bar.empty()
    st.session_state["results"], st.session_state["sig"] = results, sig

res = st.session_state.get("results")
if res is None:
    st.stop()
if st.session_state["sig"] != sig:
    st.info("Đã đổi file hoặc cài đặt. Bấm “Phân loại” để chạy lại.")
    st.stop()

# ---- Bước 3: kết quả ----
st.subheader("2. Kết quả")
rows = []
for r in res:
    top = np.argsort(-r["probs"])[:3]
    best, conf = int(top[0]), float(r["probs"][top[0]])
    ok = conf >= thr
    rows.append({
        "Tệp": r["name"],
        "Kết quả": class_names[best] if ok else "Không xác định",
        "Nhóm": group_names[best // 10] if (ok and group_names) else "",
        "Xác suất (%)": round(conf * 100, 1),
        "Top 3": ", ".join(f"{class_names[int(i)]} ({r['probs'][i] * 100:.0f}%)" for i in top),
        "Số đoạn": len(r["starts"]),
    })
df = pd.DataFrame(rows)
if not split:
    df = df.drop(columns=["Số đoạn"])
n_unknown = int((df["Kết quả"] == "Không xác định").sum())
st.write(f"Nhận diện được **{len(df) - n_unknown}/{len(df)}** file" + (f", {n_unknown} file “Không xác định”." if n_unknown else "."))
st.dataframe(df, use_container_width=True, hide_index=True)
if len(df) > 1 and len(df) > n_unknown:
    st.bar_chart(df[df["Kết quả"] != "Không xác định"]["Kết quả"].value_counts())
st.download_button("Tải kết quả (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                   file_name="ket_qua_phan_loai.csv", mime="text/csv")

# ---- Chi tiết một file ----
st.subheader("3. Chi tiết")
rnames = [r["name"] for r in res]
sel = rnames[0] if len(rnames) == 1 else st.selectbox("Chọn file để xem chi tiết", rnames)
r, it = res[rnames.index(sel)], items[rnames.index(sel)]
if len(rnames) > 1:
    st.audio(it["data"])
if len(r["starts"]) > 1:
    st.caption(f"Kết quả là trung bình trên {len(r['starts'])} đoạn 5 giây.")
st.pyplot(show_features(members[0], it["wav"], it["sr"], r["starts"][0]))

top = np.argsort(-r["probs"])[:3]
best, conf = int(top[0]), float(r["probs"][top[0]])
if conf < thr:
    st.warning(f"**Không xác định**: xác suất cao nhất chỉ {conf:.1%}, thấp hơn ngưỡng {thr:.0%}.")
else:
    st.success(f"**{class_names[best]}** ({conf:.1%})")
    if group_names:
        st.write(f"Nhóm: **{group_names[best // 10]}**")
st.markdown("**3 loại có xác suất cao nhất**")
for i in top:
    st.write(f"{class_names[int(i)]}" + (f" ({group_names[int(i) // 10]})" if group_names else ""))
    st.progress(float(r["probs"][i]), text=f"{r['probs'][i]:.1%}")
