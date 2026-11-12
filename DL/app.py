"""
Ứng dụng demo phân loại âm thanh môi trường (ESC-50) bằng Streamlit.
Hỗ trợ tải lên nhiều file cùng lúc và xếp mỗi file vào một trong 5 nhóm âm thanh.

Cách chạy:
    pip install streamlit torch torchaudio soundfile matplotlib pandas
    python -m streamlit run app.py --server.fileWatcherType none

Cần có demo_model.pt (một mô hình) hoặc demo_ensemble.pt (ensemble + TTA, tạo bằng
make_demo_model.py), đặt cạnh app.py hoặc trong thư mục demo/ cạnh app.py.
Nếu có demo_ensemble.pt, thanh bên cho phép bật/tắt ensemble + TTA.
"""
import hashlib
import io
import os
import tempfile

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # có GPU thì dùng GPU cho nhanh

APP_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [os.path.join(APP_DIR, "demo_model.pt"),
               os.path.join(APP_DIR, "demo", "demo_model.pt")]
MODEL_PATH = next((p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[0])
_ENS_CANDIDATES = [os.path.join(APP_DIR, "demo_ensemble.pt"),
                   os.path.join(APP_DIR, "demo", "demo_ensemble.pt")]
ENSEMBLE_PATH = next((p for p in _ENS_CANDIDATES if os.path.exists(p)), None)


# ----------------------------------------------------------------------------
# Mô hình (giống hệt notebook)
# ----------------------------------------------------------------------------
def conv_block(c_in, c_out, pool, n_conv):
    layers = []
    for k in range(n_conv):
        layers += [
            nn.Conv2d(c_in if k == 0 else c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        ]
    layers.append(nn.MaxPool2d(pool))
    return nn.Sequential(*layers)


class CNN(nn.Module):
    def __init__(self, n_classes=50, n_conv=2):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32, (2, 2), n_conv),
            conv_block(32, 64, (2, 2), n_conv),
            conv_block(64, 128, (2, 2), n_conv),
            conv_block(128, 256, (2, 2), n_conv),
        )
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
            conv_block(1, 32, (2, 2), n_conv),
            conv_block(32, 64, (2, 2), n_conv),
            conv_block(64, 128, (2, 1), n_conv),
            conv_block(128, 256, (2, 1), n_conv),
        )
        freq_out = n_mels // 16
        self.gru = nn.GRU(256 * freq_out, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        x = self.features(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x, _ = self.gru(x)
        x = x.mean(dim=1)
        return self.fc(self.drop(x))


class LogMel(nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, n_mels):
        super().__init__()
        self.mel = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft,
                                    hop_length=hop_length, n_mels=n_mels,
                                    f_min=0.0, f_max=sample_rate / 2, power=2.0)
        self.to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, wav):
        return self.to_db(self.mel(wav).unsqueeze(1))


class PannsFront(nn.Module):
    # Waveform 32 kHz (B, L) -> Log-Mel (B, 1, 64, T), cấu hình giống lúc pretrain PANNs
    def __init__(self):
        super().__init__()
        self.mel = T.MelSpectrogram(sample_rate=32000, n_fft=1024, win_length=1024,
                                    hop_length=320, f_min=50.0, f_max=14000.0, n_mels=64,
                                    power=2.0, norm="slaney", mel_scale="slaney",
                                    center=True, pad_mode="reflect")
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
    # Kiến trúc CNN14 của PANNs với lớp đầu ra 50 lớp
    def __init__(self, n_classes=50):
        super().__init__()
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


def build_member(bundle):
    n_classes = len(bundle["class_names"])
    n_conv = bundle.get("conv_per_block", 1)
    if bundle["model"] == "panns":
        model = PannsNet(n_classes)
        frontend = PannsFront()
    else:
        if bundle["model"] == "cnn":
            model = CNN(n_classes, n_conv)
        else:
            model = CRNN(bundle["n_mels"], n_classes, n_conv)
        frontend = LogMel(bundle["sample_rate"], bundle["n_fft"],
                          bundle["hop_length"], bundle["n_mels"])
    state = {k: (v.float() if v.is_floating_point() else v) for k, v in bundle["state"].items()}
    model.load_state_dict(state)
    model.eval()
    frontend.eval()
    return bundle, model, frontend


@st.cache_resource
def load_members(use_ensemble):
    # Trả về danh sách (bundle, model, frontend) và các mức dịch TTA
    if use_ensemble and ENSEMBLE_PATH:
        ens = torch.load(ENSEMBLE_PATH, map_location="cpu", weights_only=False)
        members = [build_member(b) for b in ens["members"]]
        shifts = tuple(ens.get("tta_shifts", (0.0,)))
    else:
        bundle = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        members, shifts = [build_member(bundle)], (0.0,)
    return [(b, m.to(DEVICE), f.to(DEVICE)) for b, m, f in members], shifts


# ----------------------------------------------------------------------------
# Bước 2: xử lý âm thanh (giống phần tiền xử lý trong notebook)
# ----------------------------------------------------------------------------
def read_audio(file_bytes, bundle):
    target_sr = bundle["sample_rate"]
    target_len = target_sr * bundle["duration"]
    try:
        wav, sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav.mean(axis=1))
    except Exception:
        import torchaudio
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
            tmp.write(file_bytes)
            path = tmp.name
        try:
            w, sr = torchaudio.load(path)
            wav = w.mean(dim=0)
        finally:
            os.remove(path)
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    if wav.numel() < target_len:
        wav = F.pad(wav, (0, target_len - wav.numel()))
    return wav[:target_len]


# ----------------------------------------------------------------------------
# Giao diện
# ----------------------------------------------------------------------------
GROUP_ICONS = ["🐾", "🌧️", "🧑", "🏠", "🚗"]
UNKNOWN = "Không xác định"
ERROR = "Lỗi đọc file"

st.set_page_config(page_title="Phân loại âm thanh ESC-50", page_icon="🔊", layout="wide")
st.title("🔊 Phân loại âm thanh môi trường (ESC-50)")
st.caption("Tải lên một hoặc nhiều file → Xử lý → Phân loại → Trả kết quả và xếp vào nhóm")

if not os.path.exists(MODEL_PATH) and ENSEMBLE_PATH is None:
    st.error("Không tìm thấy demo_model.pt hoặc demo_ensemble.pt. Hãy tạo bằng make_demo_model.py "
             "(hoặc xuất từ notebook, mục 9) và đặt cạnh file app.py.")
    st.stop()

st.sidebar.header("Thông tin mô hình")
use_ensemble = False
if ENSEMBLE_PATH:
    use_ensemble = st.sidebar.checkbox("Dùng ensemble + TTA", value=True)
if not use_ensemble and not os.path.exists(MODEL_PATH):
    use_ensemble = True     # chỉ có file ensemble
members, shifts = load_members(use_ensemble)
bundle = members[0][0]
class_names = bundle["class_names"]
group_names = bundle["group_names"]

names = " + ".join(f"{b['experiment']} ({b['model'].upper()})" for b, _, _ in members)
st.sidebar.write(f"Cấu hình: {names}")
if len(shifts) > 1:
    st.sidebar.write(f"TTA: {len(shifts)} lần dịch thời gian, lấy trung bình xác suất")
st.sidebar.write(f"Thiết bị: {DEVICE.type.upper()}")
threshold = st.sidebar.slider("Ngưỡng báo Không xác định", 0.0, 1.0,
                              float(bundle["threshold"]), 0.05)
st.sidebar.caption("Nếu xác suất cao nhất của một file thấp hơn ngưỡng này, "
                   "file đó được xếp vào Không xác định.")


def analyze(file_bytes):
    # Bước 2 và 3: xử lý âm thanh và phân loại một file (ensemble + TTA nếu có)
    probs, spec = None, None
    with torch.no_grad():
        for b, model, frontend in members:
            wav = read_audio(file_bytes, b).to(DEVICE)
            batch = torch.stack([torch.roll(wav, int(s * wav.numel())) if s else wav for s in shifts])
            sp = frontend(batch)
            p = torch.softmax(model((sp - b["mean"]) / b["std"]).float(), dim=1).mean(dim=0)
            probs = p if probs is None else probs + p
            if spec is None:
                spec = sp[0, 0].float().cpu().numpy()
    return dict(probs=(probs / len(members)).cpu().numpy(), spec=spec)


uploaded_files = st.file_uploader("Tải lên một hoặc nhiều file âm thanh",
                                  type=["wav", "mp3", "flac", "ogg"],
                                  accept_multiple_files=True)
if not uploaded_files:
    st.info("Hãy tải lên một hoặc nhiều file âm thanh để bắt đầu.")
    st.stop()

cache = st.session_state.setdefault("cache", {})     # tránh xử lý lại khi đổi ngưỡng
records = []
progress = st.progress(0.0, text="Đang xử lý và phân loại...")
for k, f in enumerate(uploaded_files):
    data = f.getvalue()
    key = f"{int(use_ensemble)}:{hashlib.md5(data).hexdigest()}"
    if key not in cache:
        try:
            cache[key] = analyze(data)
        except Exception as exc:
            cache[key] = dict(error=str(exc))
    item = cache[key]
    progress.progress((k + 1) / len(uploaded_files),
                      text=f"Đã xử lý {k + 1}/{len(uploaded_files)} file")

    if "error" in item:
        records.append(dict(name=f.name, data=data, ok=False, label=ERROR, group=ERROR,
                            best_p=np.nan, group_p=np.nan, top3=[], error=item["error"]))
        continue

    probs = item["probs"]
    best_i = int(probs.argmax())
    best_p = float(probs[best_i])
    g = best_i // 10
    group_p = float(probs[g * 10:(g + 1) * 10].sum())
    top_i = [int(i) for i in probs.argsort()[::-1][:3]]
    known = best_p >= threshold
    records.append(dict(
        name=f.name, data=data, ok=True, key=key,
        label=class_names[best_i] if known else UNKNOWN,
        group=group_names[g] if known else UNKNOWN,
        best_p=best_p, group_p=group_p,
        top3=[(class_names[i], float(probs[i])) for i in top_i],
        error=""))
progress.empty()

df = pd.DataFrame([{
    "STT": i + 1,
    "Tên file": r["name"],
    "Loại dự đoán": r["label"],
    "Nhóm": r["group"],
    "Xác suất loại (%)": round(r["best_p"] * 100, 1) if r["ok"] else None,
    "Xác suất nhóm (%)": round(r["group_p"] * 100, 1) if r["ok"] else None,
    "Top-3": "; ".join(f"{n} ({p:.0%})" for n, p in r["top3"]),
} for i, r in enumerate(records)])


def group_icon(name):
    if name in group_names:
        return GROUP_ICONS[group_names.index(name) % len(GROUP_ICONS)]
    return "❓" if name == UNKNOWN else "⚠️"


st.subheader("Kết quả")
tab_group, tab_table, tab_detail = st.tabs(["Theo nhóm", "Bảng chi tiết", "Từng file"])

# ---- Tab 1: tổng hợp theo nhóm
with tab_group:
    order = list(group_names) + [UNKNOWN]
    if (df["Nhóm"] == ERROR).any():
        order.append(ERROR)
    counts = df["Nhóm"].value_counts().reindex(order, fill_value=0)

    cols = st.columns(len(order))
    for col, name in zip(cols, order):
        col.metric(f"{group_icon(name)} {name}", int(counts[name]))

    count_df = pd.DataFrame({"Nhóm": order, "Số file": [int(counts[n]) for n in order]})
    st.altair_chart(
        alt.Chart(count_df).mark_bar().encode(
            y=alt.Y("Nhóm:N", sort=order, title=None),
            x=alt.X("Số file:Q", title="Số file", axis=alt.Axis(tickMinStep=1, format="d")),
            tooltip=["Nhóm", "Số file"],
        ).properties(height=42 * len(order)),
        use_container_width=True)

    for name in order:
        sub = df[df["Nhóm"] == name]
        if sub.empty:
            continue
        with st.expander(f"{group_icon(name)} {name} ({len(sub)} file)", expanded=True):
            st.dataframe(sub[["Tên file", "Loại dự đoán", "Xác suất loại (%)"]],
                         hide_index=True, use_container_width=True)

# ---- Tab 2: bảng đầy đủ và tải CSV
with tab_table:
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.download_button("Tải bảng kết quả (CSV)",
                       df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="ket_qua_phan_loai.csv", mime="text/csv")

# ---- Tab 3: chi tiết từng file
with tab_detail:
    idx = st.selectbox("Chọn file để xem chi tiết", range(len(records)),
                       format_func=lambda i: f"{i + 1}. {records[i]['name']}")
    r = records[idx]
    st.audio(r["data"])
    if not r["ok"]:
        st.error(f"Không đọc được file này: {r['error']}")
    else:
        if r["label"] == UNKNOWN:
            st.warning(f"Không xác định (xác suất cao nhất chỉ {r['best_p']:.0%}, "
                       f"thấp hơn ngưỡng {threshold:.0%})")
        else:
            st.success(f"**{r['label']}** ({r['best_p']:.0%})")
            st.write(f"Nhóm: **{r['group']}** (xác suất nhóm {r['group_p']:.0%})")

        st.write("3 loại có xác suất cao nhất:")
        top = pd.DataFrame({"Loại": [n for n, _ in r["top3"]],
                            "Xác suất (%)": [round(p * 100, 1) for _, p in r["top3"]]})
        st.altair_chart(
            alt.Chart(top).mark_bar().encode(
                y=alt.Y("Loại:N", sort=list(top["Loại"]), title=None),
                x=alt.X("Xác suất (%):Q", scale=alt.Scale(domain=[0, 100])),
                tooltip=["Loại", "Xác suất (%)"],
            ).properties(height=42 * len(top)),
            use_container_width=True)

        with st.expander("Xem ảnh Log-Mel đã đưa vào mô hình"):
            fig, ax = plt.subplots(figsize=(7, 3))
            ax.imshow(cache[r["key"]]["spec"], origin="lower", aspect="auto")
            ax.set_xlabel("Khung thời gian")
            ax.set_ylabel("Mel bin")
            st.pyplot(fig)
            plt.close(fig)

st.caption("Kết quả là dự đoán của mô hình nên có thể sai. "
           "File dài hơn 5 giây chỉ lấy 5 giây đầu.")
