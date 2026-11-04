"""
Ứng dụng demo phân loại âm thanh môi trường (ESC-50) bằng Streamlit.

Cách chạy:
    pip install streamlit torch torchaudio soundfile matplotlib pandas
    python -m streamlit run app.py --server.fileWatcherType none

Cần có file demo_model.pt (xuất từ notebook, mục 9), đặt cạnh file app.py
hoặc trong thư mục demo/ cạnh file app.py.
"""
import io
import os
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

APP_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [os.path.join(APP_DIR, "demo_model.pt"),
               os.path.join(APP_DIR, "demo", "demo_model.pt")]
MODEL_PATH = next((p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[0])


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


@st.cache_resource
def load_bundle():
    bundle = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
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
    model.load_state_dict(bundle["state"])
    model.eval()
    frontend.eval()
    return bundle, model, frontend


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
st.set_page_config(page_title="Phân loại âm thanh ESC-50", page_icon="🔊")
st.title("🔊 Phân loại âm thanh môi trường (ESC-50)")
st.caption("Tải lên → Xử lý → Phân loại → Trả kết quả")

if not os.path.exists(MODEL_PATH):
    st.error("Không tìm thấy file demo_model.pt. Hãy xuất mô hình từ notebook (mục 9) "
             "và đặt cạnh file app.py.")
    st.stop()

bundle, model, frontend = load_bundle()
class_names = bundle["class_names"]
group_names = bundle["group_names"]

st.sidebar.header("Thông tin mô hình")
st.sidebar.write(f"Cấu hình: {bundle['experiment']} ({bundle['model'].upper()}, "
                 f"{bundle['n_mels']} Mel bins)")
threshold = st.sidebar.slider("Ngưỡng báo Không xác định", 0.0, 1.0,
                              float(bundle["threshold"]), 0.05)
st.sidebar.caption("Nếu xác suất cao nhất thấp hơn ngưỡng này, ứng dụng báo Không xác định.")

uploaded = st.file_uploader("Tải lên file âm thanh",
                            type=["wav", "mp3", "flac", "ogg"])

if uploaded is None:
    st.info("Hãy tải lên một file âm thanh để bắt đầu.")
else:
    data = uploaded.getvalue()
    st.audio(data)

    with st.spinner("Đang xử lý và phân loại..."):
        wav = read_audio(data, bundle)
        with torch.no_grad():
            spec = frontend(wav.unsqueeze(0))
            x = (spec - bundle["mean"]) / bundle["std"]
            probs = torch.softmax(model(x), dim=1)[0]

    top_p, top_i = probs.topk(3)
    best_i, best_p = int(top_i[0]), float(top_p[0])

    st.subheader("Kết quả")
    if best_p < threshold:
        st.warning(f"Không xác định (xác suất cao nhất chỉ {best_p:.0%}, "
                   f"thấp hơn ngưỡng {threshold:.0%})")
    else:
        st.success(f"**{class_names[best_i]}** ({best_p:.0%})")
        st.write(f"Nhóm: **{group_names[best_i // 10]}**")

    st.write("3 loại có xác suất cao nhất:")
    result = pd.DataFrame({
        "Loại": [class_names[int(i)] for i in top_i],
        "Xác suất (%)": [round(float(p) * 100, 1) for p in top_p],
    }).set_index("Loại")
    st.bar_chart(result)

    with st.expander("Xem ảnh Log-Mel đã đưa vào mô hình"):
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.imshow(spec[0, 0].numpy(), origin="lower", aspect="auto")
        ax.set_xlabel("Khung thời gian")
        ax.set_ylabel("Mel bin")
        st.pyplot(fig)
        plt.close(fig)

    st.caption("Kết quả là dự đoán của mô hình nên có thể sai. "
               "File dài hơn 5 giây chỉ lấy 5 giây đầu.")
