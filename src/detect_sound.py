import sounddevice as sd
import numpy as np
import time

SAMPLE_RATE = 44100
DURATION = 0.5  # lấy mẫu mỗi 0.5 giây
CALIBRATION_TIME = 3  # thời gian đo noise nền (giây)

def get_volume(indata):
    return np.linalg.norm(indata) / len(indata)

# --- Bước 1: Đo noise nền ---
print("🔧 Đang đo noise nền... Vui lòng giữ yên lặng trong vài giây.")
volumes = []

def calibrate(indata, frames, time, status):
    volumes.append(get_volume(indata))

with sd.InputStream(callback=calibrate, channels=1, samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE*DURATION)):
    time.sleep(CALIBRATION_TIME)

noise_floor = np.mean(volumes)
THRESHOLD = noise_floor * 2.5  # hệ số nhân
print(f"✅ Noise nền: {noise_floor:.4f}, Ngưỡng phát hiện: {THRESHOLD:.4f}")

# --- Bước 2: Phát hiện tiếng động ---
def detect_sound(indata, frames, time, status):
    volume = get_volume(indata)
    if volume > THRESHOLD:
        print(f"Có tiếng ! (volume={volume:.4f})")
    else:
        print(f"... im lặng (volume={volume:.4f})")

print("🎤 Bắt đầu lắng nghe...")
with sd.InputStream(callback=detect_sound, channels=1, samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE*DURATION)):
    while True:
        pass
