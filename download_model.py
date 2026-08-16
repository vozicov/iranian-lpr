import urllib.request
import os

os.makedirs("models", exist_ok=True)
url = "https://huggingface.co/shalchianmh/Iran_license_plate_detection_YOLOv8m/resolve/main/YOLOv8m_Iran_license_plate_detection.pt"

print("⬇️ در حال دانلود مدل پلاک... (حدود 50 مگابایت)")
print("اگر اینترنت کند است، چند دقیقه صبر کن...")

try:
    urllib.request.urlretrieve(url, "models/plate_detector.pt")
    print("✅ دانلود کامل شد!")
    print("📂 فایل در مسیر: models/plate_detector.pt")
except Exception as e:
    print(f"❌ خطا در دانلود: {e}")
    print("لطفاً لینک را مستقیماً در مرورگر باز کن و فایل را در پوشه models قرار بده.")