import os
import shutil
import glob
from ultralytics import YOLO

DATA_YAML = "datasets/Iranian Plate.plate1/data.yaml"
OUT_MODEL = "models/plate_detector_v2.pt"

def main():
    if not os.path.exists(DATA_YAML):
        print("❌ اول src/prepare_plate_dataset.py را اجرا کن!")
        return

    model = YOLO("yolov8n.pt")
    print("🚀 شروع آموزش Plate Detector V2 (CPU — چند ساعت)...")

    model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=512,
        batch=8,
        patience=15,
        device="cpu",
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.5,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        perspective=0.002,
        fliplr=0.5,
        mixup=0.1,
        mosaic=1.0,
        name="plate_detector_v2",
        project="runs",           # 👈 اصلاح: فقط runs (YOLO خودش detect رو اضافه می‌کنه)
        exist_ok=True,
        save=True,
        plots=True,
    )

    # جستجوی best.pt در هر جایی که YOLO ذخیره کرده
    candidates = glob.glob("runs/**/plate_detector_v2/weights/best.pt", recursive=True)
    if candidates:
        best = candidates[0]
        shutil.copy(best, OUT_MODEL)
        print(f"✅ بهترین مدل ذخیره شد: {OUT_MODEL}")
    else:
        print("❌ best.pt پیدا نشد!")

if __name__ == "__main__":
    main()