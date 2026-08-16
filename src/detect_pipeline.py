import os
import sys
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VEHICLE_TYPES = {2: "ماشین", 3: "موتور", 5: "اتوبوس", 7: "کامیون"}

def put_persian_text(image, text, position, size=30, color=(0, 0, 255)):
    """رسم متن فارسی روی تصویر (OpenCV از پس فارسی برنمیاد!)"""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        display_text = get_display(arabic_reshaper.reshape(text))
    except ImportError:
        display_text = text

    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/tahoma.ttf", size)
    except Exception:
        font = ImageFont.load_default()
    fill = (color[2], color[1], color[0])  # BGR → RGB
    draw.text(position, display_text, font=font, fill=fill)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# رفع مشکل import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.ocr_engine import read_iranian_plate
except ModuleNotFoundError:
    from ocr_engine import read_iranian_plate

class IranianLPRPipeline:
    def __init__(self, plate_model_path=None):
        print("🔄 بارگذاری مدل‌ها...")

        # مدل خودرو (از قبل آموزش دیده COCO)
        vehicle_model = "yolov8s.pt" if os.path.exists("yolov8s.pt") else "yolov8n.pt"
        self.vehicle_detector = YOLO(vehicle_model)
        print(f"✅ مدل تشخیص خودرو: {vehicle_model}")

        # مدل پلاک: مسیر دلخواه یا پیش‌فرض (models/plate_detector.pt)
        if plate_model_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            plate_model_path = os.path.join(project_root, "models", "plate_detector.pt")
        self.plate_model_path = plate_model_path

        print(f"🔍 بررسی مدل پلاک در مسیر: {self.plate_model_path}")

        self.plate_detector = None
        if os.path.exists(self.plate_model_path):
            try:
                self.plate_detector = YOLO(self.plate_model_path)
                print(f"✅ مدل تشخیص پلاک با موفقیت بارگذاری شد!")
            except Exception as e:
                print(f"❌ خطا در بارگذاری مدل پلاک: {e}")
                print("احتمالاً فایل خالی یا خراب است. دوباره download_model.py را اجرا کن.")
        else:
            print("❌ فایل مدل پلاک در مسیر models/plate_detector.pt پیدا نشد!")
    def detect_vehicles(self, image):
        results = self.vehicle_detector(image, classes=[2, 3, 5, 7], iou=0.5, verbose=False)
        vehicles = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vehicles.append({"bbox": [x1, y1, x2, y2],
                                     "confidence": float(box.conf[0]),
                                     "class": int(box.cls[0])})
        return vehicles

    def detect_plate(self, vehicle_crop, conf=0.25):
        if self.plate_detector is None:
            return None

        results = self.plate_detector(vehicle_crop, conf=conf, verbose=False)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            return {
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf[0])
            }
        return None

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ نمی‌توان تصویر را خواند: {image_path}")
            return []

        vehicles = self.detect_vehicles(image)
        print(f"🚗 خودروهای پیدا شده: {len(vehicles)}")

        results = []
        for i, v in enumerate(vehicles):
            x1, y1, x2, y2 = v["bbox"]
            vtype = VEHICLE_TYPES.get(v["class"], "نامشخص")
            print(f"🚦 خودرو {i+1}: نوع = {vtype} (اعتماد: {v['confidence']:.2f})")

            vehicle_crop = image[y1:y2, x1:x2]

            plate = self.detect_plate(vehicle_crop)
            if plate is None:
                print(f"⚠️ خودرو {i+1}: پلاک پیدا نشد")
                continue

            px1, py1, px2, py2 = plate["bbox"]
            plate_crop = vehicle_crop[py1:py2, px1:px2]

            plate_text = read_iranian_plate(plate_crop)
            print(f"✅ خودرو {i+1}: پلاک = {plate_text}")

            results.append({
                "vehicle_bbox": [x1, y1, x2, y2],
                "vehicle_type": vtype,
                "plate_bbox": [px1, py1, px2, py2],
                "plate_confidence": plate["confidence"],
                "plate_text": plate_text,
            })
        return results

    def visualize_results(self, image_path, results):
        image = cv2.imread(image_path)
        for res in results:
            vx1, vy1, vx2, vy2 = res["vehicle_bbox"]
            cv2.rectangle(image, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)

            # نوع وسیله بالای کادر سبز
            image = put_persian_text(image, res.get("vehicle_type", ""),
                                     (vx1, max(vy1 - 45, 5)), size=22, color=(0, 255, 0))

            px1, py1, px2, py2 = res["plate_bbox"]
            cv2.rectangle(image, (vx1+px1, vy1+py1), (vx1+px2, vy1+py2), (0, 0, 255), 3)

            image = put_persian_text(image, res["plate_text"],
                                     (vx1+px1, max(vy1+py1-45, 5)))

        cv2.imshow("Iranian LPR Results", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    pipeline = IranianLPRPipeline()
    
    # تصویر تست - می‌توانی مسیر یک عکس واقعی از ماشین ایرانی را جایگزین کنی
    test_image = "data/test_car.jpg"

    if not os.path.exists(test_image):
        print(f"⚠️ تصویر {test_image} پیدا نشد.")
        # جستجوی خودکار هر تصویری در پوشه test
        import glob
        found = glob.glob("test/**/*.jpg", recursive=True) + glob.glob("datasets/**/*.jpg", recursive=True)
        if found:
            test_image = found[0]
            print(f"🔄 استفاده از تصویر: {test_image}")
        else:
            print("❌ هیچ تصویری برای تست پیدا نشد! لطفاً یک عکس ماشین ایرانی را با نام test_car.jpg در پوشه data قرار بده.")
            return

    results = pipeline.process_image(test_image)
    if results:
        pipeline.visualize_results(test_image, results)
    else:
        print("❌ هیچ پلاکی تشخیص داده نشد!")

if __name__ == "__main__":
    main()