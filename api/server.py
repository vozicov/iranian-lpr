import os
import sys
import cv2
import numpy as np
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from detect_pipeline import IranianLPRPipeline

app = FastAPI(
    title="Iranian LPR API",
    description="سرویس تشخیص و خواندن پلاک خودروهای ایرانی (مدل V2)",
    version="2.0.0",
)

print("⏳ بارگذاری مدل‌ها برای API...")
pipeline = IranianLPRPipeline()
print("✅ API آماده‌ست!")


@app.get("/")
def root():
    return {"message": "API پلاک‌خوان ایرانی فعاله 🚗", "model": "plate_detector_v2"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/detect-plate")
async def detect_plate(file: UploadFile = File(...)):
    """آپلود عکس ماشین → خروجی JSON شامل نوع وسیله و متن پلاک"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="فایل تصویر معتبر نیست!")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            tmp_path = f.name
        cv2.imwrite(tmp_path, image)
        results = pipeline.process_image(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    plates = [
        {
            "plate_text": r["plate_text"],
            "vehicle_type": r.get("vehicle_type", "نامشخص"),
            "vehicle_bbox": r["vehicle_bbox"],
            "plate_bbox": r["plate_bbox"],
            "confidence": r["plate_confidence"],
        }
        for r in results
    ]

    return {"message": f"{len(plates)} پلاک تشخیص داده شد", "plates": plates}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)