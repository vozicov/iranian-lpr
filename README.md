#  Iranian LPR

## Iranian License Plate Recognition System

An end-to-end Computer Vision system for detecting and recognizing Iranian vehicle license plates.

This project implements a complete pipeline that detects vehicles, localizes Iranian license plates, and extracts plate text using deep learning models.

---

##  Project Overview

Automatic License Plate Recognition (ALPR) is an important application in intelligent transportation systems, traffic monitoring, and smart city solutions.

The goal of this project is to build a lightweight pipeline capable of processing vehicle images and returning detected license plates along with their recognized text through an API service.

---

##  Pipeline Architecture

```
Input Image
     |
     ↓
Vehicle Detection
(YOLOv8)
     |
     ↓
Iranian License Plate Detection
(Custom YOLOv8 Model)
     |
     ↓
Plate Image Processing
     |
     ↓
Persian OCR
(CRNN-based OCR Engine)
     |
     ↓
FastAPI JSON Response
```

---

##  Features

- Vehicle detection using YOLOv8
- Iranian license plate detection with a custom trained model
- Persian license plate text recognition
- End-to-end inference pipeline
- FastAPI REST API
- Evaluation tools for measuring detection performance

---

##  Technologies

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- FastAPI
- CRNN OCR
- Roboflow Dataset

---

##  Model Evaluation

The plate detection model was evaluated on validation and test samples using IoU-based matching.

### Validation Results

| Metric | Score |
|---|---:|
| Precision | 80.8% |
| Recall | 83.0% |
| F1 Score | 81.9% |

### Test Results

| Metric | Score |
|---|---:|
| Precision | 81.1% |
| Recall | 85.7% |
| F1 Score | 83.3% |

The evaluation includes analysis of:
- True Positive detections
- False Positive detections
- False Negative cases

---

##  API Usage

The project provides a FastAPI endpoint for image-based plate recognition.

### Run API

```bash
python api/server.py
```

The API will start on:

```
http://127.0.0.1:8000
```

---

### Send Image Request

```bash
curl -X POST "http://127.0.0.1:8000/detect-plate" \
-F "file=@image.jpg"
```

---

### Example Response

```json
{
  "message": "1 پلاک تشخیص داده شد",
  "plates": [
    {
      "plate_text": "۱۱ ایران - ۲۸۸ و ۴۹",
      "vehicle_type": "ماشین",
      "vehicle_bbox": [100,73,577,385],
      "plate_bbox": [304,256,405,286],
      "confidence": 0.70
    }
  ]
}
```

---

##  Project Structure

```
iranian-lpr/
│
├── api/
│   └── server.py
│
├── models/
│   └── plate_detector.pt
│
├── src/
│   ├── detect_pipeline.py
│   ├── ocr_engine.py
│   ├── evaluate.py
│   ├── train_plate_v2.py
│   └── prepare_plate_dataset.py
│
├── data/
│   └── test_car.jpg
│
├── requirements.txt
└── README.md
```

---

##  Dataset

The plate detection model was trained using an Iranian license plate dataset from Roboflow:

https://universe.roboflow.com/sultan-space/iranian-plate

The dataset is not included in this repository due to size limitations.

---

##  Limitations

Current limitations:

- OCR performance decreases on very blurry images
- Extreme viewing angles can reduce detection accuracy
- Very small license plates may not be detected correctly
- Recognition quality depends on image resolution and lighting conditions

---

## Future Improvements

Possible improvements:

- Vehicle make and model classification
- Improved Persian OCR with larger datasets
- Real-time video stream processing
- Model optimization for edge devices
- Deployment with Docker

---

##  License

This project is released under the MIT License.