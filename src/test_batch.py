import os, sys, glob, random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detect_pipeline import IranianLPRPipeline

def main():
    pipeline = IranianLPRPipeline()

    images = (glob.glob("datasets/Iranian Plate.v1/test/images/*.jpg") +
              glob.glob("datasets/Iranian Plate.v1/test/images/*.png"))
    random.seed(42)
    random.shuffle(images)

    for i, path in enumerate(images[:5], 1):
        print(f"\n🖼️ تست {i}: {os.path.basename(path)}")
        results = pipeline.process_image(path)
        if results:
            pipeline.visualize_results(path, results)  # با هر کلید، میره بعدی
        else:
            print("❌ پلاکی پیدا نشد")

if __name__ == "__main__":
    main()