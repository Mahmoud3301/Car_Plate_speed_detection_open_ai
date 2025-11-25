import cv2
import os
import csv
import base64
import threading
import numpy as np
import winsound
from datetime import datetime
from openai import OpenAI
from shapely.geometry import LineString
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ==================== CONFIG ====================
VIDEO_PATH = "tc.mp4"
MODEL_PATH = "yolov8n.pt"
OPENROUTER_API_KEY = "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"
CSV_FILE = "vehicle_records_deepsort.csv"
SPEED_LIMIT = 40   # km/h
PPM = 8            # pixels per meter
FPS = 30           # frames per second

os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

# ==================== FOLDERS ====================
os.makedirs("crop", exist_ok=True)

# ==================== CSV INITIALIZATION ====================
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Track ID", "Speed (km/h)", "Date & Time",
            "Vehicle Model", "Color", "Company", "Number Plate", "Crop Image"
        ])
    print(f"✅ Created CSV file: {CSV_FILE}")

# ==================== OpenAI Client ====================
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ==================== DeepSORT Tracker ====================
deepsort = DeepSort(
    max_age=30,
    n_init=3,
    nms_max_overlap=1.0,
    max_cosine_distance=0.2
)

# ==================== HELPER FUNCTIONS ====================
def save_to_csv(track_id, speed, timestamp, model, color, company, plate, image_path):
    """Save record to CSV"""
    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([track_id, speed, timestamp, model, color, company, plate, image_path])
    print(f"📝 Saved to CSV → ID: {track_id} | {speed} km/h")


def safe_beep():
    """Play warning sound (cross-platform)"""
    try:
        winsound.Beep(1500, 500)
    except Exception:
        print("\a", end="", flush=True)


def analyze_vehicle_with_ai(image_path, track_id, speed, timestamp):
    """Analyze vehicle image using GPT-4o-mini"""
    try:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        completion = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "Analyze this vehicle and extract ONLY these details:\n"
                             "| Vehicle Model | Color | Company | Number Plate |\n"
                             "|--------------|--------|---------|--------------|"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }]
        )

        response_text = completion.choices[0].message.content.strip()

        # Parse AI output table
        valid_rows = [
            row.split("|")[1:-1]
            for row in response_text.split("\n")
            if "|" in row and "Vehicle Model" not in row and "---" not in row
        ]

        if valid_rows and len(valid_rows[0]) >= 4:
            model, color, company, plate = [v.strip() for v in valid_rows[0]]
        else:
            model, color, company, plate = ("Unknown", "Unknown", "Unknown", "Unknown")

        save_to_csv(track_id, speed, timestamp, model, color, company, plate, image_path)

        print("\n=======================")
        print("🚨 OVERSPEED DETECTED!")
        print(f"ID: {track_id} | Speed: {speed} km/h | Time: {timestamp}")
        print(f"Model: {model} | Color: {color} | Company: {company} | Plate: {plate}")
        print("=======================\n")

    except Exception as e:
        print(f"❌ AI Error: {e}")


def estimate_speed(p1, p2, fps, ppm):
    """Estimate speed (km/h)"""
    pixel_distance = np.linalg.norm(np.array(p2) - np.array(p1))
    meters = pixel_distance / ppm
    speed_m_s = meters * fps
    return round(speed_m_s * 3.6, 1)


def crossed_line(p1, p2, line):
    """Check if car crosses radar line"""
    return LineString([p1, p2]).intersects(LineString(line))


# ==================== MAIN ====================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

track_points = {}
saved_ids = set()
region_line = [(0, 300), (1020, 300)]

print("🚗 Starting Radar Speed Detection with YOLOv8 + DeepSORT (CSV only)...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id == 2:  # Car class
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append([[x1, y1, x2 - x1, y2 - y1], conf, cls_id])

    tracks = deepsort.update_tracks(detections, frame=frame)

    # Draw radar line
    cv2.line(frame, region_line[0], region_line[1], (0, 255, 0), 2)
    cv2.putText(frame, "Radar Line (Speed Limit 40 km/h)", (20, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = int(track.track_id)
        x1, y1, w, h = map(int, track.to_ltwh())
        x2, y2 = x1 + w, y1 + h
        center = ((x1 + x2)//2, (y1 + y2)//2)

        track_points.setdefault(track_id, []).append(center)

        # Speed calculation
        if len(track_points[track_id]) >= 2:
            p1, p2 = track_points[track_id][-2], track_points[track_id][-1]
            speed = estimate_speed(p1, p2, FPS, PPM)
        else:
            speed = 0

        color = (255, 255, 0) if speed <= SPEED_LIMIT else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{track_id} {speed} km/h", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Overspeed + radar crossing
        if (
            len(track_points[track_id]) >= 2
            and crossed_line(track_points[track_id][-2], track_points[track_id][-1], region_line)
            and speed > SPEED_LIMIT
            and track_id not in saved_ids
        ):
            cropped = frame[y1:y2, x1:x2]
            if cropped.size > 0:
                img_path = f"crop/{track_id}_{speed}kmh.jpg"
                cv2.imwrite(img_path, cropped)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                threading.Thread(target=safe_beep, daemon=True).start()
                threading.Thread(
                    target=analyze_vehicle_with_ai,
                    args=(img_path, track_id, speed, timestamp),
                    daemon=True,
                ).start()

                saved_ids.add(track_id)

    cv2.imshow("Radar Speed Detection (YOLOv8 + DeepSORT)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Overspeed results saved to: {CSV_FILE}")











# import cv2
# import os
# import csv
# import base64
# import threading
# import numpy as np
# import winsound
# from datetime import datetime
# from shapely.geometry import LineString
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
# from openai import OpenAI

# # ==================== CONFIG ====================
# VIDEO_PATH = "tc.mp4"
# MODEL_PATH = "yolov8n.pt"
# CSV_FILE = "vehicle_records_deepsort.csv"
# SPEED_LIMIT = 40   # km/h
# PPM = 8            # pixels per meter
# FPS = 30           # frames per second

# # ✅ ضع مفتاحك هنا مباشرة
# OPENAI_API_KEY = "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"

# # ==================== CHECK KEY ====================
# if not OPENAI_API_KEY:
#     raise ValueError("❌ API key not found! Please add your OpenAI key.")
# else:
#     print("✅ OpenAI API key loaded successfully.")

# # ==================== CLIENT ====================
# client = OpenAI(api_key=OPENAI_API_KEY)

# # ==================== FOLDERS ====================
# os.makedirs("crop", exist_ok=True)

# # ==================== CSV INITIALIZATION ====================
# if not os.path.exists(CSV_FILE):
#     with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as file:
#         writer = csv.writer(file)
#         writer.writerow([
#             "Track ID", "Speed (km/h)", "Date & Time",
#             "Vehicle Model", "Color", "Company", "Number Plate", "Crop Image"
#         ])
#     print(f"✅ Created CSV file: {CSV_FILE}")

# # ==================== DeepSORT Tracker ====================
# deepsort = DeepSort(
#     max_age=30,
#     n_init=3,
#     nms_max_overlap=1.0,
#     max_cosine_distance=0.2
# )

# # ==================== HELPER FUNCTIONS ====================
# def save_to_csv(track_id, speed, timestamp, model, color, company, plate, image_path):
#     """Save record to CSV"""
#     with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as file:
#         writer = csv.writer(file)
#         writer.writerow([track_id, speed, timestamp, model, color, company, plate, image_path])
#     print(f"📝 Saved to CSV → ID: {track_id} | {speed} km/h")


# def safe_beep():
#     """Play warning sound"""
#     try:
#         winsound.Beep(1500, 500)
#     except Exception:
#         print("\a", end="", flush=True)


# def analyze_vehicle_with_ai(image_path, track_id, speed, timestamp):
#     """Analyze vehicle image using GPT-4-Vision"""
#     try:
#         with open(image_path, "rb") as img_file:
#             base64_image = base64.b64encode(img_file.read()).decode("utf-8")

#         completion = client.chat.completions.create(
#             model="gpt-4o-mini",  # free-tier compatible if you have access
#             messages=[{
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": "Analyze this vehicle image and return:\nVehicle Model, Color, Company, and Number Plate."
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
#                     }
#                 ]
#             }]
#         )

#         response_text = completion.choices[0].message.content.strip()
#         print(f"🔍 AI Analysis:\n{response_text}\n")

#         # Attempt to parse details
#         model, color, company, plate = ("Unknown", "Unknown", "Unknown", "Unknown")
#         for line in response_text.splitlines():
#             if "model" in line.lower():
#                 model = line.split(":")[-1].strip()
#             elif "color" in line.lower():
#                 color = line.split(":")[-1].strip()
#             elif "company" in line.lower():
#                 company = line.split(":")[-1].strip()
#             elif "plate" in line.lower():
#                 plate = line.split(":")[-1].strip()

#         save_to_csv(track_id, speed, timestamp, model, color, company, plate, image_path)

#         print("\n=======================")
#         print("🚨 OVERSPEED DETECTED!")
#         print(f"ID: {track_id} | Speed: {speed} km/h | Time: {timestamp}")
#         print(f"Model: {model} | Color: {color} | Company: {company} | Plate: {plate}")
#         print("=======================\n")

#     except Exception as e:
#         print(f"❌ AI Error: {e}")


# def estimate_speed(p1, p2, fps, ppm):
#     """Estimate speed (km/h)"""
#     pixel_distance = np.linalg.norm(np.array(p2) - np.array(p1))
#     meters = pixel_distance / ppm
#     speed_m_s = meters * fps
#     return round(speed_m_s * 3.6, 1)


# def crossed_line(p1, p2, line):
#     """Check if car crosses radar line"""
#     return LineString([p1, p2]).intersects(LineString(line))


# # ==================== MAIN ====================
# print("🚗 Starting Radar Speed Detection with YOLOv8 + DeepSORT...")

# model = YOLO(MODEL_PATH)
# cap = cv2.VideoCapture(VIDEO_PATH)

# track_points = {}
# saved_ids = set()
# region_line = [(0, 300), (1020, 300)]

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.resize(frame, (1020, 500))
#     results = model(frame, verbose=False)[0]

#     detections = []
#     for box in results.boxes:
#         cls_id = int(box.cls[0])
#         if cls_id == 2:  # Car class
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = float(box.conf[0])
#             detections.append([[x1, y1, x2 - x1, y2 - y1], conf, cls_id])

#     tracks = deepsort.update_tracks(detections, frame=frame)

#     # Draw radar line
#     cv2.line(frame, region_line[0], region_line[1], (0, 255, 0), 2)
#     cv2.putText(frame, f"Radar Line (Speed Limit {SPEED_LIMIT} km/h)",
#                 (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     for track in tracks:
#         if not track.is_confirmed():
#             continue

#         track_id = int(track.track_id)
#         x1, y1, w, h = map(int, track.to_ltwh())
#         x2, y2 = x1 + w, y1 + h
#         center = ((x1 + x2)//2, (y1 + y2)//2)

#         track_points.setdefault(track_id, []).append(center)

#         # Speed calculation
#         if len(track_points[track_id]) >= 2:
#             p1, p2 = track_points[track_id][-2], track_points[track_id][-1]
#             speed = estimate_speed(p1, p2, FPS, PPM)
#         else:
#             speed = 0

#         color = (255, 255, 0) if speed <= SPEED_LIMIT else (0, 0, 255)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(frame, f"ID:{track_id} {speed} km/h", (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#         # Overspeed + radar crossing
#         if (
#             len(track_points[track_id]) >= 2
#             and crossed_line(track_points[track_id][-2], track_points[track_id][-1], region_line)
#             and speed > SPEED_LIMIT
#             and track_id not in saved_ids
#         ):
#             cropped = frame[y1:y2, x1:x2]
#             if cropped.size > 0:
#                 img_path = f"crop/{track_id}_{speed}kmh.jpg"
#                 cv2.imwrite(img_path, cropped)
#                 timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#                 threading.Thread(target=safe_beep, daemon=True).start()
#                 threading.Thread(
#                     target=analyze_vehicle_with_ai,
#                     args=(img_path, track_id, speed, timestamp),
#                     daemon=True,
#                 ).start()

#                 saved_ids.add(track_id)

#     cv2.imshow("Radar Speed Detection (YOLOv8 + DeepSORT)", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()
# print(f"✅ Overspeed results saved to: {CSV_FILE}")
