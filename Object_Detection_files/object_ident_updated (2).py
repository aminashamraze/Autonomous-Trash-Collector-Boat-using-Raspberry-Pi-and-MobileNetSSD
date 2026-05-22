#finds camera attached, runs recognition pipeline on each frame, figures out state machines and sends commands via USB serial to ESP32
import cv2
import glob
import time
import serial
from collections import deque

# ---------- ESP32 serial setup ----------
ESP32_PORT = "/dev/ttyUSB0"   # change to /dev/ttyACM0 if needed
BAUD_RATE = 115200

esp32 = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

cmd_history = deque(maxlen=5)
last_sent_cmd = None


def get_stable_command(new_cmd):
    global last_sent_cmd

    cmd_history.append(new_cmd)

    stable_cmd = max(set(cmd_history), key=cmd_history.count)
    count = cmd_history.count(stable_cmd)

    # same command must appear at least 3 times in last 5 frames
    if count >= 3 and stable_cmd != last_sent_cmd:
        last_sent_cmd = stable_cmd
        return stable_cmd

    return None


def send_to_esp32(cmd):
    message = cmd + "\n"
    esp32.write(message.encode())
    print("SENT TO ESP32:", cmd)


# ---------- Object detection model setup ----------
classNames = []
classFile = "/home/astrieximpact/coco_folder/coco.names"

with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/astrieximpact/coco_folder/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/astrieximpact/coco_folder/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


# ---------- Robust camera selection ----------
def find_camera():
    stable_paths = glob.glob("/dev/v4l/by-id/*video-index0")

    for path in stable_paths:
        print("Trying stable camera path:", path)
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("Using camera:", path)
                return cap

        cap.release()

    for i in range(10):
        print("Trying camera index:", i)
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("Using camera index:", i)
                return cap

        cap.release()

    raise RuntimeError("No working camera found")


# ---------- Object detection ----------
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(
        img,
        confThreshold=thres,
        nmsThreshold=nms
    )

    if len(objects) == 0:
        objects = classNames

    objectInfo = []

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]

            if className in objects:
                objectInfo.append([box, className, confidence])

                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)

                    cv2.putText(
                        img,
                        f"{className.upper()} {confidence * 100:.1f}%",
                        (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

    return img, objectInfo


# ---------- Field-of-view mapping ----------
def map_object_to_zone(frame_width, frame_height, box):
    x, y, w, h = box

    object_center_x = x + w // 2
    object_center_y = y + h // 2
    frame_center_x = frame_width // 2

    center_margin = frame_width * 0.15

    if object_center_x < frame_center_x - center_margin:
        horizontal_zone = "LEFT"
    elif object_center_x > frame_center_x + center_margin:
        horizontal_zone = "RIGHT"
    else:
        horizontal_zone = "CENTER"

    box_area = w * h
    frame_area = frame_width * frame_height
    area_ratio = box_area / frame_area

    return horizontal_zone, object_center_x, object_center_y, area_ratio


# ---------- Boat decision logic ----------
def boat_state_machine(horizontal_zone):
    if horizontal_zone == "LEFT":
        return "TURN_LEFT"

    if horizontal_zone == "RIGHT":
        return "TURN_RIGHT"

    if horizontal_zone == "CENTER":
        return "MOVE_FORWARD"

    return "SEARCH"


# ---------- Draw navigation overlay ----------
def draw_navigation_overlay(img, objectInfo):
    frame_height, frame_width, _ = img.shape

    cv2.line(
        img,
        (frame_width // 2, 0),
        (frame_width // 2, frame_height),
        (255, 0, 0),
        2
    )

    left_boundary = int(frame_width // 2 - frame_width * 0.15)
    right_boundary = int(frame_width // 2 + frame_width * 0.15)

    cv2.line(img, (left_boundary, 0), (left_boundary, frame_height), (255, 255, 0), 1)
    cv2.line(img, (right_boundary, 0), (right_boundary, frame_height), (255, 255, 0), 1)

    action = "SEARCH"

    if len(objectInfo) > 0:
        target = max(objectInfo, key=lambda obj: obj[0][2] * obj[0][3])
        box, className, confidence = target

        horizontal_zone, cx, cy, area_ratio = map_object_to_zone(
            frame_width,
            frame_height,
            box
        )

        action = boat_state_machine(horizontal_zone)

        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

        cv2.putText(
            img,
            f"TARGET: {className} | {horizontal_zone}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            img,
            f"AREA: {area_ratio:.4f} | ACTION: {action}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        print(f"Detected {className}: {horizontal_zone}, area={area_ratio:.4f} => {action}")

    else:
        cv2.putText(
            img,
            "NO TARGET | ACTION: SEARCH",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        print("No object detected => SEARCH")

    return img, action


if __name__ == "__main__":
    cap = find_camera()

    target_objects = ["bottle", "cup"]

    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()

        if not success:
            print("Failed to read frame from camera")
            break

        result, objectInfo = getObjects(img, 0.45, 0.2, objects=target_objects)

        img, action = draw_navigation_overlay(img, objectInfo)

        stable_cmd = get_stable_command(action)

        if stable_cmd is not None:
            send_to_esp32(stable_cmd)

        cv2.imshow("Output", img)

        # slow down slightly to reduce jitter/load
        time.sleep(0.10)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    send_to_esp32("STOP")

    cap.release()
    esp32.close()
    cv2.destroyAllWindows()
