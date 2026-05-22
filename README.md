
# Autonomous Trash Collector Boat using Raspberry Pi and MobileNet SSD

A Raspberry Pi based autonomous trash-collecting boat prototype that uses a camera and MobileNet SSD object detection to identify floating trash-like objects and guide the boat toward them.

This project was built as an embedded computer vision system for small ponds, still water, or controlled test environments. The main goal was to explore how a low-cost Raspberry Pi camera system can detect objects such as bottles, cups, and other floating debris, then use that detection result to support autonomous navigation.

---

## Project Goal

The goal of this project is to build a small autonomous boat that can:

1. Capture live video using a Raspberry Pi camera.
2. Run object detection locally on the Raspberry Pi.
3. Detect trash-related objects using a pre-trained MobileNet SSD model.
4. Decide whether the object is located on the left, center, or right side of the camera frame.
5. Use that information to guide future movement decisions such as turning left, turning right, or moving forward.

This repository focuses mainly on the computer vision and decision logic side of the project.

---

## Why This Project Matters

Floating trash is a common problem in small ponds, lakes, and still-water areas. A full industrial trash-collection system can be expensive, but a small autonomous prototype can demonstrate the basic idea at low cost.

This project combines:

- Embedded Linux
- Raspberry Pi
- Python
- Camera-based perception
- MobileNet SSD object detection
- Real-time decision logic
- Autonomous robotics concepts

It is a practical example of how computer vision can be used in an embedded system.

---

## System Overview

At a high level, the system works like this:

```text
Camera Feed
    ↓
Raspberry Pi
    ↓
MobileNet SSD Object Detection
    ↓
Filter for Trash-Like Objects
    ↓
Find Object Position in Frame
    ↓
Generate Movement Decision
````

The Raspberry Pi receives camera frames, runs object detection, and checks whether a detected object is likely to be trash. If a valid object is detected, the program looks at where the object appears in the image.

For example:

* Object appears on the left side → turn left
* Object appears near the center → move forward
* Object appears on the right side → turn right
* No object detected → search or stop depending on the behavior selected

---

## Important Design Decision: Why Object Area Was Not Used for Near/Far Distance

One major lesson from this project was that bounding box area was not reliable enough to decide whether an object was near or far.

At first, it may seem logical to use object area like this:

```text
Large bounding box area  → object is close
Small bounding box area  → object is far
```

In theory, this makes sense. If an object gets closer to the camera, it usually appears larger in the image.

However, in this project, the area measurement was unreliable for real movement decisions.

### Why Area Was Unreliable

The detected bounding box area changed for many reasons besides actual distance:

1. **Camera angle changed the visible size of the object**

   A bottle or cup may look large from one angle and much smaller from another angle, even if the distance is similar.

2. **Partial detection affected the bounding box**

   Sometimes the model detected only part of the object instead of the full object. This caused the bounding box area to become smaller even when the object was not actually far away.

3. **Object shape was inconsistent**

   Trash objects are not uniform. A bottle, cup, wrapper, and container all have different shapes. Comparing their bounding box areas does not give a consistent distance estimate.

4. **Water reflection and lighting affected detection**

   Since the project is meant for water-based environments, reflections, glare, and shadows can change how the object appears to the camera.

5. **Bounding boxes jitter between frames**

   Object detection models do not always draw the exact same bounding box every frame. Even when the object is still, the box can slightly move or resize.

6. **Perspective distortion matters**

   Objects near the edge of a wide camera frame can appear stretched or compressed compared to objects near the center.

Because of these issues, using area as a distance signal created unstable behavior.

### Final Decision

For this reason, this project does **not** base movement decisions on object area.

The system does **not** decide:

```text
area is large  → object is near
area is small  → object is far
```

Instead, the system focuses on a more reliable beginner-friendly control idea:

```text
Where is the object horizontally in the camera frame?
```

That means the decision logic is based mainly on object position:

```text
Object on left side   → turn left
Object in center      → move forward
Object on right side  → turn right
No object detected    → search / stop
```

This made the behavior easier to understand, easier to debug, and less sensitive to noisy area measurements.

---

## Object Detection Model

This project uses MobileNet SSD, a lightweight deep learning object detection model.

MobileNet SSD is useful for Raspberry Pi projects because it is smaller and faster than many large object detection models. It can run on low-power hardware while still detecting common objects.

The model uses a list of class names stored in:

```text
coco.names
```

This file contains the object labels that the model can recognize.

In this project, only selected labels are useful. For example, trash-related or floating-object labels may include things like:

```text
bottle
cup
```

The exact labels can be edited inside the Python script depending on what objects you want the boat to respond to.

---

## Repository Structure

```text
Autonomous-Trash-Collector-Boat-using-Raspberry-Pi-and-MobileNetSSD/
│
├── Images/
│   └── Project images, screenshots, or visual results
│
├── Object_Detection_files/
│   ├── object_ident_updated (2).py
│   ├── coco.names
│   └── Model/configuration files used for object detection
│
├── LICENSE
│
└── README.md
```

The main folder to copy to the Raspberry Pi is:

```text
Object_Detection_files/
```

---

## Hardware Used

Typical hardware for this project includes:

* Raspberry Pi
* Raspberry Pi Camera or USB camera
* Battery power source
* Boat chassis or floating platform
* Motor driver / ESC
* DC motors or water propulsion motors
* Optional microcontroller for motor control, such as ESP32 or Arduino

The Raspberry Pi handles the vision system. A separate motor controller can be used to convert the Raspberry Pi decision output into actual motor signals.

---

## Software Used

* Python
* OpenCV
* MobileNet SSD
* Raspberry Pi OS
* COCO object labels
* Camera interface for Raspberry Pi

---

## How the Program Works

The basic logic is:

1. Start the camera.
2. Load the MobileNet SSD model.
3. Read frames from the camera.
4. Run object detection on each frame.
5. Check if the detected object is one of the target objects.
6. Draw a bounding box around the detected object.
7. Find the center of the bounding box.
8. Compare the object center to the camera frame width.
9. Decide whether the object is on the left, center, or right.
10. Print or send a movement command.

Example logic:

```text
if object_center_x < left_threshold:
    turn left

elif object_center_x > right_threshold:
    turn right

else:
    move forward
```

The key point is that the decision is based on horizontal position, not object area.

---

## Beginner Explanation: What Is a Bounding Box?

When the model detects an object, it draws a rectangle around it.

That rectangle is called a bounding box.

```text
+-------------------+
|                   |
|      bottle       |
|                   |
+-------------------+
```

The program can measure different things from this box:

* x-position
* y-position
* width
* height
* area
* center point

For this project, the most useful value was the center x-position.

The center x-position tells us whether the object is more to the left, center, or right side of the camera image.

---

## Why Horizontal Position Was Better Than Area

Horizontal position was more reliable because the boat mainly needs to know which direction to turn.

For example, if the object is on the left side of the image, the boat should rotate or steer left until the object becomes centered.

Once the object is centered, the boat can move forward.

This is a simple visual servoing idea:

```text
Keep turning until the target object is centered in the camera frame.
```

This approach is easier than trying to estimate exact physical distance from a single camera.

---

## Setup Instructions

### 1. Clone the Repository

On the Raspberry Pi:

```bash
git clone https://github.com/aminashamraze/Autonomous-Trash-Collector-Boat-using-Raspberry-Pi-and-MobileNetSSD.git
cd Autonomous-Trash-Collector-Boat-using-Raspberry-Pi-and-MobileNetSSD
```

### 2. Go to the Object Detection Folder

```bash
cd Object_Detection_files
```

### 3. Install Python Dependencies

Install OpenCV and other required packages:

```bash
pip install opencv-python numpy
```

Depending on your Raspberry Pi setup, you may also need:

```bash
sudo apt update
sudo apt install python3-opencv
```

### 4. Run the Object Detection Script

Run:

```bash
python3 "object_ident_updated (2).py"
```

If your filename is changed later, run the updated Python file instead.

---

## Selecting Which Objects to Detect

The model may detect many objects from the COCO dataset, but the boat should only react to objects that make sense for trash collection.

Inside the Python script, you can filter the detected labels.

For example:

```python
target_objects = ["bottle", "cup"]
```

This prevents the boat from reacting to unrelated objects.

---

## Movement Decision Logic

The movement decision can be explained using the frame width.

Assume the camera frame is divided into three regions:

```text
+----------------+----------------+----------------+
|      LEFT      |     CENTER     |     RIGHT      |
+----------------+----------------+----------------+
```

If the detected object's center is in the left region, the boat should turn left.

If the object is in the center region, the boat should move forward.

If the object is in the right region, the boat should turn right.

Example decision table:

| Object Position     | Boat Decision  |
| ------------------- | -------------- |
| Left side of frame  | Turn left      |
| Center of frame     | Move forward   |
| Right side of frame | Turn right     |
| No object detected  | Search or stop |

This keeps the control logic simple and understandable.

---

## What This Project Demonstrates

This project demonstrates:

* Running computer vision on a Raspberry Pi
* Using MobileNet SSD for object detection
* Filtering detected object classes
* Extracting bounding box information
* Making control decisions from camera input
* Understanding why some sensor measurements are unreliable
* Designing simpler logic when real-world signals are noisy

A key engineering lesson from this project is that not every measurement should be trusted just because it is available. Bounding box area was available, but it was not reliable enough to use for near/far decisions.

---

## Limitations

This is a prototype, so there are some limitations:

* The system does not accurately estimate object distance.
* Bounding box area is not used for near/far classification.
* Detection quality depends on lighting, camera angle, and object visibility.
* Water reflection can affect detection.
* The model may miss objects or detect only part of an object.
* The project currently focuses more on perception and decision logic than full production-level boat control.

---

## Future Improvements

Possible future improvements include:

* Add motor control using ESP32, Arduino, or direct Raspberry Pi GPIO.
* Send movement commands from Raspberry Pi to a microcontroller over UART.
* Add GPS for waypoint navigation.
* Add IMU feedback for heading correction.
* Add ultrasonic sensors or depth camera for better distance estimation.
* Add object tracking to reduce detection jitter.
* Add smoothing or voting across multiple frames.
* Improve the dataset for water/trash-specific detection.
* Train a custom trash detection model instead of using only COCO labels.
* Add automatic boot startup using `systemd`.

---

## Engineering Lessons Learned

The biggest lesson from this project was that real-world computer vision is noisy.

A beginner may assume that if a model detects an object and gives a bounding box, then all information from that box is reliable. In practice, that is not always true.

The bounding box area changed too much because of object angle, partial detection, reflection, lighting, and frame-to-frame jitter. Because of that, the system avoided using area for distance decisions.

Instead of forcing unreliable area-based distance logic, the project used a simpler and more dependable approach:

```text
Use object position to decide steering direction.
```

That is a stronger design choice for this stage of the project.

---

## Credits

The object detection setup was based on a Raspberry Pi MobileNet SSD object detection workflow inspired by:

```text
https://core-electronics.com.au/guides/object-identify-raspberry-pi/
```

This repository adapts the idea for an autonomous trash-collecting boat prototype.

---

## License

This project is licensed under the MIT License.

```

One thing I’d tighten in the repo too: rename `object_ident_updated (2).py` to something cleaner like `trash_detection.py`. Filenames with spaces and parentheses work, but they look messy on GitHub and are annoying to run from terminal.
::contentReference[oaicite:1]{index=1}
```

[1]: https://github.com/aminashamraze/Autonomous-Trash-Collector-Boat-using-Raspberry-Pi-and-MobileNetSSD "GitHub - aminashamraze/Autonomous-Trash-Collector-Boat-using-Raspberry-Pi-and-MobileNetSSD: A Raspberry-Pi based autonomous trash collector robot that laverages the use of MobileNetSSD for detection of trash and collects them, useful for small ponds and still waters. · GitHub"


