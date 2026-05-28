# smooth_decision_wrapper_locked.py
from picamera2 import Picamera2
import cv2, numpy as np, time
from collections import deque

# ---------- Tunables ----------
FRAME_W, FRAME_H = 960, 544
DETECT_EVERY_N_FRAMES = 3
MIN_AREA = 350
CROP_TOP_FRACTION = 0.20
AUTO_CANNY_SIGMA = 0.33

# Presence voting
M = 12
N = 7

# Steering / approach
CENTER_TOL_PX = 70
STEER_HYSTERESIS_PX = 35
APPROACH_AREA_PX = 14000
STOP_AREA_RATIO = 0.55         # stop only when target fills enough of frame
FULL_FRAME_RATIO = 0.70

# Motion smoothing
ALPHA_CX = 0.35
ALPHA_AREA = 0.30
MAX_BOXES_FOR_STABLE_TRACK = 8

# Target scoring
AREA_WEIGHT = 1.0
CENTER_WEIGHT = 0.8
PREV_WEIGHT = 1.2

# Target lock
LOCK_DIST_PX = 140             # max center movement to still count as same object
LOCK_AREA_RATIO_MIN = 0.45
LOCK_AREA_RATIO_MAX = 2.2
LOCK_MAX_MISSES = 3            # how many detection updates to tolerate before dropping lock
SWITCH_SCORE_MARGIN = 1.35     # challenger must be this much better to steal lock

# ---------- Helpers ----------
def auto_canny(gray, sigma=AUTO_CANNY_SIGMA):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)

def smooth(prev, new, alpha):
    if prev is None:
        return new
    return alpha * new + (1.0 - alpha) * prev

def box_center(box):
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0

def box_area(box):
    return box[2] * box[3]

def find_boxes(frame_bgr):
    h = frame_bgr.shape[0]
    y0 = int(h * CROP_TOP_FRACTION)
    roi = frame_bgr[y0:, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = auto_canny(gray)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue

        x, y, w, hb = cv2.boundingRect(c)
        extent = float(area) / (w * hb + 1e-6)
        if extent < 0.15:
            continue

        boxes.append((x, y + y0, w, hb))
    return boxes

def choose_leftmost_large(boxes):
    """For clutter/fresh selection: keep only major boxes, then pick leftmost."""
    if not boxes:
        return None

    areas = [box_area(b) for b in boxes]
    max_area = max(areas)
    candidates = [b for b in boxes if box_area(b) >= 0.45 * max_area]
    return min(candidates, key=lambda b: b[0])

def target_score(box, prev_cx=None):
    """
    Higher score = more attractive target.
    Prefers larger area, closer to center, and closer to previous target.
    Small left bias helps break ties.
    """
    x, y, w, h = box
    cx = x + w / 2.0
    area = w * h
    img_cx = FRAME_W / 2.0

    center_penalty = abs(cx - img_cx) / img_cx
    prev_penalty = 0.0 if prev_cx is None else abs(cx - prev_cx) / img_cx

    score = (
        AREA_WEIGHT * area
        - CENTER_WEIGHT * center_penalty * 10000
        - PREV_WEIGHT * prev_penalty * 10000
    )

    # slight left preference as tie-breaker
    score += (FRAME_W - cx) * 0.01
    return score

def pick_target(boxes, prev_cx=None):
    """Fresh selection only. Choose best scored box."""
    if not boxes:
        return None
    return max(boxes, key=lambda b: target_score(b, prev_cx=prev_cx))

def match_locked_target(boxes, locked_target):
    """
    Try to match the previously locked target to a nearby current box.
    Returns best matching box or None.
    """
    if locked_target is None or not boxes:
        return None

    lcx, lcy = box_center(locked_target)
    larea = box_area(locked_target)

    best = None
    best_dist = float("inf")

    for b in boxes:
        cx, cy = box_center(b)
        area = box_area(b)

        dist = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
        area_ratio = area / (larea + 1e-6)

        if dist <= LOCK_DIST_PX and LOCK_AREA_RATIO_MIN <= area_ratio <= LOCK_AREA_RATIO_MAX:
            if dist < best_dist:
                best_dist = dist
                best = b

    return best

def maybe_switch_target(locked_target, fresh_candidate, prev_cx=None):
    """
    Even if a fresh candidate exists, do not switch unless it is clearly better.
    This prevents hopping between two nearby objects.
    """
    if locked_target is None:
        return fresh_candidate

    if fresh_candidate is None:
        return locked_target

    locked_score = target_score(locked_target, prev_cx=prev_cx)
    fresh_score = target_score(fresh_candidate, prev_cx=prev_cx)

    if fresh_score > locked_score * SWITCH_SCORE_MARGIN:
        return fresh_candidate
    return locked_target

# ---------- State machine ----------
SEARCH, ACQUIRE, TRACK, CONFIRM = "SEARCH", "ACQUIRE", "TRACK", "CONFIRM"

def main():
    picam2 = Picamera2()
    cfg = picam2.create_video_configuration(
        main={"size": (FRAME_W, FRAME_H)},
        controls={"FrameRate": 30}
    )
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.3)

    detections = deque(maxlen=M)

    state = SEARCH
    frame_i = 0

    last_boxes = []
    last_target = None

    # Locked target state
    locked_target = None
    locked_miss_count = 0

    # Smoothed metrics
    smoothed_cx = None
    smoothed_area = None

    last_turn = None
    confirm_count = 0

    frame_area = FRAME_W * FRAME_H

    print("ESC to quit.")
    while True:
        frame = picam2.capture_array()
        frame_i += 1

        if frame_i % DETECT_EVERY_N_FRAMES == 0:
            last_boxes = find_boxes(frame)
            cluttered = len(last_boxes) > MAX_BOXES_FOR_STABLE_TRACK

            # 1) First try to keep the currently locked target
            matched = match_locked_target(last_boxes, locked_target)

            if matched is not None:
                locked_target = matched
                locked_miss_count = 0
            else:
                # current lock not matched this round
                locked_miss_count += 1

                # 2) Only choose a new target if lock has been lost for a few updates
                if locked_miss_count > LOCK_MAX_MISSES:
                    if cluttered:
                        fresh_candidate = choose_leftmost_large(last_boxes)
                    else:
                        fresh_candidate = pick_target(last_boxes, prev_cx=smoothed_cx)

                    locked_target = fresh_candidate
                    if locked_target is not None:
                        locked_miss_count = 0
                else:
                    # 3) Keep old target alive briefly even if missed once/twice
                    if cluttered:
                        fresh_candidate = choose_leftmost_large(last_boxes)
                    else:
                        fresh_candidate = pick_target(last_boxes, prev_cx=smoothed_cx)

                    locked_target = maybe_switch_target(
                        locked_target,
                        fresh_candidate,
                        prev_cx=smoothed_cx
                    )

            last_target = locked_target
            detections.append(last_target is not None)

            if last_target is not None:
                raw_cx, _ = box_center(last_target)
                raw_area = box_area(last_target)

                smoothed_cx = smooth(smoothed_cx, raw_cx, ALPHA_CX)
                smoothed_area = smooth(smoothed_area, raw_area, ALPHA_AREA)

        positives = sum(detections)
        trash_present = (len(detections) == M and positives >= N)

        cluttered = len(last_boxes) > MAX_BOXES_FOR_STABLE_TRACK
        action = "HOLD"

        if last_target is not None and smoothed_cx is not None and smoothed_area is not None:
            x, y, w, h = last_target
            cx = smoothed_cx
            area = smoothed_area

            centered = abs(cx - FRAME_W / 2.0) <= CENTER_TOL_PX
            approach_near = area >= APPROACH_AREA_PX
            stop_ready = (area / frame_area) >= STOP_AREA_RATIO
            full_frame_like = (area / frame_area) >= FULL_FRAME_RATIO
        else:
            x = y = w = h = 0
            cx = None
            area = 0
            centered = False
            approach_near = False
            stop_ready = False
            full_frame_like = False

        # ---------- State transitions + actions ----------
        if state == SEARCH:
            confirm_count = 0
            if trash_present:
                state = ACQUIRE
            action = "FORWARD_SLOW"

        elif state == ACQUIRE:
            confirm_count = 0
            if not trash_present:
                state = SEARCH
                action = "FORWARD_SLOW"
            else:
                state = TRACK
                action = "SLOW_DOWN"

        elif state == TRACK:
            confirm_count = 0
            if not trash_present:
                state = SEARCH
                action = "FORWARD_SLOW"
            elif last_target is None or cx is None:
                action = "TURN_LEFT" if cluttered else "HOLD"
            else:
                err = cx - FRAME_W / 2.0

                # hysteresis steering to avoid left-right flicker
                if err < -(CENTER_TOL_PX + STEER_HYSTERESIS_PX):
                    action = "TURN_LEFT"
                    last_turn = "LEFT"
                elif err > (CENTER_TOL_PX + STEER_HYSTERESIS_PX):
                    action = "TURN_RIGHT"
                    last_turn = "RIGHT"
                elif abs(err) <= CENTER_TOL_PX:
                    action = "FORWARD"
                else:
                    if last_turn == "LEFT":
                        action = "TURN_LEFT"
                    elif last_turn == "RIGHT":
                        action = "TURN_RIGHT"
                    else:
                        action = "FORWARD"

                # centered but not close enough to stop -> keep approaching
                if centered and approach_near and not stop_ready:
                    action = "FORWARD"

                if centered and approach_near:
                    state = CONFIRM

        elif state == CONFIRM:
            if not trash_present:
                state = SEARCH
                confirm_count = 0
                action = "FORWARD_SLOW"
            elif last_target is None or cx is None:
                state = TRACK
                confirm_count = 0
                action = "HOLD"
            else:
                err = cx - FRAME_W / 2.0

                if cluttered or full_frame_like:
                    if err < -CENTER_TOL_PX:
                        action = "TURN_LEFT"
                    elif err > CENTER_TOL_PX:
                        action = "TURN_RIGHT"
                    else:
                        action = "TURN_LEFT"   # left-first fallback
                else:
                    if err < -CENTER_TOL_PX:
                        action = "TURN_LEFT"
                    elif err > CENTER_TOL_PX:
                        action = "TURN_RIGHT"
                    else:
                        if stop_ready:
                            confirm_count += 1
                            action = "TRASH_FOUND_STOP"
                        else:
                            confirm_count = 0
                            action = "FORWARD"

                # require repeat confirmation before final stop
                if confirm_count < 2 and action == "TRASH_FOUND_STOP":
                    action = "FORWARD"

                if not centered and abs(err) > (CENTER_TOL_PX + STEER_HYSTERESIS_PX):
                    state = TRACK
                    confirm_count = 0

        # ---------- Draw overlays ----------
        for (bx, by, bw, bh) in last_boxes:
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

        if last_target is not None:
            tx, ty, tw, th = last_target
            cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 0, 255), 3)

        cv2.line(frame, (FRAME_W // 2, 0), (FRAME_W // 2, FRAME_H), (255, 255, 0), 1)

        cv2.putText(frame, f"state={state} action={action}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"vote {positives}/{len(detections)} (need {N}/{M})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"boxes={len(last_boxes)} cluttered={cluttered}", (10, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"lock_misses={locked_miss_count}", (10, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)

        if last_target is not None and smoothed_cx is not None and smoothed_area is not None:
            cv2.putText(frame, f"smoothed area={int(smoothed_area)}", (10, 128),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"fill ratio={smoothed_area / frame_area:.2f}", (10, 154),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (int(smoothed_cx), int(ty + th / 2)), 6, (255, 0, 255), -1)

        cv2.imshow("smooth detection + decision", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
