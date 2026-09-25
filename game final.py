import json
import os
import random
import cv2
import math
import time
import numpy as np
import mediapipe as mp
import pygame

# --- AUDIO GENERATOR ---
pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)

def generate_laser_sound():
    duration = 0.12
    n_samples = int(22050 * duration)
    t = np.linspace(0, duration, n_samples, False)
    freq = np.linspace(950, 120, n_samples)
    waveform = np.sin(2 * np.pi * freq * t) * 0.35 * (1 - t / duration)
    return pygame.mixer.Sound((waveform * 32767).astype(np.int16))

def generate_explosion_sound():
    duration = 0.25
    n_samples = int(22050 * duration)
    t = np.linspace(0, duration, n_samples, False)
    noise = np.random.uniform(-1, 1, n_samples)
    envelope = np.exp(-t * 12) * 0.4
    return pygame.mixer.Sound((noise * envelope * 32767).astype(np.int16))

def generate_game_over_sound():
    duration = 0.5
    n_samples = int(22050 * duration)
    t = np.linspace(0, duration, n_samples, False)
    freq = np.linspace(400, 100, n_samples)
    waveform = np.sin(2 * np.pi * freq * t) * 0.5 * (1 - t / duration)
    return pygame.mixer.Sound((waveform * 32767).astype(np.int16))

def generate_powerup_sound():
    duration = 0.3
    n_samples = int(22050 * duration)
    t = np.linspace(0, duration, n_samples, False)
    freq = np.linspace(300, 900, n_samples)
    waveform = np.sin(2 * np.pi * freq * t) * 0.4 * (1 - t / duration)
    return pygame.mixer.Sound((waveform * 32767).astype(np.int16))

def generate_nuke_sound():
    duration = 0.6
    n_samples = int(22050 * duration)
    t = np.linspace(0, duration, n_samples, False)
    noise = np.random.uniform(-1, 1, n_samples)
    envelope = np.exp(-t * 4) * 0.6
    return pygame.mixer.Sound((noise * envelope * 32767).astype(np.int16))

snd_laser = generate_laser_sound()
snd_explosion = generate_explosion_sound()
snd_game_over = generate_game_over_sound()
snd_powerup = generate_powerup_sound()
snd_nuke = generate_nuke_sound()

# --- DIRECTORY & DATA SETUP ---
FACE_DIR = "player_faces"
if not os.path.exists(FACE_DIR):
    os.makedirs(FACE_DIR)

LEADERBOARD_FILE = "leaderboard.json"

def load_leaderboard():
    if os.path.exists(LEADERBOARD_FILE):
        try:
            with open(LEADERBOARD_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_score(name, score, photo_path):
    if not name.strip():
        name = "Pilot"
    board = load_leaderboard()
    board.append({"name": name.strip(), "score": score, "photo": photo_path})
    board = sorted(board, key=lambda x: x["score"], reverse=True)[:10]
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(board, f, indent=4)
    return board

# --- MEDIAPIPE SETUP ---
cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- DISPLAY CONFIGURATION ---
WINDOW_NAME = "IT Exhibition - AI Cyberpunk Arcade"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# --- GAME STATE & VARIABLES ---
player_name = ""
entering_name = True
current_photo_path = None
current_photo_img = None
photo_cache = {}

score = 0
player_x = 320
player_y = 400
player_size = 25

obstacles = []
lasers = []
particles = []
powerups = []
muzzle_flashes = []
combo_texts = [] # Floating combo text popups

combo_count = 0
last_kill_time = 0

stars = [[random.randint(0, 640), random.randint(0, 480), random.randint(1, 3), random.uniform(0.1, 1.0)] for _ in range(50)]

active_powerup = None
powerup_end_time = 0

last_laser_time = 0
game_over = False
score_saved = False
cached_leaderboard = []

def draw_panel(img, pt1, pt2, border_color=(0, 255, 255), fill_color=(15, 15, 20), alpha=0.75):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, fill_color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, pt1, pt2, border_color, 1)

    c_len = 10
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1, y1), (x1 + c_len, y1), border_color, 2)
    cv2.line(img, (x1, y1), (x1, y1 + c_len), border_color, 2)
    cv2.line(img, (x2, y1), (x2 - c_len, y1), border_color, 2)
    cv2.line(img, (x2, y1), (x2, y1 + c_len), border_color, 2)
    cv2.line(img, (x1, y2), (x1 + c_len, y2), border_color, 2)
    cv2.line(img, (x1, y2), (x1, y2 - c_len), border_color, 2)
    cv2.line(img, (x2, y2), (x2 - c_len, y2), border_color, 2)
    cv2.line(img, (x2, y2), (x2, y2 - c_len), border_color, 2)

# --- UPGRADED HIGH-TECH SUNGLASSES ENGINE ---
def draw_cool_sunglasses(frame, face_landmarks, h, w):
    re = face_landmarks.landmark[33]
    le = face_landmarks.landmark[263]
    temple_r = face_landmarks.landmark[127]
    temple_l = face_landmarks.landmark[356]

    p1 = (int(re.x * w), int(re.y * h))
    p2 = (int(le.x * w), int(le.y * h))
    pr = (int(temple_r.x * w), int(temple_r.y * h))
    pl = (int(temple_l.x * w), int(temple_l.y * h))

    angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    eye_dist = int(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2

    glass_w = int(eye_dist * 2.8)
    glass_h = int(eye_dist * 1.1)

    if glass_w < 12 or glass_h < 12:
        return

    overlay = np.zeros((glass_h, glass_w, 3), dtype=np.uint8)

    # Sleek Cyberpunk Double-Bridged Aviator Lenses
    left_lens = np.array([
        [int(glass_w * 0.04), int(glass_h * 0.20)],
        [int(glass_w * 0.46), int(glass_h * 0.20)],
        [int(glass_w * 0.42), int(glass_h * 0.88)],
        [int(glass_w * 0.15), int(glass_h * 0.95)]
    ], np.int32)

    right_lens = np.array([
        [int(glass_w * 0.54), int(glass_h * 0.20)],
        [int(glass_w * 0.96), int(glass_h * 0.20)],
        [int(glass_w * 0.85), int(glass_h * 0.95)],
        [int(glass_w * 0.58), int(glass_h * 0.88)]
    ], np.int32)

    # Dark Polarized Lens Fill & Neon Outlines
    cv2.fillPoly(overlay, [left_lens, right_lens], (12, 12, 18))
    cv2.polylines(overlay, [left_lens, right_lens], True, (0, 255, 255), 2)

    # High-Tech Brow & Nose Bridge
    cv2.line(overlay, (int(glass_w * 0.02), int(glass_h * 0.18)), (int(glass_w * 0.98), int(glass_h * 0.18)), (255, 0, 255), 2)
    cv2.line(overlay, (int(glass_w * 0.44), int(glass_h * 0.35)), (int(glass_w * 0.56), int(glass_h * 0.35)), (0, 255, 255), 2)

    # Dynamic Holographic Lens Reflection Streaks
    cv2.line(overlay, (int(glass_w * 0.10), int(glass_h * 0.28)), (int(glass_w * 0.28), int(glass_h * 0.80)), (255, 255, 255), 2)
    cv2.line(overlay, (int(glass_w * 0.16), int(glass_h * 0.28)), (int(glass_w * 0.32), int(glass_h * 0.75)), (255, 255, 255), 1)

    cv2.line(overlay, (int(glass_w * 0.60), int(glass_h * 0.28)), (int(glass_w * 0.78), int(glass_h * 0.80)), (255, 255, 255), 2)
    cv2.line(overlay, (int(glass_w * 0.66), int(glass_h * 0.28)), (int(glass_w * 0.82), int(glass_h * 0.75)), (255, 255, 255), 1)

    M = cv2.getRotationMatrix2D((glass_w // 2, glass_h // 2), angle, 1.0)
    rotated_glass = cv2.warpAffine(overlay, M, (glass_w, glass_h))

    x1, y1 = max(0, cx - glass_w // 2), max(0, cy - glass_h // 2)
    x2, y2 = min(w, x1 + glass_w), min(h, y1 + glass_h)

    crop_w, crop_h = x2 - x1, y2 - y1
    if crop_w <= 0 or crop_h <= 0:
        return

    glass_crop = rotated_glass[0:crop_h, 0:crop_w]
    gray_crop = cv2.cvtColor(glass_crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_crop, 1, 255, cv2.THRESH_BINARY)

    roi = frame[y1:y2, x1:x2]
    roi_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    glass_fg = cv2.bitwise_and(glass_crop, glass_crop, mask=mask)
    frame[y1:y2, x1:x2] = cv2.add(roi_bg, glass_fg)

    cv2.line(frame, (cx - glass_w // 2, cy - 2), pr, (0, 255, 255), 2)
    cv2.line(frame, (cx + glass_w // 2, cy - 2), pl, (0, 255, 255), 2)

def draw_spacecraft_and_ignition(frame, x, y, is_shooting):
    flicker = random.randint(-3, 4)
    flame_length = random.randint(22, 34) if is_shooting else random.randint(14, 22)

    outer_flame = np.array([
        [x - 12, y + 14],
        [x + 12, y + 14],
        [x, y + 14 + flame_length + flicker]
    ], np.int32)
    cv2.fillPoly(frame, [outer_flame], (0, 100, 255))

    inner_flame = np.array([
        [x - 6, y + 14],
        [x + 6, y + 14],
        [x, y + 14 + int(flame_length * 0.65)]
    ], np.int32)
    cv2.fillPoly(frame, [inner_flame], (0, 235, 255))

    if random.random() < 0.8:
        particles.append([
            x + random.randint(-8, 8),
            y + 16 + flame_length,
            random.uniform(-1.0, 1.0),
            random.uniform(3.0, 6.0),
            (0, 215, 255),
            random.randint(6, 12)
        ])

    wings = np.array([
        [x, y - 30],
        [x - 26, y + 16],
        [x - 12, y + 12],
        [x, y + 18],
        [x + 12, y + 12],
        [x + 26, y + 16]
    ], np.int32)

    ship_color = (0, 255, 255) if is_shooting else (255, 180, 0)
    cv2.fillPoly(frame, [wings], (35, 35, 45))
    cv2.polylines(frame, [wings], True, ship_color, 2)

    cockpit = np.array([
        [x, y - 18],
        [x - 6, y + 2],
        [x + 6, y + 2]
    ], np.int32)
    cv2.fillPoly(frame, [cockpit], (255, 230, 100))
    cv2.polylines(frame, [cockpit], True, (255, 255, 255), 1)

    cv2.circle(frame, (x - 22, y + 10), 3, (0, 255, 0), -1)
    cv2.circle(frame, (x + 22, y + 10), 3, (0, 255, 0), -1)

def draw_realistic_meteor(frame, mx, my, radius, vertices, angle):
    if random.random() < 0.7:
        particles.append([
            mx + random.randint(-radius // 2, radius // 2),
            my - radius,
            random.uniform(-0.8, 0.8),
            random.uniform(-4.0, -2.0),
            (0, random.randint(100, 180), 255),
            random.randint(8, 14)
        ])

    poly_pts = []
    for r_offset, a_offset in vertices:
        rad = math.radians(angle + a_offset)
        r_curr = radius + r_offset
        px = int(mx + r_curr * math.cos(rad))
        py = int(my + r_curr * math.sin(rad))
        poly_pts.append([px, py])

    pts_array = np.array(poly_pts, np.int32)

    cv2.fillPoly(frame, [pts_array], (40, 45, 55))
    cv2.polylines(frame, [pts_array], True, (0, 120, 255), 2)
    cv2.circle(frame, (mx, my + radius // 2), int(radius * 0.6), (0, 60, 200), -1)

    crater_offset_1 = (int(mx - radius * 0.3), int(my - radius * 0.2))
    crater_offset_2 = (int(mx + radius * 0.2), int(my + radius * 0.3))
    cv2.circle(frame, crater_offset_1, max(3, radius // 4), (25, 28, 35), -1)
    cv2.circle(frame, crater_offset_1, max(3, radius // 4), (15, 18, 22), 1)
    cv2.circle(frame, crater_offset_2, max(2, radius // 5), (25, 28, 35), -1)

def draw_plasma_laser(frame, x1, y1, x2, y2, color_theme=(0, 255, 255)):
    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_theme, 8)
    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)
    cv2.circle(frame, (int(x1), int(y1)), 6, color_theme, -1)
    cv2.circle(frame, (int(x1), int(y1)), 3, (255, 255, 255), -1)

    if random.random() < 0.6:
        particles.append([
            int(x1) + random.randint(-4, 4),
            int(y1) + random.randint(5, 15),
            random.uniform(-0.5, 0.5),
            random.uniform(1.0, 3.0),
            color_theme,
            random.randint(6, 10)
        ])

def trigger_muzzle_flash(x, y, color=(0, 255, 255)):
    muzzle_flashes.append([x, y, 4, color])

def draw_muzzle_flashes(frame):
    for mf in muzzle_flashes[:]:
        x, y, radius, color = mf
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            ex = int(x + (radius * 3) * math.cos(rad))
            ey = int(y + (radius * 3) * math.sin(rad))
            cv2.line(frame, (x, y), (ex, ey), color, 2)

        cv2.circle(frame, (x, y), radius * 2, (255, 255, 255), -1)
        mf[2] -= 1
        if mf[2] <= 0:
            muzzle_flashes.remove(mf)

def add_combo_popup(x, y, points, multiplier):
    text = f"+{points} COMBO! x{multiplier}" if multiplier > 1 else f"+{points}"
    color = (0, 255, 255) if multiplier > 1 else (0, 215, 255)
    combo_texts.append([x, y, text, color, 1.0]) # X, Y, Text, Color, Alpha/Life

def draw_combo_popups(frame):
    for ct in combo_texts[:]:
        x, y, text, color, alpha = ct
        
        # Render text with drop shadow
        cv2.putText(frame, text, (int(x) + 1, int(y) + 1), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1)

        ct[1] -= 1.5   # Drift upward
        ct[4] -= 0.04  # Fade out
        if ct[4] <= 0:
            combo_texts.remove(ct)

def create_explosion(x, y, color, count=16):
    for _ in range(count):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 8)
        particles.append([x, y, math.cos(angle) * speed, math.sin(angle) * speed, color, random.randint(12, 20)])

# --- MAIN LOOP ---
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    if entering_name:
        raw_frame = frame.copy()

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                draw_cool_sunglasses(frame, face_landmarks, h, w)

        draw_panel(frame, (50, 30), (590, 450), border_color=(0, 255, 255), alpha=0.85)

        cv2.putText(frame, "AI PILOT REGISTRATION", (140, 75), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 255), 2)
        cv2.putText(frame, "Enter Call-Sign / Name:", (90, 120), cv2.FONT_HERSHEY_DUPLEX, 0.6, (220, 220, 220), 1)

        draw_panel(frame, (90, 135), (550, 185), border_color=(0, 255, 0), fill_color=(10, 25, 10), alpha=0.9)
        cv2.putText(frame, player_name + "_", (105, 170), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 0), 2)

        draw_panel(frame, (220, 200), (420, 370), border_color=(0, 215, 255), alpha=0.9)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                draw_cool_sunglasses(raw_frame, face_landmarks, h, w)

        cam_crop = cv2.resize(raw_frame[80:400, 180:460], (196, 166))
        frame[202:368, 222:418] = cam_crop
        cv2.putText(frame, "[AR SCAN ACTIVE]", (250, 360), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1)

        cv2.putText(frame, "Type name & Press ENTER to Scan & Play", (100, 400), cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "Controls: 1 Finger (Move) | 2 Fingers (Fire)", (95, 425), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 215, 255), 1)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(player_name.strip()) > 0:
            snap_frame = raw_frame.copy()
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    draw_cool_sunglasses(snap_frame, face_landmarks, h, w)

            timestamp = int(time.time())
            filename = f"{FACE_DIR}/{player_name.strip()}_{timestamp}.jpg"
            face_crop = cv2.resize(snap_frame[50:430, 150:490], (100, 100))
            cv2.imwrite(filename, face_crop)

            current_photo_path = filename
            current_photo_img = cv2.resize(face_crop, (60, 60))
            entering_name = False

        elif key == 8:
            player_name = player_name[:-1]
        elif key != 255 and len(player_name) < 10 and chr(key).isprintable():
            player_name += chr(key)
        continue

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            draw_cool_sunglasses(frame, face_landmarks, h, w)

    is_shooting = False

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            index_tip = hand_landmarks.landmark[8]
            player_x = int(index_tip.x * w)

            middle_tip = hand_landmarks.landmark[12]
            middle_pip = hand_landmarks.landmark[10]

            if middle_tip.y < middle_pip.y:
                is_shooting = True

                if time.time() - last_laser_time > 0.16 and not game_over:
                    las_color = (255, 255, 0) if active_powerup == "TRIPLE" else (0, 255, 255)

                    if active_powerup == "TRIPLE":
                        lasers.append([player_x - 22, player_y - 15, -4, -20, las_color])
                        lasers.append([player_x, player_y - 30, 0, -20, las_color])
                        lasers.append([player_x + 22, player_y - 15, 4, -20, las_color])

                        trigger_muzzle_flash(player_x - 22, player_y - 15, las_color)
                        trigger_muzzle_flash(player_x, player_y - 30, las_color)
                        trigger_muzzle_flash(player_x + 22, player_y - 15, las_color)
                    else:
                        lasers.append([player_x - 14, player_y - 10, 0, -20, las_color])
                        lasers.append([player_x + 14, player_y - 10, 0, -20, las_color])

                        trigger_muzzle_flash(player_x - 14, player_y - 10, las_color)
                        trigger_muzzle_flash(player_x + 14, player_y - 10, las_color)

                    snd_laser.play()
                    last_laser_time = time.time()

    player_x = max(35, min(w - 35, player_x))

    for star in stars:
        star[1] += star[2]
        if star[1] > h: star[1] = 0; star[0] = random.randint(0, w)
        color_val = int(255 * star[3])
        cv2.circle(frame, (star[0], star[1]), star[2], (color_val, color_val, color_val), -1)

    if active_powerup and time.time() > powerup_end_time:
        active_powerup = None

    if not game_over:
        if random.random() < 0.06:
            r = random.randint(18, 28)
            num_verts = random.randint(6, 9)
            verts = [(random.randint(-r // 3, r // 3), i * (360 // num_verts)) for i in range(num_verts)]
            obstacles.append([
                random.randint(35, w - 35),
                -20,
                random.randint(5, 9),
                r,
                verts,
                random.randint(0, 360),
                random.randint(-5, 5)
            ])

        if random.random() < 0.005 and len(powerups) < 2:
            ptype = random.choice(["TRIPLE", "DOUBLE", "NUKE"])
            powerups.append([random.randint(50, w - 50), 0, 3, ptype])

        for pw in powerups[:]:
            pw[1] += pw[2]
            color = (255, 255, 0) if pw[3] == "TRIPLE" else ((0, 215, 255) if pw[3] == "DOUBLE" else (255, 0, 255))
            label = "3X" if pw[3] == "TRIPLE" else ("2X" if pw[3] == "DOUBLE" else "N")

            cv2.circle(frame, (pw[0], pw[1]), 16, color, 2)
            cv2.circle(frame, (pw[0], pw[1]), 12, (20, 20, 20), -1)
            cv2.putText(frame, label, (pw[0] - 8, pw[1] + 5), cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 1)

            if pw[1] > h: powerups.remove(pw)

        for laser in lasers[:]:
            laser[0] += laser[2]
            laser[1] += laser[3]
            tail_x = laser[0] - laser[2]
            tail_y = laser[1] - laser[3] + 20

            draw_plasma_laser(frame, laser[0], laser[1], tail_x, tail_y, laser[4])

            if laser[1] < 0 or laser[0] < 0 or laser[0] > w:
                lasers.remove(laser); continue

            for pw in powerups[:]:
                if math.hypot(laser[0] - pw[0], laser[1] - pw[1]) < 22:
                    snd_powerup.play()
                    create_explosion(pw[0], pw[1], (255, 255, 255), count=20)
                    if pw[3] == "NUKE":
                        snd_nuke.play()
                        for obs in obstacles:
                            create_explosion(obs[0], obs[1], (0, 100, 255), count=10)
                            score += 20 if active_powerup == "DOUBLE" else 10
                        obstacles.clear()
                    else:
                        active_powerup = pw[3]
                        powerup_end_time = time.time() + 8.0

                    if laser in lasers: lasers.remove(laser)
                    if pw in powerups: powerups.remove(pw)
                    break

        for obs in obstacles[:]:
            obs[1] += obs[2]
            obs[5] += obs[6]

            draw_realistic_meteor(frame, obs[0], obs[1], obs[3], obs[4], obs[5])

            for laser in lasers[:]:
                if math.hypot(laser[0] - obs[0], laser[1] - obs[1]) < obs[3] + 8:
                    create_explosion(obs[0], obs[1], (0, 140, 255))
                    snd_explosion.play()

                    # Combo Multiplier Calculation
                    now = time.time()
                    if now - last_kill_time < 2.0:
                        combo_count += 1
                    else:
                        combo_count = 1
                    last_kill_time = now

                    pts_gained = (20 if active_powerup == "DOUBLE" else 10) * combo_count
                    score += pts_gained

                    add_combo_popup(obs[0], obs[1], pts_gained, combo_count)

                    if laser in lasers: lasers.remove(laser)
                    if obs in obstacles: obstacles.remove(obs)
                    break

            if math.hypot(player_x - obs[0], player_y - obs[1]) < obs[3] + player_size:
                create_explosion(player_x, player_y, (0, 0, 255), count=30)
                snd_game_over.play()
                game_over = True

            if obs[1] > h + 30:
                obstacles.remove(obs)
                score += 2 if active_powerup == "DOUBLE" else 1

        for p in particles[:]:
            p[0] += p[2]; p[1] += p[3]; p[5] -= 1
            cv2.circle(frame, (int(p[0]), int(p[1])), max(1, p[5] // 3), p[4], -1)
            if p[5] <= 0: particles.remove(p)

    if not game_over:
        draw_spacecraft_and_ignition(frame, player_x, player_y, is_shooting)
        draw_muzzle_flashes(frame)
        draw_combo_popups(frame)

    draw_panel(frame, (15, 15), (340, 90), border_color=(0, 255, 255), alpha=0.85)

    if current_photo_img is not None:
        frame[22:82, 22:82] = current_photo_img
        cv2.rectangle(frame, (22, 22), (82, 82), (0, 255, 0), 1)

    cv2.putText(frame, f"PILOT: {player_name[:10]}", (92, 45), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"SCORE: {score:04d}", (92, 75), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)

    if active_powerup:
        rem_time = max(0.0, powerup_end_time - time.time())
        p_color = (255, 255, 0) if active_powerup == "TRIPLE" else (0, 215, 255)
        draw_panel(frame, (355, 15), (625, 55), border_color=p_color, alpha=0.85)
        cv2.putText(frame, f"{active_powerup} ({rem_time:.1f}s)", (365, 42), cv2.FONT_HERSHEY_DUPLEX, 0.55, p_color, 1)

    if game_over:
        if not score_saved:
            cached_leaderboard = save_score(player_name, score, current_photo_path)
            score_saved = True

        draw_panel(frame, (50, 30), (590, 450), border_color=(0, 0, 255), alpha=0.9)

        cv2.putText(frame, "GAME OVER", (230, 75), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, f"Final Score: {score}", (240, 105), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 1)

        cv2.putText(frame, "=== TOP PILOTS LEADERBOARD ===", (140, 135), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)

        y_offset = 155
        for idx, entry in enumerate(cached_leaderboard[:4]):
            color = (0, 215, 255) if entry['name'] == player_name and entry['score'] == score else (220, 220, 220)

            txt = f"{idx+1}. {entry['name']:<10} ..... {entry['score']:04d} pts"
            cv2.putText(frame, txt, (170, y_offset + 28), cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1)

            img_path = entry.get("photo", "")
            if img_path and os.path.exists(img_path):
                if img_path not in photo_cache:
                    p_img = cv2.imread(img_path)
                    if p_img is not None:
                        photo_cache[img_path] = cv2.resize(p_img, (40, 40))

                if img_path in photo_cache:
                    p_crop = photo_cache[img_path]
                    frame[y_offset:y_offset + 40, 115:155] = p_crop
                    cv2.rectangle(frame, (115, y_offset), (155, y_offset + 40), color, 1)

            y_offset += 50

        draw_panel(frame, (80, 395), (560, 435), border_color=(255, 255, 255), fill_color=(20, 20, 30), alpha=0.9)
        cv2.putText(frame, "PRESS [R] RESTART  |  PRESS [C] NEW PILOT", (90, 422), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if game_over:
        if key in [ord("r"), ord("R")]:
            score = 0
            combo_count = 0
            obstacles.clear()
            lasers.clear()
            particles.clear()
            powerups.clear()
            combo_texts.clear()
            active_powerup = None
            game_over = False
            score_saved = False

        elif key in [ord("c"), ord("C")]:
            player_name = ""
            entering_name = True
            current_photo_path = None
            current_photo_img = None
            score = 0
            combo_count = 0
            obstacles.clear()
            lasers.clear()
            particles.clear()
            powerups.clear()
            combo_texts.clear()
            active_powerup = None
            game_over = False
            score_saved = False

cap.release()
cv2.destroyAllWindows()
