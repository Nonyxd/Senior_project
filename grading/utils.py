import cv2
import numpy as np
import json
import os
from django.conf import settings

# ==========================================
# 1. Image Processing Helpers
# ==========================================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    maxW = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))
    maxH = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-bl)))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))

def auto_detect_paper(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (img.shape[0]*img.shape[1] * 0.1):
            return approx.reshape(4, 2)
    return None

# ==========================================
# 2. GridMapper Class (สำหรับคำนวณพิกัดบน rectified_output.jpg)
# ==========================================
class GridMapper:
    def __init__(self, img_w, img_h):
        self.w = img_w; self.h = img_h
        self.box_w = 0.032
        self.box_h = 0.024    
        self.step_x = 0.0414; self.step_y = 0.0253
        
        # ตำแหน่งเริ่มต้นของแต่ละคอลัมน์ (ปรับตาม Template จริง)
        self.c1_x, self.c1_y = 0.133, 0.303
        self.c2_x, self.c2_y = 0.4657, 0.0250
        self.c3_x, self.c3_y = 0.7950, 0.0250
        
        self.id_start_x, self.id_start_y = 0.0710, 0.0650
        self.id_step_x, self.id_step_y = 0.0310, 0.0190
        self.id_box_w, self.id_box_h = 0.028, 0.020

    def get_question_coords(self, q_num):
        # กำหนดคอลัมน์และแถวตามเลขข้อ
        if 1 <= q_num <= 26: sx, sy, r = self.c1_x, self.c1_y, q_num-1
        elif 27 <= q_num <= 63: sx, sy, r = self.c2_x, self.c2_y, q_num-27
        elif 64 <= q_num <= 100: sx, sy, r = self.c3_x, self.c3_y, q_num-64
        else: return {}
        
        coords = {}
        base_x = int(sx * self.w)
        base_y = int((sy + (r * self.step_y)) * self.h)
        step_x = int(self.step_x * self.w)
        bw, bh = int(self.box_w * self.w), int(self.box_h * self.h)

        for i, lbl in enumerate(['a','b','c','d','e']):
            cx = base_x + (i * step_x)
            cy = base_y
            coords[lbl] = (cx - bw//2, cy - bh//2, cx + bw//2, cy + bh//2)
        return coords

    def get_student_id_coords(self):
        id_grid = {}
        for col in range(10):
            id_grid[col] = {}
            for digit in range(10):
                bx = int((self.id_start_x + (col * self.id_step_x)) * self.w)
                by = int((self.id_start_y + (digit * self.id_step_y)) * self.h)
                bw, bh = int(self.id_box_w * self.w), int(self.id_box_h * self.h)
                id_grid[col][digit] = (bx - bw//2, by - bh//2, bx + bw//2, by + bh//2)
        return id_grid

# ==========================================
# 3. ROBUST SCANNING (Center Crop)
# ==========================================
def get_center_roi_pixel_count(thresh_img, x1, y1, x2, y2):
    w = x2 - x1; h = y2 - y1
    crop_x = int(w * 0.30); crop_y = int(h * 0.30)
    if w - 2*crop_x <= 0 or h - 2*crop_y <= 0: return 0
    roi = thresh_img[y1+crop_y : y2-crop_y, x1+crop_x : x2-crop_x]
    return cv2.countNonZero(roi)

def robust_scan_answers(thresh_img, mapper):
    detected_answers = {}
    for q in range(1, 101):
        coords = mapper.get_question_coords(q)
        choice_pixels = {}
        for ch, (x1, y1, x2, y2) in coords.items():
            choice_pixels[ch] = get_center_roi_pixel_count(thresh_img, x1, y1, x2, y2)

        sorted_choices = sorted(choice_pixels.items(), key=lambda x: x[1], reverse=True)
        best_ch, max_val = sorted_choices[0]
        
        if max_val < 30: 
            detected_answers[q] = []
            continue

        picked = [best_ch]
        for i in range(1, 5):
            ch_next, val_next = sorted_choices[i]
            if val_next > max_val * 0.85: 
                picked.append(ch_next)
        detected_answers[q] = sorted(picked)
    return detected_answers

def robust_scan_student_id(thresh_img, mapper):
    id_grid = mapper.get_student_id_coords()
    result_list = []
    for col in range(10):
        digit_pixels = {}
        for digit in range(10):
            x1, y1, x2, y2 = id_grid[col][digit]
            digit_pixels[digit] = get_center_roi_pixel_count(thresh_img, x1, y1, x2, y2)
        sorted_digits = sorted(digit_pixels.items(), key=lambda x: x[1], reverse=True)
        best_d, max_val = sorted_digits[0]
        
        if max_val < 30:
            result_list.append("?")
            continue
        picked = [str(best_d)]
        second_d, second_val = sorted_digits[1]
        if second_val > max_val * 0.85:
            picked.append(str(second_d))
        if len(picked) > 1: result_list.append(f"[{','.join(picked)}]")
        else: result_list.append(picked[0])
    return result_list

# ==========================================
# 4. Grading Logic
# ==========================================
def grade_exam_logic(student_ans, correct_key):
    score = 0; results = {}
    for q in range(1, 101):
        stu = student_ans.get(q, [])
        cor_list = correct_key.get(str(q), [])
        cor_str = cor_list[0] if cor_list else None
        
        if not cor_str: results[q]="UNKNOWN"; continue
        if len(stu) > 1: results[q] = "DOUBLE"
        elif len(stu) == 0: results[q] = "EMPTY"
        elif len(stu) == 1 and stu[0] == cor_str:
            score += 1; results[q] = "CORRECT"
        else: results[q] = "WRONG"
    return score, results

def draw_result_on_image(img, mapper, stu_ans, key, results):
    vis = img.copy()
    for q in range(1, 101):
        coords = mapper.get_question_coords(q)
        stu_list = stu_ans.get(q, [])
        status = results.get(q, "UNKNOWN")
        cor_list = key.get(str(q), [])
        cor_str = cor_list[0] if cor_list else None
        
        for ch, (x1, y1, x2, y2) in coords.items():
            color = None
            if ch in stu_list:
                if status == "CORRECT": color = (0, 255, 0)
                elif status == "DOUBLE": color = (0, 255, 255)
                else: color = (0, 0, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            if status in ["WRONG", "EMPTY", "DOUBLE"] and ch == cor_str:
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(vis, (x1, y2), (x2, y1), (0, 255, 0), 2)
    return vis

# ==========================================
# 5. GENERATE KEY IMAGE (ใช้ rectified_output.jpg เป็นฐาน)
# ==========================================
def generate_key_image(answer_key, output_filename, total_questions=100):
    """
    สร้างภาพเฉลย Preview โดยใช้รูป rectified_output.jpg
    - วาดกากบาทเฉลย (สีเขียว)
    - วาดเส้นแดงกั้นข้อสุดท้าย (ตาม total_questions)
    """
    # 1. โหลดรูป Template
    template_path = os.path.join(settings.MEDIA_ROOT, 'rectified_output.jpg')
    
    if not os.path.exists(template_path):
        print(f"Error: Template not found at {template_path}")
        return None 

    img = cv2.imread(template_path)
    if img is None:
        print("Error: Cannot read template image.")
        return None

    mapper = GridMapper(img.shape[1], img.shape[0])
    
    # 2. เริ่มวาดบนรูป
    vis = img.copy()
    
    # ทำให้ภาพจางลงเล็กน้อยเพื่อให้เห็นรอยปากกาชัดขึ้น (Optional)
    # overlay = vis.copy()
    # cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)

    for q in range(1, 101):
        coords = mapper.get_question_coords(q)
        if not coords: continue

        ans_list = answer_key.get(str(q), [])
        
        # วาดเฉลย (กากบาทสีเขียว)
        for ch, (x1, y1, x2, y2) in coords.items():
            if ch in ans_list:
                cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(vis, (x1, y2), (x2, y1), (0, 255, 0), 2)

        # 3. วาดเส้นแดงกั้นข้อ (Limit)
        if q == total_questions and total_questions < 100:
            # ใช้พิกัดของตัวเลือก 'a' และ 'e' เพื่อหาความกว้างของข้อ
            box_a = coords['a']
            box_e = coords['e']
            
            start_x = box_a[0] - 20 # ถอยหลังไปนิดหน่อย
            end_x = box_e[2] + 20   # เลยไปนิดหน่อย
            line_y = box_a[3] + 10  # ลงมาจากขอบล่าง 10px
            
            # ขีดเส้นแดง
            cv2.line(vis, (start_x, line_y), (end_x, line_y), (0, 0, 255), 3) # สีแดง (BGR: 0,0,255)
            
            # เขียนข้อความ * END * สีแดง
            cv2.putText(vis, "* END *", (start_x + 50, line_y + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 4. บันทึกรูป
    save_dir = os.path.join(settings.MEDIA_ROOT, 'keys')
    os.makedirs(save_dir, exist_ok=True)
    
    full_save_path = os.path.join(save_dir, output_filename)
    cv2.imwrite(full_save_path, vis)
    
    # Return path relative to MEDIA_ROOT
    return f'keys/{output_filename}'

# ==========================================
# 6. Main Process Function (Called by Django View)
# ==========================================
def process_omr(image_path, answer_key):
    """
    ฟังก์ชันหลักที่ Django จะเรียกใช้เพื่อตรวจข้อสอบ
    """
    img = cv2.imread(image_path)
    if img is None: return None, "Cannot read image"
    
    points = auto_detect_paper(img)
    if points is None: return None, "Cannot detect paper corners"
    
    warped = four_point_transform(img, points)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # ใช้ Threshold แบบ Robust (C=25) สำหรับภาพมืด
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 51, 25)
    
    mapper = GridMapper(warped.shape[1], warped.shape[0])
    
    # Scan
    stu_id_list = robust_scan_student_id(thresh, mapper)
    stu_ans = robust_scan_answers(thresh, mapper)
    
    # Grade
    score, results = grade_exam_logic(stu_ans, answer_key)
    
    # Draw Result Image
    result_img = draw_result_on_image(warped, mapper, stu_ans, answer_key, results)
    
    # Save Graded Image
    # สร้างชื่อไฟล์ใหม่ เช่น graded_S__4702211.jpg
    filename = os.path.basename(image_path)
    graded_filename = f"graded_{filename}"
    
    # เซฟลง folder เดียวกับที่อัปโหลด (เช่น media/uploads/)
    save_dir = os.path.dirname(image_path)
    save_path = os.path.join(save_dir, graded_filename)
    
    cv2.imwrite(save_path, result_img)
    
    return {
        "student_id": "".join(stu_id_list),
        "score": score,
        "image_url": graded_filename # ส่งชื่อไฟล์กลับไป
    }, None