import cv2
import numpy as np
import os
from django.conf import settings
from ultralytics import YOLO

# --- Imports สำหรับ PDF Generation (ReportLab) ---
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

# ==========================================
# PART 1: GLOBAL CONFIG & YOLO
# ==========================================
YOLO_MODEL = None

def get_yolo_model():
    """ โหลดโมเดล YOLO ครั้งเดียว """
    global YOLO_MODEL
    if YOLO_MODEL is None:
        model_path = os.path.join(settings.BASE_DIR, 'grading/models/best_new.pt')
        try:
            YOLO_MODEL = YOLO(model_path)
            print(f"✅ Loaded YOLO: {model_path}")
        except Exception as e:
            print(f"❌ Error loading YOLO: {e}")
            return None
    return YOLO_MODEL

# ==========================================
# PART 2: IMAGE PROCESSING HELPERS
# ==========================================
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # TL
    rect[2] = pts[np.argmax(s)]      # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # TR
    rect[3] = pts[np.argmax(diff)]   # BL
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

def calculate_overlap(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    if boxAArea == 0: return 0
    return interArea / float(boxAArea)

def get_pixel_count(thresh_img, box):
    x1, y1, x2, y2 = box
    margin = 4
    if x2-x1 <= 2*margin or y2-y1 <= 2*margin: return 0
    roi = thresh_img[y1+margin:y2-margin, x1+margin:x2-margin]
    return cv2.countNonZero(roi)

# ==========================================
# PART 3: GRID MAPPER
# ==========================================
class GridMapper:
    def __init__(self, img_w, img_h):
        self.w = img_w; self.h = img_h
        self.box_w = 0.032
        self.box_h = 0.024    
        self.step_x = 0.0414; self.step_y = 0.0253
        
        self.c1_x, self.c1_y = 0.133, 0.303
        self.c2_x, self.c2_y = 0.4657, 0.0250
        self.c3_x, self.c3_y = 0.7950, 0.0250
        
        self.id_start_x, self.id_start_y = 0.0710, 0.0650
        self.id_step_x, self.id_step_y = 0.0310, 0.0190
        self.id_box_w, self.id_box_h = 0.028, 0.020

    def get_question_coords(self, q_num):
        if 1 <= q_num <= 26: sx, sy, r = self.c1_x, self.c1_y, q_num-1
        elif 27 <= q_num <= 63: sx, sy, r = self.c2_x, self.c2_y, q_num-27
        elif 64 <= q_num <= 100: sx, sy, r = self.c3_x, self.c3_y, q_num-64
        else: return {}
        
        base_x = int(sx * self.w)
        base_y = int((sy + (r * self.step_y)) * self.h)
        step_x = int(self.step_x * self.w)
        bw, bh = int(self.box_w * self.w), int(self.box_h * self.h)
        coords = {}
        for i, lbl in enumerate(['a','b','c','d','e']):
            cx = base_x + (i * step_x)
            cy = base_y
            coords[lbl] = [cx - bw//2, cy - bh//2, cx + bw//2, cy + bh//2]
        return coords

    def get_student_id_coords(self):
        id_grid = {}
        for col in range(10):
            id_grid[col] = {}
            for digit in range(10):
                bx = int((self.id_start_x + (col * self.id_step_x)) * self.w)
                by = int((self.id_start_y + (digit * self.id_step_y)) * self.h)
                bw, bh = int(self.id_box_w * self.w), int(self.id_box_h * self.h)
                id_grid[col][digit] = [bx - bw//2, by - bh//2, bx + bw//2, by + bh//2]
        return id_grid

# ==========================================
# PART 4: SCANNING LOGIC (HYBRID)
# ==========================================
def scan_selective(image, mapper):
    # 1. Prepare Data
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 51, 25)
    
    # 2. YOLO
    model = get_yolo_model()
    yolo_boxes = []
    if model:
        results = model.predict(image, conf=0.10, iou=0.45, imgsz=1024, verbose=False)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            yolo_boxes.append({'box': [int(x1), int(y1), int(x2), int(y2)], 'conf': float(box.conf[0])})

    # A. Student ID (Hybrid)
    student_id_list = []
    id_grid = mapper.get_student_id_coords()
    ID_PIXEL_THRESH = 150 
    
    for col in range(10):
        found_digit = "?"
        max_score = 0 
        for digit in range(10):
            grid_box = id_grid[col][digit]
            pixels = get_pixel_count(thresh, grid_box)
            
            yolo_hit = False
            for ybox in yolo_boxes:
                if calculate_overlap(grid_box, ybox['box']) > 0.15:
                    yolo_hit = True; break
            
            score = 0
            if pixels > ID_PIXEL_THRESH: score += 100
            if yolo_hit: score += 50
            
            if score > 0 and score > max_score:
                max_score = score
                found_digit = str(digit)
        student_id_list.append(found_digit)

    # B. Exam Answers (YOLO Only)
    detected_answers = {}
    for q in range(1, 101):
        coords = mapper.get_question_coords(q)
        if not coords: continue
        found_choices = []
        for ch, grid_box in coords.items():
            yolo_conf = 0.0
            is_found = False
            for ybox in yolo_boxes:
                if calculate_overlap(grid_box, ybox['box']) > 0.15:
                    is_found = True; yolo_conf = ybox['conf']; break
            if is_found:
                found_choices.append({'choice': ch, 'conf': yolo_conf})
        
        if not found_choices: detected_answers[q] = []
        else:
            found_choices.sort(key=lambda x: x['conf'], reverse=True)
            detected_answers[q] = sorted([item['choice'] for item in found_choices])

    return student_id_list, detected_answers

# ==========================================
# PART 5: GRADING & DRAWING
# ==========================================
def grade_exam_logic(student_ans, correct_key):
    score = 0; results = {}
    # เก็บ details รายข้อเพื่อส่งกลับไป views.py
    details = {} 
    
    for q in range(1, 101):
        q_str = str(q)
        stu = student_ans.get(q, [])
        cor_list = correct_key.get(q_str, [])
        cor_str = cor_list[0] if cor_list else None
        
        status = "UNKNOWN"
        if not cor_str: status = "UNKNOWN"
        elif len(stu) > 1: status = "DOUBLE"
        elif len(stu) == 0: status = "EMPTY"
        elif len(stu) == 1 and stu[0] == cor_str:
            score += 1; status = "CORRECT"
        else: status = "WRONG"
        
        results[q] = status
        details[q_str] = stu # เก็บคำตอบดิบ เช่น ['a']
        
    return score, results, details

def draw_result_on_image(img, mapper, stu_ans, key, results, student_id_list):
    vis = img.copy()
    for q in range(1, 101):
        coords = mapper.get_question_coords(q)
        stu_list = stu_ans.get(q, [])
        status = results.get(q, "UNKNOWN")
        cor_list = key.get(str(q), [])
        cor_str = cor_list[0] if cor_list else None
        
        for ch, box in coords.items():
            if ch in stu_list:
                color = (0, 255, 0) if status == "CORRECT" else (0, 255, 255) if status == "DOUBLE" else (0, 0, 255)
                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), color, 2)
            if status in ["WRONG", "EMPTY", "DOUBLE"] and ch == cor_str:
                cv2.line(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.line(vis, (box[0], box[3]), (box[2], box[1]), (0, 255, 0), 2)
                
    id_grid = mapper.get_student_id_coords()
    for col, digit_str in enumerate(student_id_list):
        if digit_str != "?":
            digit = int(digit_str)
            box = id_grid[col][digit]
            cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    return vis

def generate_key_image(answer_key, output_filename, total_questions=100):
    template_path = os.path.join(settings.MEDIA_ROOT, 'rectified_output.jpg')
    if not os.path.exists(template_path): return None 
    img = cv2.imread(template_path)
    if img is None: return None
    mapper = GridMapper(img.shape[1], img.shape[0])
    vis = img.copy()
    for q in range(1, 101):
        coords = mapper.get_question_coords(q)
        if not coords: continue
        ans_list = answer_key.get(str(q), [])
        for ch, box in coords.items():
            if ch in ans_list:
                cv2.line(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.line(vis, (box[0], box[3]), (box[2], box[1]), (0, 255, 0), 2)
        if q == total_questions:
            box_a = coords['a']; box_e = coords['e']
            cv2.line(vis, (box_a[0]-20, box_a[3]+10), (box_e[2]+20, box_a[3]+10), (0,0,255), 3)
            cv2.putText(vis, "* END *", (box_a[0]+50, box_a[3]+35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    save_dir = os.path.join(settings.MEDIA_ROOT, 'keys')
    os.makedirs(save_dir, exist_ok=True)
    full_save_path = os.path.join(save_dir, output_filename)
    cv2.imwrite(full_save_path, vis)
    return f'keys/{output_filename}'

# ==========================================
# PART 6: PDF GENERATION (ADDED BACK!)
# ==========================================
FONT_NAME = 'THSarabunNew'
FONT_PATH = os.path.join(settings.BASE_DIR, 'static', 'fonts', 'THSarabunNew.ttf')

def register_font():
    try:
        if os.path.exists(FONT_PATH):
            pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))
            return True
    except: pass
    return False

def draw_bubble(c, x, y, text):
    c.circle(x, y, 6, stroke=1, fill=0)
    c.setFont(FONT_NAME, 10)
    c.drawCentredString(x, y - 3, str(text))

def generate_exam_pdf(buffer, exam, student=None):
    register_font()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # --- HEADER ---
    c.setFont(FONT_NAME, 20)
    c.drawCentredString(width/2, height-50, "มหาวิทยาลัยเกษตรศาสตร์")
    c.setFont(FONT_NAME, 16)
    c.drawCentredString(width/2, height-70, "กระดาษคำตอบ")
    
    header_top = height - 90
    c.rect(40, header_top-90, width-80, 90)
    
    s_name = f"{student.first_name} {student.last_name}" if student else ""
    s_id = student.student_id if student else ""
    date_str = exam.exam_date.strftime("%d / %m / %Y") if exam.exam_date else ""
    time_str = exam.start_time.strftime("%H:%M") if exam.start_time else ""
    
    c.setFont(FONT_NAME, 14)
    # Row 1
    c.drawString(50, header_top-25, f"ชื่อ-นามสกุล:  {s_name}")
    c.line(110, header_top-27, 280, header_top-27)
    c.drawString(290, header_top-25, f"ชื่อวิชา:  {exam.subject_name}")
    c.drawString(450, header_top-25, f"รหัสวิชา:  {exam.subject_code}")
    # Row 2
    c.drawString(50, header_top-50, f"รหัสนิสิต:  {s_id}")
    c.line(100, header_top-52, 200, header_top-52)
    c.drawString(220, header_top-50, f"หมู่เรียน:  {exam.section}")
    c.drawString(350, header_top-50, f"วัน/เดือน/ปี:  {date_str}")
    # Row 3
    c.drawString(50, header_top-75, f"ห้องสอบ:  {exam.room}")
    c.line(90, header_top-77, 200, header_top-77)
    c.drawString(220, header_top-75, f"เวลาเริ่มสอบ:  {time_str}")
    c.drawString(350, header_top-75, f"เวลาที่ใช้สอบ:  {exam.duration_minutes} นาที")

    # --- GRID ---
    # Student ID Grid
    start_x, start_y = 60, height - 230
    c.drawString(start_x, start_y+15, "รหัสนิสิต")
    c.rect(start_x-10, start_y-145, 160, 170)
    for col in range(10):
        c.rect(start_x+(col*14), start_y, 14, 14)
        for row in range(10):
            draw_bubble(c, start_x+7+(col*14), start_y-15-(row*14), str(row))
            
    # Answer Grid
    ans_x, ans_y = 60, height - 420
    c.drawString(ans_x, ans_y+15, "ข้อสอบ")
    for col in range(4):
        base_x = ans_x + (col*110)
        c.rect(base_x-5, ans_y-380, 100, 395)
        for row in range(25):
            q_num = (col*25) + row + 1
            y_pos = ans_y - 15 - (row*15)
            c.setFont(FONT_NAME, 11)
            c.drawString(base_x, y_pos-3, str(q_num))
            for i, opt in enumerate(['a','b','c','d','e']):
                draw_bubble(c, base_x+25+(i*14), y_pos, opt)

    c.showPage()
    c.save()

# ==========================================
# PART 7: MAIN PROCESS
# ==========================================
def process_omr(image_path, answer_key):
    img = cv2.imread(image_path)
    if img is None: return None, "Cannot read image"
    
    points = auto_detect_paper(img)
    if points is None: return None, "Cannot detect paper corners"
    warped = four_point_transform(img, points)
    mapper = GridMapper(warped.shape[1], warped.shape[0])
    
    # 1. SCAN
    stu_id_list, stu_ans = scan_selective(warped, mapper)
    
    # 2. GRADE
    score, results, details = grade_exam_logic(stu_ans, answer_key)
    
    # 3. DRAW
    result_img = draw_result_on_image(warped, mapper, stu_ans, answer_key, results, stu_id_list)
    
    filename = os.path.basename(image_path)
    graded_filename = f"graded_{filename}"
    save_dir = os.path.dirname(image_path).replace('papers', 'graded') # Auto switch folder if needed
    if not os.path.exists(save_dir): save_dir = os.path.dirname(image_path)
    
    save_path = os.path.join(save_dir, graded_filename)
    cv2.imwrite(save_path, result_img)
    
    return {
        "student_id": "".join(stu_id_list),
        "score": score,
        "image_url": graded_filename,
        "details": details # ส่งข้อมูลดิบกลับไป
    }, None