import cv2
import numpy as np
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# 1. Image Processing Helpers
# ==========================================
def order_points(pts):
    """เรียงลำดับจุด 4 มุม: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def four_point_transform(image, pts):
    """ทำ perspective transform"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # คำนวณความกว้างและสูงใหม่
    maxW = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    maxH = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    
    # จุดปลายทาง
    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype="float32")
    
    # ทำ perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))
    
    return warped

def auto_detect_paper(img):
    """ตรวจจับมุมกระดาษอัตโนมัติ"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    
    # หา contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    # หา contour ที่เป็นสี่เหลี่ยม
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > (img.shape[0] * img.shape[1] * 0.1):
            return approx.reshape(4, 2)
    
    return None

def calculate_overlap(boxA, boxB):
    """คำนวณพื้นที่ซ้อนทับ (IoA - Intersection over Area)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    
    if boxAArea == 0:
        return 0
    
    return interArea / float(boxAArea)

# ==========================================
# 2. GridMapper
# ==========================================
class GridMapper:
    """แมพตำแหน่งช่องคำตอบบนกระดาษ OMR"""
    
    def __init__(self, img_w, img_h):
        self.w = img_w
        self.h = img_h
        
        # ขนาดของแต่ละช่อง (เทียบกับขนาดรูป)
        self.box_w = 0.032
        self.box_h = 0.024
        
        # ระยะห่างระหว่างช่อง
        self.step_x = 0.0414
        self.step_y = 0.0253
        
        # จุดเริ่มต้นของแต่ละคอลัมน์
        self.c1_x, self.c1_y = 0.133, 0.303   # คอลัมน์ 1 (ข้อ 1-26)
        self.c2_x, self.c2_y = 0.4657, 0.0250 # คอลัมน์ 2 (ข้อ 27-63)
        self.c3_x, self.c3_y = 0.7950, 0.0250 # คอลัมน์ 3 (ข้อ 64-100)
    
    def get_question_coords(self, q_num):
        """
        ดึงพิกัดช่องคำตอบสำหรับข้อที่กำหนด
        
        Args:
            q_num: หมายเลขข้อ (1-100)
            
        Returns:
            dict: {'a': [x1,y1,x2,y2], 'b': [...], ...} หรือ {} ถ้าไม่มี
        """
        # กำหนดคอลัมน์และแถวตามหมายเลขข้อ
        if 1 <= q_num <= 26:
            sx, sy, r = self.c1_x, self.c1_y, q_num - 1
        elif 27 <= q_num <= 63:
            sx, sy, r = self.c2_x, self.c2_y, q_num - 27
        elif 64 <= q_num <= 100:
            sx, sy, r = self.c3_x, self.c3_y, q_num - 64
        else:
            return {}
        
        coords = {}
        base_x = int(sx * self.w)
        base_y = int((sy + (r * self.step_y)) * self.h)
        step_x = int(self.step_x * self.w)
        bw, bh = int(self.box_w * self.w), int(self.box_h * self.h)
        
        # สร้างพิกัดสำหรับแต่ละตัวเลือก (a, b, c, d, e)
        for i, lbl in enumerate(['a', 'b', 'c', 'd', 'e']):
            cx = base_x + (i * step_x)
            cy = base_y
            coords[lbl] = [cx - bw//2, cy - bh//2, cx + bw//2, cy + bh//2]
        
        return coords

# ==========================================
# 3. OMR Model Tester
# ==========================================
class OMRModelTester:
    """คลาสสำหรับทดสอบโมเดล YOLO กับกระดาษ OMR"""
    
    def __init__(self, model_path):
        """
        Args:
            model_path: path ไปยังไฟล์โมเดล .pt
        """
        print(f"🔄 Loading model from: {model_path}")
        self.model = YOLO(model_path)
        print(f"✅ Model loaded successfully!")
        
        # Class names
        self.class_names = ['AX', 'AY', 'BX', 'BY', 'CX', 'CY', 'DX', 'DY', 'EX', 'EY', 'NisitNumX']
        
    def test_single_image(self, image_path, conf_threshold=0.25, iou_threshold=0.6, 
                         auto_detect=True, visualize=True, save_output=True):
        """
        ทดสอบโมเดลกับรูปภาพเดียว
        
        Args:
            image_path: path ของรูปภาพ
            conf_threshold: confidence threshold สำหรับการ detect
            iou_threshold: IoU threshold สำหรับ NMS
            auto_detect: ใช้ auto detect มุมกระดาษหรือไม่
            visualize: แสดงผลด้วย matplotlib หรือไม่
            save_output: บันทึกผลลัพธ์หรือไม่
            
        Returns:
            dict: ผลลัพธ์การตรวจจับ
        """
        print(f"\n{'='*60}")
        print(f"🔍 TESTING IMAGE: {image_path}")
        print(f"{'='*60}")
        
        # อ่านรูปภาพ
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Error: Cannot read image from {image_path}")
            return None
        
        original_img = img.copy()
        print(f"✅ Image loaded: {img.shape[1]}x{img.shape[0]} pixels")
        
        # ตรวจจับและแก้มุมกระดาษ (ถ้าเปิดใช้งาน)
        warped_img = None
        paper_corners = None
        
        if auto_detect:
            print(f"\n📐 Detecting paper corners...")
            paper_corners = auto_detect_paper(img)
            
            if paper_corners is not None:
                print(f"✅ Paper corners detected!")
                warped_img = four_point_transform(img, paper_corners)
                img = warped_img
                print(f"✅ Perspective corrected: {img.shape[1]}x{img.shape[0]} pixels")
            else:
                print(f"⚠️ Warning: Could not detect paper corners, using original image")
        
        # Run YOLO detection
        print(f"\n🤖 Running YOLO detection...")
        print(f"   Confidence threshold: {conf_threshold}")
        print(f"   IoU threshold: {iou_threshold}")
        
        results = self.model.predict(
            img,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )[0]
        
        # วิเคราะห์ผลลัพธ์
        detections = []
        detection_summary = {name: 0 for name in self.class_names}
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            class_name = self.class_names[cls]
            
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf,
                'class': class_name,
                'class_id': cls
            })
            
            detection_summary[class_name] += 1
        
        print(f"\n📊 Detection Summary:")
        print(f"   Total detections: {len(detections)}")
        for class_name, count in detection_summary.items():
            if count > 0:
                print(f"   {class_name}: {count}")
        
        # สร้างกระดาษตอบและแมพคำตอบ
        print(f"\n📝 Mapping answers to questions...")
        grid_mapper = GridMapper(img.shape[1], img.shape[0])
        answer_sheet = {}
        
        for q_num in range(1, 101):
            expected_coords = grid_mapper.get_question_coords(q_num)
            if not expected_coords:
                continue
            
            # หาคำตอบที่ถูกทำเครื่องหมาย
            marked_choices = []
            
            for choice, bbox in expected_coords.items():
                # เช็คว่ามี detection ที่ซ้อนทับกับช่องนี้หรือไม่
                for det in detections:
                    overlap = calculate_overlap(bbox, det['bbox'])
                    if overlap > 0.3:  # ถ้าซ้อนทับมากกว่า 30%
                        marked_choices.append({
                            'choice': choice,
                            'class': det['class'],
                            'confidence': det['confidence'],
                            'overlap': overlap
                        })
            
            if marked_choices:
                answer_sheet[q_num] = marked_choices
        
        print(f"✅ Found answers for {len(answer_sheet)} questions")
        
        # สร้างภาพสำหรับแสดงผล
        if visualize or save_output:
            vis_img = self._create_visualization(
                original_img, img, warped_img, paper_corners, 
                detections, answer_sheet, grid_mapper
            )
            
            if visualize:
                plt.figure(figsize=(20, 12))
                plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title('OMR Detection Results', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.show()
            
            if save_output:
                output_dir = Path(image_path).parent / 'test_results'
                output_dir.mkdir(exist_ok=True)
                
                output_path = output_dir / f"{Path(image_path).stem}_result.jpg"
                cv2.imwrite(str(output_path), vis_img)
                print(f"\n💾 Saved result to: {output_path}")
        
        # สร้างรายงานสรุป
        report = self._generate_report(detections, answer_sheet, detection_summary)
        
        return {
            'detections': detections,
            'answer_sheet': answer_sheet,
            'summary': detection_summary,
            'report': report,
            'image_shape': img.shape,
            'paper_detected': paper_corners is not None
        }
    
    def _create_visualization(self, original_img, processed_img, warped_img, 
                             paper_corners, detections, answer_sheet, grid_mapper):
        """สร้างภาพแสดงผลลัพธ์"""
        
        # สร้าง canvas ใหญ่
        h, w = processed_img.shape[:2]
        
        # วาดกรอบ detection
        vis_img = processed_img.copy()
        
        # สร้าง color map สำหรับแต่ละ class
        colors = {
            'AX': (255, 0, 0),    # Blue
            'AY': (255, 100, 100),
            'BX': (0, 255, 0),    # Green
            'BY': (100, 255, 100),
            'CX': (0, 0, 255),    # Red
            'CY': (100, 100, 255),
            'DX': (255, 255, 0),  # Cyan
            'DY': (255, 255, 100),
            'EX': (255, 0, 255),  # Magenta
            'EY': (255, 100, 255),
            'NisitNumX': (0, 255, 255)  # Yellow
        }
        
        # วาด detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            color = colors.get(det['class'], (128, 128, 128))
            
            # วาดกรอบ
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # วาดป้ายชื่อ
            label = f"{det['class']} {det['confidence']:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # วาดกริดคำตอบ (แสดง 10 ข้อแรกเป็นตัวอย่าง)
        for q_num in range(1, 11):
            coords = grid_mapper.get_question_coords(q_num)
            for choice, bbox in coords.items():
                x1, y1, x2, y2 = bbox
                # วาดกรอบบางๆ
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (200, 200, 200), 1)
                
                # ถ้ามีคำตอบ ทำ highlight
                if q_num in answer_sheet:
                    for ans in answer_sheet[q_num]:
                        if ans['choice'] == choice:
                            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # วาดมุมกระดาษ (ถ้ามี)
        if paper_corners is not None and warped_img is None:
            cv2.polylines(original_img, [paper_corners.astype(np.int32)], True, (0, 255, 0), 3)
        
        return vis_img
    
    def _generate_report(self, detections, answer_sheet, detection_summary):
        """สร้างรายงานสรุปผลลัพธ์"""
        
        report = "\n" + "="*60 + "\n"
        report += "📋 DETECTION REPORT\n"
        report += "="*60 + "\n\n"
        
        report += "📊 Detection Summary:\n"
        report += f"   Total detections: {len(detections)}\n"
        for class_name, count in detection_summary.items():
            if count > 0:
                report += f"   {class_name}: {count}\n"
        
        report += f"\n📝 Answer Sheet:\n"
        report += f"   Total questions answered: {len(answer_sheet)}\n\n"
        
        # แสดงคำตอบ 20 ข้อแรก
        report += "   First 20 answers:\n"
        for q_num in sorted(answer_sheet.keys())[:20]:
            answers = answer_sheet[q_num]
            if len(answers) == 1:
                ans = answers[0]
                report += f"   Q{q_num:3d}: {ans['choice'].upper()} ({ans['class']}, conf={ans['confidence']:.2f})\n"
            else:
                report += f"   Q{q_num:3d}: MULTIPLE ({len(answers)} marks)\n"
                for ans in answers:
                    report += f"         - {ans['choice'].upper()} ({ans['class']}, conf={ans['confidence']:.2f})\n"
        
        if len(answer_sheet) > 20:
            report += f"   ... and {len(answer_sheet) - 20} more questions\n"
        
        report += "\n" + "="*60 + "\n"
        
        print(report)
        return report


# ==========================================
# 4. Main Testing Function
# ==========================================
def test_omr_model(model_path, image_path, conf_threshold=0.25, 
                   auto_detect=True, visualize=True, save_output=True):
    """
    ฟังก์ชันหลักสำหรับทดสอบโมเดล
    
    Args:
        model_path: path ของโมเดล .pt
        image_path: path ของรูปภาพที่จะทดสอบ
        conf_threshold: confidence threshold
        auto_detect: ใช้ auto detect มุมกระดาษ
        visualize: แสดงผลด้วย matplotlib
        save_output: บันทึกผลลัพธ์
    """
    
    # สร้าง tester
    tester = OMRModelTester(model_path)
    
    # ทดสอบ
    results = tester.test_single_image(
        image_path=image_path,
        conf_threshold=conf_threshold,
        auto_detect=auto_detect,
        visualize=visualize,
        save_output=save_output
    )
    
    return results


# ==========================================
# 5. Example Usage
# ==========================================
if __name__ == "__main__":
    # ระบุ path ของโมเดลและรูปภาพ
    MODEL_PATH = r"C:\senior_pro\omr_site\grading\models\best_new.pt"
    IMAGE_PATH = r"C:\senior_pro\omr_site\media\uploads\15703.jpg"
    
    # ทดสอบ
    results = test_omr_model(
        model_path=MODEL_PATH,
        image_path=IMAGE_PATH,
        conf_threshold=0.25,     # ปรับได้ตามต้องการ (0.1-0.5)
        auto_detect=True,        # เปิดใช้ auto detect มุมกระดาษ
        visualize=True,          # แสดงผลด้วย matplotlib
        save_output=True         # บันทึกผลลัพธ์
    )
    
    # แสดงผลลัพธ์เพิ่มเติม
    if results:
        print(f"\n✅ Testing completed!")
        print(f"   Total detections: {len(results['detections'])}")
        print(f"   Questions answered: {len(results['answer_sheet'])}")
        print(f"   Paper auto-detected: {'Yes' if results['paper_detected'] else 'No'}")