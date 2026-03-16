import cv2
import numpy as np

class GridMapper:
    def __init__(self, img_w, img_h):
        self.w = img_w; self.h = img_h
        
        # ===============================================
        # 🟢 PART 1: ฝั่งรหัสนิสิต (ของคุณเหมือนเดิมเป๊ะ)
        # ===============================================
        self.id_cols_x = [
            0.0820, 0.1135, 0.1420, 0.1735, 0.2020, 
            0.2335, 0.2620, 0.2935, 0.3220, 0.3535
        ]
        self.id_rows_y = [
            0.0740, 0.0930, 0.1120, 0.1310, 0.1510, 
            0.1690, 0.1870, 0.2050, 0.2230, 0.2410
        ]
        self.id_box_w, self.id_box_h = 0.030, 0.020

        # ===============================================
        # 🟡 PART 2: ฝั่งข้อสอบ 100 ข้อ (แบบแยกรายข้อ/รายตัวเลือก)
        # ===============================================
        self.box_w, self.box_h = 0.040, 0.023
        
        # ↔️ แกน X ของตัวเลือก (a, b, c, d, e)
        self.ans_col1_x = [0.1420, 0.1840, 0.2240, 0.2640, 0.3030] # ข้อ 1-26
        self.ans_col2_x = [0.4660, 0.5060, 0.5470, 0.5870, 0.6260] # ข้อ 27-63
        self.ans_col3_x = [0.7880, 0.8280, 0.8690, 0.9090, 0.9480] # ข้อ 64-100

        # ↕️ แกน Y ของข้อ 1 ถึง 26 (คอลัมน์ 1)
        # ↕️ แกน Y ของข้อ 1 ถึง 26 (คอลัมน์ 1)
        self.ans_col1_y = [
            0.3070, 0.3320, 0.3570, 0.3820, 0.4070, 0.4320, 0.4570, 0.4810, 0.5060, 0.5310, 
            0.5550, 0.5800, 0.6050, 0.6300, 0.6540, 0.6790, 0.7030, 0.7270, 0.7520, 0.7770, 
            0.8010, 0.8250, 0.8500, 0.8750, 0.9000, 0.9250
        ]

        # ↕️ แกน Y ของข้อ 27 ถึง 63 (คอลัมน์ 2)
        # ↕️ แกน Y ของข้อ 27 ถึง 63 (รวม 37 ข้อ)
        # เกลี่ยระยะให้ 3 ตัวแรกก้าว 0.0250 แล้วค่อยๆ ปรับให้ไปจบที่ 0.9260 พอดีเป๊ะ
        self.ans_col2_y = [
            0.0370, 0.0620, 0.0870, 0.1117, 0.1364, 0.1610, 0.1857, 0.2104, 0.2351, 0.2597, 
            0.2844, 0.3091, 0.3338, 0.3584, 0.3831, 0.4078, 0.4325, 0.4571, 0.4818, 0.5065, 
            0.5312, 0.5558, 0.5805, 0.6052, 0.6299, 0.6545, 0.6792, 0.7039, 0.7286, 0.7532, 
            0.7779, 0.8026, 0.8273, 0.8519, 0.8766, 0.9013, 0.9260
        ]

        # ↕️ แกน Y ของข้อ 64 ถึง 100 (คอลัมน์ 3)
        self.ans_col3_y = [
            0.0370, 0.0620, 0.0870, 0.1117, 0.1364, 0.1610, 0.1857, 0.2104, 0.2351, 0.2597, 
            0.2844, 0.3091, 0.3338, 0.3584, 0.3831, 0.4078, 0.4325, 0.4571, 0.4818, 0.5065, 
            0.5312, 0.5558, 0.5805, 0.6052, 0.6299, 0.6545, 0.6792, 0.7039, 0.7286, 0.7532, 
            0.7779, 0.8026, 0.8273, 0.8519, 0.8766, 0.9013, 0.9260
        ]

    def get_question_coords(self, q_num):
        # เลือกว่าจะใช้แกน X/Y ของชุดไหน ตามเลขข้อ
        if 1 <= q_num <= 26:
            x_list = self.ans_col1_x
            y_val = self.ans_col1_y[q_num - 1]
        elif 27 <= q_num <= 63:
            x_list = self.ans_col2_x
            y_val = self.ans_col2_y[q_num - 27]
        elif 64 <= q_num <= 100:
            x_list = self.ans_col3_x
            y_val = self.ans_col3_y[q_num - 64]
        else:
            return {}
        
        bw, bh = int(self.box_w * self.w), int(self.box_h * self.h)
        by = int(y_val * self.h)
        coords = {}
        
        # วาดทีละตัวเลือก (a, b, c, d, e)
        for i, lbl in enumerate(['a', 'b', 'c', 'd', 'e']):
            bx = int(x_list[i] * self.w)
            coords[lbl] = [bx - bw//2, by - bh//2, bx + bw//2, by + bh//2]
            
        return coords

    def get_student_id_coords(self):
        id_grid = {}
        for col in range(10):
            id_grid[col] = {}
            for digit in range(10):
                bx = int(self.id_cols_x[col] * self.w)
                by = int(self.id_rows_y[digit] * self.h)
                bw, bh = int(self.id_box_w * self.w), int(self.id_box_h * self.h)
                id_grid[col][digit] = [bx - bw//2, by - bh//2, bx + bw//2, by + bh//2]
        return id_grid

# ---------------------------------------------
# สคริปต์จำลองวาดกรอบ
# ---------------------------------------------
def test_draw_grid():
    image_path = 'static/rectified_output.jpg' 
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"❌ หาไฟล์ {image_path} ไม่เจอ กรุณาเช็ค Path")
        return

    mapper = GridMapper(img.shape[1], img.shape[0])
    vis = img.copy()

    id_grid = mapper.get_student_id_coords()
    for col in range(10):
        for digit in range(10):
            box = id_grid[col][digit]
            cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    for q in range(1, 101):
        coords = mapper.get_question_coords(q)
        for ch, box in coords.items():
            cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)

    output_path = 'test_bounding_box.jpg'
    cv2.imwrite(output_path, vis)
    print(f"✅ วาดกรอบสำเร็จ! เปิดดูไฟล์ '{output_path}'")

if __name__ == '__main__':
    test_draw_grid()