from django.db import models
from django.utils import timezone

# ==========================================
# 1. ตารางรายชื่อนิสิต (Student)
# ส่วนนี้ข้อมูลจะมาจากการ Upload Excel
# ==========================================
class Student(models.Model):
    student_id = models.CharField(max_length=20, unique=True, verbose_name="รหัสนิสิต")
    first_name = models.CharField(max_length=100, verbose_name="ชื่อ")
    last_name = models.CharField(max_length=100, verbose_name="นามสกุล")
    
    # ถ้าอยากเก็บภาควิชาหรือคณะเพิ่ม ใส่ตรงนี้ได้
    # department = models.CharField(max_length=100, blank=True)

    def __str__(self):
        return f"{self.student_id} {self.first_name} {self.last_name}"


# ==========================================
# 2. ตารางการสอบ (Exam)
# ส่วนนี้อาจารย์กรอกเองหน้าเว็บ หรือเลือก Dropdown
# ==========================================
class Exam(models.Model):
    # ข้อมูลวิชา (ใช้ทำ Search Dropdown)
    subject_code = models.CharField(max_length=20, verbose_name="รหัสวิชา")
    subject_name = models.CharField(max_length=100, verbose_name="ชื่อวิชา")
    section = models.CharField(max_length=10, default="1", verbose_name="หมู่เรียน")
    
    # ข้อมูลเวลาและสถานที่ (สำหรับ PDF Header)
    exam_date = models.DateField(default=timezone.now, verbose_name="วันที่สอบ")
    start_time = models.TimeField(default=timezone.now, verbose_name="เวลาเริ่มสอบ")
    duration_minutes = models.IntegerField(default=120, verbose_name="เวลาที่ใช้สอบ (นาที)") # ใช้แทน End Time
    room = models.CharField(max_length=50, blank=True, verbose_name="ห้องสอบ")
    
    # ข้อมูลข้อสอบ
    total_questions = models.IntegerField(default=100, verbose_name="จำนวนข้อ")
    answer_key = models.JSONField(default=dict, verbose_name="เฉลย (JSON)") # เก็บ {'1': 'A', '2': 'C'}
    key_image = models.ImageField(upload_to='uploads/keys/', blank=True, null=True) # รูปเฉลย (ถ้ามี)
    
    # เชื่อมโยงกับนิสิต (ใครมีสิทธิ์สอบวิชานี้บ้าง) -> มาจาก Excel
    enrolled_students = models.ManyToManyField(Student, blank=True, related_name='exams')

    # [สำคัญ] สำหรับ Soft Delete (ลบออกจากหน้าเว็บ แต่เก็บไว้ใน DB)
    is_active = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.subject_code} {self.subject_name} (Sec {self.section})"


# ==========================================
# 3. ตารางผลการตรวจ (StudentResult)
# ส่วนนี้มาจากการ Scan กระดาษ + การตรวจ
# ==========================================
class StudentResult(models.Model):
    # สถานะการตรวจ (Requirement: ตรวจปกติ, ตรวจเครื่องเดียว, กำลังแก้, แก้เสร็จ)
    STATUS_CHOICES = [
        ('OCR', 'ตรวจจากเครื่อง (OCR Scanned)'),
        ('EDITING', 'กำลังแก้ไข (Editing)'), 
        ('FINISHED', 'แก้ไขเสร็จสิ้น (Manual Finished)'),
    ]

    exam = models.ForeignKey(Exam, on_delete=models.CASCADE, related_name='results')
    
    # เชื่อมกับ Student เพื่อดึงชื่อมาโชว์ (ถ้าหาเจอ)
    student = models.ForeignKey(Student, on_delete=models.SET_NULL, null=True, blank=True)
    
    # รหัสที่อ่านได้จริงจากกระดาษ (อาจจะผิด หรือไม่มีในระบบ)
    student_id_ocr = models.CharField(max_length=50, verbose_name="รหัสที่อ่านได้")
    
    score = models.IntegerField(default=0, verbose_name="คะแนน")
    
    # ไฟล์รูปภาพ
    original_image = models.ImageField(upload_to='uploads/papers/')
    graded_image = models.ImageField(upload_to='uploads/graded/', blank=True, null=True)

    # สถานะ
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='OCR')
    
    # เก็บข้อมูลดิบรายข้อ (JSON) เพื่อสร้าง "กล่องสีส้ม" ในหน้าเว็บ
    # ตัวอย่าง: {'1': {'status': 'CORRECT', 'mark': 'A'}, '2': {'status': 'EMPTY', 'mark': ''}}
    results_data = models.JSONField(default=dict, blank=True)
    
    # เวลาที่แก้ไขล่าสุด
    last_updated = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.student_id_ocr} - {self.exam.subject_code}"