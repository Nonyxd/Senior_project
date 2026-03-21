import os
from django.db import models
from django.utils import timezone
from django.db.models.signals import post_delete, pre_save
from django.dispatch import receiver
from django.contrib.auth.models import User # 🌟 1. ดึงโมเดล User มาใช้

# ==========================================
# 1. ตารางรายชื่อนิสิต (Student)
# ==========================================
class Student(models.Model):
    student_id = models.CharField(max_length=20, unique=True, verbose_name="รหัสนิสิต")
    first_name = models.CharField(max_length=100, verbose_name="ชื่อ")
    last_name = models.CharField(max_length=100, verbose_name="นามสกุล")

    def __str__(self):
        return f"{self.student_id} {self.first_name} {self.last_name}"

# ==========================================
# 2. ตารางการสอบ (Exam)
# ==========================================
class Exam(models.Model):
    # 🌟 2. ผูกข้อสอบกับ User (ใส่ null=True ไว้ชั่วคราว เพื่อไม่ให้ฐานข้อมูลเก่าพัง!)
    user = models.ForeignKey(User, on_delete=models.CASCADE) 

    subject_code = models.CharField(max_length=20, verbose_name="รหัสวิชา")
    subject_name = models.CharField(max_length=100, verbose_name="ชื่อวิชา")
    section = models.CharField(max_length=10, default="1", verbose_name="หมู่เรียน")
    
    exam_date = models.DateField(default=timezone.now, verbose_name="วันที่สอบ")
    start_time = models.TimeField(default=timezone.now, verbose_name="เวลาเริ่มสอบ")
    duration_minutes = models.IntegerField(default=120, verbose_name="เวลาที่ใช้สอบ (นาที)")
    room = models.CharField(max_length=50, blank=True, verbose_name="ห้องสอบ")
    
    total_questions = models.IntegerField(default=100, verbose_name="จำนวนข้อ")
    answer_key = models.JSONField(default=dict, verbose_name="เฉลย (JSON)")
    key_image = models.ImageField(upload_to='uploads/keys/', blank=True, null=True)
    
    # 🔥 [เพิ่ม] เก็บไฟล์ Excel รายชื่อนิสิต เพื่อให้ลบได้ภายหลัง
    roster_file = models.FileField(upload_to='uploads/rosters/', blank=True, null=True, verbose_name="ไฟล์รายชื่อนิสิต")

    enrolled_students = models.ManyToManyField(Student, blank=True, related_name='exams')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.subject_code} {self.subject_name} (Sec {self.section})"

# ==========================================
# 3. ตารางผลการตรวจ (StudentResult)
# ==========================================
class StudentResult(models.Model):
    # 🚩 1. เปลี่ยน STATUS_CHOICES เป็นระบบสีแทนของเดิม
    STATUS_CHOICES = [
        ('BLUE', 'ตรวจผ่าน (ไม่มีข้อผิดพลาด)'), 
        ('RED', 'พบข้อผิดพลาด (ต้องตรวจสอบ)'),
        ('GREEN', 'ตรวจสอบด้วยคนแล้ว (Verified)'),
    ]

    exam = models.ForeignKey(Exam, on_delete=models.CASCADE, related_name='results')
    student = models.ForeignKey(Student, on_delete=models.SET_NULL, null=True, blank=True)
    student_id_ocr = models.CharField(max_length=50, verbose_name="รหัสที่อ่านได้")
    score = models.IntegerField(default=0, verbose_name="คะแนน")
    
    original_image = models.ImageField(upload_to='uploads/papers/')
    graded_image = models.ImageField(upload_to='uploads/graded/', blank=True, null=True)
    debug_image = models.ImageField(upload_to='uploads/graded/', blank=True, null=True)

    # 🚩 2. อัปเดตฟิลด์ status ให้ใช้ Default เป็น RED (เดี๋ยว views.py จะเปลี่ยนเป็น BLUE ให้เองถ้าไม่เจอ Error)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='RED')
    
    results_data = models.JSONField(default=dict, blank=True)
    last_updated = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.student_id_ocr} - {self.exam.subject_code}"
    
    
# ==========================================
# 🔥 SIGNALS: จัดการลบไฟล์ออกจากเครื่องอัตโนมัติ
# (ใส่ไว้ล่างสุดของไฟล์ models.py)
# ==========================================

# 1. เมื่อลบ Exam -> ให้ลบรูปเฉลย และ ไฟล์ Excel
@receiver(post_delete, sender=Exam)
def auto_delete_file_on_exam_delete(sender, instance, **kwargs):
    # ลบรูปเฉลย
    if instance.key_image:
        if os.path.isfile(instance.key_image.path):
            os.remove(instance.key_image.path)
    # ลบไฟล์ Excel
    if instance.roster_file:
        if os.path.isfile(instance.roster_file.path):
            os.remove(instance.roster_file.path)

# 2. เมื่อลบ StudentResult -> ให้ลบรูปกระดาษคำตอบ (Original, Graded, Debug)
@receiver(post_delete, sender=StudentResult)
def auto_delete_file_on_result_delete(sender, instance, **kwargs):
    for field in [instance.original_image, instance.graded_image, instance.debug_image]:
        if field and hasattr(field, 'path') and os.path.isfile(field.path):
            try:
                os.remove(field.path)
            except Exception as e:
                print(f"Error deleting file: {e}")