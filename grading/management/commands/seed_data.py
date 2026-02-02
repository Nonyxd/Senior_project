from django.core.management.base import BaseCommand
from django.contrib.auth.models import User # <--- Import User เพื่อสร้าง Admin
from grading.models import Exam, Student, Enrollment, Subject 
import datetime

class Command(BaseCommand):
    help = 'สร้างข้อมูลจำลองและ Admin User'

    def handle(self, *args, **kwargs):
        self.stdout.write("--- เริ่มต้นการสร้างข้อมูลจำลอง ---")

        # ==========================================
        # 0. สร้าง Admin User (Superuser)
        # ==========================================
        # Username: admin / Password: 1234
        if not User.objects.filter(username='admin').exists():
            User.objects.create_superuser('admin', 'admin@example.com', '1234')
            self.stdout.write(self.style.SUCCESS(">>> สร้าง Admin เรียบร้อย (User: admin / Pass: 1234)"))
        else:
            self.stdout.write(">>> User 'admin' มีอยู่แล้ว")

        # ==========================================
        # 1. สร้าง Subject (วิชาแม่แบบ)
        # ==========================================
        sub_ct, _ = Subject.objects.get_or_create(code="CT101", name="Computational Thinking")
        sub_ai, _ = Subject.objects.get_or_create(code="AI102", name="AI Learning")
        sub_math, _ = Subject.objects.get_or_create(code="MA202", name="Mathematics for AI")

        self.stdout.write(f"1. สร้าง Subject เรียบร้อย")

        # ==========================================
        # 2. สร้าง Student และจับคู่เข้า Subject
        # ==========================================
        
        # กลุ่ม A: นิสิต ก-ซ (เรียน CT)
        thai_chars = ['ก', 'ข', 'ค', 'ง', 'จ', 'ฉ', 'ช', 'ซ'] 
        eng_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        
        for i, char in enumerate(thai_chars):
            sid = f"660100{i+1}"
            stu, _ = Student.objects.get_or_create(
                student_id=sid,
                defaults={'first_name': f"นิสิต {char}", 'last_name': f"นามสกุล {eng_chars[i]}"}
            )
            sub_ct.students.add(stu)

        # กลุ่ม B: Alice-Eve (เรียน AI และ Math)
        eng_names = ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
        for i, name in enumerate(eng_names):
            sid = f"660200{i+1}"
            stu, _ = Student.objects.get_or_create(
                student_id=sid,
                defaults={'first_name': name, 'last_name': f"Smith {i+1}"}
            )
            sub_ai.students.add(stu)
            sub_math.students.add(stu)

        self.stdout.write("2. สร้าง Student และจับคู่เข้า Subject เรียบร้อย")

        # ==========================================
        # 3. สร้าง Exam ตัวอย่าง
        # ==========================================
        
        # Exam 1: CT101
        exam1, created = Exam.objects.get_or_create(
            name="Midterm Comp Thinking",
            defaults={
                'course_code': 'CT101',
                'answer_key': {}, 
                'total_questions': 50,
                'exam_date': datetime.date(2026, 3, 15),
                'start_time': datetime.time(9, 0),
                'end_time': datetime.time(12, 0),
                'room': 'LAB-1'
            }
        )
        if created:
            for stu in sub_ct.students.all():
                Enrollment.objects.get_or_create(exam=exam1, student=stu, section="Sec 1")

        # Exam 2: AI102
        exam2, created = Exam.objects.get_or_create(
            name="Quiz 1 AI Learning",
            defaults={
                'course_code': 'AI102',
                'answer_key': {}, 
                'total_questions': 20,
                'exam_date': datetime.date(2026, 3, 20),
                'start_time': datetime.time(13, 0),
                'end_time': datetime.time(14, 0),
                'room': 'Online'
            }
        )
        if created:
            for stu in sub_ai.students.all():
                Enrollment.objects.get_or_create(exam=exam2, student=stu, section="Sec 2")

        self.stdout.write(self.style.SUCCESS('--- เสร็จสิ้น! Admin และข้อมูลพร้อมใช้งานแล้ว ---'))