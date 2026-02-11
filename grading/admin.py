from django.contrib import admin
from .models import Exam, StudentResult, Student

# --- 1. ปรับแต่งตาราง Student (รายชื่อนิสิต) ---
@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ('student_id', 'first_name', 'last_name')
    search_fields = ('student_id', 'first_name', 'last_name')
    ordering = ('student_id',)

# --- 2. ปรับแต่งตาราง Exam ---
@admin.register(Exam)
class ExamAdmin(admin.ModelAdmin):
    # ปรับ field ให้ตรงกับ Model ใหม่
    list_display = ('id', 'subject_code', 'subject_name', 'section', 'exam_date', 'total_questions', 'created_at')
    search_fields = ('subject_code', 'subject_name')
    list_filter = ('exam_date',)
    ordering = ('-created_at',)
    
    # OPTIMIZATION: ไม่ดึง JSON เฉลยมาแสดงในหน้า List เพื่อความเร็ว
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.defer('answer_key') 

# --- 3. ปรับแต่งตาราง StudentResult ---
@admin.register(StudentResult)
class StudentResultAdmin(admin.ModelAdmin):
    # ปรับ field ให้ตรงกับ Model ใหม่ (timestamp -> last_updated, student_id -> student_id_ocr)
    list_display = ('id', 'exam', 'student_id_ocr', 'score', 'status', 'last_updated')
    list_filter = ('status', 'exam') 
    search_fields = ('student_id_ocr', 'exam__subject_name')
    
    # OPTIMIZATION: ดึงข้อมูล Exam และ Student มาด้วยใน Query เดียว (แก้ N+1)
    list_select_related = ('exam', 'student')