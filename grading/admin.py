from django.contrib import admin
from .models import Exam, StudentResult

# --- 1. ปรับแต่งตาราง Exam ---
class ExamAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'created_at')
    search_fields = ('name',)
    
    # OPTIMIZATION:
    # เวลาเข้าหน้าลิสต์รายการ ไม่ต้องดึง JSON เฉลย (answer_key) ออกมา
    # ช่วยให้โหลดหน้า Admin เร็วขึ้นมากถ้ามีข้อสอบเยอะๆ
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.defer('answer_key') 

# --- 2. ปรับแต่งตาราง StudentResult ---
class StudentResultAdmin(admin.ModelAdmin):
    list_display = ('student_id', 'exam', 'score', 'timestamp')
    list_filter = ('exam',) 
    
    # OPTIMIZATION:
    # ใช้ select_related เพื่อดึงชื่อ Exam มาพร้อมกันเลยในรอบเดียว
    # (แก้ปัญหา N+1 Query: ถ้าไม่ใส่ บรรทัดนี้จะยิง SQL รัวๆ ตามจำนวนผลสอบ)
    list_select_related = ('exam',)

# ลงทะเบียน
admin.site.register(Exam, ExamAdmin)
admin.site.register(StudentResult, StudentResultAdmin)