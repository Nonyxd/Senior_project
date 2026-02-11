from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views

# Import views จาก app grading
from grading import views

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # --- Authentication (Login/Logout) ---
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('logout/confirm/', views.logout_confirm_view, name='logout_confirm'),

    # --- Dashboard & Create ---
    path('', views.index, name='index'),
    path('create/', views.create_exam, name='create_exam'),
    path('save_exam/', views.save_exam_confirm, name='save_exam_confirm'),
    
    # --- API Helper ---
    # 1. อ่าน Excel (Auto Fill)
    path('api/parse_excel/', views.api_parse_excel, name='api_parse_excel'),
    # 2. ลบวิชา (Delete Subject) **[เพิ่มใหม่ตรงนี้ครับ]**
    path('api/delete_subject/', views.delete_subject_api, name='delete_subject_api'),

    # --- Grading & Students ---
    path('grade/<int:exam_id>/', views.grade_exam_view, name='grade_exam'),
    path('exam/<int:exam_id>/upload_students/', views.upload_students, name='upload_students'),
    
    # --- Edit Result (Individual) - แก้ไข/ลบคะแนนรายบุคคล ---
    path('result/<int:result_id>/edit/', views.edit_result, name='edit_result'),
    path('api/update_result/<int:result_id>/', views.api_update_result, name='api_update_result'),
    path('delete_result/<int:result_id>/', views.delete_result, name='delete_result'),

    # --- Edit Exam (Subject) - แก้ไขข้อมูลวิชา ---
    path('edit_exam/<int:exam_id>/', views.edit_exam, name='edit_exam'),
    path('edit_preview/<int:exam_id>/', views.edit_exam_preview, name='edit_exam_preview'),
    path('save_edit_confirm/<int:exam_id>/', views.save_edit_confirm, name='save_edit_confirm'),
    path('delete/<int:exam_id>/', views.delete_exam, name='delete_exam'),

    # --- PDF Printing ---
    # 1. แบบมีพื้นหลัง (สำหรับ Print)
    path('print_sheets/<int:exam_id>/', views.generate_answer_sheet, name='print_sheets'),
    
    # 2. แบบดาวน์โหลดทั่วไป (Master Sheet)
    path('exam/<int:exam_id>/download/', views.download_exam_sheet, name='download_exam_sheet'),
    
    # 3. แบบดาวน์โหลดรายคน (Student Sheet)
    path('exam/<int:exam_id>/download/<str:student_id>/', views.download_exam_sheet, name='download_student_sheet'),
]

# --- ส่วนสำคัญ: รองรับการโชว์รูปภาพในโหมด Debug (แก้รูปแตก) ---
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)