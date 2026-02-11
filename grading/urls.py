from django.contrib import admin
from django.urls import path
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
    
    # --- API (ส่วนสำคัญที่แก้ Error ของคุณ) ---
    path('api/parse_excel/', views.api_parse_excel, name='api_parse_excel'),

    # --- Grading & Students ---
    path('grade/<int:exam_id>/', views.grade_exam_view, name='grade_exam'),
    path('exam/<int:exam_id>/upload_students/', views.upload_students, name='upload_students'),
    
    # --- Edit Result (Individual) ---
    path('result/<int:result_id>/edit/', views.edit_result, name='edit_result'),
    path('api/update_result/<int:result_id>/', views.api_update_result, name='api_update_result'),
    path('delete_result/<int:result_id>/', views.delete_result, name='delete_result'),

    # --- Edit Exam (Subject) ---
    path('edit_exam/<int:exam_id>/', views.edit_exam, name='edit_exam'),
    path('edit_preview/<int:exam_id>/', views.edit_exam_preview, name='edit_exam_preview'),
    path('save_edit_confirm/<int:exam_id>/', views.save_edit_confirm, name='save_edit_confirm'),
    path('delete/<int:exam_id>/', views.delete_exam, name='delete_exam'),

    # --- PDF Printing ---
    # ใช้ generate_answer_sheet ตามที่คุณขอ (แบบมีพื้นหลัง)
    path('print_sheets/<int:exam_id>/', views.generate_answer_sheet, name='print_sheets'),
    # เผื่อไว้สำหรับแบบธรรมดา (ถ้ามี)
    path('download_sheet/<int:exam_id>/', views.download_exam_sheet, name='download_exam_sheet'),
]

# รองรับการโชว์รูปภาพในโหมด Debug
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)