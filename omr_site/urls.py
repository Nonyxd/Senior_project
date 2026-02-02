from django.contrib import admin
from django.urls import path, include
from grading import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),
    
    # --- Grading App URLs ---
    path('', views.index, name='index'),
    
    # Create Flow
    path('create/', views.create_exam, name='create_exam'),
    path('save_exam/', views.save_exam_confirm, name='save_exam_confirm'),
    
    # User Actions
    path('logout/confirm/', views.logout_confirm_view, name='logout_confirm'),
    path('delete/<int:exam_id>/', views.delete_exam, name='delete_exam'),
    path('grade/<int:exam_id>/', views.grade_exam_view, name='grade_exam'),
    path('delete_result/<int:result_id>/', views.delete_result, name='delete_result'),
    path('print_sheets/<int:exam_id>/', views.generate_answer_sheet, name='print_sheets'),

    # --- Edit Exam Flow (ต้องมีครบ 3 บรรทัดนี้ครับ) ---
    path('edit_exam/<int:exam_id>/', views.edit_exam, name='edit_exam'),
    
    # 2 บรรทัดนี้คือตัวที่ขาดไปครับ ใส่เพิ่มเข้าไปเลย
    path('edit_preview/<int:exam_id>/', views.edit_exam_preview, name='edit_exam_preview'),
    path('save_edit_confirm/<int:exam_id>/', views.save_edit_confirm, name='save_edit_confirm'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)