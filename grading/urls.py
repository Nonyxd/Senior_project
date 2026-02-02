from django.urls import path
from . import views

urlpatterns = [
    # หน้าหลัก
    path('', views.index, name='index'),
    
    # สร้างข้อสอบ
    path('create/', views.create_exam, name='create_exam'),
    path('save_exam/', views.save_exam_confirm, name='save_exam_confirm'),
    
    # User & Actions
    path('logout/confirm/', views.logout_confirm_view, name='logout_confirm'),
    path('delete/<int:exam_id>/', views.delete_exam, name='delete_exam'),
    path('grade/<int:exam_id>/', views.grade_exam_view, name='grade_exam'),
    path('delete_result/<int:result_id>/', views.delete_result, name='delete_result'),
    path('print_sheets/<int:exam_id>/', views.generate_answer_sheet, name='print_sheets'),

    # --- ส่วน Edit Exam (ที่เคย Error) ---
    path('edit_exam/<int:exam_id>/', views.edit_exam, name='edit_exam'),
    path('edit_preview/<int:exam_id>/', views.edit_exam_preview, name='edit_exam_preview'), # <--- ตัวนี้ที่ขาดไป
    path('save_edit_confirm/<int:exam_id>/', views.save_edit_confirm, name='save_edit_confirm'),
]