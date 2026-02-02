from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Exam, StudentResult, Enrollment, Subject, Student
from .utils import process_omr, generate_key_image
import os
import uuid
import datetime
import urllib.parse
from django.conf import settings
from django.core.files.storage import default_storage

# --- Imports สำหรับ ReportLab (สร้าง PDF) ---
from django.http import HttpResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- ตั้งค่า Font ภาษาไทยสำหรับ PDF ---
try:
    font_path = os.path.join(settings.BASE_DIR, 'static/fonts/THSarabunNew.ttf')
    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont('THSarabun', font_path))
        FONT_NAME = 'THSarabun'
        FONT_SIZE = 14
    else:
        FONT_NAME = 'Helvetica'
        FONT_SIZE = 12
except:
    FONT_NAME = 'Helvetica'
    FONT_SIZE = 12


# --- 1. หน้า Dashboard (Index) ---
@login_required
def index(request):
    exams = Exam.objects.only('id', 'name', 'created_at', 'key_image').order_by('-created_at')
    return render(request, 'grading/index.html', {'exams': exams})


# --- 2. หน้าสร้างเฉลย (Create Exam) ---
@login_required
def create_exam(request):
    subjects = Subject.objects.all()

    if request.method == 'POST':
        # รับค่าจากฟอร์ม
        subject_id = request.POST.get('subject_id')
        custom_name = request.POST.get('name', '').strip()
        key_type = request.POST.get('key_type')
        
        course_code = request.POST.get('course_code', '')
        room = request.POST.get('room', '')
        
        # 1. รับค่า Limit (สำคัญ: ต้องรับก่อนเริ่มลูป)
        try:
            total_questions = int(request.POST.get('total_questions', 100))
        except ValueError:
            total_questions = 100
            
        exam_date_str = request.POST.get('exam_date')
        start_time_str = request.POST.get('start_time')
        end_time_str = request.POST.get('end_time')

        # Logic จัดการชื่อวิชา
        final_name = custom_name
        if subject_id:
            try:
                selected_subject = Subject.objects.get(pk=subject_id)
                if not final_name: 
                    final_name = selected_subject.name
                if not course_code:
                    course_code = selected_subject.code
            except Subject.DoesNotExist:
                pass

        # Validation Context
        context = {
            'range_100': range(1, 101),
            'subjects': subjects,
            'saved_name': final_name,
            'saved_code': course_code,
            'saved_room': room,
            'saved_total': total_questions,
            'saved_date': exam_date_str,
            'saved_start': start_time_str,
            'saved_end': end_time_str,
        }

        # --- VALIDATION ZONE ---
        if not final_name:
            context['error'] = 'กรุณาระบุชื่อวิชา หรือเลือกจาก Dropdown'
            return render(request, 'grading/create_exam.html', context)

        if Exam.objects.filter(name=final_name).exists():
            context['error'] = f'ชื่อวิชา "{final_name}" มีอยู่แล้วในระบบ'
            return render(request, 'grading/create_exam.html', context)

        # ตรวจสอบวันที่ (ห้ามเป็นอดีต)
        if exam_date_str:
            try:
                parsed_date = datetime.datetime.strptime(exam_date_str, '%Y-%m-%d').date()
                if parsed_date < datetime.date.today():
                    context['error'] = 'วันที่สอบไม่สามารถเป็นวันที่ผ่านมาแล้วได้'
                    return render(request, 'grading/create_exam.html', context)
            except ValueError:
                pass

        # ตรวจสอบเวลา (เริ่มต้องมาก่อนจบ)
        if start_time_str and end_time_str:
            try:
                s_t = datetime.datetime.strptime(start_time_str, '%H:%M').time()
                e_t = datetime.datetime.strptime(end_time_str, '%H:%M').time()
                if s_t >= e_t:
                    context['error'] = 'เวลาเริ่มสอบต้องมาก่อนเวลาจบสอบ'
                    return render(request, 'grading/create_exam.html', context)
            except ValueError:
                pass
        # -----------------------

        # Logic สร้างเฉลย
        key = {}
        # ** FIX: วนลูปแค่ 1 ถึง total_questions **
        target_range = range(1, total_questions + 1)

        if key_type == 'sequential':
            for i in target_range:
                if 1 <= i <= 20: ch = 'a'
                elif 21 <= i <= 40: ch = 'b'
                elif 41 <= i <= 60: ch = 'c'
                elif 61 <= i <= 80: ch = 'd'
                else: ch = 'e'
                key[str(i)] = [ch]
        elif key_type == 'manual':
            missing_questions = []
            for i in target_range:
                val = request.POST.get(f'q_{i}')
                if val: 
                    key[str(i)] = [val]
                else:
                    key[str(i)] = []
                    missing_questions.append(str(i))
            
            if missing_questions:
                context['error'] = f"เลือกเฉลยไม่ครบตามจำนวน {total_questions} ข้อ (ขาดข้อ: {', '.join(missing_questions)})"
                return render(request, 'grading/create_exam.html', context)

        # สร้างรูป Preview (ส่ง total_questions ไปวาดเส้นแดงในรูป)
        temp_filename = f"preview_{uuid.uuid4().hex[:8]}.jpg"
        preview_path = generate_key_image(key, temp_filename, total_questions)
        
        if not preview_path:
            context['error'] = 'เกิดข้อผิดพลาดในการสร้างรูปเฉลย'
            return render(request, 'grading/create_exam.html', context)

        # บันทึก Session
        request.session['temp_exam_data'] = {
            'name': final_name, 
            'key': key, 
            'image_path': preview_path,
            'subject_id': subject_id,
            'course_code': course_code,
            'room': room,
            'total_questions': total_questions,
            'exam_date': exam_date_str,
            'start_time': start_time_str,
            'end_time': end_time_str
        }
        return render(request, 'grading/create_preview.html', {'name': final_name, 'image_url': preview_path})

    # GET Request
    return render(request, 'grading/create_exam.html', {
        'range_100': range(1, 101), 
        'subjects': subjects,
        'saved_total': 100
    })


# --- 3. ยืนยันบันทึก (Create Confirm) ---
@login_required
def save_exam_confirm(request):
    data = request.session.get('temp_exam_data')
    if not data: return redirect('create_exam')

    if request.method == 'POST':
        e_date = datetime.datetime.strptime(data['exam_date'], '%Y-%m-%d').date() if data.get('exam_date') else None
        s_time = datetime.datetime.strptime(data['start_time'], '%H:%M').time() if data.get('start_time') else None
        e_time = datetime.datetime.strptime(data['end_time'], '%H:%M').time() if data.get('end_time') else None

        exam = Exam.objects.create(
            name=data['name'], 
            answer_key=data['key'], 
            key_image=data['image_path'],
            course_code=data.get('course_code', ''),
            room=data.get('room', ''),
            total_questions=data.get('total_questions', 100),
            exam_date=e_date,
            start_time=s_time,
            end_time=e_time
        )

        subject_id = data.get('subject_id')
        if subject_id:
            try:
                subject = Subject.objects.get(pk=subject_id)
                students = subject.students.all()
                enrollment_list = []
                for stu in students:
                    enrollment_list.append(Enrollment(exam=exam, student=stu, section="1"))
                Enrollment.objects.bulk_create(enrollment_list)
            except Subject.DoesNotExist:
                pass 
        
        exam_name = data['name']
        del request.session['temp_exam_data']
        messages.success(request, f"สร้างวิชาสอบ '{exam_name}' เสร็จสิ้น")
        return redirect('index')
    
    if 'image_path' in data and os.path.exists(os.path.join(settings.MEDIA_ROOT, data['image_path'])):
        os.remove(os.path.join(settings.MEDIA_ROOT, data['image_path']))
    del request.session['temp_exam_data']
    return redirect('index')


# --- 4. ลบวิชาสอบ ---
@login_required
def delete_exam(request, exam_id):
    exam = get_object_or_404(Exam.objects.only('id', 'name', 'key_image'), pk=exam_id)
    if request.method == 'POST':
        exam_name = exam.name
        if exam.key_image and os.path.exists(os.path.join(settings.MEDIA_ROOT, exam.key_image)):
            os.remove(os.path.join(settings.MEDIA_ROOT, exam.key_image))
        exam.delete()
        messages.success(request, f"ลบเฉลยกระดาษข้อสอบ {exam_name} เสร็จสิ้น") 
        return redirect('index')
    return render(request, 'grading/delete_confirm.html', {'exam': exam})


# --- 5. ตรวจข้อสอบ ---
@login_required
def grade_exam_view(request, exam_id):
    exam = get_object_or_404(Exam.objects.only('id', 'name', 'answer_key'), pk=exam_id)
    if request.method == 'POST' and request.FILES.get('image'):
        files = request.FILES.getlist('image')
        c = 0
        for f in files:
            res = StudentResult.objects.create(exam=exam, student_id="Pending", score=0, original_image=f)
            full_path = os.path.join(settings.MEDIA_ROOT, res.original_image.name)
            data, err = process_omr(full_path, exam.answer_key)
            if data:
                res.student_id = data['student_id']
                res.score = data['score']
                res.graded_image = 'uploads/' + data['image_url']
                res.save()
                c += 1
        messages.success(request, f"อัปโหลดและตรวจเรียบร้อย {c} ใบ")
        return redirect('grade_exam', exam_id=exam.id)
    
    results = StudentResult.objects.filter(exam=exam).select_related('exam').order_by('-timestamp')
    return render(request, 'grading/grade.html', {'exam': exam, 'results': results})


# --- 6. ลบผลสอบรายบุคคล ---
@login_required
def delete_result(request, result_id):
    result = get_object_or_404(StudentResult.objects.select_related('exam'), pk=result_id)
    exam_id = result.exam.id
    student_id = result.student_id

    if request.method == 'POST':
        if result.original_image and os.path.exists(os.path.join(settings.MEDIA_ROOT, result.original_image.name)):
            os.remove(os.path.join(settings.MEDIA_ROOT, result.original_image.name))
        if result.graded_image and os.path.exists(os.path.join(settings.MEDIA_ROOT, result.graded_image)):
            os.remove(os.path.join(settings.MEDIA_ROOT, result.graded_image))
        result.delete()
        messages.success(request, f"ลบกระดาษคำตอบของ '{student_id}' เสร็จสิ้น")
        return redirect('grade_exam', exam_id=exam_id)

    return render(request, 'grading/delete_result_confirm.html', {'result': result})


# --- 7. แก้ไขเฉลย (Edit Exam) ---
@login_required
def edit_exam(request, exam_id):
    exam = get_object_or_404(Exam, pk=exam_id)
    subjects = Subject.objects.all()
    
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        course_code = request.POST.get('course_code', '')
        room = request.POST.get('room', '')
        subject_id = request.POST.get('subject_id')
        
        exam_date_str = request.POST.get('exam_date')
        start_time_str = request.POST.get('start_time')
        end_time_str = request.POST.get('end_time')

        # 1. รับค่า Limit
        try:
            total_questions = int(request.POST.get('total_questions', 100))
        except ValueError:
            total_questions = 100

        # 2. เก็บเฉลย (วนลูปเท่าจำนวน Limit เท่านั้น)
        key = {}
        missing_questions = []
        
        # ** FIX: วนลูป 1 ถึง total_questions เท่านั้น **
        for i in range(1, total_questions + 1):
            val = request.POST.get(f'q_{i}')
            if val: 
                key[str(i)] = [val]
            else:
                key[str(i)] = []
                missing_questions.append(str(i))
        
        context = {'exam': exam, 'range_100': range(1, 101), 'subjects': subjects}
        
        # --- VALIDATION ZONE ---
        if not name: 
            context['error'] = 'ชื่อห้ามว่าง'
            return render(request, 'grading/edit_exam.html', context)
        
        if Exam.objects.filter(name=name).exclude(pk=exam_id).exists():
            context['error'] = f'ชื่อวิชา "{name}" มีอยู่ในระบบแล้ว'
            return render(request, 'grading/edit_exam.html', context)

        if missing_questions: 
            context['error'] = f'เลือกไม่ครบตามจำนวน {total_questions} ข้อ (ขาดข้อ: {",".join(missing_questions)})'
            return render(request, 'grading/edit_exam.html', context)

        # ตรวจสอบวันที่ (ห้ามเป็นอดีต) - เช็คเฉพาะถ้ามีการส่งค่ามา
        if exam_date_str:
            try:
                parsed_date = datetime.datetime.strptime(exam_date_str, '%Y-%m-%d').date()
                if parsed_date < datetime.date.today():
                    context['error'] = 'วันที่สอบไม่สามารถเป็นวันที่ผ่านมาแล้วได้'
                    return render(request, 'grading/edit_exam.html', context)
            except ValueError:
                pass

        # ตรวจสอบเวลา
        if start_time_str and end_time_str:
            try:
                s_t = datetime.datetime.strptime(start_time_str, '%H:%M').time()
                e_t = datetime.datetime.strptime(end_time_str, '%H:%M').time()
                if s_t >= e_t:
                    context['error'] = 'เวลาเริ่มสอบต้องมาก่อนเวลาจบสอบ'
                    return render(request, 'grading/edit_exam.html', context)
            except ValueError:
                pass
        # -----------------------

        # สร้างรูป Preview สำหรับ Edit (ส่ง total_questions ไปวาดเส้นแดง)
        temp_filename = f"preview_edit_{uuid.uuid4().hex[:8]}.jpg"
        preview_path = generate_key_image(key, temp_filename, total_questions)
        
        if not preview_path:
            context['error'] = 'เกิดข้อผิดพลาดในการสร้างรูปเฉลย'
            return render(request, 'grading/edit_exam.html', context)

        # บันทึก Session
        request.session['temp_edit_data'] = {
            'name': name,
            'key': key,
            'image_path': preview_path, 
            'subject_id': subject_id,
            'course_code': course_code,
            'room': room,
            'total_questions': total_questions,
            'exam_date': exam_date_str,
            'start_time': start_time_str,
            'end_time': end_time_str
        }
        
        return redirect('edit_exam_preview', exam_id=exam.id)

    # GET Request
    return render(request, 'grading/edit_exam.html', {
        'exam': exam, 
        'range_100': range(1, 101), 
        'subjects': subjects
    })

@login_required
def edit_exam_preview(request, exam_id):
    data = request.session.get('temp_edit_data')
    if not data:
        return redirect('edit_exam', exam_id=exam_id)
    return render(request, 'grading/edit_preview.html', {
        'exam_id': exam_id,
        'name': data['name'],
        'image_url': data['image_path']
    })

@login_required
def save_edit_confirm(request, exam_id):
    data = request.session.get('temp_edit_data')
    if not data: 
        return redirect('edit_exam', exam_id=exam_id)

    exam = get_object_or_404(Exam, pk=exam_id)

    if request.method == 'POST':
        # ลบรูปเก่า
        if exam.key_image and os.path.exists(os.path.join(settings.MEDIA_ROOT, exam.key_image)):
            os.remove(os.path.join(settings.MEDIA_ROOT, exam.key_image))

        # อัปเดตข้อมูล
        exam.name = data['name']
        exam.answer_key = data['key']
        exam.key_image = data['image_path']
        exam.course_code = data['course_code']
        exam.room = data['room']
        exam.total_questions = data['total_questions']
        
        if data['exam_date']: exam.exam_date = datetime.datetime.strptime(data['exam_date'], '%Y-%m-%d').date()
        if data['start_time']: exam.start_time = datetime.datetime.strptime(data['start_time'], '%H:%M').time()
        if data['end_time']: exam.end_time = datetime.datetime.strptime(data['end_time'], '%H:%M').time()
        
        exam.save()

        # อัปเดตรายชื่อนักเรียน
        subject_id = data.get('subject_id')
        if subject_id:
            try:
                subject = Subject.objects.get(pk=subject_id)
                Enrollment.objects.filter(exam=exam).delete()
                students = subject.students.all()
                enrollment_list = []
                for stu in students:
                    enrollment_list.append(Enrollment(exam=exam, student=stu, section="1"))
                Enrollment.objects.bulk_create(enrollment_list)
            except Subject.DoesNotExist:
                pass

        del request.session['temp_edit_data']
        messages.success(request, f"บันทึกการแก้ไข '{exam.name}' เรียบร้อยแล้ว")
        return redirect('index')

    # ยกเลิก
    if 'image_path' in data and os.path.exists(os.path.join(settings.MEDIA_ROOT, data['image_path'])):
        os.remove(os.path.join(settings.MEDIA_ROOT, data['image_path']))
    del request.session['temp_edit_data']
    return redirect('edit_exam', exam_id=exam_id)


# --- 8. Confirm Logout ---
@login_required
def logout_confirm_view(request):
    return render(request, 'registration/logout_confirm.html')


# --- 9. สร้าง PDF กระดาษคำตอบ ---
@login_required
def generate_answer_sheet(request, exam_id):
    exam = get_object_or_404(Exam, pk=exam_id)
    enrollments = Enrollment.objects.filter(exam=exam).select_related('student')
    
    filename = f"OMR_{exam.name}.pdf"
    encoded_filename = urllib.parse.quote(filename)
    
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{encoded_filename}"; filename*=UTF-8\'\'{encoded_filename}'

    c = canvas.Canvas(response, pagesize=A4)
    width, height = A4
    
    bg_image_path = os.path.join(settings.MEDIA_ROOT, 'templates/omr_template.jpg') 

    # ตรวจสอบโหมด
    if not enrollments.exists():
        loop_data = [None] # โหมดกระดาษเปล่า
    else:
        loop_data = enrollments # โหมดมีรายชื่อ

    for item in loop_data:
        # A. วาดพื้นหลัง
        if os.path.exists(bg_image_path):
            c.drawImage(bg_image_path, 0, 0, width=width, height=height)
        
        # B. ใส่ข้อมูลส่วนหัว
        if item is not None:
            student = item.student
            
            c.setFont(FONT_NAME, FONT_SIZE)
            c.setFillColorRGB(0, 0, 0.5)
            
            # --- แถว 1 ---
            c.drawString(45*mm, 256*mm, f"{student.first_name} {student.last_name}")
            c.drawString(105*mm, 256*mm, f"{exam.name}")
            c.drawString(165*mm, 256*mm, f"{exam.course_code}")

            # --- แถว 2 (รหัส + หมู่เรียน) ---
            c.drawString(45*mm, 248*mm, f"{student.student_id}")
            # แสดงหมู่เรียน
            c.drawString(105*mm, 248*mm, f"{item.section}")

            # --- แถว 3 (ห้อง + เวลา + วันที่) ---
            c.drawString(45*mm, 240*mm, f"{exam.room}")
            
            time_str = ""
            if exam.start_time and exam.end_time:
                # บังคับเวลาแบบ 24 ชม.
                time_str = f"{exam.start_time.strftime('%H:%M')} - {exam.end_time.strftime('%H:%M')}"
            c.drawString(105*mm, 240*mm, time_str)
            
            date_str = exam.exam_date.strftime('%d/%m/%Y') if exam.exam_date else ""
            c.drawString(165*mm, 240*mm, date_str)
        
        # C. ไม่วาดเส้นแดงใน PDF (ตามที่ขอ เหลือแค่ในรูป preview)

        c.showPage()

    c.save()
    return response