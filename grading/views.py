import os
import uuid
import datetime
import urllib.parse
import json
import pandas as pd 

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from django.core.files.storage import default_storage
from django.http import HttpResponse, JsonResponse
from django.utils import timezone
from django.db import transaction
from django.views.decorators.http import require_POST

# --- ReportLab Imports ---
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- Models ---
from .models import Exam, StudentResult, Student

# --- Utils ---
from .utils import process_omr, generate_key_image, generate_exam_pdf

# ==========================================
# 0. API & Helpers
# ==========================================
@login_required
@require_POST
def api_parse_excel(request):
    try:
        if not request.FILES.get('file'): return JsonResponse({'success': False, 'error': 'No file uploaded'})
        excel_file = request.FILES['file']
        df = pd.read_excel(excel_file)
        if df.empty: return JsonResponse({'success': False, 'error': 'Empty file'})

        first_row = df.iloc[0]
        data = {
            'subject_code': str(first_row.get('SubjectCode', '')).strip(),
            'subject_name': str(first_row.get('SubjectName', '')).strip(),
            'section': str(first_row.get('Section', '1')).strip(),
            'student_count': len(df)
        }
        return JsonResponse({'success': True, 'data': data})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

@login_required
@require_POST
def delete_subject_api(request):
    try:
        data = json.loads(request.body)
        subject_code = data.get('subject_code')
        if not subject_code: return JsonResponse({'success': False, 'error': 'Missing Code'})
        exams = Exam.objects.filter(subject_code=subject_code)
        count = exams.count()
        for ex in exams:
            if ex.key_image and os.path.exists(os.path.join(settings.MEDIA_ROOT, ex.key_image.name)):
                try: os.remove(os.path.join(settings.MEDIA_ROOT, ex.key_image.name))
                except: pass
        exams.delete()
        return JsonResponse({'success': True, 'message': f'Deleted {count} exams for {subject_code}'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

# ==========================================
# 1. Dashboard & Create
# ==========================================
@login_required
def index(request):
    exams = Exam.objects.filter(is_active=True).order_by('-created_at')
    return render(request, 'grading/index.html', {'exams': exams})

@login_required
def create_exam(request):
    existing_subjects = Exam.objects.values('subject_code', 'subject_name').distinct()
    if request.method == 'POST':
        subject_code = request.POST.get('subject_code', '').strip()
        subject_name = request.POST.get('subject_name', '').strip()
        section = request.POST.get('section', '1')
        exam_date_str = request.POST.get('exam_date')
        start_time_str = request.POST.get('start_time')
        duration = request.POST.get('duration_minutes', 120)
        room = request.POST.get('room', '')
        total_questions = int(request.POST.get('total_questions', 100))
        key_type = request.POST.get('key_type', 'manual')

        context = {'existing_subjects': existing_subjects, 'range_100': range(1, 101), 'saved': request.POST}
        
        if not subject_code or not subject_name:
            context['error'] = 'กรุณาระบุรหัสวิชาและชื่อวิชา'
            return render(request, 'grading/create_exam.html', context)

        try:
            e_date = datetime.datetime.strptime(exam_date_str, '%Y-%m-%d').date() if exam_date_str else timezone.now().date()
            s_time = datetime.datetime.strptime(start_time_str, '%H:%M').time() if start_time_str else timezone.now().time()
        except ValueError:
            context['error'] = 'รูปแบบวันที่หรือเวลาไม่ถูกต้อง'
            return render(request, 'grading/create_exam.html', context)

        key = {}
        missing_questions = []
        for i in range(1, total_questions + 1):
            if key_type == 'sequential':
                if i <= 20: val = 'a'
                elif i <= 40: val = 'b'
                elif i <= 60: val = 'c'
                elif i <= 80: val = 'd'
                else: val = 'e'
                key[str(i)] = [val]
            else:
                val = request.POST.get(f'q_{i}')
                if val: key[str(i)] = [val]
                else:
                    key[str(i)] = []
                    missing_questions.append(str(i))
        
        if missing_questions and key_type == 'manual':
            context['error'] = f"เฉลยไม่ครบข้อ: {', '.join(missing_questions)}"
            return render(request, 'grading/create_exam.html', context)

        temp_filename = f"preview_{uuid.uuid4().hex[:8]}.jpg"
        preview_path = generate_key_image(key, temp_filename, total_questions)

        request.session['temp_exam_data'] = {
            'subject_code': subject_code, 'subject_name': subject_name,
            'section': section, 'exam_date': str(e_date), 'start_time': str(s_time),
            'duration_minutes': duration, 'room': room, 'total_questions': total_questions,
            'answer_key': key, 'key_image_path': preview_path
        }

        if request.FILES.get('roster_file'):
            excel_file = request.FILES['roster_file']
            fs = default_storage
            filename = fs.save(f"temp_rosters/{excel_file.name}", excel_file)
            request.session['temp_roster_path'] = filename

        return render(request, 'grading/create_preview.html', {
            'name': f"{subject_code} {subject_name}", 'image_url': preview_path
        })

    return render(request, 'grading/create_exam.html', {
        'existing_subjects': existing_subjects, 'range_100': range(1, 101)
    })

@login_required
def save_exam_confirm(request):
    data = request.session.get('temp_exam_data')
    if not data: return redirect('create_exam')
    
    if request.method == 'POST':
        exam = Exam.objects.create(
            subject_code=data['subject_code'], subject_name=data['subject_name'],
            section=data['section'], exam_date=data['exam_date'], start_time=data['start_time'],
            duration_minutes=data['duration_minutes'], room=data['room'],
            total_questions=data['total_questions'], answer_key=data['answer_key'],
            key_image=data['key_image_path'], is_active=True
        )
        roster_path = request.session.get('temp_roster_path')
        if roster_path:
            full_path = os.path.join(settings.MEDIA_ROOT, roster_path)
            if os.path.exists(full_path):
                try:
                    df = pd.read_excel(full_path)
                    for _, row in df.iterrows():
                        sid = str(row.iloc[3]).strip() if len(row)>3 else str(row.get('StudentID',''))
                        fn = str(row.iloc[4]).strip() if len(row)>4 else str(row.get('FirstName',''))
                        ln = str(row.iloc[5]).strip() if len(row)>5 else str(row.get('LastName',''))
                        if sid and sid.lower()!='nan':
                            stu, _ = Student.objects.get_or_create(student_id=sid, defaults={'first_name':fn, 'last_name':ln})
                            exam.enrolled_students.add(stu)
                    os.remove(full_path)
                except: pass
        
        del request.session['temp_exam_data']
        if 'temp_roster_path' in request.session: del request.session['temp_roster_path']
        messages.success(request, f"สร้างวิชา {exam.subject_code} สำเร็จ")
        return redirect('index')
    
    if 'key_image_path' in data:
        path = os.path.join(settings.MEDIA_ROOT, data['key_image_path'])
        if os.path.exists(path): os.remove(path)
    del request.session['temp_exam_data']
    return redirect('create_exam')

@login_required
def upload_students(request, exam_id):
    exam = get_object_or_404(Exam, pk=exam_id)
    if request.method == 'POST' and request.FILES.get('excel_file'):
        try:
            df = pd.read_excel(request.FILES['excel_file'])
            c = 0
            for _, row in df.iterrows():
                sid = str(row.iloc[0]).strip()
                fn = str(row.iloc[1]).strip()
                ln = str(row.iloc[2]).strip()
                stu, _ = Student.objects.get_or_create(student_id=sid, defaults={'first_name':fn, 'last_name':ln})
                exam.enrolled_students.add(stu)
                c += 1
            messages.success(request, f"เพิ่มนิสิต {c} คน")
            return redirect('grade_exam', exam_id=exam.id)
        except Exception as e:
            messages.error(request, f"Error: {e}")
    return render(request, 'grading/upload_students.html', {'exam': exam})

# ==========================================
# 5. Grade Exam (Catch-all + Missing List)
# ==========================================
@login_required
def grade_exam_view(request, exam_id):
    exam = get_object_or_404(Exam, pk=exam_id)
    
    # Upload Logic
    if request.method == 'POST' and request.FILES.get('image'):
        files = request.FILES.getlist('image')
        c = 0
        for f in files:
            res = StudentResult.objects.create(
                exam=exam, student_id_ocr="Processing...", score=0, original_image=f, status='OCR'
            )
            data, err = process_omr(res.original_image.path, exam.answer_key)
            if data:
                clean_id = str(data.get('student_id', 'Unknown')).strip()
                res.student_id_ocr = clean_id
                res.score = data.get('score', 0)
                res.results_data = data.get('details', {})
                
                # Fix Path
                if 'image_url' in data:
                    raw_path = str(data['image_url'])
                    if 'uploads' in raw_path:
                        idx = raw_path.find('uploads')
                        rel_path = raw_path[idx:]
                    elif 'media' in raw_path:
                        idx = raw_path.find('media')
                        rel_path = raw_path[idx+6:]
                    else:
                        filename = os.path.basename(raw_path)
                        rel_path = f"uploads/papers/{filename}"
                    res.graded_image = rel_path.replace('\\', '/')
                
                try:
                    res.student = Student.objects.get(student_id=clean_id)
                except Student.DoesNotExist:
                    res.student = None
                res.save()
                c += 1
            else:
                print(f"Error: {err}")
        messages.success(request, f"ตรวจเรียบร้อย {c} ใบ")
        return redirect('grade_exam', exam_id=exam.id)

    # Display Logic
    all_results = StudentResult.objects.filter(exam=exam).select_related('student')
    enrolled_students = exam.enrolled_students.all().order_by('student_id')
    
    student_dashboard = []
    missing_students = []
    submitted_count = 0
    shown_result_ids = set()

    result_map = {}
    for r in all_results:
        if r.student:
            result_map[r.student.student_id.strip()] = r

    for student in enrolled_students:
        s_id = student.student_id.strip()
        result = result_map.get(s_id)
        
        if result:
            status = 'SUBMITTED'
            submitted_count += 1
            shown_result_ids.add(result.id)
            student_dashboard.append({'info': student, 'result': result, 'status': status})
        else:
            missing_students.append(student)
            student_dashboard.append({'info': student, 'result': None, 'status': 'MISSING'})

    unknown_results = [r for r in all_results if r.id not in shown_result_ids]

    return render(request, 'grading/grade.html', {
        'exam': exam,
        'student_dashboard': student_dashboard,
        'missing_students': missing_students,
        'unknown_results': unknown_results,
        'submitted_count': submitted_count,
        'total_students': enrolled_students.count(),
        'results': all_results,
    })

# ==========================================
# 6. Edit Result
# ==========================================
@login_required
def edit_result(request, result_id):
    result = get_object_or_404(StudentResult, pk=result_id)
    exam = result.exam
    
    if result.status == 'OCR':
        result.status = 'EDITING'
        result.save()

    answer_key = exam.answer_key
    student_answers = result.results_data or {}
    combined_data = {}
    
    for i in range(1, exam.total_questions + 1):
        q = str(i)
        k = answer_key.get(q, [])
        s = student_answers.get(q, [])
        
        s_choice = s.get('choice', []) if isinstance(s, dict) else s
        if not isinstance(s_choice, list): s_choice = [s_choice] 

        is_correct = False
        error_reason = ""

        if k and s_choice and s_choice[0] in k: 
            is_correct = True
        else:
            if not s_choice: 
                error_reason = "EMPTY"
            elif len(s_choice) > 1: 
                error_reason = "MULTIPLE"
            else: 
                error_reason = "WRONG"
            
        combined_data[q] = {
            'key': k, 
            'student': s_choice, 
            'is_correct': is_correct,
            'error_reason': error_reason
        }

    return render(request, 'grading/edit_result.html', {
        'result': result, 'exam': exam, 'combined_data': combined_data
    })

# ==========================================
# 7. API Update
# ==========================================
@login_required
@require_POST
def api_update_result(request, result_id):
    try:
        data = json.loads(request.body)
        result = get_object_or_404(StudentResult, pk=result_id)
        action = data.get('action')

        if action == 'update_score':
            q_num = str(data.get('q_num'))
            is_correct = data.get('is_correct')
            student_answers = result.results_data or {}
            key_list = result.exam.answer_key.get(q_num, [])
            
            if is_correct: student_answers[q_num] = [key_list[0]] if key_list else ['Free']
            else: student_answers[q_num] = ['F']
            
            new_score = 0
            for i in range(1, result.exam.total_questions + 1):
                qn = str(i)
                k = result.exam.answer_key.get(qn, [])
                v = student_answers.get(qn, [])
                s = v.get('choice', []) if isinstance(v, dict) else v
                if k and s and s[0] in k: new_score += 1
            
            result.results_data = student_answers
            result.score = new_score
            result.status = 'EDITING'
            result.save()
            return JsonResponse({'success': True, 'new_score': new_score})

        elif action == 'update_status':
            result.status = data.get('status')
            result.save()
            return JsonResponse({'success': True})

        elif action == 'update_student_id':
            new_id = str(data.get('student_id', '')).strip()
            if not new_id: return JsonResponse({'success': False, 'error': 'Empty ID'})
            result.student_id_ocr = new_id
            try:
                stu = Student.objects.get(student_id=new_id)
                result.student = stu
                student_name = f"{stu.first_name} {stu.last_name}"
                found = True
            except Student.DoesNotExist:
                result.student = None
                student_name = "(ไม่พบข้อมูลนิสิต)"
                found = False
            result.save()
            return JsonResponse({'success': True, 'student_name': student_name, 'found': found})

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)
    return JsonResponse({'success': False}, status=400)

# ==========================================
# 8. PDF (เพิ่ม: กา X + เขียนเลขรหัสนิสิต)
# ==========================================
@login_required
def generate_answer_sheet(request, exam_id):
    exam = get_object_or_404(Exam, pk=exam_id)
    students = exam.enrolled_students.all()
    
    filename = f"OMR_{exam.subject_code}.pdf"
    encoded_filename = urllib.parse.quote(filename)
    
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{encoded_filename}"; filename*=UTF-8\'\'{encoded_filename}'

    c = canvas.Canvas(response, pagesize=A4)
    width, height = A4
    
    font_path = os.path.join(settings.BASE_DIR, 'static', 'THSARABUNNEW.TTF')
    font_name_use = 'Helvetica'
    font_size_use = 12
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont('THSarabunNew', font_path))
            font_name_use = 'THSarabunNew'
            font_size_use = 16 
        except: pass
    
    bg_image_path = os.path.join(settings.MEDIA_ROOT, 'templates', 'omr_template.jpg') 

    if not students.exists(): loop_data = [None]
    else: loop_data = students

    for student in loop_data:
        if os.path.exists(bg_image_path): 
            c.drawImage(bg_image_path, 0, 0, width=width, height=height)
        
        if student is not None:
            c.setFont(font_name_use, font_size_use)
            c.setFillColorRGB(0, 0, 0.5)
            
            # --- 1. ข้อความทั่วไป ---
            c.drawString(45*mm, 256*mm, f"{student.first_name} {student.last_name}")
            c.drawString(105*mm, 256*mm, f"{exam.subject_name}")
            c.drawString(165*mm, 256*mm, f"{exam.subject_code}")
            c.drawString(45*mm, 248*mm, f"{student.student_id}")
            c.drawString(105*mm, 248*mm, f"{exam.section}") 
            c.drawString(165*mm, 248*mm, exam.exam_date.strftime('%d/%m/%Y') if exam.exam_date else "")
            c.drawString(45*mm, 240*mm, f"{exam.room}")
            c.drawString(105*mm, 240*mm, exam.start_time.strftime('%H:%M') if exam.start_time else "")
            c.drawString(165*mm, 240*mm, f"{exam.duration_minutes} นาที")

            # --- 2. Auto Mark "X" & Student ID Number ---
            
            # (A) ตั้งค่าโครงสร้างตารางหลัก (ใช้อ้างอิงทั้งเลขและ X)
            GRID_START_X = 24 * mm
            GRID_START_Y = 214 * mm 
            STEP_X = 5.8 * mm
            STEP_Y = 4 * mm
            
            # (B) ตัวปรับตำแหน่งเฉพาะ "X" (แก้ตรงนี้เพื่อเลื่อนกากบาทอย่างเดียว)
            # ใส่ค่าบวก (+) เพื่อเลื่อนขวา/ขึ้น, ค่าลบ (-) เพื่อเลื่อนซ้าย/ลง
            OFFSET_X_ONLY = 0 * mm   # <--- ปรับซ้าย-ขวา ของ X ตรงนี้
            OFFSET_Y_ONLY = 2 * mm   # <--- ปรับบน-ล่าง ของ X ตรงนี้
            
            student_id_str = str(student.student_id).strip()
            
            c.setFillColorRGB(0, 0, 0)
            
            for i, char in enumerate(student_id_str):
                if char.isdigit():
                    digit = int(char)
                    # คำนวณตำแหน่งมาตรฐาน
                    base_x = GRID_START_X + (i * STEP_X)
                    base_y = GRID_START_Y - (digit * STEP_Y)
                    
                    # -----------------------------------------------------------
                    # 2.1 วาดกากบาท (X) -> บวก Offset เข้าไปเฉพาะตรงนี้
                    # -----------------------------------------------------------
                    c.setFont("Helvetica", 14)
                    c.drawString(base_x + OFFSET_X_ONLY, base_y + OFFSET_Y_ONLY, "x")
                    
                    # -----------------------------------------------------------
                    # 2.2 เขียนตัวเลขรหัสนิสิต (ไม่ได้รับผลกระทบจาก Offset ข้างบน)
                    # -----------------------------------------------------------
                    header_y = GRID_START_Y + 6.5 * mm 
                    c.setFont("Helvetica", 8) 
                    c.drawString(base_x + 1.5*mm, header_y, char)

        c.showPage()

    c.save()
    return response

@login_required
def download_exam_sheet(request, exam_id, student_id=None):
    exam = get_object_or_404(Exam, pk=exam_id)
    student = get_object_or_404(Student, student_id=student_id) if student_id else None
    
    filename = f"OMR_{exam.subject_code}_{student.student_id}.pdf" if student else f"OMR_{exam.subject_code}_Master.pdf"
    
    response = HttpResponse(content_type='application/pdf')
    encoded_filename = urllib.parse.quote(filename)
    response['Content-Disposition'] = f'attachment; filename="{encoded_filename}"; filename*=UTF-8\'\'{encoded_filename}'

    generate_exam_pdf(response, exam, student)
    return response

# ==========================================
# 9. Delete Exam (แก้เป็น: ลบผลสอบถาวร + ซ่อนวิชา)
# ==========================================
@login_required
def delete_exam(request, exam_id):
    exam = get_object_or_404(Exam, pk=exam_id)
    
    if request.method == 'POST':
        # 1. ลบผลสอบและรูปภาพทั้งหมด (Hard Delete) เพื่อคืนพื้นที่
        results = exam.results.all()
        deleted_count = results.count()
        
        for res in results:
            if res.original_image:
                try: os.remove(os.path.join(settings.MEDIA_ROOT, res.original_image.name))
                except: pass
            if res.graded_image:
                try: os.remove(os.path.join(settings.MEDIA_ROOT, res.graded_image.name))
                except: pass
            res.delete()

        # 2. ซ่อนวิชา (Soft Delete)
        exam.is_active = False 
        exam.save()
        
        messages.success(request, f"ลบผลสอบ {deleted_count} ใบ และนำวิชา {exam.subject_code} ออกจากรายการแล้ว")
        return redirect('index')
        
    return render(request, 'grading/delete_confirm.html', {'exam': exam})

@login_required
def delete_result(request, result_id):
    result = get_object_or_404(StudentResult, pk=result_id)
    exam_id = result.exam.id
    if request.method == 'POST':
        if result.original_image: 
            try: os.remove(os.path.join(settings.MEDIA_ROOT, result.original_image.name))
            except: pass
        if result.graded_image:
            try: os.remove(os.path.join(settings.MEDIA_ROOT, result.graded_image.name))
            except: pass
        result.delete()
        messages.success(request, "ลบผลการตรวจแล้ว")
    return redirect('grade_exam', exam_id=exam_id)

@login_required
def edit_exam(request, exam_id):
    exam = get_object_or_404(Exam, pk=exam_id)
    existing_subjects = Exam.objects.values('subject_code', 'subject_name').distinct()
    if request.method == 'POST':
        subject_code = request.POST.get('subject_code', '').strip()
        subject_name = request.POST.get('subject_name', '').strip()
        section = request.POST.get('section', '1')
        exam_date_str = request.POST.get('exam_date')
        start_time_str = request.POST.get('start_time')
        duration = request.POST.get('duration_minutes', 120)
        room = request.POST.get('room', '')
        total_questions = int(request.POST.get('total_questions', 100))
        key_type = request.POST.get('key_type', 'manual')

        key = {}
        missing_questions = []
        for i in range(1, total_questions + 1):
            if key_type == 'sequential':
                if i <= 20: val = 'a'
                elif i <= 40: val = 'b'
                elif i <= 60: val = 'c'
                elif i <= 80: val = 'd'
                else: val = 'e'
                key[str(i)] = [val]
            else:
                val = request.POST.get(f'q_{i}')
                if val: key[str(i)] = [val]
                else:
                    key[str(i)] = []
                    missing_questions.append(str(i))
        
        context = {'exam': exam, 'existing_subjects': existing_subjects, 'range_100': range(1, 101)}
        if not subject_code or not subject_name:
            context['error'] = 'ชื่อวิชาและรหัสวิชาห้ามว่าง'
            return render(request, 'grading/edit_exam.html', context)
        if missing_questions and key_type == 'manual':
            context['error'] = f'เฉลยไม่ครบ'
            return render(request, 'grading/edit_exam.html', context)

        temp_filename = f"preview_edit_{uuid.uuid4().hex[:8]}.jpg"
        preview_path = generate_key_image(key, temp_filename, total_questions)
        request.session['temp_edit_data'] = {
            'subject_code': subject_code, 'subject_name': subject_name,
            'section': section, 'exam_date': exam_date_str, 'start_time': start_time_str,
            'duration_minutes': duration, 'room': room, 'total_questions': total_questions,
            'answer_key': key, 'image_path': preview_path
        }
        return redirect('edit_exam_preview', exam_id=exam.id)
    return render(request, 'grading/edit_exam.html', {
        'exam': exam, 'range_100': range(1, 101), 'existing_subjects': existing_subjects
    })

@login_required
def edit_exam_preview(request, exam_id):
    data = request.session.get('temp_edit_data')
    if not data: return redirect('edit_exam', exam_id=exam_id)
    return render(request, 'grading/edit_preview.html', {
        'exam_id': exam_id, 'name': f"{data['subject_code']} {data['subject_name']}", 'image_url': data['image_path']
    })

@login_required
def save_edit_confirm(request, exam_id):
    data = request.session.get('temp_edit_data')
    if not data: return redirect('edit_exam', exam_id=exam_id)
    exam = get_object_or_404(Exam, pk=exam_id)
    if request.method == 'POST':
        if exam.key_image: 
            try: os.remove(os.path.join(settings.MEDIA_ROOT, exam.key_image.name))
            except: pass
        exam.subject_code = data['subject_code']
        exam.subject_name = data['subject_name']
        exam.section = data['section']
        exam.room = data['room']
        exam.total_questions = data['total_questions']
        exam.answer_key = data['answer_key']
        exam.key_image = data['image_path']
        exam.duration_minutes = data['duration_minutes']
        if data['exam_date']: exam.exam_date = datetime.datetime.strptime(data['exam_date'], '%Y-%m-%d').date()
        if data['start_time']: exam.start_time = datetime.datetime.strptime(data['start_time'], '%H:%M').time()
        exam.save()
        del request.session['temp_edit_data']
        messages.success(request, "แก้ไขวิชาเรียบร้อย")
        return redirect('index')
    if 'image_path' in data:
        try: os.remove(os.path.join(settings.MEDIA_ROOT, data['image_path']))
        except: pass
    del request.session['temp_edit_data']
    return redirect('edit_exam', exam_id=exam_id)

@login_required
def logout_confirm_view(request):
    return render(request, 'registration/logout_confirm.html')