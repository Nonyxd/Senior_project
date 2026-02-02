# grading/models.py
from django.db import models

class Subject(models.Model):
    code = models.CharField(max_length=20)
    name = models.CharField(max_length=100)
    def __str__(self): return f"{self.code} {self.name}"

class Student(models.Model):
    student_id = models.CharField(max_length=20, unique=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    subjects = models.ManyToManyField(Subject, related_name='students', blank=True)
    def __str__(self): return f"{self.student_id} {self.first_name}"

class Exam(models.Model):
    name = models.CharField(max_length=100, unique=True)
    answer_key = models.JSONField()
    key_image = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    course_code = models.CharField(max_length=20, default="", blank=True)
    exam_date = models.DateField(null=True, blank=True)
    start_time = models.TimeField(null=True, blank=True)
    end_time = models.TimeField(null=True, blank=True)
    room = models.CharField(max_length=50, default="", blank=True)
    total_questions = models.IntegerField(default=100)
    def __str__(self): return self.name

class Enrollment(models.Model):
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE)
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    section = models.CharField(max_length=10, default="1")
    class Meta: unique_together = ('exam', 'student')

class StudentResult(models.Model):
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE)
    student_id = models.CharField(max_length=50)
    score = models.IntegerField()
    original_image = models.ImageField(upload_to='uploads/')
    graded_image = models.CharField(max_length=255, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)