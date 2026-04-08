# AI-Powered OMR System (Automated Exam Grading System)

## Live Demo (Hugging Face Spaces)
The system is currently deployed and running live on Hugging Face Spaces. You can access the web application directly via your browser without any installation required.
* **Access the System here:** [https://nonyxd-nonu.hf.space](https://nonyxd-nonu.hf.space)

*(Note: If the system has been inactive for a while, it may take 1-2 minutes to wake up the server upon your first visit.)*

---

## Introduction
This project involves the development of a web application system for grading multiple-choice answer sheets using Optical Mark Recognition (OMR). The primary objective is to reduce the time and workload associated with exam evaluation for instructors. The system is designed to accept various data file formats, automatically process and grade the submissions, and summarize the results into statistical reports.

## System Architecture & Principles
The system's operation relies on the integration of web development technologies with Artificial Intelligence (AI) and Computer Vision. The core processing workflow is divided into four main stages as follows:

### 1. Data Input & Preprocessing
Users can upload answer sheets in both image formats (JPG, PNG) and document formats (PDF). In the case of PDF files, the system utilizes the PyMuPDF library to extract individual pages and convert them into temporary images in memory, preparing them for the subsequent processing stages.

### 2. Image Registration
Since uploaded images may be skewed or distorted in proportion, the system employs OpenCV to perform perspective transformation. This process crops and aligns the specific area of the answer sheet into a flat, frontal plane, significantly increasing the accuracy of coordinate calculations for the marked bubbles.

### 3. Detection & Grading
The system leverages deep learning technology via the YOLO (You Only Look Once) model to detect key reference points on the paper. Subsequently, image processing algorithms analyze the pixel intensity within each multiple-choice option area to determine which bubble the student has marked. These results are then compared against the answer key database to calculate the total score.

### 4. Reporting & Exporting
All exam results are stored in an SQLite3 database. The system can generate individual student result reports in PDF format using ReportLab. Additionally, it supports exporting comprehensive overview data in Excel/CSV formats via the Pandas library, allowing for immediate further analysis.

## Technologies Used
* **Backend Framework:** Python, Django
* **Computer Vision & AI:** Ultralytics (YOLOv8), OpenCV, NumPy
* **Document Processing:** PyMuPDF, ReportLab
* **Data Manipulation:** Pandas
* **Database:** SQLite3
* **Deployment:** Docker, Hugging Face Spaces, WhiteNoise (for static files)

## Offline Standalone Version (Windows)
For instructors or individuals who wish to test the system offline without configuring an environment or installing Python, a pre-compiled, ready-to-use executable file is available for local machine use.

**Download Link:** [https://drive.google.com/file/d/1kH091QiSHVMACPrj19HSLKrPdh7GfjvF/view?usp=sharing]

**Usage Instructions:**
1. Download the compressed file (.zip) from the link provided above.
2. Extract the file on your computer (it is recommended to place it in a directory without strict administrative permissions, such as the Desktop or Documents folder).
3. Open the extracted folder and double-click the **ExamGradingAI.exe** file.
4. The program will initialize a local server and automatically open the default web browser to the login page within 10 to 15 seconds.

---
---

# ระบบตรวจกระดาษคำตอบอัตโนมัติด้วยปัญญาประดิษฐ์ (AI-Powered OMR System)

## ทดลองใช้งานจริงบนระบบคลาวด์ (Live Demo)
ปัจจุบันระบบได้ถูกนำขึ้นไปรันบนเซิร์ฟเวอร์ของ Hugging Face Spaces เรียบร้อยแล้ว ผู้ใช้งานสามารถเข้าถึงระบบผ่านเว็บบราวเซอร์ได้ทันทีโดยไม่ต้องติดตั้งโปรแกรมใดๆ
* **ลิงก์เข้าใช้งานระบบ:** [https://nonyxd-nonu.hf.space](https://nonyxd-nonu.hf.space)

*(หมายเหตุ: หากไม่มีผู้ใช้งานระบบเป็นระยะเวลาหนึ่ง เซิร์ฟเวอร์อาจเข้าสู่โหมดพักการทำงาน ซึ่งการเข้าใช้งานครั้งแรกหลังจากนั้นอาจใช้เวลาโหลดประมาณ 1-2 นาทีเพื่อเริ่มระบบใหม่)*

---

## บทนำ
โปรเจกต์นี้เป็นการพัฒนาระบบเว็บแอปพลิเคชันสำหรับการตรวจกระดาษคำตอบแบบปรนัย (Optical Mark Recognition - OMR) โดยมีวัตถุประสงค์เพื่อลดระยะเวลาและภาระงานในการประเมินผลการสอบ ระบบถูกออกแบบมาให้สามารถรับไฟล์ข้อมูลได้หลากหลายรูปแบบ และสามารถประมวลผลให้คะแนนได้แบบอัตโนมัติ พร้อมทั้งสรุปผลลัพธ์ออกมาเป็นรายงานทางสถิติ

## หลักการทำงานของระบบ (System Architecture & Principles)
การทำงานของระบบอาศัยการบูรณาการเทคโนโลยีทางด้าน Web Development เข้ากับปัญญาประดิษฐ์ (Artificial Intelligence) และวิทยาการคอมพิวเตอร์วิทัศน์ (Computer Vision) โดยแบ่งกระบวนการทำงานหลักออกเป็น 4 ขั้นตอน ดังนี้:

### 1. การนำเข้าและจัดการข้อมูล (Data Input & Preprocessing)
ผู้ใช้งานสามารถอัปโหลดไฟล์กระดาษคำตอบได้ทั้งในรูปแบบไฟล์ภาพ (JPG, PNG) และไฟล์เอกสาร (PDF) ในกรณีที่เป็นไฟล์ PDF ระบบจะเรียกใช้ไลบรารี PyMuPDF เพื่อทำการแยกหน้าเอกสารและแปลงเป็นไฟล์ภาพชั่วคราวในหน่วยความจำ เพื่อเตรียมพร้อมสำหรับการประมวลผลในขั้นตอนต่อไป

### 2. การวิเคราะห์และปรับแต่งภาพ (Image Registration)
เนื่องจากภาพที่อัปโหลดเข้ามาอาจมีความเอียงหรือสัดส่วนที่คลาดเคลื่อน ระบบจึงใช้ OpenCV เข้ามาจัดการปรับแก้ทัศนมิติ (Perspective Transformation) เพื่อทำการตัดขอบและดึงเฉพาะส่วนที่เป็นกระดาษคำตอบให้อยู่ในระนาบตรง ทำให้การคำนวณพิกัดของจุดฝนมีความแม่นยำสูงขึ้น

### 3. การตรวจจับและประเมินผล (Detection & Grading)
ระบบใช้เทคโนโลยี Deep Learning ผ่านโมเดล YOLO (You Only Look Once) เข้ามาช่วยในการตรวจจับจุดสำคัญบนหน้ากระดาษ จากนั้นจะใช้อัลกอริทึมทางด้าน Image Processing วิเคราะห์ความเข้มของพิกเซล (Pixel Intensity) ในแต่ละบริเวณตัวเลือก เพื่อพิจารณาว่านักเรียนฝนคำตอบในช่องใด และนำไปเปรียบเทียบกับฐานข้อมูลเฉลยเพื่อคำนวณคะแนนรวม

### 4. การสร้างรายงานและส่งออกข้อมูล (Reporting & Exporting)
ข้อมูลผลการสอบทั้งหมดจะถูกจัดเก็บลงในฐานข้อมูล SQLite3 และระบบสามารถทำการสร้างรายงานผลการสอบรายบุคคลในรูปแบบไฟล์ PDF โดยใช้ ReportLab รวมถึงรองรับการส่งออกข้อมูลภาพรวมในรูปแบบไฟล์ Excel/CSV ผ่านไลบรารี Pandas เพื่อให้นำไปวิเคราะห์ผลต่อได้ทันที

## เทคโนโลยีที่ใช้ในการพัฒนา
* **Backend Framework:** Python, Django
* **Computer Vision & AI:** Ultralytics (YOLOv8), OpenCV, NumPy
* **Document Processing:** PyMuPDF, ReportLab
* **Data Manipulation:** Pandas
* **Database:** SQLite3
* **Deployment:** Docker, Hugging Face Spaces, WhiteNoise

## การดาวน์โหลดโปรแกรมสำหรับใช้งานแบบออฟไลน์ (Standalone Version)
สำหรับอาจารย์ผู้สอนหรือผู้ที่ต้องการนำระบบไปใช้งานในเครื่องคอมพิวเตอร์ส่วนตัว (Local Machine) โดยไม่ต้องพึ่งพาอินเทอร์เน็ต สามารถดาวน์โหลดไฟล์โปรแกรมสำเร็จรูปไปเปิดใช้งานได้ทันที

**ลิงก์ดาวน์โหลด (Google Drive):** [https://drive.google.com/file/d/1kH091QiSHVMACPrj19HSLKrPdh7GfjvF/view?usp=sharing]

**คำแนะนำในการใช้งาน:**
1. ดาวน์โหลดไฟล์บีบอัด (.zip) จากลิงก์ด้านบน
2. ทำการแตกไฟล์ (Extract) ไว้ในคอมพิวเตอร์ (แนะนำให้วางในโฟลเดอร์ที่ไม่ติดสิทธิ์การดูแลระบบ เช่น Desktop หรือ Documents)
3. เข้าไปในโฟลเดอร์ที่แตกไฟล์ และดับเบิลคลิกที่ไฟล์ **ExamGradingAI.exe**
4. โปรแกรมจะทำการรันเซิร์ฟเวอร์จำลองและเปิดหน้าเว็บบราวเซอร์เริ่มต้นเข้าสู่ระบบให้โดยอัตโนมัติภายใน 10-15 วินาที
