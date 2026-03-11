from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    # 🔥 เพิ่มบรรทัดนี้: เช็คว่าถ้าไม่ใช่ก้อนข้อมูล Dictionary (เช่น เป็นค่าว่างในหน้า Create) ให้ข้ามไปเลย ไม่ต้อง Error
    if not isinstance(dictionary, dict):
        return ""
    
    val = dictionary.get(str(key), "")
    
    # ดักไว้เผื่อกรณีข้อมูลเซฟเป็น List เช่น ['a'] ให้ดึงตัวแรกออกมา
    if isinstance(val, list) and len(val) > 0:
        return val[0]
        
    return val