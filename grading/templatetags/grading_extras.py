from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    # ฟังก์ชันช่วยดึงค่าจาก Dictionary ใน HTML
    # ตัวอย่าง: dictionary.get(str(key))
    val = dictionary.get(str(key))
    if val and isinstance(val, list) and len(val) > 0:
        return val[0] # ส่งค่าตัวแรกกลับไป เช่น 'a'
    return ""