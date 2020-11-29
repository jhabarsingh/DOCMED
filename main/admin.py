from django.contrib import admin
from main.models import UserCategory, Patient, Doctor, Appointment, Prescription, Contact
# Register your models here.


admin.site.register(UserCategory)
admin.site.register(Patient)
admin.site.register(Contact)
admin.site.register(Doctor)
admin.site.register(Appointment)
admin.site.register(Prescription)
