from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver 


GENDER = (
	("M", "Male"),
	("F", "Female"),
	("O", "Others")
)

BLOOD = (
	("A+", "A+ Type"),
	("A-", "A- Type"),
	("B+", "B+ Type"),
	("B-", "B- Type"),
	("AB+", "AB+ Type"),
	("AB+", "AB- Type"),
	("O+", "O+ Type"),
	("O-", "O- Type")

)

STATUS = (
	("P", "Pending"),
	("C", "Completed")
)



class UserCategory(models.Model):
	user_category = models.OneToOneField(User, on_delete=models.CASCADE, 
										related_name="user_category")
	is_patient = models.BooleanField(default=False)
	is_doctor = models.BooleanField(default=False)
	is_receptionist = models.BooleanField(default=False)
	is_hr = models.BooleanField(default=False)

	def __str__(self):
		return self.user_category.username

class Patient(models.Model):
	patient = models.OneToOneField(UserCategory, on_delete = models.CASCADE, 
									related_name="patient")
	phone = models.CharField(max_length=12, blank=True, null=True)
	gender = models.CharField(choices=GENDER, max_length=1, blank=True, null=True)
	age = models.IntegerField(blank=True, null=True)
	address = models.CharField(blank=True, null=True, max_length=200)
	blood_group = models.CharField(choices=BLOOD, max_length=4 , blank=True, null=True)

	def __str__(self):
		return self.patient.user_category.username

class Contact(models.Model):
	name = models.CharField(max_length=255)
	email = models.EmailField()
	subject = models.CharField(max_length=255)
	message = models.CharField(max_length=255)

class Doctor(models.Model):
	doctor = models.OneToOneField(UserCategory, on_delete = models.CASCADE, 
								  related_name="doctor")
	url = models.URLField(max_length=200)
	phone = models.CharField(max_length=12, blank=True, null=True)
	gender = models.CharField(choices=GENDER, max_length=1, blank=True, null=True)
	upi_id = models.CharField(max_length=255, default="None")
	city = models.CharField(max_length=255, default="NOT AVAIALBE")
	district= models.CharField(max_length=255, default="NOT AVAIALBE")
	state = models.CharField(max_length=255, default="NOT AVAIALBE")
	country = models.CharField(max_length=255, default="NOT AVAIALBE")
	address = models.CharField(blank=True, null=True, max_length=200)
	blood_group = models.CharField(choices=BLOOD, max_length=3, blank=True, null=True)
	experience = models.IntegerField(blank=True, null=True)
	age = models.IntegerField(blank=True, null=True)
	is_working = models.BooleanField(blank=True, null=True)
	department = models.CharField(blank=True, null=True, max_length=100)
	salary = models.IntegerField(blank=True, null=True)
	attendence = models.IntegerField(blank=True, null=True)

	def __str__(self):
		return self.doctor.user_category.username



class Appointment(models.Model):
	date = models.DateField(auto_now_add=True)
	time = models.TimeField(auto_now_add=True)
	patient = models.ForeignKey(Patient, on_delete=models.CASCADE, 
		                        related_name="patient_appointment")
	doctor = models.ForeignKey(Doctor, on_delete = models.CASCADE, null=True, 
		                       blank=True, related_name="doctor_appointment")
	status = models.CharField(default="p", max_length=1)
	payment = models.IntegerField(default=500, null=True)
	
	def __str__(self):
		return self.patient.patient.user_category.username

	def total_appointment(self):
		return self.objects.all().count()

	def  completed_appointment(self):
		return len(self.objects.all().filter(status="c"))

	def pending_appointment(self):
		return len(self.objects.all().filter(status="p"))			


class Prescription(models.Model):
	appoint = models.OneToOneField(Appointment, on_delete=models.CASCADE, 
		                           related_name="patient_appointment", blank=True, null=True)
	patient = models.ForeignKey(Patient, on_delete = models.CASCADE, 
		                        related_name="patient_prescription")
	doctor = models.ForeignKey(Doctor, on_delete = models.CASCADE, 
		                       related_name="doctor_prescription")
	problem = models.CharField(max_length=300)
	symptom = models.CharField(max_length=200)
	prescription = models.TextField()


	def __str__(self):
		return self.patient.patient.user_category.username



@receiver(post_save, sender=UserCategory)
def my_handler(sender, instance, created, **kwargs):
	if created:
		if instance.is_patient:
			Patient.objects.create(patient=instance)
		elif instance.is_doctor:
			Doctor.objects.create(doctor=instance)

