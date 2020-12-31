from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from main.models import Patient, Doctor, Appointment, Prescription

DETAIL = (
	("1", "Patient"),
	("2", "Doctor"),
)
class UserForm(UserCreationForm):
	# designation = forms.ChoiceField(choices = DETAIL)
	class Meta:
		model = User
		fields = ("first_name","last_name", "username", 
				  	"email", "password1", "password2", 
				  	# "designation"
				 )

class UserProfileForm(UserCreationForm):
	class Meta:
		model = User
		fields = ("first_name","last_name", "username", 
					"email", "password1", "password2"
				 )

class UserProfileForm1(forms.ModelForm):
	class Meta:
		model = User
		fields = ("first_name","last_name", "username")

class PatientForm(forms.ModelForm):
	class Meta:
		model = Patient
		fields = ("phone", "gender", "address", "age", 
				   "blood_group"
				 )

class DoctorForm(forms.ModelForm):
	class Meta:
		model = Doctor
		fields = ("doctor", "phone", "gender", "city", 
		            "district", "state", "address", 
		            "blood_group", "department", "experience"
		         )	

class DoctorForm1(forms.ModelForm):
	class Meta:
		model = Doctor
		fields = "__all__"

class AppointmentForm(forms.ModelForm):
	class Meta:
		model = Appointment
		fields = "__all__"

class PrescriptionForm(forms.ModelForm):
	class Meta:
		model = Prescription
		fields = ("patient","doctor", "problem", 
			        "symptom", "prescription"
			     )


