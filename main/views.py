from django.shortcuts import render, redirect, reverse
from main.forms import UserForm, PatientForm, DoctorForm, UserProfileForm, UserProfileForm1, AppointmentForm, PrescriptionForm, DoctorForm1
from django.contrib import messages
from main.models import UserCategory, Patient, Doctor, Appointment, Prescription
from django.contrib.auth.decorators import login_required
from django.contrib.auth import  authenticate, login as auth_login , logout
from django.http import HttpResponse, HttpResponseRedirect
from main.decorators import is_valid_patient, is_valid_doctor, is_valid_receptionist, is_valid_hr, is_valid_patientOrhr
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from time import  sleep
from main.utils import render_to_pdf
from django.views.generic import View
from django.contrib.auth.models import User
from .symptoms_production import predict_covid_from_symptoms
from .models import Contact
from django.core.paginator import Paginator
from django.db.models import Q
from django.core.files.storage import FileSystemStorage
from PIL import Image
from .production import predict
from .ctscan_production import predict1
import cv2
import os
import numpy
from datetime import datetime


def home(request):
	"""
	RENDERS HOME PAGE
	"""
	return render(request, "card.html")


""" Machine Learning Models
    ------------------------
	This section of the code will contain 
	all the views for the ML models being
	used in the project
"""

def ml_models(request):
	"""
	Entry Page for all the Ml Models;
	"""
	return render(request, "machine_learning/ml_models.html")

def covid_symptoms_detection(request):
	"""
	COVID DETECTION ML MODEL LOGIC PART
	"""
	params = [0 for i in range(23)]
	for i in range(1, 24, 1):
		if(request.POST.get(str(i))):
			params[i - 1] = 1
	age = request.POST.get("12")
	if age:	
		if(int(age) > 0):
			params[11] = 0
			params[11 + int(age)] = 1
	
	params[10] = 0
	gender = request.POST.get("17")
	if gender:
		if(int(gender) > 0):
			params[17] = 0
			params[17 + int(gender)] = 1
	pred = predict_covid_from_symptoms(params)
	if(request.POST):
		p = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
		counter = 0
		for i in range(23):
			if(p[i] != params[i]):
				break
			counter +=1
		if(counter == 23):
			pred[0] = 0
		return redirect("main:covid_symptoms_result", pred[0])
	return render(request, "machine_learning/covid_symptoms_detection.html", context={"msg" : pred[0]}) 


def covid_symptoms_result(request, *args, **kwargs):
	"""
	RENDERS PAGE TO DISPLAY COVID RESULT
	"""
	data = int(kwargs.get("result")) * 10
	return render(request, "machine_learning/covid_symptoms_result.html", {"result" : data})


def joiner(folder_name, file_name):
	paths = os.path.dirname(os.path.abspath(__file__))
	paths = os.path.dirname(paths)
	paths = os.path.join(paths, folder_name)
	paths = os.path.join(paths, file_name)
	return paths


@csrf_exempt
def covid_xray_prediction(request, *args, **kwargs):
	"""
	Covid Detectoin From Xray Report
	"""
	if request.method == 'POST' and request.FILES['file']:
		image = request.FILES['file']
		img1 = Image.open(image)
		fe = img1.format
		now = datetime.now()
		timestamp = datetime.timestamp(now)
		img1.save(joiner('media', f'{timestamp}.{fe}'))
		
		img1 = joiner('media', f'{timestamp}.{fe}')
		hasCovid = predict(img1)
		if hasCovid == 'others':
			return render(request, "machine_learning/covid_xray_prediction.html", {
				'data': 'true'
			})
		return redirect("main:covid_xray_result", hasCovid)
	return render(request, "machine_learning/covid_xray_prediction.html", {
		'data': 'false'
	})


def covid_ctscan_result(request, *args, **kwargs):
	"""
	RENDERS PAGE TO DISPLAY COVID RESULT
	"""
	data = kwargs.get("result")
	return render(request, "machine_learning/covid_ctscan_result.html", {"result" : data})


@csrf_exempt
def covid_ctscan_prediction(request, *args, **kwargs):
	"""
	Covid Detectoin From ctscan Report
	"""
	if request.method == 'POST' and request.FILES['file']:
		image = request.FILES['file']
		img1 = Image.open(image)
		fe = img1.format
		now = datetime.now()
		timestamp = datetime.timestamp(now)
		img1.save(joiner('media', f'{timestamp}.{fe}'))
		
		img1 = joiner('media', f'{timestamp}.{fe}')
		hasCovid = predict1(img1)
		if hasCovid == 'others':
			return render(request, "machine_learning/covid_ctscan_prediction.html", {
				'data': 'true'
			})
		return redirect("main:covid_ctscan_result", hasCovid)
	return render(request, "machine_learning/covid_ctscan_prediction.html", {
		'data': 'false'
	})


def covid_xray_result(request, *args, **kwargs):
	"""
	RENDERS PAGE TO DISPLAY COVID RESULT
	"""
	data = kwargs.get("result")
	return render(request, "machine_learning/covid_xray_result.html", {"result" : data})


""" Non Machine Learning Views
    ------------------------
	This section of the code will contain 
	all the views except ML models being
	used in the project
"""

def contact(request):
	"""
	PAGE TO RENDER CONTACT DETAILS 
	"""
	if request.POST:
		name = request.POST.get("name")
		email = request.POST.get("email")
		subject = request.POST.get("subject")
		message = request.POST.get("message")
		if(not len(name) or not len(email) or not len(subject) or not len(message)):
			pass
		else:
			Contact.objects.create(name=name, email=email, subject=subject, message=message)
	return render(request, "contact.html")


def register(request):
	"""
	REGISTRATION PAGE FOR PATIENTS
	"""
	if request.user.is_authenticated:
		return redirect("main:home")
	form = UserForm(request.POST or None)
	if form.is_valid():
		username = form.cleaned_data.get("username")
		# des = form.cleaned_data.get("designation")
		form.save()
		des = "1"
		if des == "1":
			UserCategory.objects.create(user_category=form.instance, is_patient=True)
		elif des == "2":
			UserCategory.objects.create(user_category=form.instance, is_doctor=True)

		messages.success(request, f"Account created for {username}")
		return redirect("main:home")
	return render(request, "register.html", {"form": form})




@login_required
def user_logout(request):
	"""
	RENDERS USER LOGOUT PAGE
	"""
	return HttpResponseRedirect(reverse("main:confirm_logout"))

@login_required
@is_valid_patient
def doctor_list(request):
	"""
	RENDERS LIST OF DOCTORS
	"""
	if request.user.is_authenticated:
		contact_list = Doctor.objects.all()
		search = request.GET.get("search")
		if(request.method == "POST"):
			doctor_babu = request.POST.get("id")
			return redirect("main:message", doctor=doctor_babu)
		if(search and search.lower() != "none"):
			contact_list = Doctor.objects.filter(Q(city__icontains=search) | 
							     Q(district__icontains=search) | 
							     Q(state__icontains=search)
						)
		paginator = Paginator(contact_list, 5)
		page_number = request.GET.get('page')
		page_obj = paginator.get_page(page_number)
		return render(request, "doctor_list.html", context={"page_obj": page_obj, "search": search})
	return render(request, "main:home")

@login_required
@is_valid_patient
def message(request, *args, **kwargs):
	"""
	RENDERS FORM THAT NEEDS TO BE FILLED BY PATIENT BEFORE CHECKUP
	"""
	if request.user.is_authenticated:
		if(request.method == "POST"):
			problem = request.POST.get("problem")
			symptoms = request.POST.get("symtoms")
			patient = request.user.user_category.patient
			doctor = User.objects.filter(username=kwargs.get("doctor")).first().user_category.doctor

			appoint = Appointment.objects.create(patient=patient, doctor=doctor)
			Prescription.objects.create(appoint=appoint, patient=patient, doctor=doctor, problem=problem, symptom=symptoms)
			return redirect("main:patient_appointment")
		data = {
			"doctor" : kwargs.get("doctor")
		}
		return render(request, "message.html", context=data)
	return render(request, "main:home")



def user_login(request):
	"""
	RENDERS USER/PATIENT LOGIN FORM
	"""
	if request.user.is_authenticated:
		return redirect("main:home")
	if request.method == "POST":
		username = request.POST.get("username")
		password = request.POST.get("password")

		user = authenticate(request, username=username,
							password = password)
		if user:
			if user.is_active:
				auth_login(request, user)	
				return HttpResponseRedirect(reverse("main:home"))
			else:
				messages.error(request, "Invalid Credentials")			
		else:
				messages.error(request, "Invalid Credentials")
	return render(request, "login.html", {})

@login_required
def confirm_logout(request):
	"""
	RENDERS LOGOUT CONFIRM PAGE
	"""
	if request.POST.get("ok" or None):
		logout(request)
		return HttpResponseRedirect(reverse("main:home"))
	if request.POST.get("cancel" or None):
		return HttpResponseRedirect(reverse("main:home"))
	return render(request, "confirm_logout.html")

@login_required
@is_valid_patient
def patient_profile(request):
	"""
	RENDERS PATIENT PROFILE TO BE UPDATED
	"""
	user_form = UserProfileForm(request.POST or None, instance=request.user)
	patient_form = PatientForm(request.POST or None, instance=request.user.user_category.patient)
	if patient_form.is_valid() and user_form.is_valid():
		user_form.save()
		patient_form.save()
		messages.success(request, "successfully updated")
		sleep(1)
		return HttpResponseRedirect(reverse("main:home"))
	else:
		pass
	return render(request, "patient_form.html", {"user_form":user_form, "patient_form":patient_form})



@login_required
@is_valid_patient
def patient_appointment(request):
	"""
	RENDERS PATIENT APPOINTMENT FORM
	"""
	queryset = request.user.user_category.patient.patient_appointment.all()
	return render(request, "patient_appointment.html", {"appointments": queryset})

@login_required
@is_valid_patient
def patient_payment(request):
	"""
	RENDERS PATIENT PAYMENT FORM
	"""
	queryset = request.user.user_category.patient.patient_appointment.all()
	return render(request, "patient_payment.html", {"appointments": queryset})


@login_required
@is_valid_patient
def patient_medical(request):
	"""
	RENDERS PATIENT PRVIOUS APPOINTMENTS
	"""
	queryset = request.user.user_category.patient.patient_prescription.all()
	return render(request, "patient_medical.html", {"appointments": queryset})

@login_required
@is_valid_patientOrhr
def generatePDF(request, id, *args, **kwargs):
	"""
	GENERATE INVOICE PDF
	"""
	data = Appointment.objects.get(id=id)
	pdf = render_to_pdf('invoice.html', {"data":data})
	return HttpResponse(pdf, content_type='application/pdf')




@login_required
@is_valid_doctor
def doctor_appointment(request):
	"""
	RENDERS DOCTOR APPOINTMENT PAGE
	"""
	queryset = request.user.user_category.doctor.doctor_appointment.all()
	return render(request, "doctor_appointment.html", {"appointments": queryset})



@login_required
@is_valid_doctor
def doctor_profile(request):
	"""
	RENDERS DOCTOR PROFILE PAGE TO BE UPDATED
	"""
	user_form = UserProfileForm(request.POST or None, instance=request.user)
	doctor_form = DoctorForm(request.POST or None, instance=request.user.user_category.doctor)
	if doctor_form.is_valid() and user_form.is_valid():
		user_form.save()
		doctor_form.save()
		messages.success(request, "successfully updated")
		sleep(1)
		return HttpResponseRedirect(reverse("main:home"))
	else:
		pass
	return render(request, "doctor_form.html", {"user_form":user_form, "doctor_form":doctor_form})




@login_required
@is_valid_doctor
def doctor_prescriptions(request):
	"""
	RENDERS PRESCRIPTIONS GIVEN BY THE DOCTOR
	"""
	queryset = request.user.user_category.doctor.doctor_prescription.all()
	return render(request, "doctor_prescriptions.html", {"prescriptions":queryset})


@login_required
def doctor_prescription(request, id):
	"""
	RENDERS PRESCRIPTIONS GIVEN BY THE DOCTOR DETAILS
	"""
	if request.user.user_category.is_doctor:
		problem = Prescription.objects.filter(appoint=Appointment.objects.filter(id=id).first()).first().problem
		symptom = Prescription.objects.filter(appoint=Appointment.objects.filter(id=id).first()).first().symptom
		form = PrescriptionForm(request.POST or None, initial={	
			"doctor":request.user.user_category.doctor, 
			"patient": Appointment.objects.get(id=id).patient,
			"problem":problem, "symptom":symptom,
		})
		if form.is_valid():
			# form.save()
			Appointment.objects.filter(id=id).update(status="c")
			app = Appointment.objects.all().filter(id=id).first()
			pres = form.cleaned_data.get("prescription")
			Prescription.objects.filter(appoint=app).update(prescription=pres)
			messages.success(request, "Prescription Saved")
			return HttpResponseRedirect(reverse("main:doctor_appointment"))
	return render(request, "doctor_prescription.html", {"form":form})


@login_required
@is_valid_receptionist
def receptionist_dashboard(request):
	"""
	RENDERS RECEPTIONIST DASHBOARD PAGE
	"""
	queryset = Appointment.objects.all().order_by("-time")[0:5]
	a = Appointment.objects.all().count
	b = len(Appointment.objects.all().filter(status="c"))
	c = len(Appointment.objects.all().filter(status="p"))
	patient = Patient.objects.all()[0:5]
	context = {
		"appointments": queryset,
		"patient": patient,
		"a": a,
		"b": b,
		"c": c
	}
	return render(request, "receptionist_dashboard.html", context)


@login_required
@is_valid_receptionist
def create_appointments(request):
	form = AppointmentForm(request.POST or None)
	if form.is_valid():
		form.save()
		messages.success(request, "successfully created")
		return HttpResponseRedirect(reverse("main:receptionist_dashboard"))
	else:
		pass
	return render(request, "create_appointments.html", {"form": form})



@login_required
def confirm_deletes(request, id):
	if request.user.user_category.is_receptionist:
		if request.POST.get("ok" or None):
			data = User.objects.get(id=id)
			data.delete()
			return HttpResponseRedirect(reverse("main:receptionist_dashboard"))
		if request.POST.get("cancel" or None):
			return HttpResponseRedirect(reverse("main:receptionist_dashboard"))
		return render(request, "confirm_delete.html")
	else:				
		return HttpResponseRedirect(reverse("main:home"))


@login_required
def patient_profiles(request, id, *args, **kwargs):
	"""
	RENDERS PATIENT PROFILE NEEDS TO BE UPDATED
	"""
	if request.user.user_category.is_receptionist:
		user_form = UserProfileForm1(request.POST or None, instance=Patient.objects.get(id=id).patient.user_category)
		patient_form = PatientForm(request.POST or None, instance=Patient.objects.get(id=id))
		if patient_form.is_valid() and user_form.is_valid():
			user_form.save()
			patient_form.save()
			messages.success(request, "successfully updated")
			sleep(1)
			return HttpResponseRedirect(reverse("main:receptionist_dashboard"))
		return render(request, "patient_form.html", {"user_form":user_form, "patient_form":patient_form})
	else:
		return render(request, "receptionist_dashboard.html")
	


@login_required
@is_valid_hr
def hr_dashboard(request):
	"""
	RENDERS HR DASHBOARD
	"""
	queryset = 	Doctor.objects.all()
	a = Appointment.objects.all().count
	b = len(Doctor.objects.all().filter(is_working=True))
	c = len(Doctor.objects.all().filter(is_working=False))
	patient = User.objects.all()[0:5]
	context = {
		"doctor": queryset,
		"a": a,
		"b": b,
		"c": c
	}
	return render(request, "hr_dashboard.html", context)



@login_required
def confirm_deletes_hr(request, id):
	"""
	RENDERS CONFORM DELETE HR PAGE
	"""
	if request.user.user_category.is_hr:
		data = User.objects.get(id=id)
		if request.POST.get("ok" or None):
			data.delete()
			return HttpResponseRedirect(reverse("main:hr_dashboard"))
		elif request.POST.get("cancel" or None):
			return HttpResponseRedirect(reverse("main:hr_dashboard"))

		return render(request, "confirm_delete.html")			


@login_required
def doctor_profile_hr(request, id, *args, **kwargs):
	"""
	RENDERS DOCTOR PROFILE THAT CAN BE UPDATED BY HR ONLY
	"""
	if request.user.user_category.is_hr:
		user_form = UserProfileForm1(request.POST or None, instance=User.objects.get(id=id))
		doctor_form = DoctorForm1(request.POST or None, instance=User.objects.get(id=id).user_category.doctor)
		if doctor_form.is_valid() and user_form.is_valid():
			user_form.save()
			doctor_form.save()
			messages.success(request, "successfully updated")
			sleep(1)
			return HttpResponseRedirect(reverse("main:hr_dashboard"))
		return render(request, "doctor_form.html", {"user_form":user_form, "doctor_form":doctor_form})
	else:
		return render(request, "hr_dashboard.html")
	

@login_required
@is_valid_hr
def accounting(request):
	"""
	RENDERS ACCOUNTING PAGE HANDELED BY HR
	"""
	queryset = Appointment.objects.all()
	return render(request, "accounting.html", {"appointments": queryset})
