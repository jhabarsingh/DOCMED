from django.shortcuts import render, redirect, reverse
from main.forms import UserForm, PatientForm, DoctorForm, UserProfileForm, UserProfileForm1, AppointmentForm, PrescriptionForm, DoctorForm1
from django.contrib import messages
from main.models import UserCategory, Patient, Doctor, Appointment, Prescription
from django.contrib.auth.decorators import login_required
from django.contrib.auth import  authenticate, login as auth_login , logout
from django.http import HttpResponse, HttpResponseRedirect
from main.decorators import is_valid_patient, is_valid_doctor, is_valid_receptionist, is_valid_hr, is_valid_patientOrhr
from time import  sleep
from main.utils import render_to_pdf
from django.views.generic import View
from django.contrib.auth.models import User

def home(request):
	return render(request, "card.html")


def contact(request):
	return render(request, "contact.html")


def register(request):
	form = UserForm(request.POST or None)
	if form.is_valid():
		username = form.cleaned_data.get("username")
		des = form.cleaned_data.get("designation")
		form.save()
		if des == "1":
			UserCategory.objects.create(user_category=form.instance, is_patient=True)
		elif des == "2":
			UserCategory.objects.create(user_category=form.instance, is_doctor=True)

		messages.success(request, f"Account created for {username}")
		return redirect("main:home")
	return render(request, "register.html", {"form": form})




@login_required
def user_logout(request):
	return HttpResponseRedirect(reverse("main:confirm_logout"))


def user_login(request):
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
	
	if request.POST.get("ok" or None):
		logout(request)
		return HttpResponseRedirect(reverse("main:home"))
	if request.POST.get("cancel" or None):
		return HttpResponseRedirect(reverse("main:home"))
	return render(request, "confirm_logout.html")

@login_required
@is_valid_patient
def patient_profile(request):
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
	queryset = request.user.user_category.patient.patient_appointment.all()
	return render(request, "patient_appointment.html", {"appointments": queryset})

@login_required
@is_valid_patient
def patient_payment(request):
	queryset = request.user.user_category.patient.patient_appointment.all()
	return render(request, "patient_payment.html", {"appointments": queryset})


@login_required
@is_valid_patient
def patient_medical(request):
	queryset = request.user.user_category.patient.patient_prescription.all()
	return render(request, "patient_medical.html", {"appointments": queryset})

@login_required
@is_valid_patientOrhr
def generatePDF(request, id, *args, **kwargs):
    data = Appointment.objects.get(id=id)
    pdf = render_to_pdf('invoice.html', {"data":data})
    return HttpResponse(pdf, content_type='application/pdf')




@login_required
@is_valid_doctor
def doctor_appointment(request):
	queryset = request.user.user_category.doctor.doctor_appointment.all()
	return render(request, "doctor_appointment.html", {"appointments": queryset})



@login_required
@is_valid_doctor
def doctor_profile(request):
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
	queryset = request.user.user_category.doctor.doctor_prescription.all()
	return render(request, "doctor_prescriptions.html", {"prescriptions":queryset})


@login_required
def doctor_prescription(request, id):
	if request.user.user_category.is_doctor:
		form = PrescriptionForm(request.POST or None, initial={	"doctor":request.user.user_category.doctor, 
																"patient": Appointment.objects.get(id=id).patient,
																"appoint":Appointment.objects.get(id=id)
																}
								)
		if form.is_valid():
			form.save()
			Appointment.objects.filter(id=id).update(status="c")
			messages.success(request, "Prescription Saved")
			return HttpResponseRedirect(reverse("main:doctor_appointment"))
	return render(request, "doctor_prescription.html", {"form":form})


@login_required
@is_valid_receptionist
def receptionist_dashboard(request):
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
	queryset = Appointment.objects.all()

	return render(request, "accounting.html", {"appointments": queryset})