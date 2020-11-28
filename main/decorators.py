from django.shortcuts import redirect, reverse
from django.http import HttpResponse, HttpResponseRedirect


def is_valid_patient(fun):
	def authorized(request, *args, **kwargs):
		if request.user.user_category.is_patient:
			return fun(request, *args, **kwargs)
		else:
			return HttpResponseRedirect(reverse("main:home"))

	return authorized


def is_valid_doctor(fun):
	def authorized(request, *args, **kwargs):
		if request.user.user_category.is_doctor:
			return fun(request, *args, **kwargs)
		else:
			return HttpResponseRedirect(reverse("main:home"))		

	return authorized


def is_valid_receptionist(fun):
	def authorized(request, *args, **kwargs):
		if request.user.user_category.is_receptionist:
			return fun(request, *args, **kwargs)
		else:
			return HttpResponseRedirect(reverse("main:home"))
	return authorized


def is_valid_hr(fun):
	def authorized(request, *args, **kwargs):
		if request.user.user_category.is_hr:
			return fun(request, *args, **kwargs)
		else:
			return HttpResponseRedirect(reverse("main:home"))
	return authorized


def is_valid_patientOrhr(fun):
	def authorized(request, *args, **kwargs):
		if request.user.user_category.is_hr  or request.user.user_category.is_patient:
			return fun(request, *args, **kwargs)
		else:
			return HttpResponseRedirect(reverse("main:home"))
	return authorized
