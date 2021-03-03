from django.urls import path, re_path
from main import views

app_name="main"

urlpatterns = [
    path("",views.home,  name="home"),
  
    path("message/<slug:doctor>/", views.message, 
          name="message"),
    path("doctor_list/", views.doctor_list, 
          name="doctor_list"),
    path("contact/", views.contact, name="contact"),
    path("register/", views.register, name="register"),
    path("logout/", views.user_logout, name="logout"),
    path("login/", views.user_login, name="login"),
    path("confirm_logout/", views.confirm_logout, 
          name="confirm_logout"),
    path("patient_profile/", views.patient_profile, 
          name="patient_profile"),
    path("patient_profile/<int:id>/", views.patient_profiles, 
          name="patient_profiles"),
    path("patient_appointment/", views.patient_appointment, 
          name="patient_appointment"),
    path("patient_payment/", views.patient_payment, 
          name="patient_payment"),
   	path("patient_medical/", views.patient_medical, 
          name="patient_medical"),
   	path("invoice/<int:id>/", views.generatePDF, 
          name="invoice"),

   	path("doctor_appointment/", views.doctor_appointment, 
          name="doctor_appointment"),
   	path("doctor_profile/", views.doctor_profile, 
          name="doctor_profile"),
   	path("doctor_prescriptions", views.doctor_prescriptions, 
          name="doctor_prescriptions"),
   	path("doctor_prescription/<int:id>", views.doctor_prescription, 
          name="doctor_prescription"),

   	path("receptionist_dashboard/", views.receptionist_dashboard, 
          name="receptionist_dashboard"),
   	path("create_appointments/", views.create_appointments, 
          name="create_appointments"),
    path("confirm_deletes/<int:id>", views.confirm_deletes, 
          name="confirm_deletes"),

    path("hr_dashboard", views.hr_dashboard, name="hr_dashboard"),
    path("doctor_profile_hr/<int:id>/", views.doctor_profile_hr, 
          name="doctor_profile_hr"),
    path("accounting", views.accounting, name="accounting"),   
    path("confirm_deletes_hr/<int:id>", views.confirm_deletes_hr, 
          name="confirm_deletes_hr"),
]

ml = [
    path("ml_models/", views.ml_models, 
          name="ml_models"),
    path("covid_symtoms_detection/", views.covid_symptoms_detection, 
          name="covid_symptoms_detection"),
    path("covid_symptoms_result/<int:result>/", views.covid_symptoms_result, 
          name="covid_symptoms_result"),
    path("covid_xray_prediction", views.covid_xray_prediction, name="covid_xray_prediction"),
    path("covid_xray_result/<slug:result>/", views.covid_xray_result, 
          name="covid_xray_result"),
    path("covid_ctscan_prediction", views.covid_ctscan_prediction, name="covid_ctscan_prediction"),
    path("covid_ctscan_result/<slug:result>/", views.covid_ctscan_result, 
          name="covid_ctscan_result"),
]


urlpatterns += ml
