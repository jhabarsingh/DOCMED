from faker import Faker
import factory
import random
import json

fake = Faker()

patients_names = []

num_of_patients = 1000
gender = ["male", "female", "other"]
blood = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

def generate_number():
	size = 10
	number = ""
	country_code = "+91-"

	for i in range(size):
		if(i == 0):
			number+= str(random.randint(1, 9))
		else:
			number += str(random.randint(0, 9))
	return country_code + number;

for i in range(num_of_patients):
	patient = {}
	patient["username"] = "_".join(fake.name().split())
	patient["password"] =  "patient@123"
	details = {}
	details["phone"] = generate_number() 
	details["gender"] = gender[random.randint(0, 2)]
	details["age"] = random.randint(1, 110)
	details["address"] = fake.address().replace("\n", " ")
	details["blood group"] = blood[random.randint(0, 7)]
	patient["details"] = details
	patients_names.append(patient)


patients = json.dumps(patients_names, indent=2)
print(patients)
	
