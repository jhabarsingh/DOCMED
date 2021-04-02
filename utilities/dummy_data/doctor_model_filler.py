from faker import Faker
import factory
import random
import json
from indian_district import get_states
from specialists import get_specialists

states = get_states().get("states")

no_of_states = len(states)

specialists = get_specialists()

fake = Faker()

doctors_name = []

num_of_doctors = 1000
gender = ["male", "female", "other"]
blood = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

MALE = 99
FEMALE = 99

def generate_state_district():
	state = states[random.randint(0, no_of_states - 1)]
	
	no_of_districts = len(state.get("districts"))
	
	district = state.get("districts")[random.randint(0, no_of_districts - 1)]
	
	state = state.get("state")
	return (state, district)

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

def generate_profile(gender):
	male = f"https://randomuser.me/api/portraits/men/{random.randint(0, MALE)}.jpg"
	female = f"https://randomuser.me/api/portraits/women/{random.randint(0, FEMALE)}.jpg"
	if(gender == "male"):		
		return male
	elif(gender == "female"):
		return female
	return male if random.randint(0, 1) == 0 else female


for i in range(num_of_doctors):
	doctor = {}
	doctor["username"] = "_".join(fake.name().split())
	doctor["password"] =  "doctor@123"
	details = {}
	details["phone"] = generate_number() 
	details["gender"] = gender[random.randint(0, 2)]	
	details["profile_pic_url"] = generate_profile(details["gender"])
	details["age"] = random.randint(22, 110)
	details["state"], details["district"] = generate_state_district()
	details["city"] = details["district"]
	details["country"] = "India"
	details["address"] = fake.address().replace("\n", " ")
	details["blood group"] = blood[random.randint(0, 7)]
	details["experience"] = random.randint(1, 100)
	details["upi_id"] = doctor["username"] + "@_bhi"
	details["department"] = specialists[random.randint(0, len(specialists) - 1)]
	doctor["details"] = details
	doctors_name.append(doctor)


doctors = json.dumps(doctors_name, indent=2)
print(doctors)
	
