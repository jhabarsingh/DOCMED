import factory
import os
import functools
import random
from faker import Faker

fake = Faker()
DEBUG = True

def generate_username(*args):
    """ returns a random username """
    return fake.profile(fields=['username'])['username']

class Profile(object):
    """ Simple Profile model """
    def __init__(self, **kwargs):
        self.username = kwargs.get('username')

class ProfileFactory(factory.Factory):
    """ Generate random profile instances """
    class Meta:
        model = Profile
    username = factory.LazyAttribute(generate_username)

def exist():
    filename = "users.txt"
    return os.path.isfile(os.path.join(os.getcwd(), filename))

users = []

if exist():
    if DEBUG:
        os.system("rm users.txt")
    else:
        print("Exiting Because File users.txt already exist")
        exit()

print("File users.txt doesn't exit")

for i in range(20):
    obj = {}
    obj["username"] = ProfileFactory().username
    obj["password"] = fake.password()
    users.append(obj)
    details = {}

    a = functools.partial(random.randint, 0, 9)
    gen = lambda: "{}{}{}{}{}{}{}{}{}{}".format(a(), a(), a(), a(), a(), a(), a(), a(), a(), a(), a())
    details["phone"] = "91" + gen()
    gender = [
        "male",
        "female",
        "others",
    ]
    details["gender"] = random.choice(gender)
    # Patient details

    obj["details"] = details
    with open("users.txt", "a") as wf:
        wf.writelines("{\n")
        wf.writelines("    username:  " + str(obj["username"]) + ",\n")
        wf.writelines("    password:  " + str(obj["password"]) + ",\n")
        
        wf.writelines("    details:  {\n")
        for i in obj["details"]:
            wf.writelines("             phone: "+ obj["details"]["phone"] +",\n")
            wf.writelines("             gender: "+ obj["details"]["gender"] +",\n")
        wf.writelines("    }\n")
        wf.writelines("}\n")


