# Hospital_Management
### Features
* Doctor Database and its login( doctor’s interface)
* Patient Database and its login(Patient’s interface)
* Prescription Management
* Appointment Management
* Human Resource Database and its login(HR interface)
* Hospital Accounting
* Financial Reporting
* Invoicing
* Patient Medical History
* Patient Medical Files Archive
* Prescription
* Patient Payment History Receptionist Database and its login(Receptionist interface)


#### Run On Localhost 

###### Step 1
```bash
 mkdir hospital_management 
 git clone https://github.com/jhabarsingh/Hospital_Management.git 
```
---
###### Step 2
* Open The File in Text-Editor(VS CODE, SUBLIME)
---

###### Step 3
```python 
 pip install -r requirement.txt #write the command in the hospital_management folder
```

###### Step 4
```python 
 python manage.py makemigrations
 python manage.py migrate
 python manage.py runserver 8000
```
###### Login Credentials

* Admin
   > * username pyjac
   > * password 9592864914
 
* HR  
 > * username Bheru
 > * password 9592864914

* Doctor  
 > * username anurag
 > * password 9592864914

* HR  
 > * username Babu
 > * password 9592864914

* Receptionist
 > * username mender
 > * password 9592864914

###### Login As Admin And create Hr, Receptionist Account
* Go to url localhost:8000/admin/
* fill the above admin credentials
* In admin You can register Hr and Receptionist
> * Go to User Column and create a new user
> * Go to user_container colums 
> * assign user_container as the above user
> * mark is_hr as true user is **HR**
> * mark is_receptionist as true user is **RECEPTIONIST**


###### create Doctor and Patient Account
 * Go to register link fill the details
 * There You can choose wheather the user is Doctor or Patientt
 * Once registeres you can login using the credentials
#### Video Link
[Vide](https://drive.google.com/file/d/1EBM0fIkShnjhjss7qcY2Xc84tgdfsL84/view?usp=drivesdk)

#### Website Link
[Website](http://pyjac.pythonanywhere.com/)
# Covide_assistant
