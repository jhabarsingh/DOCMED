# LUNG SEGMENTATION
Lung segmentation is often the first step to increase the performance of any other supervised model, as it takes away the noise created by regions of no interest for COVID-19

## Steps 
1. Download the data from [LINK](https://www.kaggle.com/andrewmvd/covid19-ct-scans/code)
2. create a folder `input`
3. create a folder `covid19-ct-scans` inside `input`
4. extract the data inside the `covid19-ct-scans`
5. download the `script.py` file present in this repo
6. download the `requirements.txt` file from this repo
6. create a virtual env
	```
	python3 -m venv env
	source env/bin/activate
	```
7. install the dependencies
	```
	pip install -r requirements.txt
	```
8. run the `script.py` file
	```
	python script.py
	```
