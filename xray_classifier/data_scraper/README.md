## MODEL TO SCAPE THE IMAGES NEEDED FOR IMAGE CLASSIFIER

**Prerequisites:** `LINUX SYSTEM`

## STEPS 
5. download the `scrape_js.py` file present in this repo
6. download the `scrape.bash` file present in this repo
7. download the `requirements.txt` file from this repo
8. create a virtual env
	```
	python3 -m venv env
	source env/bin/activate
	```
7. install the dependencies
	```
	pip install -r requirements.txt
	```
8. give  `execution` permission to the bash file
  ```
  chmod +x script.bash
  ```
10. run the `script.bash` file
	```
	./script.bash [LINK] # EXAMPLE   /script.bash https://news.google.com/topstories?hl=en-IN&gl=IN&ceid=IN:en
	```
