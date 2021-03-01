## MODEL TO SCAPE THE IMAGES NEEDED FOR IMAGE CLASSIFIER

**Prerequisites:** `LINUX SYSTEM`

## STEPS 
1. download the `scrape_js.py` file present in this repo
2. download the `scrape.bash` file present in this repo
3. download the `requirements.txt` file from this repo
4. create a virtual env
	```
	python3 -m venv env
	source env/bin/activate
	```
5. install the dependencies
	```
	pip install -r requirements.txt
	```
6. give  `execution` permission to the bash file
  ```
  chmod +x script.bash
  ```
7. run the `script.bash` file

	```
	./script.bash [LINK] # EXAMPLE   /script.bash https://news.google.com/topstories?hl=en-IN&gl=IN&ceid=IN:en
	```
