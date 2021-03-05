# DOCMED  ⚡️ [![GitHub](https://img.shields.io/github/license/jhabarsingh/DOCMED?color=blue)](https://github.com/jhabarsingh/DOCMED/blob/master/LICENSE) [![GitHub stars](https://img.shields.io/github/stars/jhabarsingh/DOCMED)](https://github.com/jhabarsingh/DDOCMED/stargazers)  [![GitHub contributors](https://img.shields.io/github/contributors/jhabarsingh/DOCMED.svg)](https://github.com/jhabarsingh/DOCMED/graphs/contributors)  [![GitHub issues](https://img.shields.io/github/issues/jhabarsingh/DOCMED.svg)](https://github.com/jhabarsingh/DOCMED/issues) [![GitHub forks](https://img.shields.io/github/forks/jhabarsingh/DOCMED.svg?style=social&label=Fork)](https://GitHub.com/jhabarsingh/DOCMED/network/)

<p align="center">
  <img src="https://github.com/jhabarsingh/DOCMED/blob/main/docs/animations/docmed.png?raw=true" />
</p>
<details>
  <summary>:zap: TECH STACK</summary>
  <br/>
  <div style="display:flex;justify-content:space-around">
  <img titlt="Dialog Flow" src="https://pbs.twimg.com/profile_images/880147119528476672/S7C-2C6t.jpg" width="50px" height="50px"  style="margin-right:5px;"/>
  <img  title="Django" src="https://icon-library.com/images/django-icon/django-icon-0.jpg" width="50px" height="50px" style="margin-right:5px;" />
<!--   <img  title="Kommunicate" src="https://ps.w.org/kommunicate-live-chat/assets/icon-256x256.png?rev=2291443" height="50px"  style="margin-right:5px;"/> -->
  <img title="Heroku"  src="https://www.thedevcoach.co.uk/wp-content/uploads/2020/04/heroku.png" height="50px"  style="margin-right:5px;"/> 
  <img  title="Tensorflow" src="https://www.altoros.com/blog/wp-content/uploads/2016/01/tensorflow-logo-cropped.png" height="50px" style="margin-right:5px;" />
  <img  title="Postgresql" src="https://pbs.twimg.com/media/EGc7jg4XoAA0bez.png" height="50px" style="margin-right:5px;" />
  <img  title="Scikit Learn" src="https://www.analyticsvidhya.com/wp-content/uploads/2015/01/scikit-learn-logo.png" height="50px" style="margin-right:5px;" />
  <img title="Reactjs" src="https://icons-for-free.com/iconfiles/png/512/design+development+facebook+framework+mobile+react+icon-1320165723839064798.png" width="50px" height="50px"  style="margin-right:5px;"/>
  <img  title="Docker" src="https://pbs.twimg.com/profile_images/1273307847103635465/lfVWBmiW_400x400.png" height="50px" style="margin-right:5px;" />
</div>
</details>

## Abstract
To improve the conventional diagnostic procedures as they are prone to human
mistakes and these are slow, expensive and not equally accessible to everyone therefore 
we developed an efficient ML model for predicting the possibility of various
diseases like covid, viral fever, dengue etc and integrate it with an interactive web
based dashboard which will also provide some additional insights and
recommendations over the user’s medical data.
[Read More](https://docs.google.com/document/d/1q19CVPYDygCHwYQ6YYb1oWLqrlC6ymcc14U_EjeX64w/edit?usp=sharing)


![HOME PAGE](https://github.com/jhabarsingh/Covid-Assistant/blob/main/docs/animations/chatbot.gif)

## Features
Features Provided By the **DOCMED**
  1. CHATBOT **COVAT** TO RESOLVE YOUR QUERIES
  2. ML MODEL TO PREDICT COVID FROM **SYMTOMS**
  3. ML MODEL TO PREDICT COVID FROM [**CHEST XRAY REPORT**](https://github.com/jhabarsingh/XRAY-COVID-PREDICTION)
  4. ML MODEL TO PREDICT COVID FROM **CHEST CTSCAN REPORT**
  6. CONSULT WITH A **DOCTORS** IN YOUR CITY
  7. [**TRACK COVID CASES**](https://github.com/jhabarsingh/COTRACK) WORLD WIDE OR COUNTRY WISE

![MACHINE LEARNING](https://github.com/jhabarsingh/DOCMED/blob/main/docs/animations/ml_new.gif)

## Project Setup

### Using venv
```bash
git clone https://github.com/jhabarsingh/DOCMED.git  
cd DOCMED
python3 -m venv env # Python 3.6.9 or 3.7.0 version 
source env/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
python manage.py runserver
```

### Using conda
```bash
git clone https://github.com/jhabarsingh/DOCMED.git  
cd DOCMED
conda create -n docmed python==3.7 
conda activate docmed
python3 -m pip install --upgrade pip
pip install -r requirements.txt
python manage.py runserver
```

## [Want To Contribute](https://medium.com/mindsdb/contributing-to-an-open-source-project-how-to-get-started-6ba812301738)
### You can contribute to this project in many ways
 1. You can create an issue if you find any bug.
 2. You can work on an existing issue and Send PR.
 3. You can make changes in the design if it is needed.
 4. Even if you find any grammatical or spelling mistakes then also you can create an issue.

> *I would be glad to see a notification saying `User {xyz} created a Pull Request`.
I promise to review it.*
