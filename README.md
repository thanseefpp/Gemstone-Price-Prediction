<img width="1440" alt="Screenshot 2023-08-06 at 6 03 30 PM" src="https://github.com/thanseefpp/Gemstone-Price-Prediction/assets/62167887/8ff4ef58-0772-4984-a4bc-05471c4019cb">

---
<h3 align="center">Gemstone Working Demo Streamlit</h3>
---

![gem](https://github.com/thanseefpp/Gemstone-Price-Prediction/assets/62167887/8bdb312a-4da0-4f26-99af-c156c4544060)


---
## Setup Environment

```
python3 -m venv env
```

## Activate Environment

```
source env/bin/activate
```

## 1 - Installation

```
pip install -r requirements.txt
```

## 2 - To Start Training 

<p>
Before running the pipeline make sure your pc is have at least 8GB Ram for faster execution
</p>

```
python3 run_pipeline.py
```

## 3 - To Run Application

```
streamlit run streamlit_app.py
```

## 4 - You can also use FASTAPI
Before Using the FastAPI

1 - You have to Install Docker on your machine
2 - before running any codes you have to make sure you created .env and done the setup
```
DATABASE_PORT=PORT_NUMBER
POSTGRES_PASSWORD=PASSWORD
POSTGRES_USER=USER_NAME
POSTGRES_DB=NAME_FOR_DB
POSTGRES_HOST=HOST_NAME
POSTGRES_HOSTNAME=HOST_ADDRESS
```
3 - Execute this command to start docker image for postgresql

```
docker-compose up -d
```
4 - Run the app.py File to get the api's

```
uvicorn app:app --reload
```

## Clean Your Code

<p>
    This will fix the indentation issues, whitespace and other basic formate
</p>

```
python3 clean_code.py
```

- For Pushing this code to your repo you have to install git lfs

```
git lfs install
```