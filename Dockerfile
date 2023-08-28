FROM python:3.9
# COPY . .
# RUN apt-get update
COPY . .
RUN apt-get update
# # set work directory
WORKDIR /usr/src/appnoise

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1 
#Prevents Python from writing pyc files to disc (equivalent to python -B option)
ENV PYTHONUNBUFFERED 1
#Prevents Python from buffering stdout and stderr (equivalent to python -u option)

# install dependencies
# RUN pip install psycopg2 fastapi[all]  
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# copy project
