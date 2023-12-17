FROM python:3.7-slim-buster

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN pip install pandas
RUN pip install opencv-python
RUN pip install scikit-learn
RUN pip install flask

CMD ["python3", "app.py"]
