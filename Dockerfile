# Use the official Python image as the base image
FROM python:3.
WORKDIR /app


COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app
EXPOSE 8080

CMD ["python", "app.py"]