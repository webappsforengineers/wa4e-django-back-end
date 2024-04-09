FROM python:3.9

WORKDIR /

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080

EXPOSE 8080

CMD ["python", "manage.py", "runserver", "0.0.0.0:8080"]
