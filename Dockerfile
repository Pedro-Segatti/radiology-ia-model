FROM python:3.9

WORKDIR /app

COPY requirements.txt /app

RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . /app

ENV FLASK_APP=src/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_DEBUG=1
ENV FLASK_ENV=development
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["flask", "run"]