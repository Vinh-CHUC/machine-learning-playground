FROM pytorch/pytorch:latest

RUN mkdir -p /opt/ml/model
RUN chmod 777 /opt/ml/model

ADD . .

ENV PYTHONUNBUFFERED=TRUE
ENTRYPOINT ["python", "src/vinh_playground/MNIST/cnn.py"]
