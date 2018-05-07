FROM pytorch/pytorch:latest

ADD . .

ENV PYTHONUNBUFFERED=TRUE
ENTRYPOINT ["python", "src/vinh_playground/MNIST/cnn.py"]
