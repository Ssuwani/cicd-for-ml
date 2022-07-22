FROM python:3.8

RUN pip install scikit-learn==1.1.0

ADD train.py .

ENTRYPOINT ["python"]
CMD [ "train.py" ]