FROM python:3.9.18
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD streamlit run sentiment.py
RUN pip install streamlit
RUN pip install tensorflow
