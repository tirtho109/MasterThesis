# Docker image as the base image
FROM python:3.7

WORKDIR /SCIANN

# Install any additional Python dependencies (if needed)
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install jupyter

# Copy the .ipynb files into the container
COPY *.ipynb .
# Start the Jupyter Notebook server in the directory with the notebooks
WORKDIR /

# Copy the .ipynb file into the container
EXPOSE 8888

# Start the Jupyter Notebook server
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]