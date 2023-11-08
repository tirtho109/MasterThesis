# Use the official Jupyter Docker image as the base image
FROM jupyter/base-notebook

WORKDIR / C:\Users\tirth\Documents\Git Code\MasterThesis

# Install any additional Python dependencies (if needed)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the .ipynb file into the container
COPY ./example/math.ipynb /MasterThesis/

# Start the Jupyter Notebook server
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--NotebookApp.token="]

