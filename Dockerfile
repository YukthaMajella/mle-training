# Use a Miniconda base image
FROM continuumio/miniconda3:latest

# Set a working directory
WORKDIR /app
COPY ./ ./

# Create the conda environment from the env.yml file
COPY env.yml .
RUN conda env create -f env.yml
RUN echo "source activate mle-dev2" >> ~/.bashrc
ENV PATH=/opt/conda/envs/mle-dev2/bin:$PATH

# Expose the port that the MLFlow UI will run on
EXPOSE 5000

# Command to run MLFlow file
CMD ["sh", "-c", "python scripts/main_mlflow.py & mlflow ui --host 0.0.0.0 --port 5000"]
