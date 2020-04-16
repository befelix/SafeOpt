FROM continuumio/miniconda

# Install build essentials and clean up
RUN apt-get update --quiet \
  && apt-get install -y --no-install-recommends --quiet build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

ARG PYTHON="3.5"

# Update conda, install packages, and clean up
RUN conda update conda --yes --quiet \
  && conda create -n safeopt python=$PYTHON pip numpy scipy --yes --quiet \
  && conda clean --yes --all \
  && hash -r

# Source the anaconda environment
ENV PATH /opt/conda/envs/safeopt/bin:$PATH

# The following are useful for developtment, but not testing
# Get the requirements files (seperate from the main body)
#COPY requirements.txt requirements.dev.txt /code/

# Install requirements and clean up
#RUN pip --no-cache-dir install -r /code/requirements.txt \
#  && pip --no-cache-dir install -r /code/requirements.dev.txt \
#  && rm -rf /root/.cache

# Copy the main code
COPY . /code
RUN cd /code \
  && pip install --no-cache-dir -e . \
  && pip --no-cache-dir install -r /code/requirements.dev.txt \
  && rm -rf /root/.cache

WORKDIR /code

