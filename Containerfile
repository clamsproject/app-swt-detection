FROM ghcr.io/clamsproject/clams-python-opencv4:1.0.9

# See https://github.com/orgs/clamsproject/packages?tab=packages&q=clams-python for more base images
# IF you want to automatically publish this image to the clamsproject organization, 
# 1. you should have generated this template without --no-github-actions flag
# 1. to add arm64 support, change relevant line in .github/workflows/container.yml 
#     * NOTE that a lots of software doesn't install/compile or run on arm64 architecture out of the box 
#     * make sure you locally test the compatibility of all software dependencies before using arm64 support 
# 1. use a git tag to trigger the github action. You need to use git tag to properly set app version anyway

################################################################################
# DO NOT EDIT THIS SECTION
ARG CLAMS_APP_VERSION
ENV CLAMS_APP_VERSION ${CLAMS_APP_VERSION}
################################################################################

################################################################################
# clams-python base images are based on debian distro
# install more system packages as needed using the apt manager
################################################################################

RUN apt-get update && apt-get install -y wget

################################################################################
# main app installation

RUN pip install --no-cache-dir torch==2.1.0
RUN pip install --no-cache-dir torchvision==0.16.0

# Getting the model at build time so we don't need to get it each time we start
# a container. This is also because without it I ran into "Connection reset by peer"
# errors once in a while.
RUN wget https://download.pytorch.org/models/vgg16-397923af.pth
RUN mkdir /root/.cache/torch /root/.cache/torch/hub /root/.cache/torch/hub/checkpoints
RUN mv vgg16-397923af.pth /root/.cache/torch/hub/checkpoints

WORKDIR /app

COPY . /app

# default command to run the CLAMS app in a production server 
CMD ["python3", "app.py", "--production"]
################################################################################
