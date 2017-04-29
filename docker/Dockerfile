# Centos
FROM sbubmi/segmentation_base:latest

# RUN yum -y update

# Install-directories
ENV SRC_DIR=/tmp/nscale
ENV BLD_DIR=$SRC_DIR/build

RUN mkdir -p $SRC_DIR
WORKDIR $SRC_DIR
COPY . $SRC_DIR

# Build
RUN mkdir $BLD_DIR && cd $WORKDIR
WORKDIR $BLD_DIR

#RUN cmake $SRC_DIR -DOpenCV_DIR=/usr/local/share/OpenCV -DNS_FEATURES=ON -DNS_NORMALIZATION=ON -DNS_SEGMENT=ON

# Do we want to build examples and tests?
RUN cmake $SRC_DIR -DOpenCV_DIR=/usr/local/share/OpenCV -DNS_FEATURES=ON \
-DNS_NORMALIZATION=ON -DNS_SEGMENT=ON -DBUILD_EXAMPLE_APPLICATIONS=ON \
-DBUILD_TEST_APPLICATIONS=ON

RUN make -j4

WORKDIR /tmp

CMD ["/bin/bash"]
