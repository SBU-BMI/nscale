#!/bin/bash

#
# Assumes you have Docker installed on your system, and wish to build
# your project in a Docker container
#

echo
echo "We'll build and start a Docker container, and execute it for you"
echo "interactively..."
echo

PROGNAME=$(basename "$0")
if [[ $# -lt 1 ]] ; then
  echo "Need project_name parameter. Please start again."
  echo 'usage: ${PROGNAME} project_name'
  echo
  echo 'example:'
  echo './build-start-docker-container.sh nscale'
  echo
else
  docker build -t $USER/$1 .
  docker run --name $USER-$1 -it -d $USER/$1 /bin/bash
  containerId=$(docker inspect --format '{{ .Id }}' $USER-$1)
  docker exec -it $containerId /bin/bash
fi
