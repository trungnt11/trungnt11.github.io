#!/bin/bash

##
# file: _commit_and_deploy.sh
# usage: ./_commit_and_deploy.sh <Commit message>
#
# This script builds the jekyll site, commits the output of
# _site to the master branch and pushes it to the master branch
# on github.
 
if [[ -z "$1" ]]; then
  echo "Please enter a git commit message"
  exit 1
fi
 
jekyll build && \
  cd _site && \
  git add . && \
  git commit -am "$1" && \
  git push origin master && \
  cd .. && \
  echo "Successfully built and pushed to GitHub."
