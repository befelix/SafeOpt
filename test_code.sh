#!/usr/bin/env bash

get_script_dir () {
     SOURCE="${BASH_SOURCE[0]}"
     # While $SOURCE is a symlink, resolve it
     while [ -h "$SOURCE" ]; do
          DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
          SOURCE="$( readlink "$SOURCE" )"
          # If $SOURCE was a relative symlink (so no "/" as prefix, need to resolve it relative to the symlink base directory
          [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
     done
     DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
     echo "$DIR"
}

# Change to script root
cd $(get_script_dir)

# Run style tests
echo "Running style tests"
flake8 safeopt --exclude test*.py,__init__.py --ignore=E402,W503

# Ignore import errors for __init__ and tests
flake8 safeopt --filename=__init__.py,test*.py --ignore=F,E402,W503

# Run unit tests
echo "Running unit tests"
nosetests --with-doctest safeopt

