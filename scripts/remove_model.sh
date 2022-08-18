#!/bin/bash

find . -not \( \
    -path "./.git*" 
    -or -path "./.gradient*" 
    -or -name "." 
    -or -name ".." \) | xargs rm -rf
