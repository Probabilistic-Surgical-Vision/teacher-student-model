#!/bin/bash

find . -not \( \
    -path "./.git*" \
    -or -path "./.gradient*" \
    -or -path "./ensemble-dataset*" \
    -or -name "." \
    -or -name ".." \) | xargs rm -rf
