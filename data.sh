#!/bin/bash

for entry in `ls -d $1/*`; do
    cat $entry | grep -i "$2"
done