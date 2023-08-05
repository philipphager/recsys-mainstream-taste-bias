#!/bin/bash

python main.py -m \
    data=faces \
    model=ease,funk,knn-item-item,knn-user-user,nmf
