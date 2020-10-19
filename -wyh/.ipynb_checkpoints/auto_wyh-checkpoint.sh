#!/bin/bash
 
for file in ./cfg/TIMIT_CGS_wyh/*
do
	python run_exp.py $file
done
