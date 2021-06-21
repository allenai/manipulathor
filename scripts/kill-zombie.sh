#!/bin/bash


kill -9 $(ps aux | grep 'Train-' | awk '{print $2}')
kill -9 $(ps aux | grep 'thor-' | awk '{print $2}')
kill -9 $(ps aux | grep 'python.*<defunct>' | awk '{print $2}')