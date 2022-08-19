#!/bin/bash


sudo kill -9 $(ps aux | grep 'Train-' | awk '{print $2}')
sudo kill -9 $(ps aux | grep 'Test-' | awk '{print $2}')
sudo kill -9 $(ps aux | grep 'Valid-' | awk '{print $2}')
sudo kill -9 $(ps aux | grep 'VectorSampledTask' | awk '{print $2}')
sudo kill -9 $(ps aux | grep 'thor-' | awk '{print $2}')
sudo kill -9 $(ps aux | grep 'python.*<defunct>' | awk '{print $2}')
sudo kill -9 $(ps aux | grep 'from multiprocessing.forkserver' | awk '{print $2}')


#kill -9 $(ps aux | grep 'ssh -NfL')