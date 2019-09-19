#!/bin/bash

cd /ghome/minsb/adapooling_car/
cd $1
/ghome/minsb/tools/caffe_ada/build/tools/caffe train --solver=./adapooling.solver $2 $3
