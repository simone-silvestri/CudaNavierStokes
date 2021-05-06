#!/bin/bash                                                                                                                                                                                                


i=$1
while [[ $i -le $2 ]]; do
     num1=`printf "%04i" $i`
     num2=`printf "%07i" $i`
     FILE=e.$num2.bin
     if test -f "$FILE"; then
        echo "$FILE exists."
        cp field.NUMBER.xmf field.$num1.xmf
        cp video_NUMBER.py  video_$num1.py

        sed -i -e "s/NUMBER/$num2/g" field.$num1.xmf
        sed -i -e "s/NUMBER/$num1/g" video_$num1.py

        ~/ParaView-5.8.1-MPI-Linux-Python2.7-64bit/bin/pvbatch video_$num1.py

        rm field.$num1.xmf
        rm video_$num1.py

        echo "done image number $num1"
        ((i=i+1))
     else
        echo "file does not exist yet"
        sleep 1m
     fi  
done

