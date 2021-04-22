#!/bin/bash                                                                                                                                                                                                


i=0
while [[ $i -le 174 ]]; do
     num1=`printf "%04i" $i`
     num2=`printf "%07i" $i`

     cp field.NUMBER2.xmf field.$num1.xmf
     cp video_NUMBER2.py  video_$num1.py

     sed -i -e "s/NUMBER/$num2/g" field.$num1.xmf
     sed -i -e "s/NUMBER/$num1/g" video_$num1.py

     ~/ParaView-5.8.1-MPI-Linux-Python2.7-64bit/bin/pvbatch video_$num1.py

     rm field.$num1.xmf
     rm video_$num1.py
      
     echo "done image number $num1"
     ((i=i+1))
done


