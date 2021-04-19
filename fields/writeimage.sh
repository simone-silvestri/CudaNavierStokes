#!/bin/bash                                                                                                                                                                                                


i=161
while [[ $i -le 202 ]]; do
     num1=`printf "%04i" $i`
     num2=`printf "%07i" $i`

     cp field.NUMBER.xmf field.$num1.xmf
     cp video_NUMBER.py  video_$num1.py

     sed -i -e "s/NUMBER/$num2/g" field.$num1.xmf
     sed -i -e "s/NUMBER/$num1/g" video_$num1.py

     ~/ParaView-5.8.1-MPI-Linux-Python2.7-64bit/bin/pvpython video_$num1.py

     rm field.$num1.xmf
     rm video_$num1.py
      
     echo "done image number $num1"
     ((i=i+1))
done


