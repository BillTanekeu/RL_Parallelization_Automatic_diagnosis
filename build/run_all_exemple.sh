#!/bin/bash
cmake -E remove CMakeCache.txt
cmake ../
make -j4

echo -e "\n Nombre de Threads : "
read th
echo -e "\n Accuracy pour la convergence (entre 0 et 1) : "
read acc

# init networks
#./app/main1 ../data 32 $acc -2

# run seqential version
   ./app/main1 ../data 1 $acc 0
   ./app/main1 ../data 1 $acc 0
   ./app/main1 ../data 1 $acc 0



run parallels versions


  for method in `seq 1 13 `
  do
    	for k in `seq 2 $th`
    	do 
    		let c=k
    		for i in `seq 3 `
    		do
    			./app/main1 ../data $c $acc $method
    		done
    	done
  done



