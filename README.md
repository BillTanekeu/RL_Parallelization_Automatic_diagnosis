## Environment

The code was developed and tested on Ubuntu 22.04, g++ version 11.4.0. 

Requirements:

* C++ compiler with C++14 support
* cmake



### Run the Algorithmes 

```bash

mkdir build && cd build && cmake ../
make -j4
./app/main1 ../data 4 0.95 1


```
### Params
*   thread number
*   accuracy for convergence
*   method to run

### methods
*   -2 ->   Initialize networks
*   -1 -> run all version
*   0 -> sequential version
*   1 -> multi-env version
*   2 -> sychronous version
*   3 -> sychronous version with implicite aggregation
*   4 -> Asychronous version 1/2  threads for environment interaction
*   5 -> Asychronous version 1/2  threads for environment interaction with implicite aggregation
*   6 -> Asychronous version 1/4  threads for environment interaction
*   7 -> Asychronous version 1/4  threads for environment interaction with implicte aggregation
*   8 -> Adaptative methode 0.60 accuracy for swap
*   9 -> Adaptative methode 0.60 accuracy for swap with implicte aggregation
*   10 -> Adaptative methode 0.70 accuracy for swap
*   11 -> Adaptative methode 0.70 accuracy for swap with implicte aggregation
*   12 -> Adaptative methode 0.80 accuracy for swap
*   13 -> Adaptative methode 0.80 accuracy for swap with implicte aggregation


### Arguments of the executable
* The first argument is a folder of dataset
* The second argument is a number of threads
* The last argument is a stop criterio (accuracy for convergence) 



