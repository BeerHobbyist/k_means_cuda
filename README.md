# Running the project
The project is implemented in two versions. One cuda paralled and one cpu sequential sourced from https://www.eecs.northwestern.edu/~wkliao/Kmeans/index.html
## Input generation
To run the project first you need some sample input. To do this run:
```
make generate_input
``` 
and then:
```
./generate_input.o <number_of_points> <dimensions> <out_filename>
```
this should generate an input file in the correct format
## Cuda version
To run cuda version build it using `make` and run:
```
./kmeans_cuda.o -i <input_file> -n <number_of_clusters>
```
## Cpu version
To run it on cpu simply got to `/sequential_kmeans` and use `make` and run:
```
./seq_main -i <input_file> -n <number_of_clusters>
```
## Algorith comparison
For an input of size 1000000 x 30 and K = 300 the running times where:
 - Cpu - 949.3460s
 - Gpu - 29.34938s