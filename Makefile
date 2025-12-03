TARGET = kmeans_cuda

all: $(TARGET)

$(TARGET): main.cu atiomic_k_means.cu
	 nvcc -std=c++17 main.cu atiomic_k_means.cu -o $(TARGET).o

generate_input: generate_input.cpp
	g++ -std=c++17 generate_input.cpp -o generate_input.o

clean:
	 rm -f $(TARGET)


