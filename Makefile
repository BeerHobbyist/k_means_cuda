TARGET = kmeans_cuda

all: $(TARGET)

$(TARGET): main.cu atiomic_k_means.cu
	 nvcc -std=c++17 main.cu atiomic_k_means.cu -o $(TARGET)

clean:
	 rm -f $(TARGET)


