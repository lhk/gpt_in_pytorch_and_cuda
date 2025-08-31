# Simple Makefile for CUDA project




NVCC = nvcc
TARGETS = matmul matmul_benchmark softmax softmax_benchmark mhsa

all: $(TARGETS)

matmul: matmul.cu
	$(NVCC) -o $@ $^


matmul_benchmark: matmul_benchmark.cu
	$(NVCC) -o $@ $^


softmax: softmax.cu
	$(NVCC) -o $@ $^


softmax_benchmark: softmax_benchmark.cu
	$(NVCC) -o $@ $^

mhsa: mhsa.cu
	$(NVCC) -o $@ $^

clean:
	rm -f $(TARGETS)