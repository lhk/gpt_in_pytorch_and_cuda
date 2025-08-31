# CUDA GPT Makefile
# Compiler settings
NVCC = nvcc
CUDA_FLAGS = -std=c++14
LIBS = -lcurand

# Source files
SOURCES = generate.cu gpt.cu modules.cu kernels.cu
HEADERS = weight_loader.h

# Default target
all: generate prompt

# Main programs  
generate: generate.cu $(HEADERS)
	$(NVCC) $(CUDA_FLAGS) -o $@ $< $(LIBS)

prompt: prompt.cu $(HEADERS)
	$(NVCC) $(CUDA_FLAGS) -o $@ $< $(LIBS)

# Convenience targets
run: prompt
	./prompt

# Clean
clean:
	rm -f generate prompt gpt_test modules_test kernels_test memory_test modules_memory_test generate_new

# Force rebuild
rebuild: clean all

.PHONY: all run test clean rebuild 
