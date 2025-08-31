# Functional Tests Makefile
# Compiler settings
NVCC = nvcc
CUDA_FLAGS = -std=c++14
LIBS = -lcurand

# Parent directory for includes
PARENT_DIR = ..

# Test targets
TESTS = modules_test kernels_test gpt_test

# Default target
all: $(TESTS)

# Individual test compilation
modules_test: modules_test.cu $(PARENT_DIR)/modules.cu $(PARENT_DIR)/kernels.cu $(PARENT_DIR)/weight_loader.h
	$(NVCC) $(CUDA_FLAGS) -I$(PARENT_DIR) -o $@ $< $(LIBS)

kernels_test: kernels_test.cu $(PARENT_DIR)/kernels.cu $(PARENT_DIR)/weight_loader.h
	$(NVCC) $(CUDA_FLAGS) -I$(PARENT_DIR) -o $@ $< $(LIBS)

gpt_test: gpt_test.cu $(PARENT_DIR)/gpt.cu $(PARENT_DIR)/modules.cu $(PARENT_DIR)/kernels.cu $(PARENT_DIR)/weight_loader.h
	$(NVCC) $(CUDA_FLAGS) -I$(PARENT_DIR) -o $@ $< $(LIBS)

# Run individual tests
run-modules: modules_test
	@echo "=== Running Modules Functional Test ==="
	./modules_test

run-kernels: kernels_test
	@echo "=== Running Kernels Functional Test ==="
	./kernels_test

run-gpt: gpt_test
	@echo "=== Running GPT Functional Test ==="
	./gpt_test

# Run all functional tests
test-all: $(TESTS)
	@echo "=== Running All Functional Tests ==="
	@$(MAKE) run-modules
	@$(MAKE) run-kernels
	@$(MAKE) run-gpt

# Clean
clean:
	rm -f $(TESTS)

# Force rebuild
rebuild: clean all

.PHONY: all clean rebuild test-all run-modules run-kernels run-gpt