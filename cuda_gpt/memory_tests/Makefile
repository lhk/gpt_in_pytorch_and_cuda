# Memory Tests Makefile
# Compiler settings
NVCC = nvcc
CUDA_FLAGS = -std=c++14
LIBS = -lcurand

# Parent directory for includes
PARENT_DIR = ..

# Test targets
TESTS = test_layernorm_memory test_mhsa_memory test_mlp_memory test_transformerblock_memory test_gpt_memory

# Default target
all: $(TESTS)

# Individual test compilation
test_layernorm_memory: test_layernorm_memory.cu $(PARENT_DIR)/modules.cu $(PARENT_DIR)/kernels.cu $(PARENT_DIR)/weight_loader.h
	$(NVCC) $(CUDA_FLAGS) -I$(PARENT_DIR) -o $@ $< $(LIBS)

test_mhsa_memory: test_mhsa_memory.cu $(PARENT_DIR)/modules.cu $(PARENT_DIR)/kernels.cu $(PARENT_DIR)/weight_loader.h
	$(NVCC) $(CUDA_FLAGS) -I$(PARENT_DIR) -o $@ $< $(LIBS)

test_mlp_memory: test_mlp_memory.cu $(PARENT_DIR)/modules.cu $(PARENT_DIR)/kernels.cu $(PARENT_DIR)/weight_loader.h
	$(NVCC) $(CUDA_FLAGS) -I$(PARENT_DIR) -o $@ $< $(LIBS)

test_transformerblock_memory: test_transformerblock_memory.cu $(PARENT_DIR)/modules.cu $(PARENT_DIR)/kernels.cu $(PARENT_DIR)/weight_loader.h
	$(NVCC) $(CUDA_FLAGS) -I$(PARENT_DIR) -o $@ $< $(LIBS)

test_gpt_memory: test_gpt_memory.cu $(PARENT_DIR)/gpt.cu $(PARENT_DIR)/modules.cu $(PARENT_DIR)/kernels.cu $(PARENT_DIR)/weight_loader.h
	$(NVCC) $(CUDA_FLAGS) -I$(PARENT_DIR) -o $@ $< $(LIBS)

# Note: No CUDA suppressions - we want to see the full picture initially

# Run individual tests
run-layernorm: test_layernorm_memory
	@echo "=== Running LayerNorm Memory Test ==="
	./test_layernorm_memory

run-mhsa: test_mhsa_memory
	@echo "=== Running MHSA Memory Test ==="
	./test_mhsa_memory

run-mlp: test_mlp_memory
	@echo "=== Running MLP Memory Test ==="
	./test_mlp_memory

run-transformerblock: test_transformerblock_memory
	@echo "=== Running TransformerBlock Memory Test ==="
	./test_transformerblock_memory

run-gpt: test_gpt_memory
	@echo "=== Running GPT Memory Test ==="
	./test_gpt_memory

# Valgrind tests with targeted CUDA suppressions
memcheck-layernorm: test_layernorm_memory cuda.supp
	@echo "=== Valgrind: LayerNorm Memory Check ==="
	@valgrind --leak-check=full --suppressions=cuda.supp --show-leak-kinds=definite,indirect --track-origins=yes --error-exitcode=1 ./test_layernorm_memory || echo "FAILED: LayerNorm has memory leaks"
	@echo "LayerNorm memory check completed"

memcheck-mhsa: test_mhsa_memory cuda.supp
	@echo "=== Valgrind: MHSA Memory Check ==="
	@valgrind --leak-check=full --suppressions=cuda.supp --show-leak-kinds=definite,indirect --track-origins=yes --error-exitcode=1 ./test_mhsa_memory || echo "FAILED: MHSA has memory leaks"
	@echo "MHSA memory check completed"

memcheck-mlp: test_mlp_memory cuda.supp
	@echo "=== Valgrind: MLP Memory Check ==="
	@valgrind --leak-check=full --suppressions=cuda.supp --show-leak-kinds=definite,indirect --track-origins=yes --error-exitcode=1 ./test_mlp_memory || echo "FAILED: MLP has memory leaks"
	@echo "MLP memory check completed"

memcheck-transformerblock: test_transformerblock_memory cuda.supp
	@echo "=== Valgrind: TransformerBlock Memory Check ==="
	@valgrind --leak-check=full --suppressions=cuda.supp --show-leak-kinds=definite,indirect --track-origins=yes --error-exitcode=1 ./test_transformerblock_memory || echo "FAILED: TransformerBlock has memory leaks"
	@echo "TransformerBlock memory check completed"

memcheck-gpt: test_gpt_memory cuda.supp
	@echo "=== Valgrind: GPT Memory Check ==="
	@valgrind --leak-check=full --suppressions=cuda.supp --show-leak-kinds=definite,indirect --track-origins=yes --error-exitcode=1 ./test_gpt_memory || echo "FAILED: GPT has memory leaks"
	@echo "GPT memory check completed"

# Run all memory checks
memcheck-all: $(TESTS) cuda.supp
	@echo "=== Running All Memory Checks ==="
	@$(MAKE) memcheck-layernorm
	@$(MAKE) memcheck-mhsa
	@$(MAKE) memcheck-mlp
	@$(MAKE) memcheck-transformerblock
	@$(MAKE) memcheck-gpt
	@echo "All memory checks completed"

# Quick test run (no valgrind)
test-all: $(TESTS)
	@echo "=== Running All Tests (No Valgrind) ==="
	@$(MAKE) run-layernorm
	@$(MAKE) run-mhsa
	@$(MAKE) run-mlp
	@$(MAKE) run-transformerblock
	@$(MAKE) run-gpt

# Clean
clean:
	rm -f $(TESTS) cuda.supp

# Force rebuild
rebuild: clean all

.PHONY: all clean rebuild test-all memcheck-all run-layernorm run-mhsa run-mlp run-transformerblock run-gpt memcheck-layernorm memcheck-mhsa memcheck-mlp memcheck-transformerblock memcheck-gpt
