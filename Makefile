CXX := mpic++
CXXFLAGS := -Wall -Wextra -Iinclude -I ${mkEigenInc}

SRC_DIR := src
TESTS_DIR := tests
OBJ_DIR := obj
BIN_DIR := bin

# List of files to exclude
EXCLUDE_FILES_SRC :=

# Find all source files in the source directory excluding specific files
SRC_FILES := $(filter-out $(addprefix $(SRC_DIR)/,$(EXCLUDE_FILES_SRC)), $(wildcard $(SRC_DIR)/*.cpp))

# Create a list of object files based on the source files
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))

EXCLUDE_FILES_TST := small_test_svd.cpp small_rsvd.cpp QRTest.cpp small_qr.cpp PMTest.cpp

# Test source files
TEST_SRC_FILES := $(filter-out $(addprefix $(TESTS_DIR)/,$(EXCLUDE_FILES_TST)), $(wildcard $(TESTS_DIR)/*.cpp))

# Test executable names
TEST_BINS := $(patsubst $(TESTS_DIR)/%.cpp,$(BIN_DIR)/%,$(TEST_SRC_FILES))

# Default target
all: $(OBJ_FILES)

# Rule to build object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to build test executables
$(BIN_DIR)/%: $(TESTS_DIR)/%.cpp $(OBJ_FILES)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(if $(filter profile,$(MAKECMDGOALS)),-pg) $< $(filter %.o,$^) -o $@

# Target to build and run tests without profiling
test: $(TEST_BINS) create_directories
	@echo "Running tests..."
	@$(foreach test, $(TEST_BINS), mpirun --allow-run-as-root -np 4 ./$(test);)

# Target to build and run tests with profiling
profile: CXXFLAGS += -pg
profile: test
	@gprof $(BIN_DIR)/rSVD_test2 > profile_output.txt

# Clean up generated files and remove files inside /data/output/
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	rm -rf data/output/QR/*
	rm -rf data/output/*

# Target to create output directories
create_directories:
	mkdir -p ./data/output/QR
	mkdir -p ./data/output/rSVD
	mkdir -p ./data/output/SVD

.PHONY: all test profile clean create_directories
