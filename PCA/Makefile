CXX := mpicxx # only for some computation in PM and rSVD
CXXFLAGS := -fopenmp -I../include -Iinclude -I ${mkEigenInc}

SRC_DIR := ../src
TESTS_DIR := tests
OBJ_DIR := obj
BIN_DIR := bin
MAIN_DIR := main
# List of files to exclude
EXCLUDE_FILES_SRC :=

# Find all source files in the source directory, excluding specific files
SRC_FILES := $(filter-out $(addprefix $(SRC_DIR)/,$(EXCLUDE_FILES_SRC)), $(wildcard $(SRC_DIR)/*.cpp))

# Create a list of object files based on the source files
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))

EXCLUDE_FILES_TST := athletic_test.cpp

# Test source files
TEST_SRC_FILES := $(filter-out $(addprefix $(TESTS_DIR)/,$(EXCLUDE_FILES_TST)), $(wildcard $(TESTS_DIR)/*.cpp))

# Test executable names
TEST_BINS := $(patsubst $(TESTS_DIR)/%.cpp,$(BIN_DIR)/%,$(TEST_SRC_FILES))

# Default target
all: $(OBJ_FILES) $(BIN_DIR)/main

# Rule to build object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to build the main executable
$(BIN_DIR)/main: $(OBJ_FILES) $(MAIN_DIR)/main.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ_FILES) $(MAIN_DIR)/main.cpp

# Rule to build test executables
$(BIN_DIR)/%: $(TESTS_DIR)/%.cpp $(OBJ_FILES)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(if $(filter profile,$(MAKECMDGOALS)),-pg) $< $(filter %.o,$^) -o $@

# Target to build and run tests without profiling
# Parameters for the test
TEST_INPUT_FILE := data/input/tourists.txt
TEST_OUTPUT_FILE := data/output/pca_tourists_results.txt
TEST_NORMALIZE := no

test: $(TEST_BINS) create_directories
	@echo "Running tests..."
	@$(foreach test, $(TEST_BINS), ./$(test) $(TEST_INPUT_FILE) $(TEST_OUTPUT_FILE) $(TEST_NORMALIZE);)

run: $(BIN_DIR)/main
	@if [ -z "$(dataset)" ]; then \
		echo "Error: 'dataset' variable is not set."; \
		echo "Usage: make run dataset=FILENAME normalize=yes/no"; \
		exit 1; \
	fi
	@if [ -z "$(normalize)" ]; then \
		echo "Error: 'normalize' variable is not set."; \
		echo "Usage: make run dataset=FILENAME normalize=yes/no"; \
		exit 1; \
	fi
	@echo "Running main on dataset: $(dataset) with normalization: $(normalize)"
	@./$(BIN_DIR)/main data/input/$(dataset) $(normalize)

# Target to build and run tests with profiling
profile: CXXFLAGS += -pg
profile: test
	@gprof $(BIN_DIR)/rSVD_test2 > profile_output.txt

# Clean up generated files and remove files inside /data/output/
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	rm -rf data/output/*

.PHONY: all test profile clean create_directories run
