CXX := g++
CXXFLAGS := -fopenmp -Wall -Wextra -Iinclude -I ${mkEigenInc}

SRC_DIR := src
TESTS_DIR := tests
OBJ_DIR := obj
BIN_DIR := bin

# List of files to exclude
EXCLUDE_FILES_SRC := PM3.cpp PM2_full.cpp PM2_one_iteration.cpp rSVD.cpp PM1.cpp SVD.cpp 

# Find all source files in the source directory excluding specific files
SRC_FILES := $(filter-out $(addprefix $(SRC_DIR)/,$(EXCLUDE_FILES_SRC)), $(wildcard $(SRC_DIR)/*.cpp))

# Create a list of object files based on the source files
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_FILES))

EXCLUDE_FILES_TST := PM3Test.cpp PMTest2.cpp SVD_test.cpp PMTest3.cpp rSVD_test.cpp PMTest.cpp QRTest.cpp

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
test: $(TEST_BINS)
	@echo "Running tests..."
	@$(foreach test, $(TEST_BINS), ./$(test);)

# Target to build and run tests with profiling
profile: CXXFLAGS += -pg
profile: test
	@gprof $(BIN_DIR)/QRTest > profile_output.txt

# Clean up generated files and remove files inside /data/output/
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	rm -f data/output/*

.PHONY: all test profile clean
