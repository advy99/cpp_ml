
CXX := g++
HOME := .
BIN := $(HOME)/bin
OBJ := $(HOME)/obj
SRC := $(HOME)/src
INC := $(HOME)/include
LIB := $(HOME)/lib

###
#
# Variables for optimization and GPROF
#
###

OPTIMIZACION ?= 2
GPROF ?= 0

ifeq ($(DEBUG), 1)
# target para debug (cambiamos flags y el mensaje)
OPTIMIZACION = g -g
endif

ifeq ($(GPROF), 1)
F_GPROF = -pg
else
F_GPROF =
endif


###
#
# Variables for GTEST suite
#
###

gtest_include = -isystem /usr/include/gtest/
gtestlibs  = /usr/lib/libgtest.so
gtestflags = $(gtest_include) $(gtestlibs)

###
#
# Compiler flags and includes
#
###

O_LEVEL := -O$(strip $(OPTIMIZACION))

CXXFLAGS = -std=c++20 -Wall -Wextra -Wpedantic -Wfloat-equal -I$(INC) -isystem /usr/include/opencv4 $(O_LEVEL) $(F_GPROF)
OPENCV_LIB = -lopencv_core


###
#
# Main target
#
###

TARGET = $(BIN)/main
TARGET_OBJECT = $(OBJ)/main.o


###
#
# GTEST target
#
###

TEST_TARGET = $(BIN)/tests
TEST_TARGET_OBJECT = $(OBJ)/tests.o

###
#
# Library targets
#
###

LIBRARY = $(LIB)/libcpp_ml.a

TRANSFORMERS_OBJECTS = $(OBJ)/label_encoder.o $(OBJ)/standard_scaler.o
CLASSIFICATION_OBJECTS = $(OBJ)/knn_classifier.o
REGRESSION_OBJECTS = $(OBJ)/knn_regressor.o $(OBJ)/linear_regression.o
MODELS_OBJECTS = $(CLASSIFICATION_OBJECTS) $(REGRESSION_OBJECTS)
METRICS_OBJECTS = $(OBJ)/classification_metrics.o $(OBJ)/regression_metrics.o
MATH_OBJECTS = $(OBJ)/distances.o
DATASETS_OBJECTS =

###
#
# Main recipes
#
###

all: make_dirs $(TARGET)

tests: make_dirs $(TEST_TARGET)

###
#
# Main target compilation
#
###

$(TARGET): $(TARGET_OBJECT) $(LIBRARY)
	$(CXX) $(TARGET_OBJECT) -lcpp_ml $(OPENCV_LIB) -L$(LIB) -o $(TARGET)


$(LIBRARY): $(TRANSFORMERS_OBJECTS) $(METRICS_OBJECTS) $(MODELS_OBJECTS) $(MATH_OBJECTS) $(DATASETS_OBJECTS)
	ar rvs $(LIBRARY) $^


###
#
# Test target compilation
#
###


$(TEST_TARGET): $(TEST_TARGET_OBJECT) $(LIBRARY)
	$(CXX) $(TEST_TARGET_OBJECT) -lcpp_ml $(OPENCV_LIB) $(gtestflags) -L$(LIB) -o $(TEST_TARGET)


###
#
# Objects compilation
#
###

$(TARGET_OBJECT): $(SRC)/main.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(TEST_TARGET_OBJECT): $(SRC)/tests.cpp
	$(CXX) -c $(CXXFLAGS) $(gtest) $< -o $@


$(OBJ)/label_encoder.o: $(SRC)/transformers/label_encoder.cpp $(INC)/transformers/label_encoder.hpp $(INC)/transformers.hpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(OBJ)/standard_scaler.o: $(SRC)/transformers/standard_scaler.cpp $(INC)/transformers/standard_scaler.hpp $(INC)/transformers.hpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(OBJ)/knn_classifier.o: $(SRC)/classifiers/knn_classifier.cpp $(INC)/classifiers/knn_classifier.hpp $(INC)/classifiers.hpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(OBJ)/knn_regressor.o: $(SRC)/regressors/knn_regressor.cpp $(INC)/regressors/knn_regressor.hpp $(INC)/regressors.hpp
	$(CXX) -c $(CXXFLAGS) $< -o $@


$(OBJ)/linear_regression.o: $(SRC)/regressors/linear_regression.cpp $(INC)/regressors/linear_regression.hpp $(INC)/regressors.hpp
	$(CXX) -c $(CXXFLAGS) $< -o $@


$(OBJ)/classification_metrics.o: $(SRC)/metrics/classification_metrics.cpp $(INC)/metrics/classification_metrics.hpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(OBJ)/regression_metrics.o: $(SRC)/metrics/regression_metrics.cpp $(INC)/metrics/regression_metrics.hpp
	$(CXX) -c $(CXXFLAGS) $< -o $@


$(OBJ)/distances.o: $(SRC)/math/distances.cpp $(INC)/math/distances.hpp
	$(CXX) -c $(CXXFLAGS) $< -o $@


###
#
# Auxiliar recipes
#
###

folder_exists_message = "This folder alredy exists"

make_dirs:
	@printf "\e[36mCreating folders ...\e[0m\n"
	-@mkdir $(OBJ) 2> /dev/null || printf "\t\e[33m%s $(OBJ)\e[0m\n" $(folder_exists_message)
	-@mkdir $(BIN) 2> /dev/null || printf "\t\e[33m%s $(BIN)\e[0m\n" $(folder_exists_message)
	-@mkdir $(LIB) 2> /dev/null || printf "\t\e[33m%s $(LIB)\e[0m\n" $(folder_exists_message)


clean:
	@printf "\e[36mCleaning $(OBJ)\e[0m\n"
	-@rm $(OBJ)/*.o 2> /dev/null || printf "\t\e[33m$(OBJ) is empty. Nothing to clean\e[0m\n"
	@printf "\e[36mCleaning $(BIN)\e[0m\n"
	-@rm $(BIN)/* 2> /dev/null || printf "\t\e[33m$(BIN) is empty. Nothing to clean\e[0m\n"
	@printf "\e[36mCleaning $(LIB)\e[0m\n"
	-@rm $(LIB)/*.a 2> /dev/null || printf "\t\e[33m$(LIB) is empty. Nothing to clean\e[0m\n"

update_compiledb:
	@make clean
	@compiledb make -n

.PHONY: all make_dirs
