
CXX := g++
HOME := .
BIN := $(HOME)/bin
OBJ := $(HOME)/obj
SRC := $(HOME)/src
INC := $(HOME)/include
LIB := $(HOME)/lib

CXXFLAGS = -std=c++20 -Wall -Wextra -Wpedantic -Wfloat-equal -g -Og


TARGET = $(BIN)/main
TARGET_OBJECT = $(OBJ)/main.o


LIBRARY = $(LIB)/libcpp_ml.a

ENCODERS_OBJECTS = $(OBJ)/label_encoder.o
CLASSIFICATION_OBJECTS = $(OBJ)/knn_classifier.o
REGRESSION_OBJECTS = $(OBJ)/knn_regressor.o
MODELS_OBJECTS = $(CLASSIFICATION_OBJECTS) $(REGRESSION_OBJECTS)
METRICS_OBJECTS = $(OBJ)/classification_metrics.o
MATH_OBJECTS = $(OBJ)/distances.o
DATASETS_OBJECTS =

all: make_dirs $(TARGET)


$(TARGET): $(TARGET_OBJECT) $(LIBRARY)
	$(CXX) $(TARGET_OBJECT) -lcpp_ml -L$(LIB) -o $(TARGET)


$(LIBRARY): $(ENCODERS_OBJECTS) $(METRICS_OBJECTS) $(MODELS_OBJECTS) $(MATH_OBJECTS) $(DATASETS_OBJECTS)
	ar rvs $(LIBRARY) $^

$(TARGET_OBJECT): $(SRC)/main.cpp
	$(CXX) -c $(CXXFLAGS) $< -I$(INC) -o $@


$(OBJ)/label_encoder.o: $(SRC)/transformers/label_encoder.cpp $(INC)/transformers/label_encoder.hpp $(INC)/transformers.hpp
	$(CXX) -c $(CXXFLAGS) $< -I$(INC) -o $@


$(OBJ)/knn_classifier.o: $(SRC)/classifiers/knn_classifier.cpp $(INC)/classifiers/knn_classifier.hpp $(INC)/classifiers.hpp
	$(CXX) -c $(CXXFLAGS) $< -I$(INC) -o $@

$(OBJ)/knn_regressor.o: $(SRC)/regressors/knn_regressor.cpp $(INC)/regressors/knn_regressor.hpp $(INC)/regressors.hpp
	$(CXX) -c $(CXXFLAGS) $< -I$(INC) -o $@

$(OBJ)/classification_metrics.o: $(SRC)/metrics/classification_metrics.cpp $(INC)/metrics/classification_metrics.hpp
	$(CXX) -c $(CXXFLAGS) $< -I$(INC) -o $@

$(OBJ)/distances.o: $(SRC)/math/distances.cpp $(INC)/math/distances.hpp
	$(CXX) -c $(CXXFLAGS) $< -I$(INC) -o $@


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
