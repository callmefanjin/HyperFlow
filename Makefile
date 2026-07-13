CXX      ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O3 -march=native

SRC_DIR  := src

JSON_CFLAGS ?= $(shell pkg-config --cflags nlohmann_json 2>/dev/null)
ifeq ($(JSON_CFLAGS),)
  JSON_CFLAGS := -I$(CONDA_PREFIX)/include
endif

EIGEN_CFLAGS ?= $(shell pkg-config --cflags eigen3 2>/dev/null)
ifeq ($(EIGEN_CFLAGS),)
  EIGEN_CFLAGS := -I$(CONDA_PREFIX)/include/eigen3
endif

CANTERA_CFLAGS ?= $(shell pkg-config --cflags cantera 2>/dev/null)
CANTERA_LIBS   ?= $(shell pkg-config --libs cantera 2>/dev/null)
ifeq ($(CANTERA_LIBS),)
  CANTERA_LIBS := -L$(CONDA_PREFIX)/lib -lcantera
endif

HDF5_CFLAGS ?= $(shell pkg-config --cflags hdf5 2>/dev/null)
HDF5_LIBS   ?= $(shell pkg-config --libs hdf5 2>/dev/null)
ifeq ($(HDF5_CFLAGS),)
  HDF5_CFLAGS := -I$(CONDA_PREFIX)/include
  HDF5_LIBS   := -L$(CONDA_PREFIX)/lib -lhdf5_hl -lhdf5
endif

CXXFLAGS += -I$(SRC_DIR) $(JSON_CFLAGS) $(EIGEN_CFLAGS) $(CANTERA_CFLAGS) $(HDF5_CFLAGS) -fopenmp
LIBS      = $(CANTERA_LIBS) $(HDF5_LIBS) -lgomp -Wl,-rpath,$(CONDA_PREFIX)/lib

COMMON_SRCS = \
	$(SRC_DIR)/tool_Function.C \
	$(SRC_DIR)/preComputation.C \
	$(SRC_DIR)/coreMethod.C \
	$(SRC_DIR)/sourceTerm.C \
	$(SRC_DIR)/boundaryCondition.C

.PHONY: all clean

all: HyperFlow

HyperFlow: $(SRC_DIR)/HyperFlow_main.C $(COMMON_SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC_DIR)/HyperFlow_main.C $(COMMON_SRCS) $(LIBS)

clean:
	rm -f HyperFlow
