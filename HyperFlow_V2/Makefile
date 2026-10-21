CXX      ?= g++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O3 -march=native

ifeq ($(strip $(CONDA_PREFIX)),)
  $(error CONDA_PREFIX is unset; activate the Cantera environment or run make CONDA_PREFIX=/path/to/environment)
endif

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

CXXFLAGS += $(JSON_CFLAGS) $(EIGEN_CFLAGS) $(CANTERA_CFLAGS) $(HDF5_CFLAGS) -I$(CONDA_PREFIX)/include -fopenmp
LIBS      = $(CANTERA_LIBS) $(HDF5_LIBS) -lgomp -Wl,-rpath,$(CONDA_PREFIX)/lib
SUNDIALS_LIBS = -L$(CONDA_PREFIX)/lib -lsundials_cvode -lsundials_nvecserial -lsundials_sunlinsolspgmr -lsundials_core

SRC_DIR = src
COMMON_SRCS = $(addprefix $(SRC_DIR)/,tool_Function.C preComputation.C profileFunction.C coreMethod.C sourceTerm.C boundaryCondition.C timeIntegrator.C)
HEADERS = $(wildcard $(SRC_DIR)/*.H)

.PHONY: all clean

all: HyperFlow_nochem_steady HyperFlow_implicit

HyperFlow_nochem_steady: $(SRC_DIR)/HyperFlow_nochem_steady.C $(COMMON_SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -o $@ $(filter %.C,$^) $(LIBS)

HyperFlow_implicit: $(SRC_DIR)/HyperFlow_implicit.C $(SRC_DIR)/implicitIntegrator.C $(COMMON_SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -o $@ $(filter %.C %.cpp,$^) $(LIBS) $(SUNDIALS_LIBS)

clean:
	rm -f HyperFlow_nochem_steady HyperFlow_implicit
