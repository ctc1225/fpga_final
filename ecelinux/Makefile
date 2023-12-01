#==========================================================================
# Makefile
#==========================================================================
# @brief: A makefile the compiles and synthesizes the retnet_recurrent program
#
# @desc: 1. "make" runs csim by default
#        2. "make csim" compiles & executes the software implementation
#        3. "make clean" cleans up the directory


# Extract Vivado HLS include path
XILINX_VIVADO?=/opt/xilinx/Xilinx_Vivado_vitis_2019.2/Vivado/2019.2
XIL_HLS=source $(XILINX_VIVADO)/settings64.sh; vivado_hls
VHLS_INC=$(XILINX_VIVADO)/include
# Specify compilation flags
CFLAGS=-g -I${VHLS_INC} -DHLS_NO_XIL_FPO_LIB -std=c++11 

TCL_SCRIPT=run.tcl

.PHONY: all csim bitstream clean

all: csim

retnet_projection: retnet_projection.cpp retnet_projection_test.cpp
	g++ ${CFLAGS} $^ -o $@ -lrt

test_projection: retnet_projection
	@echo "Running test_retnet_projection sim..."
	mkdir -p result
	./$< | tee $@
	@echo "Result recorded to result/test_retnet_projection.txt"

retnet_recurrent: retnet_projection.cpp retnet_projection_test.cpp
	g++ ${CFLAGS} $^ -o $@ -lrt

result/retnet_recurrent_csim.txt: retnet_recurrent
	@echo "Running retnet_recurrent sim..."
	mkdir -p result
	./$< | tee $@

csim: result/retnet_recurrent_csim.txt
	@echo "Result recorded to $<"

xillydemo.bit:
	@echo "=================================================================" 
	@echo "Synthesizing retnet_recurrent and creating bitstream with $(TCL_SCRIPT)..."
	@echo "=================================================================" 
	$(XIL_HLS) -f $(TCL_SCRIPT)
	source $(XILINX_VIVADO)/settings64.sh; ./run_bitstream.sh

bitstream: xillydemo.bit
	@echo "Bitstream saved to $<"

clean:
	rm -rf retnet_recurrent *.dat *.prj *.log
	rm -rf zedboard_project* xillydemo.bit
