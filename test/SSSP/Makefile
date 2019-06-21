
NAME = sssp

NVCC = nvcc
KLAP = klap

ARCH = -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52
FLAGS = -Xcompiler '-fPIC' -dc -m64 --default-stream per-thread

INCLUDE  = -I. -I../../include

SRC = main.cu
OBJ = $(SRC:.cu=.o)
BIN = $(NAME).base $(NAME).aw $(NAME).ab $(NAME).ag

base: $(NAME).base

all: $(BIN)

.SECONDARY:

kernel.a%.cu: kernel.de.cu
	$(KLAP) $< -t `echo $@ | cut -d '.' -f2` -o $@ -- --cuda-device-only $(INCLUDE) --no-cuda-version-check -nocudalib

kernel.de.cu: kernel.base.cu
	$(KLAP) $< -t de -o $@ -- --cuda-device-only $(INCLUDE) --no-cuda-version-check -nocudalib

%.o: %.cu
	$(NVCC) $(ARCH) $(FLAGS) $(INCLUDE) -o $@ -c $<

$(NAME).%: kernel.%.o $(OBJ)
	$(NVCC) $(ARCH) -o $@ $+ -lcudadevrt 

clean:
	rm -f *.o *.csv $(BIN) kernel.a?.cu kernel.de.cu outputs/*

test:
	./test.sh

test-all: all
	./test.sh

