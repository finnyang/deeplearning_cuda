DEBUG:=1
SRCS := $(shell find ./src -name "*.cu" -type f)
OBJS := $(SRCS:%.cu=%.o)
TARGET=main
arch=--gpu-architecture=compute_52 --gpu-code=compute_52
INCLUDE:=-I./include -I/usr/local/cuda-7.5/include
LIB:=-L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -latlas -lcblas -lm
FLAGS:=-DSSD
ifeq ($(DEBUG), 1)
FLAGS+=-DDEBUG
endif
all:$(OBJS)
	gcc $(OBJS) -o main $(INCLUDE) $(LIB)
$(OBJS):%.o:%.cu
	nvcc $(arch) $(FLAGS) $(INCLUDE) -c $< -o $@
clean:
	rm -fr $(OBJS) $(TARGET) main
