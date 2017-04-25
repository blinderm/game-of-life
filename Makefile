CC := nvcc -arch sm_20

CFLAGS := -g --sdl2

all: life

clean:
	@rm -f life

life: life.cu
	$(CC) $(CFLAGS) -o life life.cu
