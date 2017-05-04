CC := nvcc -arch sm_20

CFLAGS := -g `sdl2-config --libs --cflags` -Wno-deprecated-gpu-targets

all: life

clean:
	@rm -f life

life: life.o 
	$(CC) $(CFLAGS) -o life life.o 

life.o: life.cu life.hh
	$(CC) $(CFLAGS) -c life.cu

