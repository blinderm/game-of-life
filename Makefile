CC := nvcc -arch sm_20

CFLAGS := -g `sdl2-config --libs --cflags`

all: life

clean:
	@rm -f life

life: life.o scheduler.o
	$(CC) $(CFLAGS) -o life life.o scheduler.o 

life.o: life.cu
	$(CC) $(CFLAGS) -c life.cu

scheduler.o: scheduler.cc scheduler.hh
	$(CC) $(CFLAGS) -c scheduler.cc

