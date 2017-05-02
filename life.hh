#if !defined(LIFE_HH)
#define LIFE_HH

#include <math.h>

#include <stdio.h>
#include <time.h>
#include <math.h>

#include <stdio.h>
#include <time.h>
#include <vector>

#include <SDL.h>
#include <pthread.h>

#include "bitmap.hh"
#include "gui.hh"
#include "util.hh"

using namespace std;

// screen size
#define BMP_WIDTH 800
#define BMP_HEIGHT 600

// cell dimension
#define CELL_DIM 10

// grid size
#define GRID_WIDTH (BMP_WIDTH/CELL_DIM)
#define GRID_HEIGHT (BMP_HEIGHT/CELL_DIM)

// threads per block
#define THREADS_PER_BLOCK 64

// colors!
#define WHITE rgb32(255.,255.,255.)
#define BLACK rgb32(0.,0.,0.)

// update delay
#define DELAY 25

// barrier for threads
static pthread_barrier_t barrier;

// grid struct
struct grid {
    int board[(int) GRID_HEIGHT][(int) GRID_WIDTH];

    grid(int value) {
        memset(board, value, sizeof(int) * GRID_HEIGHT * GRID_WIDTH);
    }
};

// coordinate struct
struct coord {
    int x;
    int y;

    coord() : x(0), y(0) {}
}; 

bool running = true;

// function parameter struct
struct input_args {
    coord loc;
    uint32_t mouse_state;
    SDL_Event* event;

    input_args(SDL_Event* e) {
        loc = coord();
        mouse_state = SDL_GetMouseState(&(loc.x), &(loc.y));
        event = e;
    }
};

// indicate whether or not the simulation is paused
bool paused = true;

// bitmap screen variable
bitmap* bmp;

// grid for indicating cell state (dead or alive)
grid* g;

// Create a GUI window
gui ui("Conway's Game of Life", BMP_WIDTH, BMP_HEIGHT);


// Get input from the keyboard and execute proper command 
void* get_keyboard_input(void* params);

// Get input from the mouse and toggle the appropriate cell's state/color
void* get_mouse_input(void* params);

// Update each cell in order to advance the simulation
void update_cells();

// Toggle the cell's state, change the color accordingly
void let_there_be_light(coord loc);

// Set up the grid with an existing layout specified by a file
void load_grid(FILE * layout);








#endif
