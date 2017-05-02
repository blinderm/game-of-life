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
typedef struct grid {
    bool board[(int) GRID_HEIGHT][(int) GRID_WIDTH];
} grid_t;


// coordinate struct
typedef struct coord {
    int x;
    int y;
} coord_t;

bool running = true;

// mouse function parameter struct
typedef struct mouse_args {
    coord_t loc;
    uint32_t mouse_state;
    bool mouse_up;
    SDL_Event* event;
} mouse_args_t;

// keyboard function parameter struct
typedef struct keyboard_args {
    coord_t loc;
    const uint8_t* keyboard_state;
    SDL_Event* event;
} keyboard_args_t;

bool paused = true;

// bitmap screen variable
bitmap* bmp;

// grid for indicating cell state (dead or alive)
grid_t* g;

// Create a GUI window
gui ui("Conway's Game of Life", BMP_WIDTH, BMP_HEIGHT);


// Get input from the keyboard and execute proper command 
void* get_keyboard_input(void* params);

// Get input from the mouse and toggle the appropriate cell's state/color
void* get_mouse_input(void* params);

// Update each cell in order to advance the simulation
void update_cells(void* params);

// Toggle the cell's state, change the color accordingly
void let_there_be_light(coord_t loc);

// Set up the grid with an existing layout specified by a file
void load_grid(FILE * layout);








#endif
