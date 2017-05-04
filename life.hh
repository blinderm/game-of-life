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
#define GRID_WIDTH ((BMP_WIDTH)/(CELL_DIM))
#define GRID_HEIGHT ((BMP_HEIGHT)/(CELL_DIM))

// region dimension
#define REGION_DIM 10

// threads per block
#define THREADS_PER_BLOCK 64

// update delay
#define DELAY 25

// barrier for threads
static pthread_barrier_t barrier;

// grid struct
struct grid {
    int board[GRID_HEIGHT][GRID_WIDTH];

    grid(int val) {
        memset(board, val, sizeof(int) * GRID_HEIGHT * GRID_WIDTH);
    }

    void fill(int value) {
        memset(board, value, sizeof(int) * GRID_HEIGHT * GRID_WIDTH);
    }

    __host__ __device__ int get(int row, int col) {
        return this->board[row][col];
    }
    __host__ __device__ void set(int row, int col, int value) {
        this->board[row][col] = value;
    }
    __host__ __device__ void inc(int row, int col) {
        this->board[row][col]++;
    }
    __host__ __device__ void dec(int row, int col) {
        this->board[row][col]--;
    }
};


// grid struct
struct reggrid {
    int board[(int) GRID_HEIGHT / REGION_DIM][(int) GRID_WIDTH / REGION_DIM];

    reggrid(int val) { 
        memset(board, val, sizeof(int) * (GRID_HEIGHT/REGION_DIM) * (GRID_WIDTH/REGION_DIM));
    }

    __host__ __device__ int get(int row, int col) {
        return this->board[row][col];
    }
    __host__ __device__ void set(int row, int col, int value) {
        this->board[row][col] = value;
    }
    __host__ __device__ void inc(int row, int col) {
        this->board[row][col]++;
    }
    __host__ __device__ void dec(int row, int col) {
        this->board[row][col]--;
    }

    void fill(int value) {
        memset(board, value, sizeof(int) * GRID_HEIGHT * GRID_WIDTH);
    }
};

// coordinate struct
struct coord {
    int x;
    int y;

    coord() : x(0), y(0) {}
    coord(int x, int y) : x(x), y(y) {}
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

// colors!
#define NUM_COLORS 3

// array of colors for interpolation
rgb_f32 colors[NUM_COLORS + 1] = { 
    rgb_f32(0, 0, 255),
    rgb_f32(255, 0, 255),
    rgb_f32(0, 0, 255),
    rgb_f32(0, 0, 255),
};

// Preset colors
#define BLACK 0
#define WHITE 1

rgb32 preset_colors[2] = {
    rgb32(0, 0, 0),
    rgb32(255, 255, 255),
};

// indicate whether or not the simulation is paused
bool paused = true;

// bitmap screen variable
bitmap* bmp;

// grid for indicating cell state (dead or alive)
grid* g;

// grid for indicating alive cells in region
reggrid* regions; 

// create a GUI window
gui ui("Conway's Game of Life", BMP_WIDTH, BMP_HEIGHT);

// get input from the keyboard and execute proper command 
void* get_keyboard_input(void* params);

// get input from the mouse and toggle the appropriate cell's state/color
void* get_mouse_input(void* params);

// update each cell in order to advance the simulation
void update_cells();

// Toggle the cell's state, change the color accordingly
void fill_cell_with(coord loc, rgb32 color);

// Toggle with WHITE
void let_there_be_light(coord loc);

// Toggle with BLACK
void darkness_in_the_deep(coord loc);

// Set up the grid with an existing layout specified by a file
void load_grid(FILE * layout);

// Clear the board and bitmap
void clear_pixels();

void add_glider(coord loc);

rgb32 age_to_color(int age);


#endif
