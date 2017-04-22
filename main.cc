#include <cmath>
#include <cstdio>
#include <ctime>
#include <vector>

#include <SDL.h>

#include "bitmap.hh"
#include "gui.hh"
#include "util.hh"
#include "scheduler.hh"

using namespace std;

// Screen size
#define WIDTH 800
#define HEIGHT 600

// Cell dimension
#define CELL_DIM 10

// grid struct
typedef struct grid {
    bool board[HEIGHT/CELL_DIM][WIDTH/CELL_DIM];
} grid_t;

// coordinate struct
typedef struct coord {
    int x;
    int y;
} coord_t;

// function parameter struct
typedef struct args {
    coord_t loc;
    const uint8_t* keyboard_state;
    uint32_t mouse_state;
    bool mouse_up;
} args_t;

// bitmap screen variable
bitmap* bmp;

// grid for indicating cell state (dead or alive)
grid_t* g;




// Get input from the keyboard and execute proper command 
void getKeyboardInput(void* params);

// Get input from the mouse and toggle the appropriate cell's state/color
void getMouseInput(void* params);

// Update each cell in order to advance the simulation
void updateCells(void* params);


// Toggle the cell's state, change the color accordingly
void toggleCell(coord_t loc);





// Get input from the keyboard and execute proper command 
void getKeyboardInput(void* params) {

    args_t* args = (args_t*) params;

    // If the "c" key is pressed, clear the board
    if(args->keyboard_state[SDL_SCANCODE_C]) {

        rgb32 color = rgb32(0.0, 0.0, 0.0);

        // Loop over points in the bitmap to change color
        for (int x = 0; x < WIDTH; x++) {
            for (int y = 0; y < HEIGHT; y++) {
                bmp->set(x, y, color);
            }
        }
        memset(g->board, 0, sizeof(grid_t));
    }

    // If the "p" key is pressed, toggle the pause-ness the simulation
    if(args->keyboard_state[SDL_SCANCODE_P]) {
        // add a thing in the scheduler thing to be able to pause the thing
        printf("I pressed 'p'\n");
    }

    // If the "q" key is pressed, quit the simulation
    if(args->keyboard_state[SDL_SCANCODE_Q]) {
        // add a thing in the scheduler thing to be able to quit the thing 
        // (alternatively, just try to free the mouse and see what happens!)
    }
}


// Get input from the mouse and toggle the appropriate cell's state/color
void getMouseInput(void* params) {

    args_t* args = (args_t*) params;
    
    // If the left mouse button is pressed, get position and toggle cell
    if(args->mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) {
        // Only create one if the mouse button has been released
        if(args->mouse_up) {
            toggleCell(args->loc);
            // Don't create another one until the mouse button is released
            args->mouse_up = false;
        }
    } else {
        // The mouse button was released
        args->mouse_up = true;
    }
}



// Update each cell in order to advance the simulation
void updateCells(void* params) {

    args_t* args = (args_t*) params;
    // To do: things
}


// Toggle cell's color 
void toggleCell(coord_t loc) {
    
    printf("clicked coordinate: (%d, %d)\n", loc.x, loc.y);
    // Indicate in the boolean grid that cell's state has been changed
    g->board[loc.y][loc.x] = !g->board[loc.y][loc.x]; 
    // color for cell to be set
    rgb32 color = g->board[loc.y][loc.x] ? rgb32(255.0, 255.0, 255.0) : rgb32(0.0, 0.0, 0.0);

    // Find upper-left corner in boolean grid of cell
    int x_start = loc.x * CELL_DIM;
    int y_start = loc.y * CELL_DIM;

    // Loop over points in the bitmap to change color
    for (int x = x_start; x < x_start + CELL_DIM; x++) {
        for (int y = y_start; y < y_start + CELL_DIM; y++) {
            bmp->set(x, y, color);
        }
    }
}


/**
 * Entry point for the program
 * \param argc  The number of command line arguments
 * \param argv  An array of command line arguments
 */
int main(int argc, char** argv) {

    // Seed the random number generator
    srand(time(NULL));

    // Create a GUI window
    gui ui("Conway's Game of Life", WIDTH, HEIGHT);

    bitmap bits(WIDTH, HEIGHT);

    bmp = &bits;
    // Render everything using this bitmap
    // bitmap bmp(WIDTH, HEIGHT);

    // Create the grid
    g = (grid_t*) malloc(sizeof(grid_t));
    memset(g->board, 0, sizeof(grid_t));

    /*
    // Create a job to read user input every 150ms
    add_job(getKeyboardInput, 150);
    add_job(getMouseInput, 150);

    // Create a job to update apples every 120ms
    add_job(updateCells, 120);
    */

    while(1) {

        SDL_Event event;
        while(SDL_PollEvent(&event) == 1) {
            // If the event is a quit event, then leave the loop
            if(event.type == SDL_QUIT) {
                stop_scheduler();
            }
        }

        args_t* args = (args_t*) malloc(sizeof(args_t));
        args->keyboard_state = SDL_GetKeyboardState(NULL);
        args->mouse_state = SDL_GetMouseState(&(args->loc.x), &(args->loc.y));
        args->mouse_up = true;

        getKeyboardInput((void*) args); 
        getMouseInput((void*) args);
        updateCells((void*) args);

        ui.display(*bmp);
    }
    // TO DO: Generate a couple cells immediately

    // Run the task queue

    //run_scheduler();


    return 0;
}



