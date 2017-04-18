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

// Get input from the keyboard and execute proper command 
void getKeyboardInput(bitmap* bmp, grid_t* g);

// Get input from the mouse and toggle the appropriate cell's state/color
void getMouseInput(bitmap* bmp, grid_t* g);

// Toggle the cell's state, change the color accordingly
void toggleCell(bitmap* bmp, grid_t* g, coord_t loc);

// Update each cell in order to advance the simulation
void updateCells(bitmap* bmp, grid_t* g);


// Get input from the keyboard and execute proper command 
void getKeyboardInput(bitmap* bmp, grid_t* g) {

    // Get the keyboard state
    const uint8_t* keyboard = SDL_GetKeyboardState(NULL);

    // If the "c" key is pressed, clear the board
    if(keyboard[SDL_SCANCODE_C]) {

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
    if(keyboard[SDL_SCANCODE_P]) {
        // add a thing in the scheduler thing to be able to pause the thing
    }

    // If the "q" key is pressed, quit the simulation
    if(keyboard[SDL_SCANCODE_Q]) {
        // add a thing in the scheduler thing to be able to quit the thing 
        // (alternatively, just try to free the mouse and see what happens!)
    }
}


// Get input from the mouse and toggle the appropriate cell's state/color
void getMouseInput(bitmap* bmp, grid_t* g) {

    // Get the current mouse state
    coord_t loc;
    uint32_t mouse_state = SDL_GetMouseState(&loc.x, &loc.y);
    // Save the last time the mouse was clicked (IS THIS NECESSARY?)
    bool mouse_up = true;

    // If the left mouse button is pressed, get position and toggle cell
    if(mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) {
        // Only create one if the mouse button has been released
        if(mouse_up) {
            toggleCell(bmp, g, loc);

            // Don't create another one until the mouse button is released
            mouse_up = false;
        }
    } else {
        // The mouse button was released
        mouse_up = true;
    }
}


// Toggle cell's color 
void toggleCell(bitmap* bmp, grid_t* g, coord_t loc) {
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


// Update each cell in order to advance the simulation
void updateCells(bitmap* bmp, grid_t* g) {

    // To do: things
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

    // Render everything using this bitmap
    bitmap bmp(WIDTH, HEIGHT);

    // Create the grid
    grid_t* g = (grid_t*) malloc(sizeof(grid_t));
    memset(g->board, 0, sizeof(grid_t));

    
    // TO DO: ALL THE SCHEDULING


    
    ui.display(bmp);

    return 0;
}



