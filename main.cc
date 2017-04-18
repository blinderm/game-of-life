#include <cmath>
#include <cstdio>
#include <ctime>
#include <vector>

#include <SDL.h>

#include "bitmap.hh"
#include "gui.hh"
#include "util.hh"

using namespace std;

// Screen size
#define WIDTH 800
#define HEIGHT 600

// Cell dimension
#define CELL_DIM 10

// Minimum time between clicks
#define CREATE_INTERVAL 1000

// Time step size
#define DT 0.04

// grid struct
typedef struct grid {
    bool board[HEIGHT/CELL_DIM][WIDTH/CELL_DIM];
} grid_t;

// coordinate struct
typedef struct coord {
    int x;
    int y;
} coord_t;

// Toggle 
void toggleCell(bitmap* bmp, grid_t* g, coord_t loc);


/**
 * Entry point for the program
 * \param argc  The number of command line arguments
 * \param argv  An array of command line arguments
 */
int main(int argc, char** argv)   {

    // Seed the random number generator
    srand(time(NULL));

    // Create a GUI window
    gui ui("Conway's Game of Life", WIDTH, HEIGHT);

    // Render everything using this bitmap
    bitmap bmp(WIDTH, HEIGHT);

    // Save the last time the mouse was clicked
    bool mouse_up = true;

    // Start with the running flag set to true
    bool running = true;

    // Loop until we get a quit event
    while(running) {

        // Process events
        SDL_Event event;
        while(SDL_PollEvent(&event) == 1) {
            // If the event is a quit event, then leave the loop
            if(event.type == SDL_QUIT) running = false;
        }


        /* TO DO 
        // Get the current mouse state
        int mouse_x, mouse_y;
        uint32_t mouse_state = SDL_GetMouseState(&mouse_x, &mouse_y);


        // If the left mouse button is pressed, get position and toggle cell
        if(mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) {
            // Only create one if the mouse button has been released
            if(mouse_up) {
                addRandomGalaxy(mouse_x - x_offset, mouse_y - y_offset);

                // Don't create another one until the mouse button is released
                mouse_up = false;
            }
        } else {
            // The mouse button was released
            mouse_up = true;
        }
        // Get the keyboard state
        const uint8_t* keyboard = SDL_GetKeyboardState(NULL);


        // If the "c" key is pressed, clear the board
        if(keyboard[SDL_SCANCODE_UP]) {
            y_offset++;
            bmp.shiftDown();  // Shift pixels so scrolling doesn't create trails
        }

        // If the "p" key is pressed, toggle the pause-ness the simulation
        if(keyboard[SDL_SCANCODE_DOWN]) {
            y_offset--;
            bmp.shiftUp();  // Shift pixels so scrolling doesn't create trails
        }
        */

        ui.display(bmp);
    }
    return 0;
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
