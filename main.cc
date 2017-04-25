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
  bool board[(int) (HEIGHT/CELL_DIM)][(int) (WIDTH/CELL_DIM)];
} grid_t;

// coordinate struct
typedef struct coord {
  int x;
  int y;
} coord_t;

bool quit = false;

// function parameter struct
// TO DO: not this? 
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

// Create a GUI window
gui ui("Conway's Game of Life", WIDTH, HEIGHT);


// Get input from the keyboard and execute proper command 
void getKeyboardInput(void* params);

// Get input from the mouse and toggle the appropriate cell's state/color
void getMouseInput(void* params);

// Update each cell in order to advance the simulation
void updateCells(void* params);

// display the screen 
void displayBMP(void* params);

// Toggle the cell's state, change the color accordingly
void toggleCell(coord_t loc);

void loadGrid(FILE * layout);

// Get input from the keyboard and execute proper command 
void getKeyboardInput(void* params) {

  puts("getting keyboard input");
  args_t* args = (args_t*) params;

  // If the "c" key is pressed, clear the board
  if(args->keyboard_state[SDL_SCANCODE_C]) {
    puts("Cleared!\n");

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
    puts("Pause!\n");
    // TO DO: add a thing in the scheduler thing to be able to pause the thing
  }

  // If the "q" key is pressed, quit the simulation
  if(args->keyboard_state[SDL_SCANCODE_Q]) {
    puts("quit!\n");
    quit = true;
    stop_scheduler();
    return;
  }
}


// Get input from the mouse and toggle the appropriate cell's state/color
void getMouseInput(void* params) {

  puts("getting mouse input");

  args_t* args = (args_t*) params;

  // If the left mouse button is pressed, get position and toggle cell
  // TO DO: make this thing toggle only once per click/release
  // why is this bit & instead of logical &&?
  if((args->mouse_state = SDL_GetMouseState(&(args->loc.x), &(args->loc.y))) 
        & SDL_BUTTON(SDL_BUTTON_LEFT)) {
    printf("left down!\n");
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

  puts("updating cell");

  args_t* args = (args_t*) params;
  // TO DO: GPU WOOOOO
}


// Toggle cell's color 
void toggleCell(coord_t loc) {

  printf("clicked coordinate: (%d, %d)\n", loc.x, loc.y);
  // Indicate in the boolean grid that cell's state has been changed
  printf("before: %d \n", g->board[loc.y][loc.x]);
  g->board[loc.y][loc.x] = !g->board[loc.y][loc.x]; 
  printf("after: %d \n", g->board[loc.y][loc.x]);
  // color for cell to be set
  rgb32 color = g->board[loc.y][loc.x] ? rgb32(255.0, 255.0, 255.0) : rgb32(0.0, 0.0, 0.0);

  // Find upper-left corner in boolean grid of cell
  int x_start = (loc.x / CELL_DIM) * CELL_DIM;
  int y_start = (loc.y / CELL_DIM) * CELL_DIM;
  printf("start loc: (%d, %d)\n", x_start, y_start);

  // Loop over points in the bitmap to change color
  for (int x = x_start; x < x_start + CELL_DIM; x++) {
    for (int y = y_start; y < y_start + CELL_DIM; y++) {
      bmp->set(x, y, color);
    }
  }
}

// display screen
void displayBMP(void* args) {
  ui.display(*bmp);
}

void loadGrid(FILE * layout) {
  coord_t loc;
  loc.x = 0, loc.y = 0;
  char ch;

  while ((ch = getc(layout)) != EOF) {
    if (ch == '\n') {
      loc.x = 0;
      loc.y ++;
    }
    else {
      if (ch != ' ') {
        toggleCell(loc);
      }
      loc.x ++;
    }
  }
}

/**
 * Entry point for the program
 */
int main(int argc, char ** argv) {

  // Create the bitmap 
  bitmap bits(WIDTH, HEIGHT);
  bmp = &bits;

  // Create the grid
  g = (grid_t*) malloc(sizeof(grid_t));
  memset(g->board, 0, sizeof(grid_t));

  if (argc > 1) {
    FILE * fp;
    fp = fopen(argv[1], "r");
    loadGrid(fp);
    fclose(fp);
  }


  // struct of arguments for functions in scheduler
  args_t* args = (args_t*) malloc(sizeof(args_t));
  args->keyboard_state = SDL_GetKeyboardState(NULL);
  args->mouse_state = SDL_GetMouseState(&(args->loc.x), &(args->loc.y));
  args->mouse_up = true;

  /*
  // Add jobs to scheduler
  add_job(displayBMP, 1, (void*) args);
  add_job(getKeyboardInput, 1, (void*) args);
  add_job(getMouseInput, 1, (void*) args);
  add_job(updateCells, 2, (void*) args);

  run_scheduler();
  */

  int i = 0;

  while(!quit) {
    printf("%d\n", i++);
    SDL_Event event;
    while(SDL_PollEvent(&event) == 1) {
      // If the event is a quit event, then leave the loop
      if(event.type == SDL_QUIT) {
        quit = true;
        stop_scheduler();
      }
    }

    getKeyboardInput((void*) args); 
    getMouseInput((void*) args);
    updateCells((void*) args);

    ui.display(*bmp);
  }

  return 0;
}
