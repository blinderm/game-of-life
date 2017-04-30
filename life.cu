#include <math.h>

#include <stdio.h>
#include <time.h>
#include <vector>

#include <SDL.h>
#include <pthread.h>

#include "bitmap.hh"
#include "gui.hh"
#include "util.hh"
#include "scheduler.hh"

using namespace std;

// Screen size
#define BMP_WIDTH 800
#define BMP_HEIGHT 600

// Cell dimension
#define CELL_DIM 10

// Grid size
#define GRID_WIDTH (BMP_WIDTH/CELL_DIM)
#define GRID_HEIGHT (BMP_HEIGHT/CELL_DIM)

// Threads per block
#define THREADS_PER_BLOCK 64

// Colors!
#define WHITE rgb32(255.,255.,255.)
#define BLACK rgb32(0.,0.,0.)



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

// for old times' sake
bool we_tried = true;

// mouse function parameter struct
typedef struct mouse_args {
    coord_t loc;
    uint32_t mouse_state;
    bool mouse_up;
} mouse_args_t;

// keyboard function parameter struct
typedef struct keyboard_args {
    coord_t loc;
    const uint8_t* keyboard_state;
} keyboard_args_t;


// bitmap screen variable
bitmap* bmp;

// grid for indicating cell state (dead or alive)
grid_t* g;

// Create a GUI window
gui ui("Conway's Game of Life", BMP_WIDTH, BMP_HEIGHT);


// Get input from the keyboard and execute proper command 
void* getKeyboardInput(void* params);

// Get input from the mouse and toggle the appropriate cell's state/color
void* getMouseInput(void* params);

// Update each cell in order to advance the simulation
void updateCells(void* params);

// display the screen 
void displayBMP(void* params);

// Toggle the cell's state, change the color accordingly
void toggleCell(coord_t loc);

// Set up the grid with an existing layout specified by a file
void loadGrid(FILE * layout);


/*
   void print_grid(grid_t* g) {
   for (int i = 0; i < GRID_WIDTH; i++) {
   for (int j = 0; j < GRID_HEIGHT; j++) {
   putchar(g->board[i][j] ? 'X' : ' ');
   }
   putchar('\n');
   }
   putchar('\n');
   }
 */



//  Use Conway's update algorithm to decide whether or not to toggle cell 
__global__ void life_or_death(grid_t* gpu_g) {

    size_t index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    // first thing's first: establish boundaries
    int row = index / GRID_WIDTH;
    int col = index % GRID_WIDTH;
    int left = max(0, col - 1);
    int right = min(GRID_WIDTH - 1, col + 1);
    int top = max(0, row - 1);
    int bottom = min(GRID_HEIGHT - 1, row + 1);

    int i,j;
    int alive_neighbors = 0;
    for(j = left; j <= right; j++) {
        for(i = top; i <= bottom; i++) {
            alive_neighbors += (!(col == j && row == i) && gpu_g->board[i][j]) ? 1 : 0;
        }
    }

    /*
       if (alive_neighbors > 0) {
       printf("high\n");
       }
     */

    // if (col > row) printf(" akdfjn\n");

    if (alive_neighbors < 2 || alive_neighbors > 3) {
        // clear the cell
        gpu_g->board[row][col] = false;
    }
    else if (alive_neighbors == 2) {
        // do nothing!
    }
    else { // (alive_neighbors == 3)
        // light up the cell
        gpu_g->board[row][col] = true;
    }
}



/* "this should absolutely happen in parallel, but i'm okay proceeding in serial so that we can work on other things" - a nervous david who transcends temporal dimensions
   __global__ void updateBMP(grid_t* gpu_g, bitmap* gpu_bmp) {
   size_t bmp_index = blockIdx.x * blockDim.x + threadIdx.x;
   int bmp_row = bmp_index / BMP_WIDTH;
   int bmp_col = bmp_index % BMP_WIDTH;
   bool alive = gpu_g->board[bmp_row / CELL_DIM][bmp_col / CELL_DIM];
   rgb32 color = alive ? WHITE : BLACK;
// gpu_bmp->set(bmp_row, bmp_col, color);
}
 */






// Get input from the keyboard and execute proper command 
void* getKeyboardInput(void* params) {

    while (we_tried) {

        pthread_barrier_wait(&barrier);
        //puts("getting keyboard input");
        keyboard_args_t* args = (keyboard_args_t*) params;

        // If the "c" key is pressed, clear the board
        if(args->keyboard_state[SDL_SCANCODE_C]) {
            puts("Cleared!\n");

            // Loop over points in the bitmap to change color
            for (int x = 0; x < BMP_WIDTH; x++) {
                for (int y = 0; y < BMP_HEIGHT; y++) {
                    bmp->set(x, y, BLACK);
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
            we_tried = false;
            //return NULL;
        }

        pthread_barrier_wait(&barrier);
    }

    return NULL;
}


// Get input from the mouse and toggle the appropriate cell's state/color
void* getMouseInput(void* params) {

    while(we_tried) {
        pthread_barrier_wait(&barrier);

        mouse_args_t* args = (mouse_args_t*) params;

        // If the left mouse button is pressed, get position and toggle cell
        // TO DO: make this thing toggle only once per click/release
        // why is this bit & instead of logical &&?
        args->mouse_state = SDL_GetMouseState(&(args->loc.x), &(args->loc.y));

        if (args->mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) {
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

        pthread_barrier_wait(&barrier);
    }

    return NULL;
}



// Update each cell in order to advance the simulation
void updateCells(void* params) {

    // allocate space for GPU grid
    grid_t* gpu_g;
    if (cudaMalloc(&gpu_g, sizeof(grid_t)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate grid on GPU\n");
        exit(2);
    }

    // alocate space for GPU bitmap
    bitmap* gpu_bmp;
    if (cudaMalloc(&gpu_bmp, sizeof(bitmap)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate bitmap on GPU\n");
        exit(2);
    }

    // copy the CPU grid to the GPU grid
    if (cudaMemcpy(gpu_g, g, sizeof(grid_t), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy grid to the GPU\n");
    }

    // copy the CPU bitmap to the GPU bitmap
    if (cudaMemcpy(gpu_bmp, bmp, sizeof(bitmap), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy bitmap to the GPU\n");
    }

    // number of block to run (rounding up to include all threads)
    size_t grid_blocks = (GRID_WIDTH*GRID_HEIGHT + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    life_or_death<<<grid_blocks, THREADS_PER_BLOCK>>>(gpu_g);
    cudaDeviceSynchronize();
    //print_grid(gpu_g);

    // copy the GPU grid back to the CPU
    if (cudaMemcpy(g, gpu_g, sizeof(grid_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy grid from the GPU\n");
    }

    // copy the CPU bitmap to the GPU bitmap
    if (cudaMemcpy(bmp, gpu_bmp, sizeof(bitmap), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy bitmap from the GPU\n");
    }


    g->board[40][10] = 1;
    g->board[40][11] = 1;
    g->board[40][12] = 1;

    // Loop over points in the bitmap to change color
    for(int row = 0; row < BMP_HEIGHT; row++){
        for(int col = 0; col < BMP_WIDTH; col++){
            rgb32 color = g->board[row / CELL_DIM][col / CELL_DIM] ? WHITE : BLACK;
            bmp->set(col, row, color);
        }
    }


    /*
       updateBMP<<<bmp_blocks, THREADS_PER_BLOCK>>>(gpu_g, gpu_bmp);
       cudaDeviceSynchronize();
     */

    // free everything we malloc'ed
    cudaFree(gpu_g);
    cudaFree(gpu_bmp);

    ui.display(*bmp);

}


// Toggle cell's color 
void toggleCell(coord_t loc) {

    
    // Indicate in the boolean grid that cell's state has been changed
    printf("before: %d \n", g->board[loc.y][loc.x]);
    
    if (g->board[loc.y][loc.x]) {
        g->board[loc.y][loc.x] = false;
        puts("set to false");
    } else {
        g->board[loc.y][loc.x] = true;
    } 
    // color for cell to be set
    rgb32 color = g->board[loc.y][loc.x] ? WHITE : BLACK;
    printf("after: %d \n", g->board[loc.y][loc.x]);

    // Find upper-left corner in boolean grid of cell
    int x_start = (loc.x / CELL_DIM) * CELL_DIM;
    int y_start = (loc.y / CELL_DIM) * CELL_DIM;
    //printf("start loc: (%d, %d)\n", x_start, y_start);

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
    bitmap bits(BMP_WIDTH, BMP_HEIGHT);
    bmp = &bits;

    // Create the grid
    g = (grid_t*) malloc(sizeof(grid_t));
    for (int col = 0; col < GRID_WIDTH; col++) {
        for (int row = 0; row < GRID_HEIGHT; row++) {
            g->board[row][col] = false;
        }
    }

    if (argc > 1) {
        FILE * fp;
        fp = fopen(argv[1], "r");
        loadGrid(fp);
        fclose(fp);
    }

    // struct of arguments for mouse function
    mouse_args_t* mouse_args = (mouse_args_t*) malloc(sizeof(mouse_args_t));
    mouse_args->mouse_state = SDL_GetMouseState(&(mouse_args->loc.x), &(mouse_args->loc.y));
    mouse_args->mouse_up = true;

    // struct of arguments for keyboard function
    keyboard_args_t* keyboard_args = (keyboard_args_t*) malloc(sizeof(keyboard_args_t));
    keyboard_args->keyboard_state = SDL_GetKeyboardState(NULL);

    ui.display(*bmp);

    // Initialize barrier
    pthread_barrier_init(&barrier, NULL, 3);

    // Set up threads
    pthread_t mouse_thread, keyboard_thread;

    if (pthread_create(&mouse_thread, NULL, getMouseInput, (void*) mouse_args)) {
        perror("error in pthread_create.\n");
        exit(2);
    }
    if (pthread_create(&keyboard_thread, NULL, getKeyboardInput, (void*) keyboard_args)) {
        perror("error in pthread_create.\n");
        exit(2);
    }

    // Loop until we get a quit event
    while(we_tried) {
        // Process events
        SDL_Event event;
        while(SDL_PollEvent(&event) == 1) {
            // If the event is a quit event, then leave the loop
            if(event.type == SDL_QUIT) we_tried = false;
        }

        // thread barriers
        pthread_barrier_wait(&barrier);
        pthread_barrier_wait(&barrier);

        // Display the rendered frame
        ui.display(*bmp);
   }

    // join threads
    if (pthread_join(mouse_thread, NULL)) {
        perror("Failed joining.\n");
        exit(2);
    }
    if (pthread_join(keyboard_thread, NULL)) {
        perror("Failed joining.\n");
        exit(2);    
    }

    return 0;
}

// for zachary
