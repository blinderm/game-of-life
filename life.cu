#include "life.hh"


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






// Get input from the keyboard and execute proper command 
void* getKeyboardInput(void* params) {

    while (you_failed) {

        pthread_barrier_wait(&barrier); // Waits for the Poll Event in main
        //puts("getting keyboard input");
        keyboard_args_t* args = (keyboard_args_t*) params;

        // If the "c" key is pressed, clear the board
        if(args->keyboard_state[SDL_SCANCODE_C]) {
            //puts("Cleared!\n");

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
            //puts("Pause!\n");
            paused = !(paused);
        }

        // If the "q" key is pressed, quit the simulation
        if(args->keyboard_state[SDL_SCANCODE_Q]) {
            //puts("quit!\n");
            you_failed = false;
        }
        

        pthread_barrier_wait(&barrier); // Releases the main function to run updates
    }

    return NULL;
}


// Get input from the mouse and toggle the appropriate cell's state/color
void* getMouseInput(void* params) {

    mouse_args_t* args = (mouse_args_t*) params;
    while(you_failed) {
        pthread_barrier_wait(&barrier); // Waits for the Poll Event in main

        // If the left mouse button is pressed, get position and toggle cell
        // TO DO: make this thing toggle only once per click/release
        // why is this bit & instead of logical &&?
        args->mouse_state = SDL_GetMouseState(&(args->loc.x), &(args->loc.y));
        //printf("[%d,%d]\n", args->loc.x, args->loc.y);

        if (args->mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) {
            //printf("left down!\n");
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

        pthread_barrier_wait(&barrier); // Releases the main function to run updates
    }

    return NULL;
}



// Update each cell in order to advance the simulation
void updateCells(void* params) {

    int counter = 0;
    for (int i = 0; i < GRID_HEIGHT; i++) {
      for(int j = 0; j < GRID_WIDTH; j++) {
        if (g->board[j][i]) {
          //printf("(%d,%d)\n", j, i);
          counter++;
        }
      }
    }
    //printf("%d alive\n", counter);

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

    // ui.display(*bmp);

}


// Toggle cell's color 
void toggleCell(coord_t loc) {

    
    // Indicate in the boolean grid that cell's state has been changed
    //printf("before: %d \n", g->board[loc.y/CELL_DIM][loc.x/CELL_DIM]);
    
    // if (g->board[loc.y][loc.x]) {
    //     g->board[loc.y][loc.x] = false;
    //     puts("set to false");
    // } else {
    //     g->board[loc.y][loc.x] = true;
    // } 
    // color for cell to be set
    g->board[loc.y/CELL_DIM][loc.x/CELL_DIM] = true;
    rgb32 color = g->board[loc.y/CELL_DIM][loc.x/CELL_DIM] ? WHITE : BLACK;
    //printf("after: %d \n", g->board[loc.y/CELL_DIM][loc.x/CELL_DIM]);

    // Find upper-left corner in boolean grid of cell
    int x_start = (loc.x / CELL_DIM) * CELL_DIM;
    int y_start = (loc.y / CELL_DIM) * CELL_DIM;

    // Loop over points in the bitmap to change color
    for (int x = x_start; x < x_start + CELL_DIM; x++) {
        for (int y = y_start; y < y_start + CELL_DIM; y++) {
            bmp->set(x, y, color);
        }
    }
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
    while(you_failed) {
        // Process events
        SDL_Event event;

        while(SDL_PollEvent(&event) == 1) {
            // If the event is a quit event, then leave the loop
            if(event.type == SDL_QUIT) {
                you_failed = false;
            }
        }


        // thread barrier for input threads
        pthread_barrier_wait(&barrier); // Releases the input threads to get input;
        pthread_barrier_wait(&barrier); // Waits for the input threads to finish

        if (!paused) {
          updateCells(keyboard_args);
          sleep_ms(25);
        }


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




/* some shitty thing david made a while back. keeping it here for old TIME's sake
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



