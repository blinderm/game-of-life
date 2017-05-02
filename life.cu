#include "life.hh"
// use Conway's update algorithm to decide whether or not to toggle cell 
__global__ void life_or_death(grid* gpu_g) {

    size_t index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    // establish boundaries for checking neighbors
    int row = index / GRID_WIDTH;
    int col = index % GRID_WIDTH;
    int left = max(0, col - 1);
    int right = min(GRID_WIDTH - 1, col + 1);
    int top = max(0, row - 1);
    int bottom = min(GRID_HEIGHT - 1, row + 1);

    int alive_neighbors = 0;
    for(int j = left; j <= right; j++) {
        for(int i = top; i <= bottom; i++) {
            alive_neighbors += (!(col == j && row == i) && gpu_g->board[i][j]) ? 1 : 0;
        }
    }

    if (alive_neighbors < 2 || alive_neighbors > 3) {
        // clear the cell
        gpu_g->board[row][col] = 0;
    }
    else if (alive_neighbors == 2) {
        // do nothing!
        if (gpu_g->board[row][col]) gpu_g->board[row][col] ++;
    }
    else { // (alive_neighbors == 3)
        // light up the cell
        gpu_g->board[row][col] ++;
    }
}


// get input from the keyboard and execute proper command 
// UPDATE: currently, this only works as CTRL-P, CTRL-C, and CTRL-Q
void* get_keyboard_input(void* params) {

    bool clear = false;
    bool pause = false;
    bool quit = false;

    input_args* args = (input_args*) params;
    while (running) {
        // waits for the Poll Event in main
        pthread_barrier_wait(&barrier);

        // if the "c" key is pressed, clear the board
        switch (args->event->type) {
            case SDL_KEYDOWN:
                switch (args->event->key.keysym.scancode) {
                    case SDL_SCANCODE_C:
                        clear = true;
                        break;
                    case SDL_SCANCODE_P:
                        pause = true;
                        break;
                    case SDL_SCANCODE_Q:
                        quit = true;
                        break;
                    default:
                        break;
                }
                break;
            case SDL_KEYUP:
                switch (args->event->key.keysym.scancode) {
                    case SDL_SCANCODE_C:
                        if (clear) {
                            for (int x = 0; x < BMP_WIDTH; x++) {
                                for (int y = 0; y < BMP_HEIGHT; y++) {
                                    bmp->set(x, y, BLACK);
                                }
                            }
                            memset(g->board, 0, sizeof(grid));
                            puts("Cleared");
                            clear = false;
                        }
                        break;
                    case SDL_SCANCODE_P:
                        if (pause) {
                            paused = !(paused);
                            pause = false;
                            puts("Pause toggle!");
                        }
                        break;
                    case SDL_SCANCODE_Q:
                        if (quit) {
                            running = false;
                            quit = false;
                        }
                        break;
                    default:
                        break;
                }
                break;
            default:
                break;
        }
        // releases the main function to run updates
        pthread_barrier_wait(&barrier);
    }

    return NULL;
}


// get input from the mouse and toggle the appropriate cell's state/color
void* get_mouse_input(void* params) {

    input_args* args = (input_args*) params;
    while(running) {

        // waits for the Poll Event in main
        pthread_barrier_wait(&barrier); 

        // if the left mouse button is pressed, get position and toggle cell
        // TO DO: make this thing toggle only once per click/release
        args->mouse_state = SDL_GetMouseState(&(args->loc.x), &(args->loc.y));

        if (args->mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) {

            let_there_be_light(args->loc);
        }

        pthread_barrier_wait(&barrier); // releases the main function to run updates
    }

    return NULL;
}



// update each cell in order to advance the simulation
void update_cells() {

    // allocate space for GPU grid
    grid* gpu_g;
    if (cudaMalloc(&gpu_g, sizeof(grid)) != cudaSuccess) {
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
    if (cudaMemcpy(gpu_g, g, sizeof(grid), cudaMemcpyHostToDevice) != cudaSuccess) {
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

    // copy the GPU grid back to the CPU
    if (cudaMemcpy(g, gpu_g, sizeof(grid), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy grid from the GPU\n");
    }

    // copy the CPU bitmap to the GPU bitmap
    if (cudaMemcpy(bmp, gpu_bmp, sizeof(bitmap), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy bitmap from the GPU\n");
    }

    // loop over points in the bitmap to change color
    for(int row = 0; row < BMP_HEIGHT; row++){
        for(int col = 0; col < BMP_WIDTH; col++){
            rgb32 color = age_to_color(g->board[row / CELL_DIM][col / CELL_DIM]);
            bmp->set(col, row, color);
        }
    }

    // free everything we malloc'ed
    cudaFree(gpu_g);
    cudaFree(gpu_bmp);
}

// 
void let_there_be_light(coord loc) {
    // indicate in the boolean grid that cell's state has been changed
    g->board[loc.y/CELL_DIM][loc.x/CELL_DIM] = true;
    rgb32 color = g->board[loc.y/CELL_DIM][loc.x/CELL_DIM] ? WHITE : BLACK;

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

void load_grid(FILE * layout) {
    coord loc;
    loc.x = 0, loc.y = 0;
    char ch;

    while ((ch = getc(layout)) != EOF) {
        if (ch == '\n') {
            loc.x = 0;
            loc.y ++;
        }
        else {
            if (ch != ' ') {
                let_there_be_light(loc);
            }
            loc.x ++;
        }
    }
}



rgb32 age_to_color(int age) {

    return age == 0 ? BLACK : rgb32(0, max(255 - 50 * age, 0), 200);
}

/**
 * Entry point for the program
 */
int main(int argc, char ** argv) {

    // Create the bitmap 
    bitmap bits(BMP_WIDTH, BMP_HEIGHT);
    bmp = &bits;

    // Create the grid
    grid grd(0);
    g = &grd;

    if (argc > 1) {
        FILE * fp;
        fp = fopen(argv[1], "r");
        load_grid(fp);
        fclose(fp);
    }

    SDL_Event event;

    // struct of arguments for mouse function
    input_args mouse_args(&event);
    input_args keyboard_args(&event);

    ui.display(*bmp);

    // initialize barrier
    pthread_barrier_init(&barrier, NULL, 3);

    // set up threads
    pthread_t mouse_thread, keyboard_thread;

    if (pthread_create(&mouse_thread, NULL, get_mouse_input, (void*) (&mouse_args))) {
        perror("error in pthread_create.\n");
        exit(2);
    }
    if (pthread_create(&keyboard_thread, NULL, get_keyboard_input, (void*) (&keyboard_args))) {
        perror("error in pthread_create.\n");
        exit(2);
    }

    // loop until we get a quit event
    while(running) {

        // process events
        while(SDL_PollEvent(&event) == 1); 

        // releases the input threads to get input;
        pthread_barrier_wait(&barrier); 
        // waits for the input threads to finish
        pthread_barrier_wait(&barrier); 

        if (!paused) {
            update_cells();
            sleep_ms(DELAY);
        }

        // display the rendered frame
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
