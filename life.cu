#include "life.hh"

// count neighbors
__global__ void count_neighbors(grid* gpu_g, grid* gpu_neighbors) {

    size_t index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    // establish boundaries for checking neighbors
    int row = index / GRID_WIDTH;
    int col = index % GRID_WIDTH;
    int left = max(0, col - 1);
    int right = min(GRID_WIDTH - 1, col + 1);
    int top = max(0, row - 1);
    int bottom = min(GRID_HEIGHT - 1, row + 1);

    gpu_neighbors->set(row, col, 0);

    for (int r = top; r <= bottom; r++) {
        for (int c = left; c <= right; c++) {
            if (!(col == c && row == r) && (gpu_g->get(r, c) > 0)) {
                gpu_neighbors->inc(row, col);
            }
        }
    }

}

// use Conway's update algorithm to decide whether or not to toggle cell 
__global__ void life_or_death(grid* gpu_g, grid* gpu_neighbors) {

    size_t index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    // establish boundaries for checking neighbors
    int row = index / GRID_WIDTH;
    int col = index % GRID_WIDTH;

    switch(gpu_neighbors->get(row, col)) {
        case 2: // alive cell stays alive; dead cell stays dead
            if(gpu_g->get(row, col) > 0) { // alive cell stays alive
                gpu_g->inc(row, col);
            }
            break;
        case 3: // alive cell stays alive; dead cell comes alive
            gpu_g->inc(row, col);
            break;
        default: // alive cell dies; dead cell stays dead
            gpu_g->set(row, col, 0);
            break;
    }

}


// get input from the keyboard and execute proper command 
void* get_keyboard_input(void* params) {

    input_args* args = (input_args*) params;

    bool clear = false;
    bool pause = false;
    bool step = false;
    bool quit = false;
    bool glider = false;

    while (running) {
        // waits for the Poll Event in main
        pthread_barrier_wait(&barrier);

        switch (args->event->type) {
            case SDL_KEYDOWN:
                switch (args->event->key.keysym.scancode) {
                    case SDL_SCANCODE_C:
                        clear = true;
                        break;
                    case SDL_SCANCODE_G:
                        glider = true;
                        break;
                    case SDL_SCANCODE_P:
                        pause = true;
                        break;
                    case SDL_SCANCODE_SPACE:
                        step = paused;
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
                            clear_pixels();
                            puts("Cleared");
                            clear = false;
                        }
                        break;
                    case SDL_SCANCODE_G:
                        if (glider) {
                            add_glider(args->loc);
                            puts("Glider");
                            glider = false;
                        }
                        break;
                    case SDL_SCANCODE_P:
                        if (pause) {
                            paused = !(paused);
                            puts("Pause toggle!");
                            pause = false;
                        }
                        break;
                    case SDL_SCANCODE_SPACE:
                        if (step) {
                            update_cells();
                            step = false;
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

        // if the left mouse button is pressed, get position and toggle cell appropriately
        args->mouse_state = SDL_GetMouseState(&(args->loc.x), &(args->loc.y));

        if (args->mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) {
            // bring cell to life
            let_there_be_light(args->loc);
        }
        if (args->mouse_state & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
            // kill cell
            darkness_in_the_deep(args->loc);
        }

        // releases the main function to run updates
        pthread_barrier_wait(&barrier); 
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
    // allocate space for GPU neighbors
    grid* gpu_neighbors;
    if (cudaMalloc(&gpu_neighbors, sizeof(grid)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate grid on GPU\n");
        exit(2);
    }
  
    // copy the CPU grid to the GPU grid
    if (cudaMemcpy(gpu_g, g, sizeof(grid), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy grid to the GPU\n");
    }
    // copy the CPU neighbors to the GPU neighbors 
    if (cudaMemcpy(gpu_neighbors, g, sizeof(grid), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy neighbors grid to the GPU\n");
    }

    // number of block to run (rounding up to include all threads)
    size_t grid_blocks = (GRID_WIDTH*GRID_HEIGHT + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    count_neighbors<<<grid_blocks, THREADS_PER_BLOCK>>>(gpu_g, gpu_neighbors);
    cudaDeviceSynchronize();
    life_or_death<<<grid_blocks, THREADS_PER_BLOCK>>>(gpu_g, gpu_neighbors);
    cudaDeviceSynchronize();

    // copy the GPU grid back to the CPU
    if (cudaMemcpy(g, gpu_g, sizeof(grid), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy grid from the GPU\n");
    }    

    // loop over points in the bitmap to change color
    for(int row = 0; row < BMP_HEIGHT; row++){
        for(int col = 0; col < BMP_WIDTH; col++){
            rgb32 color = age_to_color(g->get(row / CELL_DIM, col / CELL_DIM));
            bmp->set(col, row, color);
        }
    }

    // free everything we malloc'ed
    cudaFree(gpu_g->board);
    cudaFree(gpu_g);
}

// fill an entire cell with the given color
void fill_cell_with(coord loc, rgb32 color) {

    // find upper-left corner in grid of cells
    int x_start = (loc.x / CELL_DIM) * CELL_DIM;
    int y_start = (loc.y / CELL_DIM) * CELL_DIM;

    // loop over points in the bitmap to change color
    for (int x = x_start; x < x_start + CELL_DIM; x++) {
        for (int y = y_start; y < y_start + CELL_DIM; y++) {
            bmp->set(x, y, color);
        }
    }
}

// toggle cell with WHITE
void let_there_be_light(coord loc) {

    g->set(loc.y/CELL_DIM, loc.x/CELL_DIM, 1);
    fill_cell_with(loc, colors[0]);
}

// toggle cell with BLACK
void darkness_in_the_deep(coord loc) {

    g->set(loc.y/CELL_DIM, loc.x/CELL_DIM, 0);
    fill_cell_with(loc, preset_colors[BLACK]);
}

// set up the grid with an existing layout specified by a file
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

// helper function for linear color interpolation
rgb_f32 interpolate_colors(int current_age, int old_age, int new_age, 
        rgb_f32 old_color, rgb_f32 new_color) {

    rgb_f32 slope = (new_color - old_color) * (1. / (float) (new_age- old_age));
    rgb_f32 current_color = old_color + slope * current_age;

    return current_color;
}

// function for linear color interpolation
rgb32 age_to_color(int age) {

    // dead cells are black, which is different behavior from living cells
    if (age == 0) {
        return preset_colors[BLACK];
    }

    // living cells "age" in the following way:
    int transition_time = 7;

    int interp = min(NUM_COLORS - 1, age / transition_time);
    rgb_f32 color = interpolate_colors(
            age, 
            interp * transition_time, 
            (interp + 1) * transition_time,
            colors[interp],
            colors[interp+1]);

    return rgb32(color);
}


// entry point for the program
int main(int argc, char ** argv) {

    // create the bitmap 
    bitmap bits(BMP_WIDTH, BMP_HEIGHT);
    bmp = &bits;

    // create the grid
    grid grd(0);
    g = &grd;

    // load grid from file specified by user where appropriate
    if (argc > 1) {
        FILE * fp;
        fp = fopen(argv[1], "r");
        load_grid(fp);
        fclose(fp);
    }

    // function parameter structs
    SDL_Event event;
    input_args mouse_args(&event);
    input_args keyboard_args(&event);

    // initialize barrier
    pthread_barrier_init(&barrier, NULL, 3);

    // display the bitmap
    ui.display(*bmp);

    // set up threads and run
    pthread_t mouse_thread, keyboard_thread;

    if (pthread_create(&mouse_thread, NULL, get_mouse_input, (void*) (&mouse_args))) {
        perror("error in pthread_create.\n");
        exit(2);
    }
    if (pthread_create(&keyboard_thread, NULL, get_keyboard_input, (void*) (&keyboard_args))) {
        perror("error in pthread_create.\n");
        exit(2);
    }

    // create file to export evaluations data
    char name[30];
    sprintf(name, "data/%dTPB_0RD.csv", THREADS_PER_BLOCK);
    FILE *data = fopen(name, "w");
    if (data == NULL) {
        printf("error in fopen\n");
        exit(2);
    }
    fprintf(data, "threads_per_block,region_dim,num_iterations,time\n");

    size_t start_time, end_time;

    int iterations = 0;

    // loop until we get a quit event
    while(running) {

        // process events
        while(SDL_PollEvent(&event) == 1) {
            // if the event is a quit event, then leave the loop
            if(event.type == SDL_QUIT) running = false;
        }

        // releases the input threads to get input
        pthread_barrier_wait(&barrier); 
        // waits for the input threads to finish
        pthread_barrier_wait(&barrier); 

        // time the update function only for evaluations
        if (!paused) {
            start_time = time_ms();
            update_cells();
            end_time = time_ms();
            fprintf(data, "%d,0,%Iu,%d\n", THREADS_PER_BLOCK,
                    iterations++, end_time - start_time);
            sleep_ms(DELAY);
        }

        if (iterations > 1000){
          puts("Over 1000");
        }

        // display the rendered frame
        ui.display(*bmp);
    }

    fclose(data);

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

// clear the board and the bitmap
void clear_pixels() {
    bmp->fill(preset_colors[BLACK]);
    g->fill(0);
}

// add a glider shape to the board
void add_glider(coord loc) {

    SDL_GetMouseState(&(loc.x), &(loc.y));

    let_there_be_light(loc);
    let_there_be_light(coord(loc.x + CELL_DIM, loc.y + CELL_DIM));
    let_there_be_light(coord(loc.x + CELL_DIM, loc.y + 2 * CELL_DIM));
    let_there_be_light(coord(loc.x , loc.y + 2 * CELL_DIM));
    let_there_be_light(coord(loc.x - CELL_DIM, loc.y + 2 * CELL_DIM));
}
