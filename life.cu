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
__global__ void life_or_death(grid* gpu_g, grid* gpu_neighbors, grid* gpu_regions) {

    size_t index = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

    // establish boundaries for checking neighbors
    int row = index / GRID_WIDTH;
    int col = index % GRID_WIDTH;

    if (gpu_regions->get(row / REGION_DIM, col / REGION_DIM) > 0) {
        switch(gpu_neighbors->get(row, col)) {
            case 2: // alive cell stays alive; dead cell stays dead
                if(gpu_g->get(row, col) > 0) { // alive cell stays alive
                    gpu_g->inc(row, col);
                }
                break;
            case 3: // alive cell stays alive; dead cell comes alive
                if (gpu_g->get(row, col) == 0) { // dead cell comes alive
                    gpu_regions->inc(row / REGION_DIM, col / REGION_DIM);
                }
                gpu_g->inc(row, col);
                break;
            default: // alive cell dies; dead cell stays dead
                if (gpu_g->get(row, col) > 0) { // alive cell dies
                    gpu_regions->dec(row / REGION_DIM, col / REGION_DIM);
                }
                gpu_g->set(row, col, 0);
                break;
        }

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
                                    bmp->set(x, y, preset_colors[BLACK]);
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

    if (cudaMalloc(&(gpu_g->board), sizeof(int) * gpu_g->height * gpu_g->width) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate grid board on GPU\n");
        exit(2);
    } 

    // alocate space for GPU bitmap
    bitmap* gpu_bmp;
    if (cudaMalloc(&gpu_bmp, sizeof(bitmap)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate bitmap on GPU\n");
        exit(2);
    }

    // allocate space for neighbors
    grid* gpu_neighbors;
    if (cudaMalloc(&gpu_neighbors, sizeof(grid)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate grid on GPU\n");
        exit(2);
    }

    // allocate space for neighbors
    grid* gpu_regions;
    if (cudaMalloc(&gpu_regions, sizeof(grid)) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate regions grid on GPU\n");
        exit(2);
    }
    if (cudaMalloc(&gpu_regions->board, sizeof(int) * gpu_regions->height * gpu_regions->width) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate regions board on GPU\n");
        exit(2);
    } 



    // copy the CPU grid to the GPU grid
    if (cudaMemcpy(gpu_g, g, sizeof(grid), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy grid to the GPU\n");
    }

    // copy the CPU grid array to the GPU grid array
    if (cudaMemcpy(gpu_g->board, g->board, sizeof(int) * gpu_g->height * gpu_g->width,
                cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy grid array to the GPU\n");
    }

    // copy the CPU bitmap to the GPU bitmap
    if (cudaMemcpy(gpu_bmp, bmp, sizeof(bitmap), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy bitmap to the GPU\n");
    }

    // copy the CPU neighbors grid to the GPU neighbors grid
    if (cudaMemcpy(gpu_neighbors, g, sizeof(grid), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy neighbors grid to the GPU\n");
    }

    // copy the GPU regions grid to the GPU regions grid
    if (cudaMemcpy(gpu_regions, regions, sizeof(grid), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy regions grid to the GPU\n");
    }

    // copy the CPU regions array to the GPU regions array
    if (cudaMemcpy(gpu_regions->board, regions->board, sizeof(int) * gpu_regions->height * gpu_regions->width,
                cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy regions array to the GPU\n");
    }


    // number of block to run (rounding up to include all threads)
    size_t grid_blocks = (GRID_WIDTH*GRID_HEIGHT + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    count_neighbors<<<grid_blocks, THREADS_PER_BLOCK>>>(gpu_g, gpu_neighbors);
    cudaDeviceSynchronize();
    life_or_death<<<grid_blocks, THREADS_PER_BLOCK>>>(gpu_g, gpu_neighbors, gpu_regions);
    cudaDeviceSynchronize();

    // copy the GPU grid back to the CPU
    if (cudaMemcpy(g, gpu_g, sizeof(grid), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy grid from the GPU\n");
    }

    // copy the GPU grid array back to the CPU
    if (cudaMemcpy(g->board, gpu_g->board, sizeof(int) * gpu_g->height * gpu_g->width,
                cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy grid array from the GPU\n");
    }

    // copy the GPU bitmap back to the CPU 
    if (cudaMemcpy(bmp, gpu_bmp, sizeof(bitmap), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy bitmap from the GPU\n");
    }

    // copy the GPU regions grid back to the CPU
    if (cudaMemcpy(regions, gpu_regions, sizeof(grid), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy regions grid from the GPU\n");
    }

    // copy the GPU grid array back to the CPU
    if (cudaMemcpy(regions->board, gpu_regions->board, sizeof(int) * gpu_regions->height * gpu_regions->width,
                cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to regions grid array from the GPU\n");
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
    cudaFree(gpu_bmp);
    cudaFree(gpu_regions->board);
    cudaFree(gpu_regions);
    
}

// 
void let_there_be_light(coord loc) {
    // indicate in the boolean grid that cell's state has been changed
    g->set(loc.y/CELL_DIM,loc.x/CELL_DIM, 1);
    rgb32 color = g->get(loc.y/CELL_DIM,loc.x/CELL_DIM) ? preset_colors[WHITE] : preset_colors[BLACK];
    regions->inc((loc.y/CELL_DIM)/REGION_DIM,(loc.x/CELL_DIM)/REGION_DIM);

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


rgb_f32 interpolate_colors(int current_age, int old_age, int new_age, rgb_f32 old_color,
        rgb_f32 new_color) {
    rgb_f32 slope = (new_color - old_color) * (1. / (float) (new_age- old_age));
    rgb_f32 current_color = old_color + slope * current_age;

    return current_color;
}

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


/**
 * Entry point for the program
 */
int main(int argc, char ** argv) {

    // create the bitmap 
    bitmap bits(BMP_WIDTH, BMP_HEIGHT);
    bmp = &bits;

    // create the grid
    grid grd(GRID_HEIGHT, GRID_WIDTH);
    g = &grd;

    // create the regions grid
    grid rgns((GRID_HEIGHT/REGION_DIM), (GRID_WIDTH/REGION_DIM));
    regions = &rgns;

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
        while(SDL_PollEvent(&event) == 1) {
            // If the event is a quit event, then leave the loop
            if(event.type == SDL_QUIT) running = false;
        }

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
