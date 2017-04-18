#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <stdint.h>
#include <stdlib.h>

/// The type of a task function
typedef void (*job_t)();

// Add a periodic job to the job list. The function `job_fn` should run every `interval` ms
void add_job(job_t task_fn, size_t interval);

// Remove the current job from the job list
void remove_job();

// Change the interval of the current job
void update_job_interval(size_t interval);

// Stop running the scheduler after the current job finishes
void stop_scheduler();

// Run the game scheduler until there are no jobs left to run or the scheduler is stopped
void run_scheduler();

#endif