#include "scheduler.hh"
#include "util.hh"
#include <stdlib.h>
#include <stdbool.h>

typedef struct node {
  job_t job;
  size_t interval;
  size_t time_left;
  struct node* next;
} node_t;

node_t* queue;
bool running;
bool free_me = false;

// adds the job, maintaining the sorted order of the queue
void add_job(job_t task_fn, size_t interval) {
  node_t* new_node = (node_t*) malloc(sizeof(node_t));
  new_node->job = task_fn;
  new_node->interval = interval;
  new_node->time_left = interval;
  if(queue == NULL) {
    new_node->next = NULL;
    queue = new_node;
  } else {
    // case: new node's interval comes before first node's time left
    if(queue->time_left > new_node->time_left) {
      new_node->next = queue;
      queue = new_node;
    } else { // step thru list to find correct spot
      node_t* cur = queue;
      while(cur->next != NULL && cur->next->time_left < new_node->time_left) {
        cur = cur->next;
      }
      new_node->next = cur->next;
      cur->next = new_node;
    }
  }
}

void remove_job() {
  free_me = true;
}

void update_job_interval(size_t interval) {
  queue->interval = interval;
}

void stop_scheduler() {
  running = false;
}

void run_scheduler() {
  running = true;
  while(running && queue != NULL) {
    sleep_ms(queue->time_left);
    queue->job();
    // reduce all time_left by current node's time_left
    for(node_t* cur = queue->next; cur != NULL; cur = cur->next) {
      cur->time_left -= queue->time_left;
    }
    // remove head of queue
    node_t* ran = queue;
    queue = queue->next;
    if(free_me) {
      free(ran);
      free_me = false;
    } else {
      // re-enqueue head
      add_job(ran->job, ran->interval);
    }
  }
}
