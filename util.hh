#if !defined(UTIL_HH)
#define UTIL_HH

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <sys/time.h>

/**
 * Sleep for a given number of milliseconds
 * \param   ms  The number of milliseconds to sleep for
 */
static void sleep_ms(size_t ms) {
  struct timespec ts;
  size_t rem = ms % 1000;
  ts.tv_sec = (ms - rem)/1000;
  ts.tv_nsec = rem * 1000000;
  
  // Sleep repeatedly as long as nanosleep is interrupted
  while(nanosleep(&ts, &ts) != 0) {}
}

/**
 * Get the time in milliseconds since UNIX epoch
 */
static size_t time_ms() {
  struct timeval tv;
  if(gettimeofday(&tv, NULL) == -1) {
    perror("gettimeofday");
    exit(2);
  }
  
  // Convert timeval values to milliseconds
  return tv.tv_sec*1000 + tv.tv_usec/1000;
}

// Generate a random float in a given range
static float drand(float min, float max) {
  return ((float)rand() / RAND_MAX) * (max - min) + min;
}

#endif