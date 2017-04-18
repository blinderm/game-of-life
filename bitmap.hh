#if !defined(BITMAP_HH)
#define BITMAP_HH

#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

struct rgb32 {
  uint8_t alpha;
  uint8_t blue;
  uint8_t green;
  uint8_t red;
  
  rgb32() : red(0), green(0), blue(0) {}
  
  rgb32(uint8_t r, uint8_t g, uint8_t b) : red(r), green(g), blue(b) {}
};

class bitmap {
public:
  // Constructor: set up the bitmap width, height, and data array
  bitmap(size_t width, size_t height) : _width(width), _height(height) {
    _data = new rgb32[width*height];
  }
  
  // Destructor: free the data array
  ~bitmap() {
    delete _data;
  }
  
  // Get the size of this bitmap's image data
  size_t size() { return _width*_height*sizeof(rgb32); }
  
  // Copy this bitmap to a given data location
  void copy_to(void* dest) {
    memcpy(dest, _data, size());
  }
  
  // Disallow the copy constructor for bitmaps
  bitmap(const bitmap&) = delete;
  bitmap(bitmap&&) = delete;
  
  // Disallow copying assignment for bitmaps
  bitmap& operator=(const bitmap&) = delete;
  bitmap& operator=(bitmap&&) = delete;
  
  // Get the height of the bitmap
  size_t height() { return _height; }
  
  // Get the width of the bitmap
  size_t width() { return _width; }
  
  // Set the color at a given location
  void set(int x, int y, rgb32 color) {
    // Instead of failing assertions for out-of-bounds pixels, just ignore them
    if(x < 0 || x >= _width || y < 0 || y >= _height) return;
    _data[y*_width+x] = color;
  }
  
  // Scale the color of each point by a given multiplier
  void darken(float multiplier) {
    for(int x=0; x<_width; x++) {
      for(int y=0; y<_height; y++) {
        _data[x+y*_width].alpha *= multiplier;
        _data[x+y*_width].blue *= multiplier;
        _data[x+y*_width].green *= multiplier;
        _data[x+y*_width].red *= multiplier;
      }
    }
  }
  
  // Shift all of the pixels in this bitmap up one position
  void shiftUp() {
    for(int y=0; y<_height-1; y++) {
      for(int x=0; x<_width; x++) {
        _data[x+y*_width] = _data[x+(y+1)*_width];
      }
    }
    
    for(int x=0; x<_width; x++) {
      _data[x+(_height-1)*_width] = rgb32();
    }
  }
  
  // Shift all of the pixels in this bitmap down one position
  void shiftDown() {
    for(int y=_height-1; y>0; y--) {
      for(int x=0; x<_width; x++) {
        _data[x+y*_width] = _data[x+(y-1)*_width];
      }
    }
    
    for(int x=0; x<_width; x++) {
      _data[x] = rgb32();
    }
  }
  
  // Shift all of the pixels in this bitmap left one position
  void shiftLeft() {
    for(int y=0; y<_height; y++) {
      for(int x=0; x<_width-1; x++) {
        _data[x+y*_width] = _data[x+y*_width+1];
      }
    }
    
    for(int y=0; y<_height; y++) {
      _data[_width-1+y*_width] = rgb32();
    }
  }
  
  // Shift all of the pixels in this bitmap right one position
  void shiftRight() {
    for(int y=0; y<_height; y++) {
      for(int x=_width-1; x>0; x--) {
        _data[x+y*_width] = _data[x+y*_width-1];
      }
    }
    
    for(int y=0; y<_height; y++) {
      _data[y*_width] = rgb32();
    }
  }
  
private:
  size_t _width;
  size_t _height;
  rgb32* _data;
};

#endif