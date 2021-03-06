#if !defined(BITMAP_HH)
#define BITMAP_HH

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define fmin(x,y) ( ( (x) < (y) ) ? (x) : (y) )
#define fmax(x,y) ( ( (x) > (y) ) ? (x) : (y) )

struct rgb_f32 { 
    double blue;
    double green;
    double red;

    rgb_f32() : red(0.), green(0.), blue(0.) {}
    rgb_f32(double r, double g, double b) : red(r), green(g), blue(b) {}

    rgb_f32 operator+(const rgb_f32& color) {
        return rgb_f32(
                fmin(255, this->red + color.red), 
                fmin(255, this->green + color.green), 
                fmin(255, this->blue + color.blue)
                );
    }
    rgb_f32 operator-(const rgb_f32& color) {
        return rgb_f32(
                fmax(0, this->red - color.red), 
                fmax(0, this->green - color.green), 
                fmax(0, this->blue - color.blue)
                );
    }
    rgb_f32 operator-(void) {
        return rgb_f32(
                255 - this->red, 
                255 - this->green, 
                255 - this->blue
                );
    }
    rgb_f32 operator*(const float scalar) {
        if (scalar < 0) {
            return rgb_f32(0.,0.,0.);
        } else { 
            return rgb_f32(
                    fmin(255, scalar * this->red), 
                    fmin(255, scalar * this->green), 
                    fmin(255, scalar * this->blue)
                    );
        }
    }
};

struct rgb32 {
    uint8_t alpha;
    uint8_t blue;
    uint8_t green;
    uint8_t red;

    rgb32() : red(0), green(0), blue(0) {}

    __host__ __device__ rgb32(uint8_t r, uint8_t g, uint8_t b) : red(r), green(g), blue(b) {}

    rgb32(rgb_f32 f) : red((int) f.red), green((int) f.green), blue((int) f.blue) {}
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

        // Get the height of the bitmap
        size_t height() { return _height; }

        // Get the width of the bitmap
        size_t width() { return _width; }

        // Set the color at a given location
        __host__ __device__ void set(int x, int y, rgb32 color) {
            assert(!(x < 0 || x >= _width || y < 0 || y >= _height)) ;
            _data[y*_width+x] = color;
        }

        void fill(rgb32 color) {
            for (int x = 0; x < _width; x++)
                for (int y = 0; y < _height; y++)
                    set(x, y, color);
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
