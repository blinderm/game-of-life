#if !defined(GUI_HH)
#define GUI_HH

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <SDL.h>

#include "bitmap.hh"

/**
 * A class that creates an SDL window and can display bitmaps in that window
 */
class gui {
public:
  /**
   * Create a window with the given name, width, and height
   * \param name    The name to display at the top of the window
   * \param width   The width of the window in pixels
   * \param heigt   The height of the window in pixels
   */
  gui(const char* name, size_t width, size_t height) : _width(width), _height(height) {
    // Initialize SDL
    if(SDL_Init(SDL_INIT_VIDEO) != 0) {
      fprintf(stderr, "Failed to initialize SDL: %s\n", SDL_GetError());
      exit(2);
    }
  
    // Create SDL window
    _window = SDL_CreateWindow(name,
                               SDL_WINDOWPOS_UNDEFINED,
                               SDL_WINDOWPOS_UNDEFINED,
                               _width,
                               _height,
                               SDL_WINDOW_OPENGL);
    if(_window == NULL) {
      fprintf(stderr, "Failed to create SDL window: %s\n", SDL_GetError());
      SDL_Quit(); // Shut down SDL
      exit(2);
    }
  
    // Create an SDL renderer
    _renderer = SDL_CreateRenderer(_window, -1, 0);
    if(_renderer == NULL) {
      fprintf(stderr, "Failed to create SDL renderer: %s\n", SDL_GetError());
      SDL_DestroyWindow(_window);
      SDL_Quit();
      exit(2);
    }
  
    // Clear the display
    SDL_SetRenderDrawColor(_renderer, 255, 255, 255, 255);
    SDL_RenderClear(_renderer);
  
    // Create a texture that the renderer will display
    _texture = SDL_CreateTexture(_renderer,
                                 SDL_PIXELFORMAT_RGBA8888,
                                 SDL_TEXTUREACCESS_STREAMING,
                                 _width,
                                 _height);
                                 
    if(_texture == NULL) {
      fprintf(stderr, "Failed to create SDL texture: %s\n", SDL_GetError());
      SDL_DestroyRenderer(_renderer);
      SDL_DestroyWindow(_window);
      SDL_Quit();
      exit(2);
    }
  }
  
  /**
   * Destructor for the window: clean up all SDL resources
   */
  ~gui() {
    // Clean up before exiting
    SDL_DestroyTexture(_texture);
    SDL_DestroyRenderer(_renderer);
    SDL_DestroyWindow(_window);
    SDL_Quit();
  }
  
  /**
   * Display a bitmap at a specific location on the screen
   * \param bmp     The bitmap to display
   * \param xstart  The horizontal position where bmp should be displayed
   * \param ystart  The vertical position wher ebmp should be displayed
   * \param width   The width of the area where bmp should be displayed
   * \param height  The height of the area where bmp should be displayed
   */
  void display(bitmap& bmp, int xstart, int ystart, int width, int height) {
    uint32_t* data;
    int pitch;
  
    SDL_LockTexture(_texture, NULL, (void**)&data, &pitch);
  
    bmp.copy_to(data);
  
    SDL_UnlockTexture(_texture);
    SDL_Rect destination = { 0, 0, (int)_width, (int)_height };
    SDL_RenderCopy(_renderer, _texture, NULL, &destination);
    SDL_RenderPresent(_renderer);
  }
  
  /**
   * Display a bitmap in the entire window
   * \param bmp   The bitmap to display
   */
  void display(bitmap& bmp) {
    display(bmp, 0, 0, _width, _height);
  }
  
private:
  size_t _width;
  size_t _height;
  SDL_Window* _window;
  SDL_Renderer* _renderer;
  SDL_Texture* _texture;
};

#endif
