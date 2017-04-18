ROOT := .
TARGETS := life
CXXFLAGS := `sdl2-config --cflags` -g -O2 --std=c++11
LDFLAGS  := `sdl2-config --libs` -lpthread

include $(ROOT)/common.mk