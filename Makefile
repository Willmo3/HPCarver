CC = g++
CFLAGS = -g -Wall
TARGET = hpcarver
REQS = hpcarver.cpp
BUILD_DIR = out

# NOTE: Magick++-config must be used to build successfully!
all: default
default: $(REQS)
	$(CC) `Magick++-config --cxxflags --cppflags` $(CFLAGS) -O2 -o $(BUILD_DIR)/$(TARGET) $(REQS)  `Magick++-config --ldflags --libs`

clean:
	rm $(BUILD_DIR)/*
