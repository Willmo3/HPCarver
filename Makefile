all: tests exe

exe:
	cmake cmake-build-debug
	cmake --build cmake-build-debug
	./out/hpcarver

tests: test/test_imagemagick.cpp
	cmake cmake-build-debug
	cmake --build cmake-build-debug
	./out/test_imagemagick
