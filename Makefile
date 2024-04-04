all: tests exe

simple:
	cmake -DRAJA_DIR=raja/lib/cmake/raja cmake-build-debug

exe:
	cmake -DRAJA_DIR=raja/lib/cmake/raja cmake-build-debug
	cmake --build cmake-build-debug

tests: test/test_hpimage.cpp
	cmake -DRAJA_DIR=raja/lib/cmake/raja cmake-build-debug
	cmake --build cmake-build-debug
	valgrind ./out/test_hpimage

clean:
	cd out && rm *
