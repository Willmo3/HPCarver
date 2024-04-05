RAJA_OPTS=-DRAJA_DIR=raja/lib/cmake/raja

all: tests exe

simple:
	cmake ${RAJA_OPTS} cmake-build-debug

exe:
	cmake -DRAJA_DIR=raja/lib/cmake/raja cmake-build-debug
	cmake --build cmake-build-debug

tests: test/test_hpimage.cpp test/test_serialcarve.cpp
	cmake -DRAJA_DIR=raja/lib/cmake/raja cmake-build-debug
	cmake --build cmake-build-debug
	./out/test_hpimage
	./out/test_serialcarve

valgrind: test/test_hpimage.cpp test/test_serialcarve.cpp
	cmake -DRAJA_DIR=raja/lib/cmake/raja cmake-build-debug
	cmake --build cmake-build-debug
	valgrind ./out/test_hpimage
	valgrind ./out/test_serialcarve

clean:
	cd out && rm *
