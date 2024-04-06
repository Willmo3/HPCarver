RAJA_OPTS=-DRAJA_DIR=raja/lib/cmake/raja

all: tests exe

simple:
	cmake ${RAJA_OPTS} cmake-build-debug

exe:
	cmake ${RAJA_OPTS} cmake-build-debug
	cmake --build cmake-build-debug

tests: 
	cmake ${RAJA_OPTS} cmake-build-debug
	cmake --build cmake-build-debug
	./out/test_serialcarve

valgrind:
	cmake ${RAJA_OPTS} cmake-build-debug
	cmake --build cmake-build-debug
	valgrind ./out/test_serialcarve

clean:
	cd out && rm *
