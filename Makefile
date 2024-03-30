all: tests exe

exe:
	cmake cmake-build-debug
	cmake --build cmake-build-debug

tests: test/test_hpimage.cpp
	cmake cmake-build-debug
	cmake --build cmake-build-debug
	valgrind ./out/test_hpimage

clean:
	cd out && rm *
