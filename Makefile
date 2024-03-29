all: tests exe

exe:
	cmake cmake-build-debug
	cmake --build cmake-build-debug
	./out/hpcarver

tests: test/test_hpimage.cpp
	cmake cmake-build-debug
	cmake --build cmake-build-debug
	./out/test_hpimage

clean:
	cd out && rm *
