 # GoogleTest install:
 # GoogleTest requires at least C++14
 set(CMAKE_CXX_STANDARD 14)
 set(CMAKE_CXX_STANDARD_REQUIRED ON)
 include(FetchContent)
 FetchContent_Declare(
 googletest
 URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
 )
 # Windows compatibility
 set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
 FetchContent_MakeAvailable(googletest)

 # Build tests
 add_executable(test_imagemagick test_imagemagick.cpp)
 target_link_libraries(test_imagemagick GTest::gtest_main)
 include(GoogleTest)
 gtest_discover_tests(test_imagemagick)

 # Link all to ImageMagick
 target_link_libraries(test_imagemagick
 ${ImageMagick_LIBRARIES}
 hpimage)