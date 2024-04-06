 # GoogleTest install:
 # GoogleTest requires at least C++14
 include(FetchContent)
 FetchContent_Declare(
 googletest
 URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
 )
 # Windows compatibility
 set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
 FetchContent_MakeAvailable(googletest)

 # Build tests
 add_executable(test_serialcarve test_serialcarve.cpp)
 target_link_libraries(test_serialcarve GTest::gtest_main)
 include(GoogleTest)

 gtest_discover_tests(test_serialcarve)

 # Link all to HPImage
 target_link_libraries(test_serialcarve
         hpimage
         serial)
