# Install script for directory: /home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/okvis/brisk/src/brisk_external

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/lib/libbrisk.a")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/lib" TYPE STATIC_LIBRARY FILES "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/okvis/brisk/src/brisk_external-build/lib/libbrisk.a")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/bin/demo" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/bin/demo")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/bin/demo"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/bin/demo")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/bin" TYPE EXECUTABLE FILES "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/okvis/brisk/src/brisk_external-build/bin/demo")
  if(EXISTS "$ENV{DESTDIR}/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/bin/demo" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/bin/demo")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/bin/demo"
         OLD_RPATH "/opt/ros/kinetic/lib/x86_64-linux-gnu:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/bin/demo")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "dev")
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/include/")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/include" TYPE DIRECTORY FILES "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/okvis/brisk/src/brisk_external/include/" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/lib/CMake/brisk/briskConfig.cmake;/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/lib/CMake/brisk/briskConfigVersion.cmake")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/lib/CMake/brisk" TYPE FILE FILES
    "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/okvis/brisk/src/brisk_external-build/CMakeFiles/briskConfig.cmake"
    "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/okvis/brisk/src/brisk_external-build/briskConfigVersion.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/lib/CMake/brisk/briskTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/lib/CMake/brisk/briskTargets.cmake"
         "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/okvis/brisk/src/brisk_external-build/CMakeFiles/Export/_home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/lib/CMake/brisk/briskTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/lib/CMake/brisk/briskTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/lib/CMake/brisk/briskTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/lib/CMake/brisk/briskTargets.cmake")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/lib/CMake/brisk" TYPE FILE FILES "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/okvis/brisk/src/brisk_external-build/CMakeFiles/Export/_home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/lib/CMake/brisk/briskTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
     "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/lib/CMake/brisk/briskTargets-release.cmake")
    if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
    if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
        message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
    endif()
file(INSTALL DESTINATION "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/lib/CMake/brisk" TYPE FILE FILES "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/okvis/brisk/src/brisk_external-build/CMakeFiles/Export/_home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/lib/CMake/brisk/briskTargets-release.cmake")
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/okvis/brisk/src/brisk_external-build/agast/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/okvis/brisk/src/brisk_external-build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
