message(STATUS "downloading...
     src='https://www.doc.ic.ac.uk/~sleutene/software/brisk-2.0.3.zip'
     dst='/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/okvis/brisk/src/brisk-2.0.3.zip'
     timeout='none'")




file(DOWNLOAD
  "https://www.doc.ic.ac.uk/~sleutene/software/brisk-2.0.3.zip"
  "/home/xcy/catkin_ws/src/MYNT-EYE-OKVIS-Sample/build/okvis/brisk/src/brisk-2.0.3.zip"
  SHOW_PROGRESS
  # no TIMEOUT
  STATUS status
  LOG log)

list(GET status 0 status_code)
list(GET status 1 status_string)

if(NOT status_code EQUAL 0)
  message(FATAL_ERROR "error: downloading 'https://www.doc.ic.ac.uk/~sleutene/software/brisk-2.0.3.zip' failed
  status_code: ${status_code}
  status_string: ${status_string}
  log: ${log}
")
endif()

message(STATUS "downloading... done")
