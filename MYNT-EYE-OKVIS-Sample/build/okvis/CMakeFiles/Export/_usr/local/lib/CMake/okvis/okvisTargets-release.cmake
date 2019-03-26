#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "okvis_util" for configuration "Release"
set_property(TARGET okvis_util APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(okvis_util PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "/usr/local/lib/libokvis_util.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS okvis_util )
list(APPEND _IMPORT_CHECK_FILES_FOR_okvis_util "/usr/local/lib/libokvis_util.a" )

# Import target "okvis_kinematics" for configuration "Release"
set_property(TARGET okvis_kinematics APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(okvis_kinematics PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "okvis_util"
  IMPORTED_LOCATION_RELEASE "/usr/local/lib/libokvis_kinematics.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS okvis_kinematics )
list(APPEND _IMPORT_CHECK_FILES_FOR_okvis_kinematics "/usr/local/lib/libokvis_kinematics.a" )

# Import target "okvis_time" for configuration "Release"
set_property(TARGET okvis_time APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(okvis_time PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "/usr/local/lib/libokvis_time.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS okvis_time )
list(APPEND _IMPORT_CHECK_FILES_FOR_okvis_time "/usr/local/lib/libokvis_time.a" )

# Import target "okvis_cv" for configuration "Release"
set_property(TARGET okvis_cv APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(okvis_cv PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "okvis_util;okvis_kinematics;okvis_time;opencv_calib3d;opencv_core;opencv_dnn;opencv_features2d;opencv_flann;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_shape;opencv_stitching;opencv_superres;opencv_video;opencv_videoio;opencv_videostab;opencv_viz;opencv_aruco;opencv_bgsegm;opencv_bioinspired;opencv_ccalib;opencv_cvv;opencv_datasets;opencv_dpm;opencv_face;opencv_fuzzy;opencv_hdf;opencv_img_hash;opencv_line_descriptor;opencv_optflow;opencv_phase_unwrapping;opencv_plot;opencv_reg;opencv_rgbd;opencv_saliency;opencv_stereo;opencv_structured_light;opencv_surface_matching;opencv_text;opencv_tracking;opencv_xfeatures2d;opencv_ximgproc;opencv_xobjdetect;opencv_xphoto;opencv_core;opencv_highgui;opencv_imgproc;opencv_features2d;brisk;agast"
  IMPORTED_LOCATION_RELEASE "/usr/local/lib/libokvis_cv.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS okvis_cv )
list(APPEND _IMPORT_CHECK_FILES_FOR_okvis_cv "/usr/local/lib/libokvis_cv.a" )

# Import target "okvis_common" for configuration "Release"
set_property(TARGET okvis_common APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(okvis_common PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "okvis_util;okvis_kinematics;okvis_time;okvis_cv"
  IMPORTED_LOCATION_RELEASE "/usr/local/lib/libokvis_common.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS okvis_common )
list(APPEND _IMPORT_CHECK_FILES_FOR_okvis_common "/usr/local/lib/libokvis_common.a" )

# Import target "okvis_ceres" for configuration "Release"
set_property(TARGET okvis_ceres APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(okvis_ceres PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "okvis_util;okvis_cv;okvis_common;ceres;gomp;/usr/lib/x86_64-linux-gnu/libspqr.so;/usr/lib/x86_64-linux-gnu/libtbb.so;/usr/lib/x86_64-linux-gnu/libtbbmalloc.so;/usr/lib/x86_64-linux-gnu/libcholmod.so;/usr/lib/x86_64-linux-gnu/libccolamd.so;/usr/lib/x86_64-linux-gnu/libcamd.so;/usr/lib/x86_64-linux-gnu/libcolamd.so;/usr/lib/x86_64-linux-gnu/libamd.so;/usr/lib/liblapack.so;/usr/lib/libblas.so;/usr/lib/libblas.so;/usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so;/usr/lib/x86_64-linux-gnu/librt.so;/usr/lib/x86_64-linux-gnu/libcxsparse.so;/usr/lib/x86_64-linux-gnu/libglog.so;opencv_calib3d;opencv_core;opencv_dnn;opencv_features2d;opencv_flann;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_shape;opencv_stitching;opencv_superres;opencv_video;opencv_videoio;opencv_videostab;opencv_viz;opencv_aruco;opencv_bgsegm;opencv_bioinspired;opencv_ccalib;opencv_cvv;opencv_datasets;opencv_dpm;opencv_face;opencv_fuzzy;opencv_hdf;opencv_img_hash;opencv_line_descriptor;opencv_optflow;opencv_phase_unwrapping;opencv_plot;opencv_reg;opencv_rgbd;opencv_saliency;opencv_stereo;opencv_structured_light;opencv_surface_matching;opencv_text;opencv_tracking;opencv_xfeatures2d;opencv_ximgproc;opencv_xobjdetect;opencv_xphoto"
  IMPORTED_LOCATION_RELEASE "/usr/local/lib/libokvis_ceres.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS okvis_ceres )
list(APPEND _IMPORT_CHECK_FILES_FOR_okvis_ceres "/usr/local/lib/libokvis_ceres.a" )

# Import target "okvis_timing" for configuration "Release"
set_property(TARGET okvis_timing APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(okvis_timing PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "okvis_util"
  IMPORTED_LOCATION_RELEASE "/usr/local/lib/libokvis_timing.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS okvis_timing )
list(APPEND _IMPORT_CHECK_FILES_FOR_okvis_timing "/usr/local/lib/libokvis_timing.a" )

# Import target "okvis_matcher" for configuration "Release"
set_property(TARGET okvis_matcher APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(okvis_matcher PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "okvis_util"
  IMPORTED_LOCATION_RELEASE "/usr/local/lib/libokvis_matcher.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS okvis_matcher )
list(APPEND _IMPORT_CHECK_FILES_FOR_okvis_matcher "/usr/local/lib/libokvis_matcher.a" )

# Import target "okvis_frontend" for configuration "Release"
set_property(TARGET okvis_frontend APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(okvis_frontend PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "brisk;agast;opengv;ceres;gomp;/usr/lib/x86_64-linux-gnu/libspqr.so;/usr/lib/x86_64-linux-gnu/libtbb.so;/usr/lib/x86_64-linux-gnu/libtbbmalloc.so;/usr/lib/x86_64-linux-gnu/libcholmod.so;/usr/lib/x86_64-linux-gnu/libccolamd.so;/usr/lib/x86_64-linux-gnu/libcamd.so;/usr/lib/x86_64-linux-gnu/libcolamd.so;/usr/lib/x86_64-linux-gnu/libamd.so;/usr/lib/liblapack.so;/usr/lib/libblas.so;/usr/lib/libblas.so;/usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so;/usr/lib/x86_64-linux-gnu/librt.so;/usr/lib/x86_64-linux-gnu/libcxsparse.so;/usr/lib/x86_64-linux-gnu/libglog.so;okvis_util;okvis_cv;okvis_ceres;okvis_timing;okvis_matcher"
  IMPORTED_LOCATION_RELEASE "/usr/local/lib/libokvis_frontend.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS okvis_frontend )
list(APPEND _IMPORT_CHECK_FILES_FOR_okvis_frontend "/usr/local/lib/libokvis_frontend.a" )

# Import target "okvis_multisensor_processing" for configuration "Release"
set_property(TARGET okvis_multisensor_processing APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(okvis_multisensor_processing PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "okvis_time;okvis_util;okvis_kinematics;okvis_cv;okvis_common;okvis_ceres;okvis_frontend;/usr/lib/x86_64-linux-gnu/libglog.so"
  IMPORTED_LOCATION_RELEASE "/usr/local/lib/libokvis_multisensor_processing.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS okvis_multisensor_processing )
list(APPEND _IMPORT_CHECK_FILES_FOR_okvis_multisensor_processing "/usr/local/lib/libokvis_multisensor_processing.a" )

# Import target "okvis_app_synchronous" for configuration "Release"
set_property(TARGET okvis_app_synchronous APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(okvis_app_synchronous PROPERTIES
  IMPORTED_LOCATION_RELEASE "/usr/local/bin/okvis_app_synchronous"
  )

list(APPEND _IMPORT_CHECK_TARGETS okvis_app_synchronous )
list(APPEND _IMPORT_CHECK_FILES_FOR_okvis_app_synchronous "/usr/local/bin/okvis_app_synchronous" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
