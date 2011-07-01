PROJECT( nscale_proj )
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE( segment segment.cxx )
TARGET_LINK_LIBRARIES( segment ${OpenCV_LIBS} )