file(GLOB GGML_SOURCES_TOPS "*.cpp")

set(TOPS_INSTALL_DIR "/opt/tops")

set(TOPS_INCLUDE_DIRS
${TOPS_INSTALL_DIR}/include
${TOPS_INSTALL_DIR}/include/gcu/umd
${TOPS_INSTALL_DIR}/include/tops
${TOPS_INSTALL_DIR}/lib
/usr/include/gcu/
)

list(APPEND TOPS_LIBRARIES
efrt
topsrt
topsaten
)

ggml_add_backend_library(ggml-tops ${GGML_SOURCES_TOPS})
target_link_libraries(ggml-tops PRIVATE ${TOPS_LIBRARIES})
target_include_directories(ggml-tops PRIVATE ${TOPS_INCLUDE_DIRS})
target_link_directories(ggml-tops PRIVATE ${TOPS_INSTALL_DIR}/lib /usr/lib)