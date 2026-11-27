# Add our C ABI to the ScaNN shared library, retaining its registrations and dependencies.
set_property(GLOBAL PROPERTY MEMORYH_SCANN_SOURCE "${CMAKE_CURRENT_LIST_DIR}/bridge.cc")
function(memoryh_add_scann_bridge)
  get_property(source GLOBAL PROPERTY MEMORYH_SCANN_SOURCE)
  target_sources(scann_core_objects PRIVATE "${source}")
endfunction()
cmake_language(DEFER CALL memoryh_add_scann_bridge)
