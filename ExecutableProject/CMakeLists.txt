# CMakeList.txt: progetto CMake per OptixProjectCMake. Includere l'origine e definire
# qui la logica specifica del progetto.
#

# Aggiungere l'origine all'eseguibile di questo progetto.
add_executable (OptixProjectCMake "OptixProjectCMake.cpp" "OptixProjectCMake.h")

if (CMAKE_VERSION VERSION_GREATER 3.10)
  set_property(TARGET OptixProjectCMake PROPERTY CXX_STANDARD 20)
endif()

