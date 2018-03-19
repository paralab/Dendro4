cmake_minimum_required(VERSION 2.8)
project(dendro)

# version number
set (Dendro_VERSION_MAJOR 4)
set (Dendro_VERSION_MINOR 0)

#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${dendro_SOURCE_DIR}/lib)
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${dendro_SOURCE_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${dendro_SOURCE_DIR}/bin)



# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
        "${PROJECT_SOURCE_DIR}/include/dendro.h.in"
        "${PROJECT_BINARY_DIR}/dendro.h"
)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

#set (PAPI_LIB /home/milinda/Programs/papi-5.4.3/src)
#set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -l/home/milinda/Research/Utah/DENDRO/Libs/petsc/lib/libpapi.a")

if(OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(MPI_COMPILE_FLAGS)
  set(COMPILE_FLAGS "${COMPILE_FLAGS} ${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set(LINK_FLAGS "${LINK_FLAGS} ${MPI_LINK_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
set(LINK_FLAGS "${LINK_FLAGS} -rdynamic")

# Options
option(USE_64BIT_INDICES "Use 64-Bit indices. Reverts to 32-bit if turned off" ON)
option(PRIVATE_DA_IN_MG "Use private DA in Multigrid" ON)
option(HILBERT_ORDERING "Use Hilbert ordering instead of Morton" ON)
option(TREE_SORT "Use the treesort algorithm instead of sample sort" ON)
option(__PAPI_PROFILING__ "Use Papi to compute L1 and L2 cache misses to MatVec Loop" OFF)
option(BUILD_TESTS "Build test applications" ON)
option(BUILD_DA_EXAMPLES "Build examples using dendro::DA" ON)
option(BUILD_MG_EXAMPLES "Build test applications using dendro::MG" ON)
option(ALLTOALLV_FIX "Use K-way all to all v" OFF)
option(POWER_MEASUREMENT_TIMESTEP "Print the time step for mat vec loops" OFF)
option (SPLITTER_SELECTION_FIX "use the splitter fix for the treeSort" ON)
option (DIM_2 "To enable DIM2 version of Sorting. Tree sort part is tested and works wioth DIM 2 but rest of the dendro might not " OFF)
set(KWAY 128 CACHE INT 128)
set(NUM_NPES_THRESHOLD 16 CACHE INT 16)


if(REMOVE_DUPLICATES)
    add_definitions(-DREMOVE_DUPLICATES)
endif()


add_definitions(-DNUM_NPES_THRESHOLD=${NUM_NPES_THRESHOLD})

if(SPLITTER_SELECTION_FIX)
    add_definitions(-DSPLITTER_SELECTION_FIX)
endif()

if(USE_64BIT_INDICES)
    add_definitions(-DUSE_64BIT_INDICES)
endif()

if(HILBERT_ORDERING)
    add_definitions(-DHILBERT_ORDERING)
endif()

if(USE_NCA_PROPERTY)
    add_definitions(-DUSE_NCA_PROPERTY)
endif()


if(__PAPI_PROFILING__)
    add_definitions(-D__PAPI_PROFILING__)
endif()

if(TREE_SORT)
    add_definitions(-DTREE_SORT)
endif()

if(DIM_2)
    add_definitions(-DDIM_2)
endif()


if(ALLTOALLV_FIX)
    add_definitions(-DALLTOALLV_FIX)
    add_definitions(-DKWAY=${KWAY})
endif()


if(POWER_MEASUREMENT_TIMESTEP)
    add_definitions(-DPOWER_MEASUREMENT_TIMESTEP)
endif()


##------
include_directories(${PROJECT_BINARY_DIR}
                    ${MPI_INCLUDE_PATH}
                    include
                    include/fem
                    include/oda
                    include/omg
                    include/test
                    examples/include
                    examples/heat
                    $ENV{PETSC_DIR}/include
                    $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include
                    )

link_directories($ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib
                 $ENV{PAPI_LIB}
        )

# Sources
file(GLOB OMG_SOURCES src/omg/*.cpp)
file(GLOB ODA_SOURCES src/oda/*.cpp)
file(GLOB ODA_PRIVATE_SOURCES src/oda/private/*.cpp)

file(GLOB DENDRO_SOURCES src/*.cpp)
file(GLOB DENDRO_HEADERS include/*.h)
file(GLOB DENDRO_TEMPLATES include/*.tcc)

#### Libraries ...
set(DENDRO_LIBS dendroMG dendroDA dendroTest dendro)

add_library(dendroMG ${OMG_SOURCES})
# include/hcurvedata.h src/pickBdy.cpp src/blockPart.cpp src/construct.cpp src/balance.cpp)
add_library(dendroTest src/test/testUtils.C)

add_library(dendro ${DENDRO_SOURCES}
                   ${DENDRO_HEADERS}
                   ${DENDRO_TEMPLATES} include/hcurvedata.h src/hcurvedata.cpp src/blockPart.cpp)

add_library(dendroDA ${ODA_SOURCES} ${ODA_PRIVATE_SOURCES})
target_link_libraries(dendroDA dendro)

## executables

##---------------------------------------------------------------------------------------
##  Helper APPS
##---------------------------------------------------------------------------------------
add_executable(genGauss scripts/genGaussSeq.C)
add_executable(splitPoints scripts/splitPoints.C)

add_executable(gen2spheres scripts/gen2spheres.C)
target_link_libraries(gen2spheres dendro petsc ${MPI_LIBRARIES} m)

add_executable(genPrlFBMpts scripts/genPrlFBMpts.C)
target_link_libraries(genPrlFBMpts ${MPI_LIBRARIES} m)
##---------------------------------------------------------------------------------------

##---------------------------------------------------------------------------------------
##----------- integration with Taly -----------------------------------------------------
##---------------------------------------------------------------------------------------
add_executable(heat examples/heat/main.cpp examples/heat/parabolic.h
  examples/heat/timeStepper.cpp examples/heat/rhs.h)
target_link_libraries(heat dendroDA dendro petsc ${MPI_LIBRARIES} m)


add_executable(testFnViz examples/src/drivers/testFnViz.cpp examples/include/genPts_par.h examples/src/drivers/genPts_par.C include/test/testUtils.h include/test/testUtils.tcc src/test/testUtils.C )
    target_link_libraries(testFnViz dendroDA dendro petsc ${MPI_LIBRARIES} m)

add_executable(holeMesh examples/src/drivers/holeMesh.cpp include/test/testUtils.h include/test/testUtils.tcc src/test/testUtils.C )
    target_link_libraries(holeMesh dendroDA dendro petsc ${MPI_LIBRARIES} m)

add_executable(testIntergrid examples/src/drivers/testIntergrid.cpp include/test/testUtils.h include/test/testUtils.tcc src/test/testUtils.C )
    target_link_libraries(testIntergrid dendroDA dendro petsc ${MPI_LIBRARIES} m)

##---------------------------------------------------------------------------------------
## Octree Construct-Balance Apps
##---------------------------------------------------------------------------------------
add_executable(testRipple examples/src/drivers/tstRipple.C)
target_link_libraries(testRipple dendro petsc ${MPI_LIBRARIES} m)



add_executable(runScal examples/src/drivers/runScal.C)
target_link_libraries(runScal dendro petsc ${MPI_LIBRARIES} m)

add_executable(rippleBal examples/src/drivers/rippleBal.C)
target_link_libraries(rippleBal dendro petsc ${MPI_LIBRARIES} m)

add_executable(seqRipple examples/src/drivers/seqRipple.C)
target_link_libraries(seqRipple dendro petsc ${MPI_LIBRARIES} m)

add_executable(seqRipplePtr src/pickBdy.cpp examples/src/drivers/seqRipplePtr.C)
target_link_libraries(seqRipplePtr dendro petsc ${MPI_LIBRARIES} m)

add_executable(justBal examples/src/drivers/justBal.C)
target_link_libraries(justBal dendro petsc ${MPI_LIBRARIES} m)

add_executable(testConAndBal examples/src/drivers/testConAndBal)
target_link_libraries(testConAndBal dendro petsc ${MPI_LIBRARIES} m)

add_executable(testConBalAndCoarsen
  examples/src/drivers/testConBalAndCoarsen)
target_link_libraries(testConBalAndCoarsen dendro dendroDA petsc ${MPI_LIBRARIES} m)
##---------------------------------------------------------------------------------------

##---------------------------------------------------------------------------------------
##    Dendro DA APPS
##---------------------------------------------------------------------------------------
if(BUILD_DA_EXAMPLES)
    add_executable(testPartition include/sfcSort.h examples/src/drivers/testPartition.C examples/include/octreeStatistics.h examples/src/drivers/octreeStatistics.cpp include/hcurvedata.h src/hcurvedata.cpp examples/include/octreeStatistics.h examples/src/drivers/octreeStatistics.cpp include/test/testUtils.h src/test/testUtils.C examples/include/genPts_par.h examples/src/drivers/genPts_par.C include/dynamicPartition.h src/dynamicPartition.C
            )
    target_link_libraries(testPartition dendroDA dendro petsc ${MPI_LIBRARIES} m)

    add_executable(diffOctree include/sfcSort.h examples/src/drivers/diff_morton_hilbert.C include/hcurvedata.h src/hcurvedata.cpp examples/include/octreeStatistics.h examples/src/drivers/octreeStatistics.cpp include/test/testUtils.h src/test/testUtils.C )
    target_link_libraries(diffOctree dendroDA dendro petsc ${MPI_LIBRARIES} m)

    add_executable(flexPartition include/sfcSort.h examples/src/drivers/flexPartition.C include/hcurvedata.h src/hcurvedata.cpp examples/include/octreeStatistics.h examples/src/drivers/octreeStatistics.cpp include/test/testUtils.h src/test/testUtils.C)
    target_link_libraries(flexPartition dendroDA dendro petsc ${MPI_LIBRARIES} m)

    add_executable(tstMatVec include/sfcSort.h include/hcurvedata.h src/hcurvedata.cpp examples/src/drivers/tstMatVec.C examples/src/backend/odaJac.C examples/src/backend/handleType2Stencils.C examples/include/genPts_par.h examples/src/drivers/genPts_par.C include/test/testUtils.h include/test/testUtils.tcc src/test/testUtils.C )
    target_link_libraries(tstMatVec dendroDA dendro petsc ${MPI_LIBRARIES} m)

    add_executable(tstTreeSortMatVec include/sfcSort.h include/hcurvedata.h src/hcurvedata.cpp examples/src/drivers/tsttreeSortMatVec.cpp examples/src/backend/odaJac.C examples/src/backend/handleType2Stencils.C examples/include/genPts_par.h examples/src/drivers/genPts_par.C examples/include/octreeStatistics.h examples/src/drivers/octreeStatistics.cpp)
    target_link_libraries(tstTreeSortMatVec dendroDA dendro petsc ${MPI_LIBRARIES} m)

    add_executable(buildDA include/sfcSort.h examples/src/drivers/buildDA.C include/hcurvedata.h examples/include/octreeStatistics.h examples/src/drivers/octreeStatistics.cpp)
    target_link_libraries(buildDA dendroDA dendro petsc ${MPI_LIBRARIES} m)

    add_executable(buildRgDA include/sfcSort.h examples/src/drivers/buildRgDA.C)
    target_link_libraries(buildRgDA dendroDA dendro petsc ${MPI_LIBRARIES} m)

    #add_executable(octLaplacian examples/src/drivers/octLaplacian.C)
    #target_link_libraries(octLaplacian dendroDA dendro petsc ${MPI_LIBRARIES} m)
endif()
##---------------------------------------------------------------------------------------

##---------------------------------------------------------------------------------------
##    Dendro MG APPS
##---------------------------------------------------------------------------------------
if(BUILD_MG_EXAMPLES)
add_executable(checkError   examples/src/drivers/checkError.C
                            examples/src/backend/omgJac.C
                            examples/src/backend/matProp.C
                            examples/src/backend/odaJac.C
                            examples/src/backend/omgRhs.C
                            examples/src/backend/handleType2Stencils.C
                            examples/src/backend/handleType1Stencils.C)
target_link_libraries(checkError dendroMG dendroDA dendro petsc ${MPI_LIBRARIES} m)

add_executable(newElasMesh  examples/src/drivers/newElasMesh.C
                            examples/src/backend/elasticityJac.C
                            examples/src/backend/elasticityRhs.C
                            examples/src/backend/omgJac.C
                            examples/src/backend/odaJac.C
                            examples/src/backend/omgRhs.C
                            examples/src/backend/vecMass.C
                            examples/src/backend/handleType2Stencils.C )
target_link_libraries(newElasMesh dendroMG dendroDA dendro petsc ${MPI_LIBRARIES} m)

add_executable(newElasSolver    examples/src/drivers/newElasSolver.C
                                examples/src/backend/elasticityJac.C
                                examples/src/backend/elasticityRhs.C
                                examples/src/backend/omgJac.C
                                examples/src/backend/odaJac.C
                                examples/src/backend/omgRhs.C
                                examples/src/backend/vecMass.C
                                examples/src/backend/handleType2Stencils.C )
target_link_libraries(newElasSolver dendroMG dendroDA dendro petsc ${MPI_LIBRARIES} m)

add_executable(elasticitySolver examples/src/drivers/elasticitySolver.C
                                examples/src/backend/elasticityJac.C
                                examples/src/backend/elasticityRhs.C
                                examples/src/backend/omgJac.C
                                examples/src/backend/odaJac.C
                                examples/src/backend/omgRhs.C
                                examples/src/backend/vecMass.C
                                examples/src/backend/handleType2Stencils.C )
target_link_libraries(elasticitySolver dendroMG dendroDA dendro petsc ${MPI_LIBRARIES} m)

add_executable(tstMg    examples/src/drivers/tstMg.C
                        examples/src/backend/omgJac.C
                        examples/src/backend/matProp.C
                        examples/src/backend/odaJac.C
                        examples/src/backend/omgRhs.C
                        examples/src/backend/handleType2Stencils.C
                        examples/src/backend/handleType1Stencils.C )
target_link_libraries(tstMg dendroMG dendroDA dendro petsc ${MPI_LIBRARIES} m)

add_executable(omgNeumann_ex1   examples/src/drivers/omgNeumann_ex1.C
                                examples/src/backend/omgNeumann.C
                                examples/src/backend/handleType2Stencils.C )
target_link_libraries(omgNeumann_ex1 dendroMG dendroDA dendro petsc ${MPI_LIBRARIES} m)

add_executable(omgNeumann_ex2   examples/src/drivers/omgNeumann_ex2.C
                                examples/src/backend/omgNeumann.C
                                examples/src/backend/handleType2Stencils.C )
target_link_libraries(omgNeumann_ex2 dendroMG dendroDA dendro petsc ${MPI_LIBRARIES} m)

add_executable(omgNeumann_ex3   src/pickBdy.cpp
                                examples/src/drivers/omgNeumann_ex3.C
                                examples/src/backend/omgNeumann.C
                                examples/src/backend/handleType2Stencils.C )
target_link_libraries(omgNeumann_ex3 dendroMG dendroDA dendro petsc ${MPI_LIBRARIES} m)

add_executable(omgNeumann_2spheres  examples/src/drivers/omgNeumann_2spheres.C
                                    examples/src/backend/omgNeumann.C
                                    examples/src/backend/handleType2Stencils.C )
target_link_libraries(omgNeumann_2spheres dendroMG dendroDA dendro petsc ${MPI_LIBRARIES} m)

add_executable(tstMgReg examples/src/drivers/tstRegularMg.C
                        examples/src/backend/omgJac.C
                        examples/src/backend/matProp.C
                        examples/src/backend/odaJac.C
                        examples/src/backend/omgRhs.C
                        examples/src/backend/handleType2Stencils.C
                        examples/src/backend/handleType1Stencils.C )
target_link_libraries(tstMgReg dendroMG dendroDA dendro petsc ${MPI_LIBRARIES} m)

add_executable(testElasMatVec   examples/src/drivers/testElasMatVec.C
                                examples/src/backend/elasticityJac.C
                                examples/src/backend/elasticityRhs.C
                                examples/src/backend/omgJac.C
                                examples/src/backend/odaJac.C
                                examples/src/backend/omgRhs.C
                                examples/src/backend/vecMass.C
                                examples/src/backend/handleType2Stencils.C )
target_link_libraries(testElasMatVec dendroMG dendroDA dendro petsc ${MPI_LIBRARIES} m)

add_executable(pts2Mesh examples/src/drivers/pts2Mesh.C
                        examples/src/backend/elasticityJac.C
                        examples/src/backend/elasticityRhs.C
                        examples/src/backend/omgJac.C
                        examples/src/backend/odaJac.C
                        examples/src/backend/omgRhs.C
                        examples/src/backend/vecMass.C
                        examples/src/backend/handleType2Stencils.C )
target_link_libraries(pts2Mesh dendroMG dendroDA dendro petsc ${MPI_LIBRARIES} m)

add_executable(createCoarser  src/pickBdy.cpp  examples/src/drivers/createCoarser.C)
target_link_libraries(createCoarser dendroMG dendroDA dendro petsc ${MPI_LIBRARIES} m)
endif()
##---------------------------------------------------------------------------------------

##---------------------------------------------------------------------------------------
##    Dendro Test APPS
##---------------------------------------------------------------------------------------
if(BUILD_TESTS)
    add_executable(checkNl  src/test/compareNlFiles.C)
    target_link_libraries(checkNl dendro petsc ${MPI_LIBRARIES} m)

    add_executable(checkMl  src/pickBdy.cpp src/test/compareMFiles.C)
    target_link_libraries(checkMl dendro petsc ${MPI_LIBRARIES} m)

    add_executable(dumpMesh src/pickBdy.cpp examples/src/drivers/dumpMesh.C)
    target_link_libraries(dumpMesh dendroDA dendro petsc ${MPI_LIBRARIES} m)

    add_executable(downSample examples/src/drivers/downSample.C)
    target_link_libraries(downSample dendroMG dendroDA dendro petsc ${MPI_LIBRARIES} m)

    add_executable(checkResults src/pickBdy.cpp src/test/checkResults.C)
    target_link_libraries(checkResults dendroTest dendro dendroDA  petsc ${MPI_LIBRARIES} m)

    add_executable(checkScatterValues src/test/checkScatterValues.C)
    target_link_libraries(checkScatterValues dendroMG dendroDA dendro petsc ${MPI_LIBRARIES} m)

    add_executable(checkCoarsening src/test/checkCoarsening.C)
    target_link_libraries(checkCoarsening dendroTest dendro dendroDA  petsc ${MPI_LIBRARIES} m)

    add_executable(justCoarsen src/pickBdy.cpp examples/src/drivers/justCoarsen.C)
    target_link_libraries(justCoarsen dendroDA dendro petsc ${MPI_LIBRARIES} m)

    add_executable(testNodeAndVals src/test/testNodeAndVals.C)
    target_link_libraries(testNodeAndVals dendroTest dendro petsc ${MPI_LIBRARIES} m)

    #add_executable(testPetscInt src/pickBdy.cpp src/blockPart.cpp src/test/testPetscInt.C)
    #target_link_libraries(testPetscInt dendroTest dendro dendroDA petsc ${MPI_LIBRARIES} m)


    add_executable(tstFBM   examples/src/drivers/tstFBM.C
                            examples/src/backend/omgJac.C
                            examples/src/backend/matProp.C
                            examples/src/backend/odaJac.C
                            examples/src/backend/omgRhs.C
                            examples/src/backend/handleType2Stencils.C
                            examples/src/backend/handleType1Stencils.C )
    target_link_libraries(tstFBM dendroMG dendroDA dendro petsc ${MPI_LIBRARIES} m)

    add_executable(tstFBM2  examples/src/drivers/tstFBM2.C
                            examples/src/backend/omgJac.C
                            examples/src/backend/matProp.C
                            examples/src/backend/odaJac.C
                            examples/src/backend/omgRhs.C
                            examples/src/backend/handleType2Stencils.C
                            examples/src/backend/handleType1Stencils.C )
    target_link_libraries(tstFBM2 dendroMG dendroDA dendro petsc ${MPI_LIBRARIES} m)

    add_executable(tstDirichlet src/pickBdy.cpp
                                src/blockPart.cpp
                                examples/src/drivers/tstDirichlet.C
                                examples/src/backend/omgJac.C
                                examples/src/backend/matProp.C
                                examples/src/backend/odaJac.C
                                examples/src/backend/omgRhs.C
                                examples/src/backend/handleType2Stencils.C
                                examples/src/backend/handleType1Stencils.C )
    target_link_libraries(tstDirichlet dendroMG dendroDA dendro petsc ${MPI_LIBRARIES} m)

endif()

## copy config files etc ...
file(GLOB COPY_FILES ${CMAKE_SOURCE_DIR}/bin/options* DESTINATION ${PROJECT_BINARY_DIR})
file(GLOB COPY_FILES ${CMAKE_SOURCE_DIR}/Stencils/*.inp DESTINATION ${PROJECT_BINARY_DIR})
