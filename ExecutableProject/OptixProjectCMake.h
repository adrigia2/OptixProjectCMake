// OptixProjectCMake.h: file di inclusione per file di inclusione di sistema standard,
// o file di inclusione specifici del progetto.

#pragma once

#include <iostream>

// TODO: Fare riferimento qui alle intestazioni aggiuntive richieste dal programma.

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#define OPTIX_CHECK( call )                                             \
  {                                                                     \
    OptixResult res = call;                                             \
    if( res != OPTIX_SUCCESS )                                          \
      {                                                                 \
        fprintf( stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); \
        exit( 2 );                                                      \
      }                                                                 \
  }