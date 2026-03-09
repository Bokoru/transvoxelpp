# Transvoxelpp Library
An implementation of the [Transvoxel](https://transvoxel.org/) algorithm by [Eric Lengyel](https://terathon.com/lengyel/) in C++.

# Consuming the Library
To use this library in your project, simply clone the repository and add it to your cmake project like so:
```cmake
add_subdirectory(path/to/transvoxelpp)
```

Alternatively you can also use cmakes fetch content system:
```cmake
FetchContent_Declare(transvoxel 
    GIT_REPOSITORY https://github.com/Bokoru/transvoxelpp.git
    GIT_TAG        v1.0.0
    )
```

Once the library is part of the cmake project, link against it like so:
```cmake
target_link_libraries(some_target PUBLIC transvoxelpp)
```

# Using the Library
To use this library in code, `#include <transvoxel.hpp>`

The library uses a ~brain melting~ amount of template metaprogramming in an attempt to make integration as easy as possible.

All you realistically need is the following:
* A voxel type with:
    * A signed `volume` field, preferrably a float, this library expects the "inside of the terrain" to be negative, and "open air" to be positive.
    * \[OPTIONAL] A `gradient` field (must be a vec3 which points in the same direction as the volume gradient.)
* A vertex type with:
    * A vec3 `position` field
    * \[OPTIONAL] A vec3 `normal` field (preferrably with floating point components)
    * \[OPTIONAL] A vec3 `tangent` field (with floating point components, NOTE: tangents are only generated if `normal`s are also generated)

If you are using `glm` you must also `#include <transvoxel/glm_traits.hpp>` after including `transvoxel.hpp`.
If you are not using `glm`, you must specialize `transvoxel::vector_traits` for all vector types you intend to use, you can use `glm_traits.hpp` as a reference.

If your voxel or vertex types use different names for the fields, you will need to provide custom specializations for `transvoxel::voxel_traits` and `transvoxel::vertex_traits` respectively, you can use the base specialization in `transvoxel.hpp` as a reference.

If your voxel or vertex types use the same names but the fields are private, you must give `transvoxel::voxel_traits` and `transvoxel::vertex_traits` access by including the following in the class/struct definition of the voxel/vertex types:
```cpp
class voxel 
{
    // allow transvoxel::voxel_traist<> to acces private fields.
    friend struct transvoxel::voxel_traits<voxel>;

    float volume; // Field is private.
};

class vertex 
{
    // allow transvoxel::vertex_traits<> to acces private fields.
    friend struct transvoxel::vertex_traits<voxel>;

    glm::vec3 position; // Field is private.
};
```

To invoke the algorithm, use one of the following methods:

```cpp
// containers to store the generated geometry
std::vector<vertex> verts;
std::vector<uint16_t> indices; // can be any (preferrably unsigned) int type, however at least 16 bits of width is recommended.

// Call the algorithm.
transvoxel::march_regular_cell<0>( 
    glm::ivec3(x, y, z), // the `where` of the left-bottom-back voxel of the cell. `where` can be any vec3 type (preferrably with integer components)

    // 0 = LOD, as LOD increases the stepsize of `where` passed to voxel_getter also increases, effectively reducing the resolution. 
    // Each LOD increase cuts the resolution on each axis in half.

    [](const glm::ivec3& where) -> voxel { /* voxel getter impl */ } // voxel_getter `where` must be same type as the argument above.

    [&](const vertex& vert) -> uint16_t // vertex_putter
    { 
        uint16_t result = static_cast<uint16_t>(verts.size()); 
        verts.push_back(vert); 
        return result; 
    },

    [&](uint16_t a, uint16_t b, uint16_t c) -> void // triangle_putter
    {
        indices.push_back(a);
        indices.push_back(b);
        indices.push_back(c);
    });

transvoxel::march_transition_cell<1, true, true, true>( 
    // Template Param 0 = LOD, same as above, MUST be > 0
    // Template Param 1 = The cell directly to the left   (-x) is higher resolution
    // Template Param 2 = The cell directly to the bottom (-y) is higher resolution
    // Template Param 3 = The cell directly to the back   (-z) is higher resolution

    glm::ivec3(x, y, z), // the `where` of the left-bottom-back voxel of the cell.

    [](const glm::ivec3& where) -> voxel { /* voxel getter impl */ } // voxel_getter

    [&](const vertex& vert) -> uint16_t // vertex_putter
    { 
        uint16_t result = static_cast<uint16_t>(verts.size()); 
        verts.push_back(vert); 
        return result; 
    },

    [&](uint16_t a, uint16_t b, uint16_t c) -> void // triangle_putter
    {
        indices.push_back(a);
        indices.push_back(b);
        indices.push_back(c);
    });

```
