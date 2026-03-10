#pragma once
#include <transvoxel/inttypes.hpp>
#include <atomic>
#include <vector>

namespace transvoxel
{
    struct alignas(16) fvec
    {
        float x, y, z, w;
    };

    struct alignas(16) ivec
    {
        int x, y, z, w;
    };

    struct vertex
    {
        fvec position;
        fvec normal;
        fvec tangent;
    };

    struct triangle
    {
        uint16_t v0, v1, v2;
    };

    struct triangle_materials
    {
        uint8_t m0, m1, m2;
    };

    class mesh
    {
        friend class mesh_accessor;

        mutable std::atomic_flag sync;
        bool _freeze;

        std::vector<vertex> vertices;
        std::vector<triangle> triangles;
        std::vector<triangle_materials> materials;

        bool frozen() const
        {
            return _freeze;
        }

    public:
        mesh() : sync(ATOMIC_FLAG_INIT), _freeze(false) {}

        void freeze()
        {
            while (sync.test_and_set(std::memory_order_acquire));
            _freeze = true;
            sync.clear(std::memory_order_release);
        }

        uint16_t add_vertex(const vertex& vert)
        {
            while (sync.test_and_set(std::memory_order_acq_rel));
            uint16_t result = 0;
            if (!frozen()) 
            {
                result = static_cast<uint16_t>(vertices.size());
                vertices.push_back(vert);
            }
            sync.clear(std::memory_order_release);
            return result;
        }

        void add_triangle(const triangle& tri, const triangle_materials& mats)
        {
            while (sync.test_and_set(std::memory_order_acq_rel));
            if (!frozen()) 
            {
                triangles.push_back(tri);
                materials.push_back(mats);
            }
            sync.clear(std::memory_order_release);
        }

        void clear()
        {
            while (sync.test_and_set(std::memory_order_acq_rel));
            vertices.clear();
            triangles.clear();
            materials.clear();
            _freeze = false;
            sync.clear(std::memory_order_release);
        }
    };

    class mesh_accessor
    {
        const mesh* _mesh;

    public:
        mesh_accessor(const mesh& mesh) : _mesh(&mesh)
        {
            while (_mesh->sync.test_and_set(std::memory_order_acquire));
            if (!_mesh->frozen()) 
            {
                _mesh->sync.clear(std::memory_order_release);
                _mesh = nullptr;
            }
        }

        ~mesh_accessor()
        {
            if (_mesh) _mesh->sync.clear(std::memory_order_release);
        }

        mesh_accessor(const mesh_accessor&) = delete;
        mesh_accessor& operator=(const mesh_accessor&) = delete;

        mesh_accessor(mesh_accessor&& move) : _mesh(move._mesh)
        {
            move._mesh = nullptr;
        }

        mesh_accessor& operator=(mesh_accessor&& move)
        {
            if (this == &move) return *this;
            
            if (_mesh) _mesh->sync.clear(std::memory_order_release);

            _mesh = move._mesh;
            move._mesh = nullptr;

            return *this;
        }

        std::size_t vertex_count() const
        {
            if (_mesh) return _mesh->vertices.size();
            return 0;
        }

        std::size_t triangle_count() const
        {
            if (_mesh) return _mesh->triangles.size();
            return 0;
        }

        const vertex* vertex_data() const
        {
            if (_mesh) return _mesh->vertices.data();
            return nullptr;
        }

        const triangle* triangle_data() const
        {
            if (_mesh) return _mesh->triangles.data();
            return nullptr;
        }

        const triangle_materials* material_data() const
        {
            if (_mesh) return _mesh->materials.data();
            return nullptr;
        }
    };
}
