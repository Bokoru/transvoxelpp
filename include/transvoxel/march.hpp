#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <shared_mutex>

#include <transvoxel/lut.hpp>
#include <transvoxel/voxel.hpp>
#include <transvoxel/mesh.hpp>

#include <immintrin.h>

namespace transvoxel
{
    enum chunk_neighbor
    {
        negative_x = 0,
        positive_x,
        negative_y,
        positive_y,
        negative_z,
        positive_z
    };

    class chunk
    {
        std::weak_ptr<chunk> neighbors[6];
        std::array<voxel, 4096> voxels;
        bool modified;
        unsigned lod;

    public:
        unsigned level_of_detail() const
        {
            return lod;
        }

        void level_of_detail(unsigned _lod)
        {
            lod = _lod;
        }

        void neighbor(chunk_neighbor index, std::shared_ptr<chunk> neighbor) noexcept
        {
            neighbors[index] = neighbor;
        }

        std::shared_ptr<chunk> neighbor(chunk_neighbor index) const noexcept
        {
            return neighbors[index].lock();
        }

        voxel get(int x, int y, int z) const
        {
            const chunk* current = this;

            if (x & (~0xF))
            {
                if (auto next = neighbors[x < 0 ? negative_x : positive_x].lock()) current = next.get();
                x &= 0xF;
            }

            if (y & (~0xF))
            {
                if (auto next = neighbors[y < 0 ? negative_y : positive_y].lock()) current = next.get();
                y &= 0xF;
            }

            if (z & (~0xF))
            {
                if (auto next = neighbors[z < 0 ? negative_z : positive_z].lock()) current = next.get();
                z &= 0xF;
            }

            return current->voxels[x | (y << 4) | (z << 8)];
        }

        voxel get(__m128i v) const
        {
            return get(_mm_extract_epi32(v, 0), _mm_extract_epi32(v, 1), _mm_extract_epi32(v, 2));
        }

        void set(int x, int y, int z, voxel voxel)
        {
            if ((x < 0) || (x >= 16)) return;
            if ((y < 0) || (y >= 16)) return;
            if ((z < 0) || (z >= 16)) return;

            voxels[x | (y << 4) | (z << 8)] = voxel;

            modified = true;

            if (x == 0)       if(auto neighbor = neighbors[negative_x].lock()) neighbor->modified = true;
            else if (x == 15) if(auto neighbor = neighbors[positive_x].lock()) neighbor->modified = true;

            if (y == 0)       if(auto neighbor = neighbors[negative_y].lock()) neighbor->modified = true;
            else if (y == 15) if(auto neighbor = neighbors[positive_y].lock()) neighbor->modified = true;

            if (z == 0)       if(auto neighbor = neighbors[negative_z].lock()) neighbor->modified = true;
            else if (z == 15) if(auto neighbor = neighbors[positive_z].lock()) neighbor->modified = true;
        }

        void march(mesh& result) const;
    };
}
