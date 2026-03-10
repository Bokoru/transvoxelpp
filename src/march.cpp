#include <transvoxel/march.hpp>
using namespace transvoxel;

__m128 cross(__m128 a, __m128 b)
{
    __m128 a1 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 b1 = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 c = _mm_fmsub_ps(a, b1, _mm_mul_ps(a1, b));
    return _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 2, 1));
}

float dot(__m128 a, __m128 b)
{
    return _mm_cvtss_f32(_mm_dp_ps(a, b, 0x71));
}

__m128 normalize(__m128 x)
{
    __m128 y = _mm_mul_ps(x, x);
    y = _mm_hadd_ps(y, y);
    y = _mm_rsqrt_ps(_mm_hadd_ps(y, y));
    return _mm_mul_ps(x, y);
}

template<unsigned lod>
constexpr static unsigned step_size = 1 << lod;

template<unsigned lod>
constexpr static unsigned half_size = step_size<lod> >> 1;

template<unsigned lod, bool t>
static constexpr float small_step = t ? (step_size<lod> * 0.1f) : 0.0f;

template<unsigned lod, bool t>
static constexpr float large_step = step_size<lod> - small_step<lod, t>;

template<unsigned lod>
constexpr static int regular_lookup_offsets[8][3]
{
    { 0, 0, 0 },
    { step_size<lod>, 0, 0 },
    { 0, step_size<lod>, 0 },
    { step_size<lod>, step_size<lod>, 0 },
    { 0, 0, step_size<lod> },
    { step_size<lod>, 0, step_size<lod> },
    { 0, step_size<lod>, step_size<lod> },
    { step_size<lod>, step_size<lod>, step_size<lod> }
};

template<unsigned lod>
constexpr static int transition_lookup_offsets[9][2]
{
    { 0, 0 },
    { half_size<lod>, 0 },
    { step_size<lod>, 0 },
    { step_size<lod>, half_size<lod> },
    { step_size<lod>, step_size<lod> },
    { half_size<lod>, step_size<lod> },
    { 0, step_size<lod> },
    { 0, half_size<lod> },
    { half_size<lod>, half_size<lod> }
};


template<unsigned lod, bool l, bool d, bool b, bool r, bool u, bool f>
static constexpr fvec regular_cell_corner_positions[8]
{
    { small_step<lod, l>, small_step<lod, d>, small_step<lod, b>, 0.0f },
    { large_step<lod, r>, small_step<lod, d>, small_step<lod, b>, 0.0f },
    { small_step<lod, l>, small_step<lod, d>, large_step<lod, f>, 0.0f },
    { large_step<lod, r>, small_step<lod, d>, large_step<lod, f>, 0.0f },
    { small_step<lod, l>, large_step<lod, u>, small_step<lod, b>, 0.0f },
    { large_step<lod, r>, large_step<lod, u>, small_step<lod, b>, 0.0f },
    { small_step<lod, l>, large_step<lod, u>, large_step<lod, f>, 0.0f },
    { large_step<lod, r>, large_step<lod, u>, large_step<lod, f>, 0.0f }
};

void march_transition_cell(voxel cell[9], fvec positions[13], fvec gradient[13], const transvoxel::chunk& chunk, transvoxel::mesh& mesh)
{
    uint16_t case_index = 0;
    for (int i = 0; i < 9; ++i) if (cell[i].volume & 0x80) case_index |= 1 << i;
    if (case_index == 0 || case_index == 511) return;

    uint8_t class_index = transition_cell_class[case_index];
    const transition_cell& cell_data = transition_cell_data[class_index & 0x7F];

    uint16_t verts[12];
    // uint8_t  vmats[12];
    
    for (int i = 0; i < cell_data.vertex_count(); ++i)
    {
        vertex vert;
        uint16_t data = transition_vertex_data[case_index][i];

        uint8_t corner0 = data & 0xF, corner1 = (data >> 4) & 0xF;
        __m128 factor = _mm_set1_ps(factor_lookup[cell[corner0].volume | (cell[corner1].volume << 8)]);

        // store interpolated position
        __m128 cv0 = _mm_load_ps(&positions[corner0].x), cv1 = _mm_load_ps(&positions[corner1].x);
        _mm_store_ps(&vert.position.x, _mm_fmadd_ps(factor, _mm_sub_ps(cv1, cv0), cv0));

        // load gradient for normal calculation
        cv0 = _mm_load_ps(&gradient[corner0].x);
        cv1 = _mm_load_ps(&gradient[corner1].x);
        
        // lerp
        cv0 = _mm_fmadd_ps(factor, _mm_sub_ps(cv1, cv0), cv0);
        
        // normalize and store normal
        cv1 = _mm_mul_ps(cv0, cv0);
        cv1 = _mm_hadd_ps(cv1, cv1);
        cv1 = _mm_rsqrt_ps(_mm_hadd_ps(cv1, cv1));
        cv0 = _mm_mul_ps(cv0, cv1);
        _mm_store_ps(&vert.normal.x, cv0);

        // calculate tangent vector
        if (dot(cv0, _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f)) < 0.999f) _mm_store_ps(&vert.tangent.x, cross(cv0, _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f)));
        else _mm_store_ps(&vert.tangent.x, cross(cv0, _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f)));

        verts[i] = mesh.add_vertex(vert);
    }

    int l = cell_data.triangle_count() * 3;
    for (int i = 0; i < l; i += 3)
    {
        triangle tri;
        triangle_materials mats;

        if (class_index & 0x80)
        {
            tri.v0 = verts[cell_data.vertex_index[i + 2]];
            tri.v1 = verts[cell_data.vertex_index[i + 1]];
            tri.v2 = verts[cell_data.vertex_index[i    ]];

            /* mats.m0 = vmats[cell_data.vertex_index[i + 2]];
            mats.m1 = vmats[cell_data.vertex_index[i + 1]];
            mats.m2 = vmats[cell_data.vertex_index[i    ]]; */
        }
        else
        {
            tri.v0 = verts[cell_data.vertex_index[i    ]];
            tri.v1 = verts[cell_data.vertex_index[i + 1]];
            tri.v2 = verts[cell_data.vertex_index[i + 2]];

            /* mats.m0 = vmats[cell_data.vertex_index[i    ]];
            mats.m1 = vmats[cell_data.vertex_index[i + 1]];
            mats.m2 = vmats[cell_data.vertex_index[i + 2]]; */
        }

        mats.m0 = cell[0].material;
        mats.m1 = mats.m0;
        mats.m2 = mats.m0;
        
        mesh.add_triangle(tri, mats);
    }
}

template<unsigned lod, bool l, bool d, bool b, bool r, bool u, bool f>
void march_regular_cell(voxel cell[8], fvec gradient[8], const transvoxel::chunk& chunk, transvoxel::mesh& mesh)
{
    uint8_t case_index = 0;
    for (int i = 0; i < 8; ++i) case_index |= (cell[i].volume & 0x80) >> (7 - i);
    if (case_index == 0 || case_index == 255) return;

    const regular_cell& cell_data = regular_cell_data[regular_cell_class[case_index]];

    uint16_t verts[12];
    // uint8_t  vmats[12];
    
    for (int i = 0; i < cell_data.vertex_count(); ++i)
    {
        vertex vert;

        uint16_t data = regular_vertex_data[case_index][i];

        uint8_t corner0 = data & 0xF, corner1 = (data >> 4) & 0xF;
        __m128 factor = _mm_set1_ps(factor_lookup[cell[corner0].volume | ((uint16_t)cell[corner1].volume << 8)]);

        // store interpolated position
        __m128 cv0 = _mm_load_ps(&regular_cell_corner_positions<lod, l, d, b, r, u, f>[corner0].x), cv1 = _mm_load_ps(&regular_cell_corner_positions<lod, l, d, b, r, u, f>[corner1].x);
        _mm_store_ps(&vert.position.x, _mm_fmadd_ps(factor, _mm_sub_ps(cv1, cv0), cv0));

        // load gradient for normal calculation
        cv0 = _mm_load_ps(&gradient[corner0].x);
        cv1 = _mm_load_ps(&gradient[corner1].x);
        
        // lerp
        cv0 = _mm_fmadd_ps(factor, _mm_sub_ps(cv1, cv0), cv0);
        
        // normalize and store normal
        cv0 = normalize(cv0);
        _mm_store_ps(&vert.normal.x, cv0);

        // calculate tangent vector
        if (dot(cv0, _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f)) < 0.999f) _mm_store_ps(&vert.tangent.x, normalize(cross(cv0, _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f))));
        else _mm_store_ps(&vert.tangent.x, normalize(cross(cv0, _mm_setr_ps(0.0f, 0.0f, 0.0f, 1.0f))));

        verts[i] = mesh.add_vertex(vert);
    }

    int len = cell_data.triangle_count() * 3;
    for (int i = 0; i < len; i += 3)
    {
        triangle tri;
        tri.v0 = verts[cell_data.vertex_index[i    ]];
        tri.v1 = verts[cell_data.vertex_index[i + 1]];
        tri.v2 = verts[cell_data.vertex_index[i + 2]];
        
        triangle_materials mats;
        /* mats.m0 = vmats[cell_data.vertex_index[i    ]];
        mats.m1 = vmats[cell_data.vertex_index[i + 1]];
        mats.m2 = vmats[cell_data.vertex_index[i + 2]]; */

        mats.m0 = cell[0].material;
        mats.m1 = mats.m0;
        mats.m2 = mats.m0;

        mesh.add_triangle(tri, mats);
    }
}

template<unsigned lod, bool l, bool d, bool b, bool r, bool u, bool f>
void march_cell(int x, int y, int z, const transvoxel::chunk& chunk, transvoxel::mesh& mesh)
{
    fvec regular_gradient[8];
    {
        voxel cell[8];

        for (int i = 0; i < 8; ++i)
        {
            int xl, yl, zl;
            xl = x + regular_lookup_offsets<lod>[i][0];
            yl = y + regular_lookup_offsets<lod>[i][1];
            zl = z + regular_lookup_offsets<lod>[i][2];

            cell[i] = chunk.get(xl, yl, zl);
            if (i == 0 && (cell[0].material == 0xFF)) return;

            uint16_t x0, y0, z0;
            x0 = chunk.get(xl - step_size<lod>, yl, zl).volume;
            y0 = chunk.get(xl, yl - step_size<lod>, zl).volume;
            z0 = chunk.get(xl, yl, zl - step_size<lod>).volume;

            uint16_t x1, y1, z1;
            x1 = chunk.get(xl + step_size<lod>, yl, zl).volume;
            y1 = chunk.get(xl, yl + step_size<lod>, zl).volume;
            z1 = chunk.get(xl, yl, zl + step_size<lod>).volume;

            regular_gradient[i].x = gradient_lookup[x0 | (x1 << 8)];
            regular_gradient[i].y = gradient_lookup[y0 | (y1 << 8)];
            regular_gradient[i].z = gradient_lookup[z0 | (z1 << 8)];
            regular_gradient[i].w = 0.0f;
        }
        
        march_regular_cell<lod, l, d, b, r, u, f>(cell, regular_gradient, chunk, mesh);
    }

    if constexpr (l)
    {
        voxel cell[9];
        fvec positions[13];
        fvec gradient[13];

        for (int i = 0; i < 9; ++i)
        {
            int xl, yl, zl;
            xl = x;
            yl = y + transition_lookup_offsets<lod>[i][0];
            zl = z + transition_lookup_offsets<lod>[i][1];

            positions[i].x = xl;
            positions[i].y = yl;
            positions[i].z = zl;
            positions[i].w = 0;

            uint16_t x0, y0, z0;
            x0 = chunk.get(xl - step_size<lod>, yl, zl).volume;
            y0 = chunk.get(xl, yl - step_size<lod>, zl).volume;
            z0 = chunk.get(xl, yl, zl - step_size<lod>).volume;

            uint16_t x1, y1, z1;
            x1 = chunk.get(xl + step_size<lod>, yl, zl).volume;
            y1 = chunk.get(xl, yl + step_size<lod>, zl).volume;
            z1 = chunk.get(xl, yl, zl + step_size<lod>).volume;

            gradient[i].x = gradient_lookup[x0 | (x1 << 8)];
            gradient[i].y = gradient_lookup[y0 | (y1 << 8)];
            gradient[i].z = gradient_lookup[z0 | (z1 << 8)];
            gradient[i].w = 0.0f;
        }

        positions[9]  = regular_cell_corner_positions<lod, l, d, b, r, u, f>[0];
        positions[10] = regular_cell_corner_positions<lod, l, d, b, r, u, f>[2];
        positions[11] = regular_cell_corner_positions<lod, l, d, b, r, u, f>[4];
        positions[12] = regular_cell_corner_positions<lod, l, d, b, r, u, f>[6];

        gradient[9]  = regular_gradient[0];
        gradient[10] = regular_gradient[2];
        gradient[11] = regular_gradient[4];
        gradient[12] = regular_gradient[6];

        march_transition_cell(cell, positions, gradient, chunk, mesh);
    }

    if constexpr (r)
    {
        voxel cell[9];
        fvec positions[13];
        fvec gradient[13];

        for (int i = 0; i < 9; ++i)
        {
            int xl, yl, zl;
            xl = x + step_size<lod>;
            yl = y + transition_lookup_offsets<lod>[i][0];
            zl = z + transition_lookup_offsets<lod>[i][1];

            positions[i].x = xl;
            positions[i].y = yl;
            positions[i].z = zl;
            positions[i].w = 0;

            uint16_t x0, y0, z0;
            x0 = chunk.get(xl - step_size<lod>, yl, zl).volume;
            y0 = chunk.get(xl, yl - step_size<lod>, zl).volume;
            z0 = chunk.get(xl, yl, zl - step_size<lod>).volume;

            uint16_t x1, y1, z1;
            x1 = chunk.get(xl + step_size<lod>, yl, zl).volume;
            y1 = chunk.get(xl, yl + step_size<lod>, zl).volume;
            z1 = chunk.get(xl, yl, zl + step_size<lod>).volume;

            gradient[i].x = gradient_lookup[x0 | (x1 << 8)];
            gradient[i].y = gradient_lookup[y0 | (y1 << 8)];
            gradient[i].z = gradient_lookup[z0 | (z1 << 8)];
            gradient[i].w = 0.0f;
        }

        positions[9]  = regular_cell_corner_positions<lod, l, d, b, r, u, f>[1];
        positions[10] = regular_cell_corner_positions<lod, l, d, b, r, u, f>[3];
        positions[11] = regular_cell_corner_positions<lod, l, d, b, r, u, f>[5];
        positions[12] = regular_cell_corner_positions<lod, l, d, b, r, u, f>[7];

        gradient[9]  = regular_gradient[1];
        gradient[10] = regular_gradient[3];
        gradient[11] = regular_gradient[5];
        gradient[12] = regular_gradient[7];

        march_transition_cell(cell, positions, gradient, chunk, mesh);
    }

    if constexpr (d)
    {
        voxel cell[9];
        fvec positions[13];
        fvec gradient[13];

        for (int i = 0; i < 9; ++i)
        {
            int xl, yl, zl;
            xl = x + transition_lookup_offsets<lod>[i][0];
            yl = y;
            zl = z + transition_lookup_offsets<lod>[i][1];

            positions[i].x = xl;
            positions[i].y = yl;
            positions[i].z = zl;
            positions[i].w = 0;

            uint16_t x0, y0, z0;
            x0 = chunk.get(xl - step_size<lod>, yl, zl).volume;
            y0 = chunk.get(xl, yl - step_size<lod>, zl).volume;
            z0 = chunk.get(xl, yl, zl - step_size<lod>).volume;

            uint16_t x1, y1, z1;
            x1 = chunk.get(xl + step_size<lod>, yl, zl).volume;
            y1 = chunk.get(xl, yl + step_size<lod>, zl).volume;
            z1 = chunk.get(xl, yl, zl + step_size<lod>).volume;

            gradient[i].x = gradient_lookup[x0 | (x1 << 8)];
            gradient[i].y = gradient_lookup[y0 | (y1 << 8)];
            gradient[i].z = gradient_lookup[z0 | (z1 << 8)];
            gradient[i].w = 0.0f;
        }

        positions[9]  = regular_cell_corner_positions<lod, l, d, b, r, u, f>[0];
        positions[10] = regular_cell_corner_positions<lod, l, d, b, r, u, f>[1];
        positions[11] = regular_cell_corner_positions<lod, l, d, b, r, u, f>[2];
        positions[12] = regular_cell_corner_positions<lod, l, d, b, r, u, f>[3];

        gradient[9]  = regular_gradient[0];
        gradient[10] = regular_gradient[1];
        gradient[11] = regular_gradient[2];
        gradient[12] = regular_gradient[3];

        march_transition_cell(cell, positions, gradient, chunk, mesh);
    }

    if constexpr (u)
    {
        voxel cell[9];
        fvec positions[13];
        fvec gradient[13];

        for (int i = 0; i < 9; ++i)
        {
            int xl, yl, zl;
            xl = x + transition_lookup_offsets<lod>[i][0];
            yl = y + step_size<lod>;
            zl = z + transition_lookup_offsets<lod>[i][1];

            positions[i].x = xl;
            positions[i].y = yl;
            positions[i].z = zl;
            positions[i].w = 0;

            uint16_t x0, y0, z0;
            x0 = chunk.get(xl - step_size<lod>, yl, zl).volume;
            y0 = chunk.get(xl, yl - step_size<lod>, zl).volume;
            z0 = chunk.get(xl, yl, zl - step_size<lod>).volume;

            uint16_t x1, y1, z1;
            x1 = chunk.get(xl + step_size<lod>, yl, zl).volume;
            y1 = chunk.get(xl, yl + step_size<lod>, zl).volume;
            z1 = chunk.get(xl, yl, zl + step_size<lod>).volume;

            gradient[i].x = gradient_lookup[x0 | (x1 << 8)];
            gradient[i].y = gradient_lookup[y0 | (y1 << 8)];
            gradient[i].z = gradient_lookup[z0 | (z1 << 8)];
            gradient[i].w = 0.0f;
        }

        positions[9]  = regular_cell_corner_positions<lod, l, d, b, r, u, f>[4];
        positions[10] = regular_cell_corner_positions<lod, l, d, b, r, u, f>[5];
        positions[11] = regular_cell_corner_positions<lod, l, d, b, r, u, f>[6];
        positions[12] = regular_cell_corner_positions<lod, l, d, b, r, u, f>[7];

        gradient[9]  = regular_gradient[4];
        gradient[10] = regular_gradient[5];
        gradient[11] = regular_gradient[6];
        gradient[12] = regular_gradient[7];

        march_transition_cell(cell, positions, gradient, chunk, mesh);
    }

    if constexpr (b)
    {
        voxel cell[9];
        fvec positions[13];
        fvec gradient[13];

        for (int i = 0; i < 9; ++i)
        {
            int xl, yl, zl;
            xl = x + transition_lookup_offsets<lod>[i][0];
            yl = y + transition_lookup_offsets<lod>[i][1];
            zl = z;

            positions[i].x = xl;
            positions[i].y = yl;
            positions[i].z = zl;
            positions[i].w = 0;

            uint16_t x0, y0, z0;
            x0 = chunk.get(xl - step_size<lod>, yl, zl).volume;
            y0 = chunk.get(xl, yl - step_size<lod>, zl).volume;
            z0 = chunk.get(xl, yl, zl - step_size<lod>).volume;

            uint16_t x1, y1, z1;
            x1 = chunk.get(xl + step_size<lod>, yl, zl).volume;
            y1 = chunk.get(xl, yl + step_size<lod>, zl).volume;
            z1 = chunk.get(xl, yl, zl + step_size<lod>).volume;

            gradient[i].x = gradient_lookup[x0 | (x1 << 8)];
            gradient[i].y = gradient_lookup[y0 | (y1 << 8)];
            gradient[i].z = gradient_lookup[z0 | (z1 << 8)];
            gradient[i].w = 0.0f;
        }

        positions[9]  = regular_cell_corner_positions<lod, l, d, b, r, u, f>[0];
        positions[10] = regular_cell_corner_positions<lod, l, d, b, r, u, f>[1];
        positions[11] = regular_cell_corner_positions<lod, l, d, b, r, u, f>[4];
        positions[12] = regular_cell_corner_positions<lod, l, d, b, r, u, f>[5];

        gradient[9]  = regular_gradient[0];
        gradient[10] = regular_gradient[1];
        gradient[11] = regular_gradient[4];
        gradient[12] = regular_gradient[5];

        march_transition_cell(cell, positions, gradient, chunk, mesh);
    }

    if constexpr (f)
    {
        voxel cell[9];
        fvec positions[13];
        fvec gradient[13];

        for (int i = 0; i < 9; ++i)
        {
            int xl, yl, zl;
            xl = x + transition_lookup_offsets<lod>[i][0];
            yl = y + transition_lookup_offsets<lod>[i][1];
            zl = z + step_size<lod>;

            positions[i].x = xl;
            positions[i].y = yl;
            positions[i].z = zl;
            positions[i].w = 0;

            uint16_t x0, y0, z0;
            x0 = chunk.get(xl - step_size<lod>, yl, zl).volume;
            y0 = chunk.get(xl, yl - step_size<lod>, zl).volume;
            z0 = chunk.get(xl, yl, zl - step_size<lod>).volume;

            uint16_t x1, y1, z1;
            x1 = chunk.get(xl + step_size<lod>, yl, zl).volume;
            y1 = chunk.get(xl, yl + step_size<lod>, zl).volume;
            z1 = chunk.get(xl, yl, zl + step_size<lod>).volume;

            gradient[i].x = gradient_lookup[x0 | (x1 << 8)];
            gradient[i].y = gradient_lookup[y0 | (y1 << 8)];
            gradient[i].z = gradient_lookup[z0 | (z1 << 8)];
            gradient[i].w = 0.0f;
        }

        positions[9]  = regular_cell_corner_positions<lod, l, d, b, r, u, f>[2];
        positions[10] = regular_cell_corner_positions<lod, l, d, b, r, u, f>[3];
        positions[11] = regular_cell_corner_positions<lod, l, d, b, r, u, f>[6];
        positions[12] = regular_cell_corner_positions<lod, l, d, b, r, u, f>[7];

        gradient[9]  = regular_gradient[2];
        gradient[10] = regular_gradient[3];
        gradient[11] = regular_gradient[6];
        gradient[12] = regular_gradient[7];

        march_transition_cell(cell, positions, gradient, chunk, mesh);
    }
}

template<unsigned lod>
void march_chunk(const chunk& chunk, mesh& mesh)
{
    constexpr static unsigned step_size = 1 << lod;
    for (int z = 0; z < 16; z += step_size)
    {
        for (int y = 0; y < 16; y += step_size)
        {
            for (int x = 0; x < 16; x += step_size)
            {
                uint8_t cell_type_index = 0;
                if (x == 0)  if (auto neighbor = chunk.neighbor(negative_x)) if (neighbor->level_of_detail() < chunk.level_of_detail()) cell_type_index  = 0b000001;
                if (y == 0)  if (auto neighbor = chunk.neighbor(negative_y)) if (neighbor->level_of_detail() < chunk.level_of_detail()) cell_type_index |= 0b000010;
                if (z == 0)  if (auto neighbor = chunk.neighbor(negative_z)) if (neighbor->level_of_detail() < chunk.level_of_detail()) cell_type_index |= 0b000100;
                if (x == 15) if (auto neighbor = chunk.neighbor(positive_x)) if (neighbor->level_of_detail() < chunk.level_of_detail()) cell_type_index |= 0b001000;
                if (y == 15) if (auto neighbor = chunk.neighbor(positive_y)) if (neighbor->level_of_detail() < chunk.level_of_detail()) cell_type_index |= 0b010000;
                if (z == 15) if (auto neighbor = chunk.neighbor(positive_z)) if (neighbor->level_of_detail() < chunk.level_of_detail()) cell_type_index |= 0b100000;
                
                switch(cell_type_index)
                { // OMG OMG OMG OMG OMG OMG
                    case 0b000000: march_cell<lod, false, false, false, false, false, false>(x, y, z, chunk, mesh); break;
                    case 0b000001: march_cell<lod, true , false, false, false, false, false>(x, y, z, chunk, mesh); break;
                    case 0b000010: march_cell<lod, false, true , false, false, false, false>(x, y, z, chunk, mesh); break;
                    case 0b000011: march_cell<lod, true , true , false, false, false, false>(x, y, z, chunk, mesh); break;
                    case 0b000100: march_cell<lod, false, false, true , false, false, false>(x, y, z, chunk, mesh); break;
                    case 0b000101: march_cell<lod, true , false, true , false, false, false>(x, y, z, chunk, mesh); break;
                    case 0b000110: march_cell<lod, false, true , true , false, false, false>(x, y, z, chunk, mesh); break;
                    case 0b000111: march_cell<lod, true , true , true , false, false, false>(x, y, z, chunk, mesh); break;
                    case 0b001000: march_cell<lod, false, false, false, true , false, false>(x, y, z, chunk, mesh); break;
                    case 0b001001: march_cell<lod, true , false, false, true , false, false>(x, y, z, chunk, mesh); break;
                    case 0b001010: march_cell<lod, false, true , false, true , false, false>(x, y, z, chunk, mesh); break;
                    case 0b001011: march_cell<lod, true , true , false, true , false, false>(x, y, z, chunk, mesh); break;
                    case 0b001100: march_cell<lod, false, false, true , true , false, false>(x, y, z, chunk, mesh); break;
                    case 0b001101: march_cell<lod, true , false, true , true , false, false>(x, y, z, chunk, mesh); break;
                    case 0b001110: march_cell<lod, false, true , true , true , false, false>(x, y, z, chunk, mesh); break;
                    case 0b001111: march_cell<lod, true , true , true , true , false, false>(x, y, z, chunk, mesh); break;
                    case 0b010000: march_cell<lod, false, false, false, false, true , false>(x, y, z, chunk, mesh); break;
                    case 0b010001: march_cell<lod, true , false, false, false, true , false>(x, y, z, chunk, mesh); break;
                    case 0b010010: march_cell<lod, false, true , false, false, true , false>(x, y, z, chunk, mesh); break;
                    case 0b010011: march_cell<lod, true , true , false, false, true , false>(x, y, z, chunk, mesh); break;
                    case 0b010100: march_cell<lod, false, false, true , false, true , false>(x, y, z, chunk, mesh); break;
                    case 0b010101: march_cell<lod, true , false, true , false, true , false>(x, y, z, chunk, mesh); break;
                    case 0b010110: march_cell<lod, false, true , true , false, true , false>(x, y, z, chunk, mesh); break;
                    case 0b010111: march_cell<lod, true , true , true , false, true , false>(x, y, z, chunk, mesh); break;
                    case 0b011000: march_cell<lod, false, false, false, true , true , false>(x, y, z, chunk, mesh); break;
                    case 0b011001: march_cell<lod, true , false, false, true , true , false>(x, y, z, chunk, mesh); break;
                    case 0b011010: march_cell<lod, false, true , false, true , true , false>(x, y, z, chunk, mesh); break;
                    case 0b011011: march_cell<lod, true , true , false, true , true , false>(x, y, z, chunk, mesh); break;
                    case 0b011100: march_cell<lod, false, false, true , true , true , false>(x, y, z, chunk, mesh); break;
                    case 0b011101: march_cell<lod, true , false, true , true , true , false>(x, y, z, chunk, mesh); break;
                    case 0b011110: march_cell<lod, false, true , true , true , true , false>(x, y, z, chunk, mesh); break;
                    case 0b011111: march_cell<lod, true , true , true , true , true , false>(x, y, z, chunk, mesh); break;
                    case 0b100000: march_cell<lod, false, false, false, false, false, true >(x, y, z, chunk, mesh); break;
                    case 0b100001: march_cell<lod, true , false, false, false, false, true >(x, y, z, chunk, mesh); break;
                    case 0b100010: march_cell<lod, false, true , false, false, false, true >(x, y, z, chunk, mesh); break;
                    case 0b100011: march_cell<lod, true , true , false, false, false, true >(x, y, z, chunk, mesh); break;
                    case 0b100100: march_cell<lod, false, false, true , false, false, true >(x, y, z, chunk, mesh); break;
                    case 0b100101: march_cell<lod, true , false, true , false, false, true >(x, y, z, chunk, mesh); break;
                    case 0b100110: march_cell<lod, false, true , true , false, false, true >(x, y, z, chunk, mesh); break;
                    case 0b100111: march_cell<lod, true , true , true , false, false, true >(x, y, z, chunk, mesh); break;
                    case 0b101000: march_cell<lod, false, false, false, true , false, true >(x, y, z, chunk, mesh); break;
                    case 0b101001: march_cell<lod, true , false, false, true , false, true >(x, y, z, chunk, mesh); break;
                    case 0b101010: march_cell<lod, false, true , false, true , false, true >(x, y, z, chunk, mesh); break;
                    case 0b101011: march_cell<lod, true , true , false, true , false, true >(x, y, z, chunk, mesh); break;
                    case 0b101100: march_cell<lod, false, false, true , true , false, true >(x, y, z, chunk, mesh); break;
                    case 0b101101: march_cell<lod, true , false, true , true , false, true >(x, y, z, chunk, mesh); break;
                    case 0b101110: march_cell<lod, false, true , true , true , false, true >(x, y, z, chunk, mesh); break;
                    case 0b101111: march_cell<lod, true , true , true , true , false, true >(x, y, z, chunk, mesh); break;
                    case 0b110000: march_cell<lod, false, false, false, false, true , true >(x, y, z, chunk, mesh); break;
                    case 0b110001: march_cell<lod, true , false, false, false, true , true >(x, y, z, chunk, mesh); break;
                    case 0b110010: march_cell<lod, false, true , false, false, true , true >(x, y, z, chunk, mesh); break;
                    case 0b110011: march_cell<lod, true , true , false, false, true , true >(x, y, z, chunk, mesh); break;
                    case 0b110100: march_cell<lod, false, false, true , false, true , true >(x, y, z, chunk, mesh); break;
                    case 0b110101: march_cell<lod, true , false, true , false, true , true >(x, y, z, chunk, mesh); break;
                    case 0b110110: march_cell<lod, false, true , true , false, true , true >(x, y, z, chunk, mesh); break;
                    case 0b110111: march_cell<lod, true , true , true , false, true , true >(x, y, z, chunk, mesh); break;
                    case 0b111000: march_cell<lod, false, false, false, true , true , true >(x, y, z, chunk, mesh); break;
                    case 0b111001: march_cell<lod, true , false, false, true , true , true >(x, y, z, chunk, mesh); break;
                    case 0b111010: march_cell<lod, false, true , false, true , true , true >(x, y, z, chunk, mesh); break;
                    case 0b111011: march_cell<lod, true , true , false, true , true , true >(x, y, z, chunk, mesh); break;
                    case 0b111100: march_cell<lod, false, false, true , true , true , true >(x, y, z, chunk, mesh); break;
                    case 0b111101: march_cell<lod, true , false, true , true , true , true >(x, y, z, chunk, mesh); break;
                    case 0b111110: march_cell<lod, false, true , true , true , true , true >(x, y, z, chunk, mesh); break;
                    case 0b111111: march_cell<lod, true , true , true , true , true , true >(x, y, z, chunk, mesh); break;
                }
            }
        }
    }
}

void chunk::march(mesh& mesh) const
{
    if (lod >= 4) return;

    switch(lod)
    {
    case 0: march_chunk<0>(*this, mesh); break;
    case 1: march_chunk<1>(*this, mesh); break;
    case 2: march_chunk<2>(*this, mesh); break;
    case 3: march_chunk<3>(*this, mesh); break;
    }
}
