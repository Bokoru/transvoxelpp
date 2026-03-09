#pragma once

#include <array>
#include <limits>
#include <type_traits>
#include <cmath>

namespace transvoxel
{
    using byte_t = unsigned char;
    using ushort_t = unsigned short;
    
    struct regular_cell_data
    {
        byte_t geometry_counts;
        byte_t vertex_index[15];

        int vertex_count() const noexcept
        {
            return geometry_counts >> 4;
        }

        int triangle_count() const noexcept
        {
            return geometry_counts & 0x0F;
        }
    };

    struct transition_cell_data
    {
        byte_t geometry_counts;
        byte_t vertex_index[36];

        int vertex_count() const noexcept
        {
            return geometry_counts >> 4;
        }

        int triangle_count() const noexcept
        {
            return geometry_counts & 0x0F;
        }
    };

    struct lut
    {
        static const byte_t regular_cell_class[256];
        static const regular_cell_data regular_cell_data[16];
        static const ushort_t regular_vertex_data[256][12];

        static const byte_t transition_cell_class[512];
        static const transition_cell_data transition_cell_data[56];
        static const byte_t transition_corner_data[13];
        static const ushort_t transition_vertex_data[512][12];
    };

    template<typename vector_t>
    struct vector_traits 
    {
        // using value_type = vector_t;
        // using component_type = <...>;
        // static component_type& x_accessor(vector_t&);
        // static component_type& y_accessor(vector_t&);
        // static component_type& z_accessor(vector_t&);
        // constexpr static vector_t construct(component_type x, component_type y, component_type z);
        // static vector_t add(const vector_t& x, const vector_t& y);
        // static vector_t sub(const vector_t& x, const vector_t& y);
        // static vector_t scale(const vector_t& x, component_type y);
        // static component_type dot(const vector_t& x, const vector_t& y);
        // static vector_t cross(const vector_t& x, const vector_t& y);
        // static vector_t normalize(const vector_t& x);
    };

    template<typename voxel_t>
    struct voxel_traits 
    {
        using volume_type = decltype(std::declval<voxel_t>().volume);
        static const volume_type& get_volume(const voxel_t& voxel) { return voxel.volume; }

        template<typename dummy_t = void, typename = void>
        struct material
        {
            using value_type = void;
            static constexpr bool enabled = false;
            static const value_type get(const voxel_t& voxel) {}
        };

        template<typename dummy_t>
        struct material<dummy_t, std::void_t<decltype(std::declval<voxel_t>().material)>>
        {
            using value_type = std::remove_reference_t<decltype(std::declval<voxel_t>().material)>;
            static constexpr bool enabled = true;
            static const value_type& get(const voxel_t& voxel) { return voxel.material; }
        };

        template<typename dummy_t = void, typename = void>
        struct gradient
        {
            using value_type = void;
            static constexpr bool enabled = false;
            static const value_type get(const voxel_t& voxel) {}
        };

        template<typename dummy_t>
        struct gradient<dummy_t, std::void_t<decltype(std::declval<voxel_t>().gradient)>>
        {
            using value_type = std::remove_reference_t<decltype(std::declval<voxel_t>().gradient)>;
            static constexpr bool enabled = true;
            static const value_type& get(const voxel_t& voxel) { return voxel.gradient; }
        };
    };

    template<typename vertex_t, typename = std::void_t<decltype(std::declval<vertex_t>().position)>>
    struct vertex_traits 
    {
        template<typename dummy_t = void, typename = void>
        struct position {};

        template<typename dummy_t = void, typename = void>
        struct normal {};

        template<typename dummy_t = void, typename = void>
        struct tangent {};

        template<typename dummy_t = void, typename = void, typename = void>
        struct material {};
        
        template<typename dummy_t>
        struct position<dummy_t, std::void_t<decltype(std::declval<vertex_t>().position)>>
        {
            using value_type = std::remove_reference_t<decltype(std::declval<vertex_t>().position)>;
            using vector_traits = transvoxel::vector_traits<value_type>;
            static constexpr bool enabled = true;
            static value_type& access(vertex_t& vertex) { return vertex.position; }
        };

        template<typename dummy_t>
        struct normal<dummy_t, std::void_t<decltype(std::declval<vertex_t>().normal)>>
        {
            using value_type = std::remove_reference_t<decltype(std::declval<vertex_t>().normal)>;
            using vector_traits = transvoxel::vector_traits<value_type>;
            static constexpr bool enabled = true;
            static value_type& access(vertex_t& vertex) { return vertex.normal; }
        };

        template<typename dummy_t>
        struct tangent<dummy_t, std::void_t<decltype(std::declval<vertex_t>().tangent)>>
        {
            using value_type = std::remove_reference_t<decltype(std::declval<vertex_t>().tangent)>;
            using vector_traits = transvoxel::vector_traits<value_type>;
            static constexpr bool enabled = true;
            static value_type& access(vertex_t& vertex) { return vertex.tangent; }
        };

        template<typename dummy_t>
        struct material<dummy_t, std::void_t<decltype(std::declval<vertex_t>().material_indices)>, std::void_t<decltype(std::declval<vertex_t>().material_weights)>>
        {
            using indices_value_type = std::remove_reference_t<decltype(std::declval<vertex_t>().material_indices)>;
            using weights_value_type = std::remove_reference_t<decltype(std::declval<vertex_t>().material_weights)>;
            static constexpr bool enabled = true;
            static indices_value_type& access_indices(vertex_t& vertex) { return vertex.material_indices; }
            static weights_value_type& access_weights(vertex_t& vertex) { return vertex.material_weights; }
        };
    };

    // An extremely basic vec3 used for transition cells in-place of an actual vec3 if the vertex type does not specify normals.
    // This exists because transition cells require vertex normals during generation regardless of if the generated normals are kept or used afterwords.
    // Using this class in your own code is not recommended.
    struct basic_normal
    {
        float x, y, z;
    };

    template<>
    struct vector_traits<basic_normal>
    {
        using value_type = basic_normal;
        using component_type = float;
        static component_type& x_accessor(basic_normal& v) { return v.x; }
        static component_type& y_accessor(basic_normal& v) { return v.y; }
        static component_type& z_accessor(basic_normal& v) { return v.z; }
        constexpr static basic_normal construct(component_type x, component_type y, component_type z) { return { x, y, z }; }
        static basic_normal add(const basic_normal& x, const basic_normal& y) { return {x.x + y.x, x.y + y.y, x.z + y.z}; }
        static basic_normal sub(const basic_normal& x, const basic_normal& y) { return {x.x - y.x, x.y - y.y, x.z - y.z}; }
        static basic_normal scale(const basic_normal& x, component_type y) { return { x.x * y, x.y * y, x.z * y }; }
        static component_type dot(const basic_normal& x, const basic_normal& y) { return x.x * y.x + x.y * y.y + x.z * y.z; }
        static basic_normal cross(const basic_normal& x, const basic_normal& y) { return { x.y * y.z - x.z * y.y, x.z * y.x - x.x * y.z, x.x * y.y - x.y * y.x }; }
        static basic_normal normalize(const basic_normal& x) { float l = std::sqrtf(x.x * x.x + x.y * x.y + x.z * x.z); return { x.x / l, x.y / l, x.z / l }; }
    };

    template<typename voxel_getter_t, typename vertex_putter_t, typename triangle_putter_t, unsigned lod>
    struct algorithm
    {
        template<typename func_t>
        struct first_argument;
        
        template <typename result_t, typename first_arg_t, typename... remaining_arg_ts>
        struct first_argument<result_t(*)(first_arg_t, remaining_arg_ts...)> 
        {
            using type = first_arg_t;
        };
        
        template <typename result_t, typename owner_t, typename first_arg_t, typename... remaining_arg_ts>
        struct first_argument<result_t(owner_t::*)(first_arg_t, remaining_arg_ts...)> 
        {
            using type = first_arg_t;
        };

        template <typename result_t, typename owner_t, typename first_arg_t, typename... remaining_arg_ts>
        struct first_argument<result_t(owner_t::*)(first_arg_t, remaining_arg_ts...) const> 
        {
            using type = first_arg_t;
        };

        template <typename generic_func_t>
        struct first_argument 
        {
            using type = typename first_argument<decltype(&generic_func_t::operator())>::type;
        };

        using voxel_getter_index_t = typename first_argument<voxel_getter_t>::type;
        using voxel_getter_index_type = std::remove_reference_t<voxel_getter_index_t>;
        using voxel_getter_index_vector_traits = vector_traits<voxel_getter_index_type>;

        using voxel_type = std::remove_reference_t<std::invoke_result_t<voxel_getter_t, voxel_getter_index_t>>;
        using voxel_traits = transvoxel::voxel_traits<voxel_type>;

        using vertex_type = std::remove_reference_t<typename first_argument<vertex_putter_t>::type>;
        using vertex_traits = transvoxel::vertex_traits<vertex_type>;

        using index_type = std::remove_reference_t<typename first_argument<triangle_putter_t>::type>;
        using index_limits = std::numeric_limits<index_type>;

        using voxel_volume_type = typename voxel_type::volume_type;
        static constexpr auto voxel_volume_getter = voxel_traits::get_volume;

        using vertex_position_traits = typename vertex_traits::position<>::vector_traits;
        using vertex_position_type = typename vertex_position_traits::value_type;
        using vertex_position_component_type = typename vertex_position_traits::component_type;
        static constexpr auto vertex_position_accessor = vertex_traits::position<>::access;

        using voxel_getter_index_component_type = typename voxel_getter_index_vector_traits::component_type;

        static constexpr voxel_getter_index_component_type step_size = static_cast<voxel_getter_index_component_type>(1 << lod);
        static constexpr voxel_getter_index_component_type half_step = static_cast<voxel_getter_index_component_type>(1 << lod) >> 1;

        static constexpr voxel_getter_index_type getter_index_offsets[8]
        {
            voxel_getter_index_vector_traits::construct(0,         0,         0        ),
            voxel_getter_index_vector_traits::construct(step_size, 0,         0        ),
            voxel_getter_index_vector_traits::construct(0,         0,         step_size),
            voxel_getter_index_vector_traits::construct(step_size, 0,         step_size),
            voxel_getter_index_vector_traits::construct(0,         step_size, 0        ),
            voxel_getter_index_vector_traits::construct(step_size, step_size, 0        ),
            voxel_getter_index_vector_traits::construct(0,         step_size, step_size),
            voxel_getter_index_vector_traits::construct(step_size, step_size, step_size)
        };

        static constexpr vertex_position_component_type pstep_size = static_cast<vertex_position_component_type>(1 << lod);

        template<bool transition>
        static constexpr vertex_position_component_type psmall_off = transition ? static_cast<vertex_position_component_type>(0.1 * (1 << lod)) : static_cast<vertex_position_component_type>(0);

        template<bool left_transition, bool bottom_transition, bool back_transition>
        static constexpr vertex_position_type getter_position_offsets_transition_cell[8]
        {
            voxel_getter_index_vector_traits::construct(psmall_off<left_transition>, psmall_off<bottom_transition>, psmall_off<back_transition>),
            voxel_getter_index_vector_traits::construct(pstep_size, psmall_off<bottom_transition>, psmall_off<back_transition>),
            voxel_getter_index_vector_traits::construct(psmall_off<left_transition>, psmall_off<bottom_transition>, pstep_size),
            voxel_getter_index_vector_traits::construct(pstep_size, psmall_off<bottom_transition>, pstep_size),
            voxel_getter_index_vector_traits::construct(psmall_off<left_transition>, pstep_size, psmall_off<back_transition>),
            voxel_getter_index_vector_traits::construct(pstep_size, pstep_size, psmall_off<back_transition>),
            voxel_getter_index_vector_traits::construct(psmall_off<left_transition>, pstep_size, pstep_size),
            voxel_getter_index_vector_traits::construct(pstep_size, pstep_size, pstep_size)
        };

        static constexpr voxel_getter_index_type add_x = voxel_getter_index_vector_traits::construct(step_size, 0, 0);
        static constexpr voxel_getter_index_type add_y = voxel_getter_index_vector_traits::construct(0, step_size, 0);
        static constexpr voxel_getter_index_type add_z = voxel_getter_index_vector_traits::construct(0, 0, step_size);

        template<typename value_t, typename factor_t>
        static value_t lerp(const value_t& x, const value_t& y, factor_t t)
        {
            return y * t + x * (static_cast<factor_t>(1.0) - t);
        }

        template<typename value_t>
        static value_t lerp(const value_t& x, const value_t& y, typename vector_traits<value_t>::component_type t)
        {
            using vector_traits = transvoxel::vector_traits<value_t>;
            return vector_traits::add(vector_traits::scale(y, t), vector_traits::scale(x, static_cast<typename vector_traits<value_t>::component_type>(1.0) - t));
        }

        template<typename vector_t>
        static vector_t calculate_tangent_vector(const vector_t& original)
        {
            using vector_traits = transvoxel::vector_traits<vector_t>;
            if (std::fabs(vector_traits::dot(original, vector_t{0, 1, 0})) < 0.999f) return vector_traits::normalize(vector_traits::cross(original, vector_t{0, 1, 0}));
            return vector_traits::normalize(vector_traits::cross(original, vector_t{1, 0, 0}));
        }

        template<typename cell_t, typename corner_position_container>
        static index_type generate_vertex(ushort_t vertex_data, const cell_t& cell, const corner_position_container& corner_positions)
        {
            vertex_type vert{};
            byte_t corner_x = vertex_data & 0x0F, corner_y = (vertex_data & 0xF0) >> 4;

            float factor = static_cast<float>(-voxel_volume_getter(cell[corner_x])) / static_cast<float>(voxel_volume_getter(cell[corner_y]) - voxel_volume_getter(cell[corner_x]));
            vertex_position_accessor(vert) = lerp(corner_positions[corner_x], corner_positions[corner_y], factor);

            if constexpr (vertex_traits::normal<>::enabled)
            {
                using vertex_normal_type = typename vertex_traits::normal<>::value_type;
                constexpr auto vertex_normal_accessor = vertex_traits::normal<>::access;

                vertex_normal_type& vert_normal = vertex_normal_accessor(vert);
                using vertex_normal_vector_traits = vector_traits<vertex_normal_type>;

                constexpr auto voxel_gradient_getter = voxel_traits::gradient<>::get;
                vert_normal = vertex_normal_vector_traits::normalize(lerp(voxel_gradient_getter(cell[corner_x]), voxel_gradient_getter(cell[corner_y]), factor));
                
                if constexpr (vertex_traits::tangent<>::enabled)
                {
                    using vertex_tangent_type = typename vertex_traits::tangent<>::value_type;
                    constexpr auto vertex_tangent_accessor = vertex_traits::tangent<>::access;
                    vertex_tangent_accessor(vert) = static_cast<vertex_tangent_type>(calculate_tangent_vector(vert_normal));
                }
            }

            return vertex_putter(vert);
        }

        template<typename cell_t, typename corner_gradient_container, typename corner_position_container>
        static index_type generate_vertex(ushort_t vertex_data, const cell_t& cell, const corner_gradient_container& corner_gradient, const corner_position_container& corner_positions)
        {
            vertex_type vert{};
            byte_t corner_x = vertex_data & 0x0F, corner_y = (vertex_data & 0xF0) >> 4;

            float factor = static_cast<float>(-voxel_volume_getter(cell[corner_x])) / static_cast<float>(voxel_volume_getter(cell[corner_y]) - voxel_volume_getter(cell[corner_x]));
            vertex_position_accessor(vert) = lerp(corner_positions[corner_x], corner_positions[corner_y], factor);

            if constexpr (vertex_traits::normal<>::enabled)
            {
                using vertex_normal_type = typename vertex_traits::normal<>::value_type;
                constexpr auto vertex_normal_accessor = vertex_traits::normal<>::access;

                vertex_normal_type& vert_normal = vertex_normal_accessor(vert);
                using vertex_normal_vector_traits = vector_traits<vertex_normal_type>;

                constexpr auto voxel_gradient_getter = voxel_traits::gradient<>::get;
                vert_normal = vertex_normal_vector_traits::normalize(lerp(corner_gradient[corner_x], corner_gradient[corner_y], factor));
                
                if constexpr (vertex_traits::tangent<>::enabled)
                {
                    using vertex_tangent_type = typename vertex_traits::tangent<>::value_type;
                    constexpr auto vertex_tangent_accessor = vertex_traits::tangent<>::access;
                    vertex_tangent_accessor(vert) = static_cast<vertex_tangent_type>(calculate_tangent_vector(vert_normal));
                }
            }

            // Not correct anyway
            /* if constexpr (vertex_traits::material<>::enabled)
            {
                constexpr auto voxel_material_getter = voxel_traits::material<>::get;

                using vertex_material_indices_type = typename voxel_traits::material<>::indices_value_type;
                using material_indices_vector_traits = vector_traits<vertex_material_indices_type>;
                using material_indices_component_type = typename material_indices_vector_traits::component_type;

                using vertex_material_weights_type = typename voxel_traits::material<>::weights_value_type;
                using material_weights_vector_traits = vector_traits<vertex_material_weights_type>;
                using material_weights_component_type = typename material_weights_vector_traits::component_type;
                
                constexpr auto vertex_material_indices_accessor = voxel_traits::material<>::access_indices;
                constexpr auto vertex_material_weights_accessor = voxel_traits::material<>::access_weights;

                vertex_material_indices_type& material_indices = vertex_material_indices_accessor(vert);
                material_indices_vector_traits::x_accessor(material_indices) = static_cast<material_indices_component_type>(voxel_material_getter(cell[corner_x]));
                material_indices_vector_traits::y_accessor(material_indices) = static_cast<material_indices_component_type>(voxel_material_getter(cell[corner_y]));
                
                vertex_material_weights_type& material_weights = vertex_material_weights_accessor(vert);
                material_weights_vector_traits::x_accessor(material_weights) = static_cast<material_weights_component_type>(1.0) - static_cast<material_weights_component_type>(factor);
                material_weights_vector_traits::y_accessor(material_weights) = static_cast<material_weights_component_type>(factor);
            } */

            return vertex_putter(vert);
        }
        
        template<typename corner_position_container>
        static void march_regular_cell(const voxel_getter_index_type& where, const corner_position_container& corner_positions, const voxel_getter_t& voxel_getter, const vertex_putter_t& vertex_putter, const triangle_putter_t& triangle_putter)
        {
            voxel_type cell[8];
            for (int i = 0; i < 8; ++i) cell[i] = voxel_getter(voxel_getter_index_vector_traits::add(where, getter_index_offsets[i]));

            byte_t case_index = 0;
            for (int i = 0; i < 8; ++i) if(voxel_volume_getter(cell[i]) < 0) case_index |= 1 << i;
            if (case_index == 0x00 || case_index == 0xFF) return;

            const regular_cell_data& cell_data = lut::regular_cell_data[lut::regular_cell_class[case_index]];
            
            std::array<index_type, 12> vertex_indices{};
            for (int i = 0; i < cell_data.vertex_count(); ++i)
            {
                vertex_indices[i] = generate_vertex(lut::regular_vertex_data[case_index][i], cell, corner_positions);
            }

            int l = cell_data.triangle_count() * 3;
            for (int i = 0; i < l; i += 3)
            {
                triangle_putter(vertex_indices[cell_data.vertex_index[i]], vertex_indices[cell_data.vertex_index[i + 1]], vertex_indices[cell_data.vertex_index[i + 2]]);
            }
        }

        template<typename corner_position_container, typename corner_gradient_container>
        static void march_regular_cell(const voxel_getter_index_type& where, const corner_position_container& corner_positions, const corner_gradient_container& corner_gradients, const voxel_getter_t& voxel_getter, const vertex_putter_t& vertex_putter, const triangle_putter_t& triangle_putter)
        {
            voxel_type cell[8];
            for (int i = 0; i < 8; ++i) cell[i] = voxel_getter(voxel_getter_index_vector_traits::add(where, getter_index_offsets[i]));

            byte_t case_index = 0;
            for (int i = 0; i < 8; ++i) if(voxel_volume_getter(cell[i]) < 0) case_index |= 1 << i;
            if (case_index == 0x00 || case_index == 0xFF) return;

            const regular_cell_data& cell_data = lut::regular_cell_data[lut::regular_cell_class[case_index]];
            
            std::array<index_type, 12> vertex_indices{};
            for (int i = 0; i < cell_data.vertex_count(); ++i)
            {
                vertex_indices[i] = generate_vertex(lut::regular_vertex_data[case_index][i], cell, corner_gradients, corner_positions);
            }

            int l = cell_data.triangle_count() * 3;
            for (int i = 0; i < l; i += 3)
            {
                triangle_putter(vertex_indices[cell_data.vertex_index[i]], vertex_indices[cell_data.vertex_index[i + 1]], vertex_indices[cell_data.vertex_index[i + 2]]);
            }
        }

        template<typename = void, typename = void>
        static void march_regular_cell(const voxel_getter_index_type& where, const voxel_getter_t& voxel_getter, const vertex_putter_t& vertex_putter, const triangle_putter_t& triangle_putter)
        {
            vertex_position_type corner_positions[8];
            for (int i = 0; i < 8; ++i) { corner_positions[i] = static_cast<vertex_position_type>(voxel_getter_index_vector_traits::add(where, getter_index_offsets[i])); }
            march_regular_cell(where, corner_positions, voxel_getter, vertex_putter, triangle_putter);
        }
        
        template<typename corner_indices_container, typename corner_positions_container, typename corner_gradients_container>
        static void march_transition_cell(const corner_indices_container& corner_indices, const corner_positions_container& corner_positions, const corner_gradients_container& corner_gradients, const voxel_getter_t& voxel_getter, const vertex_putter_t& vertex_putter, const triangle_putter_t& triangle_putter)
        {
            voxel_type cell[9];
            for (int i = 0; i < 9; ++i) cell[i] = voxel_getter(corner_indices[i]); 

            ushort_t case_index = 0;
            for (int i = 0; i < 9; ++i) if(voxel_volume_getter(cell[i]) < 0) case_index |= 1 << i;
            if (case_index == 0 || case_index == 511) return;

            byte_t class_index = lut::transition_cell_class[case_index];
            bool reverse_winding = (class_index & 0x80) == 0x80;
            if (reverse_winding) class_index &= 0x7F;

            const transition_cell_data& cell_data = lut::transition_cell_data[class_index];

            std::array<index_type, 12> vertex_indices{};
            for (byte_t i = 0; i < cell_data.vertex_count(); ++i)
            {
                vertex_indices[i] = generate_vertex(lut::transition_vertex_data[case_index][i], cell, corner_gradients, corner_positions);
            }

            int l = cell_data.triangle_count() * 3;
            for (int i = 0; i < l; i += 3)
            {
                if (reverse_winding) triangle_putter(vertex_indices[cell_data.vertex_index[i + 2]], vertex_indices[cell_data.vertex_index[i + 1]], vertex_indices[cell_data.vertex_index[i]]);
                else triangle_putter(vertex_indices[cell_data.vertex_index[i]], vertex_indices[cell_data.vertex_index[i + 1]], vertex_indices[cell_data.vertex_index[i + 2]]);
            }
        }

        template<typename = void>
        using transition_gradient_vector_type = basic_normal;

        template<typename = std::enable_if_t<vertex_traits::normal<>::enabled>>
        using transition_gradient_vector_type = vertex_traits::normal<>::value_type;

        using transition_gradient_vector_traits = vector_traits<transition_gradient_vector_type>;
        using transition_gradient_component_type = typename transition_gradient_vector_traits::component_type;
        
        template<typename corner_indices_container, typename corner_positions_container>
        static void march_transition_cell(const corner_indices_container& corner_indices, const corner_positions_container& corner_positions, const voxel_getter_t& voxel_getter, const vertex_putter_t& vertex_putter, const triangle_putter_t& triangle_putter)
        {
            transition_gradient_vector_type corner_gradients[13];
            for (int i = 0; i < 9; ++i)
            {
                voxel_getter_index_type index = voxel_getter_index_vector_traits::add(corner_indices[i], getter_index_offsets[i]);
                transition_gradient_vector_traits::x_accessor(corner_gradients[i]) = (static_cast<transition_gradient_component_type>(voxel_volume_getter(voxel_getter(voxel_getter_index_vector_traits::add(index, add_x))) - voxel_volume_getter(voxel_getter(voxel_getter_index_vector_traits::sub(index, add_x)))) * static_cast<transition_gradient_component_type>(step_size * 0.5);
                transition_gradient_vector_traits::y_accessor(corner_gradients[i]) = (static_cast<transition_gradient_component_type>(voxel_volume_getter(voxel_getter(voxel_getter_index_vector_traits::add(index, add_y))) - voxel_volume_getter(voxel_getter(voxel_getter_index_vector_traits::sub(index, add_y)))) * static_cast<transition_gradient_component_type>(step_size * 0.5);
                transition_gradient_vector_traits::z_accessor(corner_gradients[i]) = (static_cast<transition_gradient_component_type>(voxel_volume_getter(voxel_getter(voxel_getter_index_vector_traits::add(index, add_z))) - voxel_volume_getter(voxel_getter(voxel_getter_index_vector_traits::sub(index, add_z)))) * static_cast<transition_gradient_component_type>(step_size * 0.5);
            }

            corner_gradients[9]  = corner_gradients[0];
            corner_gradients[10] = corner_gradients[2];
            corner_gradients[11] = corner_gradients[4];
            corner_gradients[12] = corner_gradients[6];

            return march_transition_cell(corner_indices, corner_positions, corner_gradients, voxel_getter, vertex_putter, triangle_putter);
        }

        template<typename = std::enable_if_t<vertex_traits::normal<>::enabled>, typename = std::enable_if_t<!voxel_traits::gradient<>::enabled>>
        static void march_regular_cell(const voxel_getter_index_type& where, const voxel_getter_t& voxel_getter, const vertex_putter_t& vertex_putter, const triangle_putter_t& triangle_putter)
        {
            vertex_position_type corner_positions[8];
            for (int i = 0; i < 8; ++i) { corner_positions[i] = static_cast<vertex_position_type>(voxel_getter_index_vector_traits::add(where, getter_index_offsets[i])); }

            using vertex_normal_type = typename vertex_traits::normal<>::value_type;
            using vertex_normal_vector_traits = vector_traits<vertex_normal_type>;
            using vertex_normal_component_type = typename vertex_normal_vector_traits::component_type;

            vertex_normal_type corner_gradients[8];
            for (int i = 0; i < 8; ++i)
            {
                voxel_getter_index_type index = voxel_getter_index_vector_traits::add(where, getter_index_offsets[i]);
                vertex_normal_vector_traits::x_accessor(corner_gradients[i]) = (static_cast<vertex_normal_component_type>(voxel_volume_getter(voxel_getter(voxel_getter_index_vector_traits::add(index, add_x))) - voxel_volume_getter(voxel_getter(voxel_getter_index_vector_traits::sub(index, add_x)))) * static_cast<vertex_normal_component_type>(step_size * 0.5);
                vertex_normal_vector_traits::y_accessor(corner_gradients[i]) = (static_cast<vertex_normal_component_type>(voxel_volume_getter(voxel_getter(voxel_getter_index_vector_traits::add(index, add_y))) - voxel_volume_getter(voxel_getter(voxel_getter_index_vector_traits::sub(index, add_y)))) * static_cast<vertex_normal_component_type>(step_size * 0.5);
                vertex_normal_vector_traits::z_accessor(corner_gradients[i]) = (static_cast<vertex_normal_component_type>(voxel_volume_getter(voxel_getter(voxel_getter_index_vector_traits::add(index, add_z))) - voxel_volume_getter(voxel_getter(voxel_getter_index_vector_traits::sub(index, add_z)))) * static_cast<vertex_normal_component_type>(step_size * 0.5);
            }

            march_regular_cell(where, corner_positions, corner_gradients, voxel_getter, vertex_putter, triangle_putter);
        }

        constexpr static voxel_getter_index_component_type transition_index_offset[9][2]
        {
            // [6]--[5]--[4]
            //  |    |    | 
            // [7]--[8]--[3]
            //  |    |    | 
            // [0]--[1]--[2]

            { 0,         0         },
            { half_step, 0         },
            { step_size, 0         },

            { step_size, half_step },
            { step_size, step_size },
            { half_step, step_size },

            { 0,         step_size },
            { 0,         half_step },
            { half_step, half_step },
        };

        template<bool left_transition, bool bottom_transition, bool back_transition, typename = std::enable_if_t<lod > 0>>
        static void march_transition_cell(const voxel_getter_index_type& where, const voxel_getter_t& voxel_getter, const vertex_putter_t& vertex_putter, const triangle_putter_t& triangle_putter)
        {
            if constexpr (!left_transition && !bottom_transition && !back_transition)
            {
                march_regular_cell(where, voxel_getter, vertex_putter, triangle_putter);
            }
            else
            {
                vertex_position_type regular_corner_positions[8];
                for (int i = 0; i < 8; ++i) { regular_corner_positions[i] = vertex_position_traits::add(static_cast<vertex_position_type>(where), getter_position_offsets_transition_cell<left_transition, bottom_transition, back_transition>[i]); }
                march_regular_cell(where, regular_corner_positions, voxel_getter, vertex_putter, triangle_putter);

                if constexpr (left_transition)
                {
                    constexpr static int reuse_corner_indices[4] = 
                    {
                        0, 2,
                        4, 6
                    };

                    voxel_getter_index_type corner_getter_indices[9];
                    vertex_position_type corner_positions[13];

                    for (int i = 0; i < 9; ++i)
                    {
                        const voxel_getter_index_component_type z = transition_index_offset[i][0];
                        const voxel_getter_index_component_type y = transition_index_offset[i][1];
                        corner_getter_indices[i] = voxel_getter_index_vector_traits::add(where, voxel_getter_index_vector_traits::construct(-half_step, y, z));
                        corner_positions[i]      = static_cast<vertex_position_type>(voxel_getter_index_vector_traits::add(where, voxel_getter_index_vector_traits::construct(0, y, z)));
                    }

                    for (int i = 0; i < 4; ++i)
                    {
                        corner_positions[9 + i] = regular_corner_positions[reuse_corner_indices[i]];
                    }

                    march_transition_cell(corner_getter_indices, corner_positions, voxel_getter, vertex_putter, triangle_putter);
                }

                if constexpr (bottom_transition)
                {
                    constexpr static int reuse_corner_indices[4] = 
                    {
                        0, 1,
                        2, 3
                    };

                    voxel_getter_index_type corner_getter_indices[9];
                    vertex_position_type corner_positions[13];

                    for (int i = 0; i < 9; ++i)
                    {
                        const voxel_getter_index_component_type x = transition_index_offset[i][0];
                        const voxel_getter_index_component_type z = transition_index_offset[i][1];
                        corner_getter_indices[i] = voxel_getter_index_vector_traits::add(where, voxel_getter_index_vector_traits::construct(x, -half_step, z));
                        corner_positions[i]      = static_cast<vertex_position_type>(voxel_getter_index_vector_traits::add(where, voxel_getter_index_vector_traits::construct(x, 0, z)));
                    }

                    for (int i = 0; i < 4; ++i)
                    {
                        corner_positions[9 + i] = regular_corner_positions[reuse_corner_indices[i]];
                    }

                    march_transition_cell(corner_getter_indices, corner_positions, voxel_getter, vertex_putter, triangle_putter);
                }

                if constexpr (back_transition)
                {
                    constexpr static int reuse_corner_indices[4] = 
                    {
                        0, 1,
                        4, 5
                    };

                    voxel_getter_index_type corner_getter_indices[9];
                    vertex_position_type corner_positions[13];

                    for (int i = 0; i < 9; ++i)
                    {
                        const voxel_getter_index_component_type x = transition_index_offset[i][0];
                        const voxel_getter_index_component_type y = transition_index_offset[i][1];
                        corner_getter_indices[i] = voxel_getter_index_vector_traits::add(where, voxel_getter_index_vector_traits::construct(x, y, -half_step));
                        corner_positions[i]      = static_cast<vertex_position_type>(voxel_getter_index_vector_traits::add(where, voxel_getter_index_vector_traits::construct(x, y, 0)));
                    }

                    for (int i = 0; i < 4; ++i)
                    {
                        corner_positions[9 + i] = regular_corner_positions[reuse_corner_indices[i]];
                    }

                    march_transition_cell(corner_getter_indices, corner_positions, voxel_getter, vertex_putter, triangle_putter);
                }
            }
        }
    };

    template<unsigned lod, typename where_t, typename voxel_getter_t, typename vertex_putter_t, typename triangle_putter_t>
    void march_regular_cell(const where_t& where, const voxel_getter_t& voxel_getter, const vertex_putter_t& vertex_putter, const triangle_putter_t& triangle_putter)
    {
        algorithm<voxel_getter_t, vertex_putter_t, triangle_putter_t, lod>::march_regular_cell(where, voxel_getter, vertex_putter, triangle_putter);
    }

    template<unsigned lod, bool left_transition, bool bottom_transition, bool back_transition, typename where_t, typename voxel_getter_t, typename vertex_putter_t, typename triangle_putter_t>
    void march_transition_cell(const where_t& where, const voxel_getter_t& voxel_getter, const vertex_putter_t& vertex_putter, const triangle_putter_t& triangle_putter)
    {
        algorithm<voxel_getter_t, vertex_putter_t, triangle_putter_t, lod>::march_transition_cell<left_transition, bottom_transition, back_transition>(where, voxel_getter, vertex_putter, triangle_putter);
    }
}
