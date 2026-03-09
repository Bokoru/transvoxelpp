#pragma once
#include <transvoxel.hpp>
#include <glm/glm.hpp>

namespace transvoxel
{
    template<glm::length_t L, typename T, glm::qualifier Q>
    struct vector_traits<glm::vec<L, T, Q>>
    {
        using value_type = glm::vec<L, T, Q>;

        using component_type = T;

        template<typename = std::enable_if_t<L == 1>>
        constexpr static value_type construct(component_type x, component_type y, component_type z)
        {
            return value_type(x);
        }

        template<typename = std::enable_if_t<L == 2>>
        constexpr static value_type construct(component_type x, component_type y, component_type z)
        {
            return value_type(x, y);
        }

        template<typename = std::enable_if_t<L == 3>>
        constexpr static value_type construct(component_type x, component_type y, component_type z)
        {
            return value_type(x, y, z);
        }

        template<typename = std::void_t<decltype(std::declval<value_type>().x)>
        static component_type& x_accessor(value_type& v) { return v.x; }

        template<typename = std::void_t<decltype(std::declval<value_type>().y)>
        static component_type& y_accessor(value_type& v) { return v.y; }

        template<typename = std::void_t<decltype(std::declval<value_type>().z)>
        static component_type& z_accessor(value_type& v) { return v.z; }
        
        static value_type add(const value_type& x, const value_type& y)
        {
            return x + y;
        }

        static value_type sub(const value_type& x, const value_type& y)
        {
            return x - y;
        }

        static value_type scale(const value_type& x, component_type y)
        {
            return x * y;
        }

        static component_type dot(const value_type& x, const value_type& y)
        {
            return glm::dot(x, y);
        }

        template<typename = std::enable_if_t<L == 3>>
        static value_type cross(const value_type& x, const value_type& y)
        {
            return glm::cross(x, y);
        }

        static value_type normalize(const value_type& x)
        {
            return glm::normalize(x);
        }
    };
}

/*

Example vertex:

struct transvoxel_vertex
{
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 tangent;
    glm::uvec2 material_indices; // can also be u8vec2 if voxel.material is uint8_t
    glm::vec2 material_weights;
};

If your vertex type uses different names, you will need to provide a custom specialization of transvoxel::vertex_traits

*/
