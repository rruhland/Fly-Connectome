#include <cstdint>
#include <cstring>

extern "C" void glReadPixels(int, int, int, int, unsigned, unsigned, void*);

// The unchanged MuJoCo draw uses packed GL_RGBA8 and GL_DEPTH32F_STENCIL8.
// GL_FLOAT_32_UNSIGNED_INT_24_8_REV copies the native 32-bit depth and
// companion stencil word with no depth quantization or reinterpretation.
extern "C" void packed_capture(int w, int h, std::uint8_t* bgra, void* depth,
                               const std::uint8_t* background, const float* backdrop,
                               std::uint8_t* result) noexcept {
    constexpr unsigned gl_bgra=0x80E1, gl_uint_rev=0x8367;
    constexpr unsigned gl_depth_stencil=0x84F9, gl_float_stencil_rev=0x8DAD;
    glReadPixels(0,0,w,h,gl_bgra,gl_uint_rev,bgra);
    glReadPixels(0,0,w,h,gl_depth_stencil,gl_float_stencil_rev,depth);
    const auto n=static_cast<std::size_t>(w)*h;
    std::memcpy(result,background,3*n);
    const auto* z=static_cast<const float*>(depth);
    for (std::size_t i=0;i<n;++i) {
        if (z[2*i]>backdrop[i]) {
            result[3*i]=bgra[4*i+2]; result[3*i+1]=bgra[4*i+1]; result[3*i+2]=bgra[4*i];
        }
    }
}
