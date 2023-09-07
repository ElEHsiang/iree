#include "iree/builtins/ukernel/arch/riscv_64/common_riscv_64.h"
#include "iree/builtins/ukernel/softmax_internal.h"
#include "iree/schemas/cpu_data.h"

#include <riscv_vector.h>
#include <float.h>

inline float fredmax_max(const float *input_data, size_t N) {
    size_t avl;
    size_t vl = __riscv_vsetvl_e32m8(N);
    vfloat32m8_t t = __riscv_vle32_v_f32m8(input_data, vl);
    input_data += vl;

    for (avl = N - vl; avl; avl -= vl, input_data += vl) {
        vl = __riscv_vsetvl_e32m8(N);
        vfloat32m8_t vec = __riscv_vle32_v_f32m8(input_data, vl);
        t = __riscv_vfmax_vv_f32m8(t, vec, vl);
    }

    vfloat32m1_t f = __riscv_vfmv_s_f_f32m1(-FLT_MAX, 1);
    f = __riscv_vfredmax_vs_f32m8_f32m1(t, f, N);

    return __riscv_vfmv_f_s_f32m1_f32(f);
}

static inline vfloat32m4_t eval_poly_horner(vfloat32m4_t x, vfloat32m4_t y, size_t vl,
        float c, float c1, float c2, float c3, float c4) {
    vfloat32m4_t z, a;

    z = __riscv_vfmv_v_f_f32m4(c, vl);
    a = __riscv_vfmadd_vv_f32m4(y, x, z, vl);

    y = a;
    z = __riscv_vfmv_v_f_f32m4(c1, vl);
    a = __riscv_vfmadd_vv_f32m4(y, x, z, vl);

    y = a;
    z = __riscv_vfmv_v_f_f32m4(c2, vl);
    a = __riscv_vfmadd_vv_f32m4(y, x, z, vl);

    y = a;
    z = __riscv_vfmv_v_f_f32m4(c3, vl);
    a = __riscv_vfmadd_vv_f32m4(y, x, z, vl);

    z = __riscv_vfmv_v_f_f32m4(c4, vl);
    z = __riscv_vfmadd_vv_f32m4(a, x, z, vl);
    return z;
}

inline vfloat32m4_t softexp_f32m4(vfloat32m4_t x, size_t vl) {
  // Ensure that q = RN(x/log(2)) >= e_min, so that 2^q can be computed safely
  // with a simple shift into the exponent field.
  // xmin = round(-126.5 * log(2), single, RU) ~ -87.68311309814453125
  const float xmin = -0x1.5ebb82p6;
  // x = rvv::vfmax(x, xmin, vl);
  x = __riscv_vfmax_vf_f32m4(x, xmin, vl);

  // 0. Reduction
  const float r_ln2f = 0x1.715476p0f; // single(1/log(2));
  const float l2uf = 0x1.62e4p-1f;    // round(log(2), 24-8, RN);
  const float l2lf = 0x1.7f7d1cp-20f; // round(log(2) - l2uf, single, RN);
  // T v = rvv::vfmul(x, r_ln2f, vl);
  vfloat32m4_t v = __riscv_vfmul_vf_f32m4(x, r_ln2f, vl);

  // q is narrowed to free some register space for masks
  // auto q = rvv::vfncvt_x(v, vl);
  vint16m2_t q = __riscv_vfncvt_x_f_w_i16m2(v, vl);
  // T z = rvv::vfwcvt_f(q, vl);
  vfloat32m4_t z = __riscv_vfwcvt_f_x_v_f32m4(q, vl);

  // T s = rvv::vfnmsac(x, l2uf, z, vl); // s = x - l2uf * z
  vfloat32m4_t s = __riscv_vfnmsac_vf_f32m4(x, l2uf, z, vl);
  // c5 moved up here for better scheduling of two vlse32.v instructions
  // T c5 = rvv::vfmv<T>(0x1.123bccp-7f, vl);
  vfloat32m4_t c5 = __riscv_vfmv_v_f_f32m4(0x1.123bccp-7f, vl);
  // s = rvv::vfnmsac(s, l2lf, z, vl); // s = s - l2lf * z
  s = __riscv_vfnmsac_vf_f32m4(s, l2lf, z, vl);

  // 1. Approximate e^s
  //
  // sollya> l = log(2)/2; d = [-l;l];
  // sollya> p = fpminimax(exp(x), 6, [|SG...|], d, relative, floating);
  // sollya> supnorm(p, exp(x), d, relative, 1b-24);
  // [0x1.fbad01f097cp-29;0x1.fbad03dc6759e11302p-29]
  // sollya> display=decimal!; -log2(0x1.fbad03dc6759e11302p-29);
  // 28.0122362050833425660857275511423589911147971511534
  const float c6 = 0x1.6850e4p-10f;
  // T u = rvv::vfmacc(c5, c6, s, vl);
  vfloat32m4_t u = __riscv_vfmacc_vf_f32m4(c5, c6, s, vl);
  // TODO: Not validate
  u = eval_poly_horner(s, u, vl, 0x1.555b98p-5f, 0x1.55548ep-3f, 0x1.fffff8p-2f,
                     1.0f, 1.0f);

  // 2. Reconstruction: compute u = u*2^q
  // constexpr int16_t p = std::numeric_limits<F>::digits;
  const int16_t p = FLT_MANT_DIG;
  // constexpr int16_t bias = std::numeric_limits<F>::max_exponent - 1;
  const int16_t bias = FLT_MAX_EXP - 1;
  // auto qw = rvv::vwadd(q, bias, vl);
  vint32m4_t qw = __riscv_vwadd_vx_i32m4(q, bias, vl);
  // auto qq = rvv::vsll(qw, p - 1, vl);
  vint32m4_t qq = __riscv_vsll_vx_i32m4(qw, p - 1, vl);
  // auto qf = rvv::vreinterpret<T>(qq);
  vfloat32m4_t qf = __riscv_vreinterpret_v_i32m4_f32m4(qq);
  // u = rvv::vfmul(u, qf, vl);
  u = __riscv_vfmul_vv_f32m4(u, qf, vl);
  return u;
}

void iree_uk_softmax_tile_riscv_64_f32_1d(
    const float* input_data,
    float* output_data,
    iree_uk_int32_t N) {
    size_t avl, vl;
    size_t current;
    float beta = 1.0f;
    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    // auto max =
    //     fredmax_fast<typename ::rvv::type_info<ST>::m8>(input_data, length);
    float max = fredmax_max(input_data, N);

    // Compute sum
    //  auto v_add_data =
    //      ::rvv::vfmv<vtype>(ST{0.0f}, ::rvv::type_info<vtype>::vsetvl(length));
    vfloat32m4_t v_add_data = __riscv_vfmv_v_f_f32m4(0.0f, __riscv_vsetvl_e32m4(N));
    for (current = 0, avl = N; avl; avl -= vl, current += vl) {
        // vl = ::rvv::type_info<vtype>::vsetvl(avl);
        vl = __riscv_vsetvl_e32m4(avl);
        // Compute x = (input - max) * beta => x <= 0, exp(x) <= 1
        // auto v_input_data = ::rvv::vle<vtype>(input_data + current, vl);
        vfloat32m4_t v_input_data = __riscv_vle32_v_f32m4(input_data + current, vl);
        // auto v_exp_c = ::rvv::vfsub(v_input_data, max, vl);
        vfloat32m4_t v_exp_c = __riscv_vfsub_vf_f32m4(v_input_data, max, vl);
        // v_exp_c = ::rvv::vfmul(v_exp_c, beta, vl);
        v_exp_c = __riscv_vfmul_vf_f32m4(v_exp_c, beta, vl);

        // exp_value = skl::softexp(x, vl)
        v_exp_c = softexp_f32m4(v_exp_c, vl);

        // store back exp(x) to output location and divide it by sum later
        //::rvv::vse(output_data + current, v_exp_c, vl);
        __riscv_vse32_v_f32m4(output_data + current, v_exp_c, vl);

        // Transform the reduce sum sequences into multiple elementwise additions
        // along with last reduce sum operation

        // Use elemtwise add with tail-undisturbed (to handle the case where
        // two inputs of differnet effective length do elementwise add)
        // v_add_data = ::rvv::vfadd_tu(v_add_data, v_add_data, v_exp_c, vl);
        v_add_data = __riscv_vfadd_vv_f32m4_tu(v_add_data, v_add_data, v_exp_c, vl);
    }

    // Unordered sum change the numerics, but it won't necessarily hurt accuracy
    // (the inputs are in fact randomly distributed)
    // using vredtype = typename ::rvv::type_info<vtype>::reduce_type;
    // auto v_sum_data = ::rvv::vfmv<vredtype>(ST{0.0f}, length);
    vfloat32m1_t v_sum_data = __riscv_vfmv_v_f_f32m1(0.0f, N);
    // v_sum_data = ::rvv::vfredusum(v_add_data, v_sum_data,
    //                               ::rvv::type_info<vtype>::vsetvl(length));
    v_sum_data = __riscv_vfredusum_vs_f32m4_f32m1(v_add_data, v_sum_data, __riscv_vsetvl_e32m4(N));

    // The reduce sum sclar is stored in v_sum_data[0], assign the value to sum
    // scalar
    // const ST sum = ::rvv::vfmv_f_s(v_sum_data);
    const float sum = __riscv_vfmv_f_s_f32m1_f32(v_sum_data);
    // ST reciprocal_sum = 1.0f / sum;
    float reciprocal_sum = 1.0 / sum;

    // if (max == std::numeric_limits<ST>::infinity())
    //   reciprocal_sum = std::numeric_limits<ST>::quiet_NaN();
    //   TODO: Can't use math.h, no NaN & HUGE_VALF

    // Compute result.
    for (current = 0, avl = N; avl; avl -= vl, current += vl) {
      // vl = ::rvv::type_info<vtype>::vsetvl(avl);
      vl = __riscv_vsetvl_e32m4(avl);
      // auto v_exp_c = ::rvv::vle<vtype>(output_data + current, vl);
      vfloat32m4_t v_exp_c = __riscv_vle32_v_f32m4(output_data + current, vl);
      // v_exp_c = ::rvv::vfmul(v_exp_c, reciprocal_sum, vl);
      v_exp_c = __riscv_vfmul_vf_f32m4(v_exp_c, reciprocal_sum, vl);
      // ::rvv::vse(output_data + current, v_exp_c, vl);
      __riscv_vse32_v_f32m4(output_data + current, v_exp_c, vl);
    }
}

void iree_uk_softmax_tile_riscv_64_f32_rvv(
    const void* IREE_UK_RESTRICT src_buffer,
    void* IREE_UK_RESTRICT dst_buffer,
    iree_uk_int32_t M,
    iree_uk_int32_t N) {
    int i;
    int offset;
    const float *input_data = (const float *)src_buffer;
    float *output_data = (float *)dst_buffer;

    for (i = 0, offset = 0; i < M; i++, offset += N) {
        iree_uk_softmax_tile_riscv_64_f32_1d(input_data + offset, output_data + offset, N);
    }
}
