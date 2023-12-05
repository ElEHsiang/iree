// Copyright (c) 2023 SiFive, Inc. -- Proprietary and Confidential All Rights
// Reserved.
//
// NOTICE: All information contained herein is, and remains the property of
// SiFive, Inc. The intellectual and technical concepts contained herein are
// proprietary to SiFive, Inc. and may be covered by U.S. and Foreign Patents,
// patents in process, and are protected by trade secret or copyright law.
//
// This work may not be copied, modified, re-published, uploaded, executed, or
// distributed in any way, in any medium, whether in whole or in part, without
// prior written permission from SiFive, Inc.
//
// The copyright notice above does not evidence any actual or intended
// publication or disclosure of this source code, which includes information
// that is confidential and/or proprietary, and is a trade secret, of SiFive,
// Inc.
//===----------------------------------------------------------------------===//

#include <riscv_vector.h>

// vector inverse 


// 1 / asm(fsqrt)
static inline float asm_rsqrt(float number) {
  float val;

  __asm__ __volatile__("fsqrt.s %0, %1\n": "=f"(val): "f"(number));

  return 1 / val;
}

// From Quake III
static inline float Q_rsqrt( float number )
{
    long i;
    float x2, y;
    const float threehalfs = 1.5F;

    x2 = number * 0.5F;
    y  = number;
    i  = * ( long * ) &y;                       // evil floating point bit level hacking
    i  = 0x5f3759df - ( i >> 1 );               // what the fuck?
    y  = * ( float * ) &i;
    y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
//  y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

    return y;
}


static inline void LayerNorm1D(const float *input, const float *gamma, const float *beta,
                        float *output, float epsilon, int32_t axis_size) {
  // v_type = vfloat32m4_t
  // v_reduct_type = vfloat32m1_t
  // v_sq_type = vfloat32m4_t
  // v_sq_reduce_type = vfloat32m1_t
  
  const size_t init_vl = __riscv_vsetvl_e32m4(axis_size);

  float mean0, mean1, mean2, mean3;
  float var_rsqrt0, var_rsqrt1, var_rsqrt2, var_rsqrt3;
  float reciprocal_axis_size = 1 / (float) axis_size;

  // LN_CAL_MEAN_VAR_TWO_PASS_MACRO(0)
  vfloat32m4_t v_sum;                                                              
                                                                             
  for (size_t avl = axis_size, vl = 0, last_dim_offset = 0, first = 1;
       avl > 0; avl -= vl, last_dim_offset += vl) {
    vl = __riscv_vsetvl_e32m4(avl);

    if (first) {
      v_sum = __riscv_vle32_v_f32m4(input + axis_size * 0 + last_dim_offset, vl);
      first = 0;
    } else {
      vfloat32m4_t v_data = __riscv_vle32_v_f32m4(input + axis_size * 0 + last_dim_offset, vl);
      v_sum = __riscv_vfadd_vv_f32m4_tu(v_sum, v_sum, v_data, vl);
    }
  }

  vfloat32m1_t v_redusum = __riscv_vfmv_v_f_f32m1(0.0f, init_vl);
  v_redusum = __riscv_vfredusum_vs_f32m4_f32m1(v_sum, v_redusum, init_vl);
  mean0 = __riscv_vfmv_f_s_f32m1_f32(v_redusum) * reciprocal_axis_size;

  vfloat32m4_t v_var = __riscv_vfmv_v_f_f32m4(0.0f, init_vl);

  for (size_t avl = axis_size, vl = 0, last_dim_offset = 0; avl > 0;
      avl -= vl, last_dim_offset += vl) {
    vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t v_data = __riscv_vle32_v_f32m4(input + axis_size * 0 + last_dim_offset, vl);
    v_data = __riscv_vfsub_vf_f32m4_tu(v_data, v_data, mean0, vl);
    v_var = __riscv_vfmacc_vv_f32m4_tu(v_var, v_data, v_data, vl);
  }

  vfloat32m1_t v_var_redusum = __riscv_vfmv_v_f_f32m1(0.0f, init_vl);
  v_var_redusum = __riscv_vfredusum_vs_f32m4_f32m1(v_var, v_var_redusum, init_vl);
  const float var = __riscv_vfmv_f_s_f32m1_f32(v_var_redusum) * reciprocal_axis_size; 
  var_rsqrt0 = asm_rsqrt(var + epsilon);
  // End of LN_CAL_MEAN_VAR_TWO_PASS_MACRO

  // Calculate z-score
  for (size_t avl = axis_size, vl = 0, last_dim_offset = 0; avl > 0; avl -= vl, last_dim_offset +=vl) {
    vl = __riscv_vsetvl_e32m4(avl);
    vfloat32m4_t v_data0, vdata1, vdata2, vdata3;

    // LN_LOAD_AND_VFSUB_MACRO
    v_data0 = __riscv_vle32_v_f32m4(input + last_dim_offset + axis_size * 0, vl);
    v_data0 = __riscv_vfsub_vf_f32m4(v_data0, mean0, vl);
    // end of LN_LORD_AND_VFSUB_MACRO

    if (gamma != NULL && beta != NULL) {
      vfloat32m4_t v_gamma = __riscv_vle32_v_f32m4(gamma + last_dim_offset, vl);
      vfloat32m4_t v_beta = __riscv_vle32_v_f32m4(beta + last_dim_offset, vl);

      v_data0 = __riscv_vfmul_vf_f32m4(v_data0, var_rsqrt0, vl);
      v_data0 = __riscv_vfmadd_vv_f32m4(v_data0, v_gamma, v_beta, vl);

    } else {
      // No gamma & beta, multiply rsqrt directly
      v_data0 = __riscv_vfmul_vf_f32m4(v_data0, var_rsqrt0, vl);
    }
    
    // LN_STORE_RESULT_MACRO
    __riscv_vse32_v_f32m4(output + last_dim_offset + axis_size * 0, v_data0, vl);
    // end of LN_STORE_RESULT_MACRO
  }
}
