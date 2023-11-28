// y = gamma * (x-mean(x)) / rsqrt(var(x) + epsilon) + beta
// Setting gamma = 1.0 and beta = 0.0 for simplicity.
//
// Generated from this TOSA input:
//
// func.func @layernorm() {
//   %x = util.unfoldable_constant dense<5.0> : tensor<128x384xf32>
//   %c384 = util.unfoldable_constant dense<384.0> : tensor<128x1xf32>
//   %sum = tosa.reduce_sum %x {axis = 1 : i64} : (tensor<128x384xf32>) -> tensor<128x1xf32>
//   %r384 = tosa.reciprocal %c384 : (tensor<128x1xf32>) -> tensor<128x1xf32>
//   %mean = tosa.mul %sum, %r384 {shift = 0 : i8} : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
//   %x_sub_mean = tosa.sub %x, %mean : (tensor<128x384xf32>, tensor<128x1xf32>) -> tensor<128x384xf32>
//   %square = tosa.mul %x_sub_mean, %x_sub_mean {shift = 0 : i8} : (tensor<128x384xf32>, tensor<128x384xf32>) -> tensor<128x384xf32>
//   %square_sum = tosa.reduce_sum %square {axis = 1 : i64} : (tensor<128x384xf32>) -> tensor<128x1xf32>
//   %variance = tosa.mul %square_sum, %r384 {shift = 0 : i8} : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
//   %epsilon = util.unfoldable_constant dense<9.99999996E-13> : tensor<128x1xf32>
//   %var_eps = tosa.add %variance, %epsilon : (tensor<128x1xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
//   %rsigma = tosa.rsqrt %var_eps : (tensor<128x1xf32>) -> tensor<128x1xf32>
//   %norm = tosa.mul %x_sub_mean, %rsigma {shift = 0 : i8} : (tensor<128x384xf32>, tensor<128x1xf32>) -> tensor<128x384xf32>
//   check.expect_almost_eq_const(%norm, dense<0.0> : tensor<128x384xf32>) : tensor<128x384xf32>
//   return
// }

func.func @layernorm() -> tensor<128x384xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x384xf32>
  %epsilon = arith.constant 9.99999996E-13 : f32
  %gamma = arith.constant dense<1.000000e+00> : tensor<128x384xf32>
  %beta = arith.constant dense<0.000000e+00> : tensor<128x384xf32>
  %cst_4 = arith.constant dense<5.000000e+00> : tensor<128x384xf32>
  %dst = tensor.empty() : tensor<128x384xf32>

  %0 = flow.dispatch.region -> (tensor<128x384xf32>) {
    %5 = iree_codegen.ukernel.generic "layernorm_uk"
      ins(%cst_4, %gamma, %beta, %epsilon: tensor<128x384xf32>, tensor<128x384xf32>, tensor<128x384xf32>, f32)
      outs(%dst : tensor<128x384xf32>)
      (%c128 : index) 
      // We can include some additional fields on the parameters struct as
      // needed. Here we request which processor is executing the call and
      // its data fields as defined by runtime/src/iree/schemas/cpu_data.h.
      fn_def_attrs {hal.import.fields = ["processor_data"]}
      strided_outer_dims(1) -> tensor<128x384xf32>
      flow.return %5 : tensor<128x384xf32>
  } 

  check.expect_almost_eq(%0, %cst_1) : tensor<128x384xf32>
  return %0 : tensor<128x384xf32>
}

