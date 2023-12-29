// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_DIALECT_VENDORKERNEL_OPS
#define IREE_CODEGEN_DIALECT_VENDORKERNEL_OPS

include "iree/compiler/Codegen/Dialect/IREECodegenDialect.td"
include "iree/compiler/Codegen/Interfaces/UKernelOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/TilingInterface.td"

class IREECodegen_VendorKernelOp<string mnemonic, list<Trait> traits = []> : 
  Op<IREECodegen_Dialect, mnemonic, !listconcat(traits,
    [DeclareOpInterfaceMethods<UKernelOpInterface,
        ["lowerToFunctionCall"]>,
     DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
     DeclareOpInterfaceMethods<TilingInterface,
     ["getIterationDomain",
      "getLoopIteratorTypes",
      "getResultTilePosition",
      "getTiledImplementation"]>])> {}
  
def IREECodegen_VendorKernelSoftmaxOp :
    IREECodegen_VendorKernelOp<"vendorkernel.softmax", [
      AttrSizedOperandSegments]> {
  let summary = "Vendor softmax ukernel";

  let description = [{
    Operation to wrap a softmax forwarded to a microkernel.
  }];

  let arguments = (ins
    StrAttr:$u_kernel_fn_name,
    Variadic<AnyType>:$inputs,
    Variadic<AnyRankedTensorOrMemRefType>:$outputs,
    Variadic<AnyType>:$other_operands,
    OptionalAttr<DictionaryAttr>:$fn_def_attrs,
    OptionalAttr<IndexAttr>:$strided_outer_dims);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    attr-dict $u_kernel_fn_name
    (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    (`outs` `(` $outputs^  `:` type($outputs) `)`)?
    (`(` $other_operands^ `:` type($other_operands) `)`)?
    ( `fn_def_attrs` $fn_def_attrs^ )?
    (`strided_outer_dims` `(` $strided_outer_dims^ `)`)? (`->` type($results)^)?
  }];
  let extraClassDeclaration = [{
    Value getInput() {
      return (*getODSOperands(0).begin());
    }

    Value getOutput() {
      return (*getODSOperands(1).begin());
    }

    ShapedType getInputOperandType() {
      return getInput().getType().cast<ShapedType>();
    }
    ShapedType getOutputOperandType() {
      return getOutput().getType().cast<ShapedType>();
    }
    int64_t getInputOperandRank() {
      return getInputOperandType().getRank();
    }
    int64_t getOutputOperandRank() {
      return getOutputOperandType().getRank();
    }
  }];
}

#endif // IREE_CODEGEN_DIALECT_VENDORKERNEL_OPS
