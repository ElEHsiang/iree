// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iree/compiler/Codegen/Dialect/VendorKernelOps.h>
#include <iree/compiler/Codegen/Utils/Utils.h>
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/UKernelOps.h"
#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "mlir-c/IR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

static std::optional<CastOpInterface>
getCastOpOfElementWiseCast(linalg::GenericOp genericOp) {
  if (!genericOp || genericOp.getNumDpsInputs() != 1 ||
      genericOp.getNumDpsInits() != 1 ||
      genericOp.getBody()->getOperations().size() != 2 ||
      !isElementwise(genericOp)) {
    return std::nullopt;
  }

  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  auto castOp = yieldOp->getOperand(0).getDefiningOp<CastOpInterface>();
  if (!castOp) {
    return std::nullopt;
  }
  Value castIn = castOp->getOperand(0);
  if(castIn.isa<BlockArgument>() &&
     castIn.cast<BlockArgument>().getArgNumber() != 0) {
    return std::nullopt;
  }
  return castOp;
}

namespace {
class LLVMCPUSoftmaxToUKernelPass 
    : public LLVMCPUSoftmaxToUKernelBase<LLVMCPUSoftmaxToUKernelPass> {
public:
  LLVMCPUSoftmaxToUKernelPass() {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect>();
  }

  void runOnOperation() override;
private:
};

} // namespace

/// Holds a function name and attributes.
struct FnNameAndDefAttrs {
  std::string name;
  SmallVector<NamedAttribute> defAttrs;
};

/// Returns the function name and attributes to use for a ukernel with given
/// `ukernelName` on the target described by `targetAttr`.
static FnNameAndDefAttrs
getFnNameAndDefAttrs(const char *ukernelName, RewriterBase &rewriter,
                     IREE::HAL::ExecutableTargetAttr targetAttr) {
  FnNameAndDefAttrs result;
  if (isVMVXBackend(targetAttr)) {
    result.name = std::string("vmvx.") + ukernelName;
    // TODO(#12327): Based on description in the issue, add an attribute
    // `vm.import.module` and set it to `vmvx`. This only works on `vmvx`
    // backend (obviously), but is enough to unblock while the proper fix
    // lands. For now there are a bunch of attributes set on the function, but
    // this should be made more controllable based on the backend.
    result.defAttrs.emplace_back(rewriter.getStringAttr("vm.import.module"),
                                 rewriter.getStringAttr("vmvx"));
  } else {
    result.name = std::string("iree_uk_") + ukernelName;
    result.defAttrs.emplace_back(
        rewriter.getStringAttr("hal.import.fields"),
        rewriter.getArrayAttr({rewriter.getStringAttr("processor_data")}));
    /*
    result.defAttrs.emplace_back(rewriter.getStringAttr("hal.import.bitcode"),
                                 rewriter.getBoolAttr(true));
    result.defAttrs.emplace_back(
        rewriter.getStringAttr("hal.import.cconv"),
        IREE::HAL::CallingConventionAttr::get(
            rewriter.getContext(),
            IREE::HAL::CallingConvention::ParameterStruct));
    */
  }
  return result;
}

// If the defining op of `input` is an element-wise cast, return the input to
// the casting `linalg.generic` op. Otherwise, return `input`.
static Value getInputForUKernel(Value input) {
  auto genericOp = input.getDefiningOp<linalg::GenericOp>();
  std::optional<CastOpInterface> castOp = getCastOpOfElementWiseCast(genericOp);
  if (!castOp) {
    return input;
  }
  return genericOp->getOperand(0);
}

// If the defining op of `input` is an element-wise cast, return the element
// type of the cast source with explicit signedness. Otherwise, return the
// element type of `input`.
static Type getElementTypeForUKernel(Value input) {
  auto genericOp = input.getDefiningOp<linalg::GenericOp>();
  std::optional<CastOpInterface> castOp = getCastOpOfElementWiseCast(genericOp);
  if (!castOp) {
    return llvm::cast<ShapedType>(input.getType()).getElementType();
  }
  Type castOpSrcType = castOp.value()->getOperand(0).getType();
  if (isa<arith::ExtUIOp>(*castOp)) {
    return IntegerType::get(castOp->getContext(),
                            castOpSrcType.getIntOrFloatBitWidth(),
                            IntegerType::SignednessSemantics::Unsigned);
  }
  return castOpSrcType;
}

static FailureOr<IREE::Codegen::UKernelOpInterface>
matchDAGForUKernel(RewriterBase &rewriter, linalg::SoftmaxOp op) {
  // TODO(yunh): Fix it to get executableTargetAttr 
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  const char ukernelName[] = "softmax";
  if (!hasUkernel(targetAttr, ukernelName)) {
    return failure();
  }
  Value lhs = getInputForUKernel(op.getDpsInputOperand(0)->get());
  Value out = op.getDpsInitOperand(0)->get();
  auto outType = llvm::cast<ShapedType>(out.getType());
  int64_t rank = outType.getRank();
  int64_t rows = 1;
  int64_t size = outType.getDimSize(rank - 1);

  for (int i = 0; i < rank - 1; i++) {
    rows *= outType.getDimSize(i);
  }

  Location loc = op.getLoc();

  Value rowsOp = rewriter.create<arith::ConstantIndexOp>(
      loc, rows);
  Value sizeOp = rewriter.create<arith::ConstantIndexOp>(
      loc, size);
  auto fn = getFnNameAndDefAttrs(ukernelName, rewriter, targetAttr);
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::VendorKernelSoftmaxOp>(
      loc, outType, fn.name, ValueRange{lhs}, out,
      ValueRange{rowsOp, sizeOp},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/rewriter.getIndexAttr(1));
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

namespace {

template <typename OpType>
struct LowerToUKernelPattern : OpRewritePattern<OpType> {
  LowerToUKernelPattern(MLIRContext *context)
      : OpRewritePattern<OpType>(context) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {

    FailureOr<IREE::Codegen::UKernelOpInterface> ukernelOp =
        matchDAGForUKernel(rewriter, op);
    if (failed(ukernelOp)) {
      return rewriter.notifyMatchFailure(
          op, "failed to find microkernel op to replace with");
    }
    rewriter.replaceOp(op, ukernelOp.value()->getResults());
    return success();
  }
};

} // namespace

void LLVMCPUSoftmaxToUKernelPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);

  // TODO(yunh): Add isRISCV condition, it need the executableTargetOp
  patterns.insert<LowerToUKernelPattern<linalg::SoftmaxOp>>(
    context
  );

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<>>
createLLVMCPUSoftmaxToUKernelPass() {
  return std::make_unique<LLVMCPUSoftmaxToUKernelPass>();
}
} // namespace iree_compiler
} // namespace mlir
