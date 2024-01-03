// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/VendorKernelOps.h"

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpImplementation.h"

// clang-format off
#define GET_OP_CLASSES
#include "iree/compiler/Codegen/Dialect/VendorKernelOps.cpp.inc" // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::IREE::Codegen {

/// Return a `memref.dim` or `tensor.dim` for the shape of `v` at `dim`.
static OpFoldResult getDimValue(OpBuilder &builder, Location loc, Value v,
                                int64_t dim) {
  auto type = cast<ShapedType>(v.getType());
  if (!type.isDynamicDim(dim))
    return builder.getIndexAttr(type.getDimSize(dim));

  return getAsOpFoldResult(
      TypeSwitch<Type, Value>(v.getType())
          .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
            return builder.create<tensor::DimOp>(loc, v, dim);
          })
          .Case<MemRefType>([&](MemRefType t) -> Value {
            return builder.create<memref::DimOp>(loc, v, dim);
          }));
}

/// Returns a memref.subview or a tensor.extract_slice based on the type of the
/// `source`.
static Value getSlice(OpBuilder &b, Location loc, Value source,
                      ArrayRef<OpFoldResult> offsets,
                      ArrayRef<OpFoldResult> sizes,
                      ArrayRef<OpFoldResult> strides) {
  return TypeSwitch<Type, Value>(source.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return b.create<tensor::ExtractSliceOp>(loc, source, offsets, sizes,
                                                strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Value {
        return b.create<memref::SubViewOp>(loc, source, offsets, sizes,
                                           strides);
      })
      .Default([&](Type t) { return nullptr; });
}
//===---------------------------------------------------------------------===//
// Helpers
//===---------------------------------------------------------------------===//

/// Helper method to generate a function declaration at a module scope,
/// and a call to that function
static FailureOr<func::CallOp>
createFunctionCall(RewriterBase &rewriter, Operation *op, StringRef fnName,
                   TypeRange callArgumentTypes, TypeRange callReturnTypes,
                   ValueRange callOperands,
                   ArrayRef<NamedAttribute> fnDefAttrs) {
  FunctionType functionType =
      rewriter.getFunctionType(callArgumentTypes, callReturnTypes);

  // Create a declaration for the function type.
  Location loc = op->getLoc();
  auto moduleOp = SymbolTable::getNearestSymbolTable(op);
  // Check for duplicates.
  auto fnDecl = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(moduleOp, fnName));
  if (!fnDecl) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(&moduleOp->getRegion(0).front());
    fnDecl = rewriter.create<func::FuncOp>(loc, fnName, functionType);
    SymbolTable::setSymbolVisibility(fnDecl, SymbolTable::Visibility::Private);
    for (auto attr : fnDefAttrs) {
      fnDecl->setAttr(attr.getName(), attr.getValue());
    }
    fnDecl->setAttr("llvm.bareptr", rewriter.getBoolAttr(true));
  } else if (fnDecl.getFunctionType() != functionType) {
    return rewriter.notifyMatchFailure(
        op, llvm::formatv("mismatch in function type computed during lowering "
                          "({0}) and already declared function ({1})",
                          functionType, fnDecl.getFunctionType()));
  }

  // Insert the function call.
  return rewriter.create<func::CallOp>(loc, fnDecl, callOperands);
}

//===---------------------------------------------------------------------===//
// VendorKernelOp
//===---------------------------------------------------------------------===//

/// Map type of operand of a `iree_codegen.vendorkernel.softmax` operation to
/// the type(s) of the function call arguments(s) it lowers to.
static LogicalResult getCallOpType(MLIRContext *context,
                                   Type microKernelOpOperandType,
                                   IntegerAttr stridedOuterDimsAttr,
                                   SmallVector<Type> &callOperandTypes) {
  return TypeSwitch<Type, LogicalResult>(microKernelOpOperandType)
      .Case<FloatType, IndexType, IntegerType>([&](auto scalarType) {
        callOperandTypes.push_back(scalarType);
        return success();
      })
      .Case<MemRefType>([&](MemRefType memrefType) {
        // Base ptr.
        callOperandTypes.push_back(MemRefType::get(
            ArrayRef<int64_t>{}, memrefType.getElementType(),
            MemRefLayoutAttrInterface{}, memrefType.getMemorySpace()));
        // Offset
        auto indexType = IndexType::get(context);
        callOperandTypes.push_back(indexType);
        // Strides.
        int stridedOuterDims = stridedOuterDimsAttr
                                   ? stridedOuterDimsAttr.getInt()
                                   : memrefType.getRank();
        callOperandTypes.resize(callOperandTypes.size() + stridedOuterDims,
                                indexType);
        return success();
      })
      .Default([&](Type t) { return failure(); });
}

/// Map `operand` of a `vendorkernel.softmax` operation to the operand(s) of
/// the function call it lowers to.
static LogicalResult lowerToCallOperands(Location loc, RewriterBase &rewriter,
                                         Value operand,
                                         IntegerAttr stridedOuterDimsAttr,
                                         SmallVector<Value> &callOperands) {
  return TypeSwitch<Type, LogicalResult>(operand.getType())
      .Case<FloatType, IndexType, IntegerType>([&](auto scalarType) {
        callOperands.push_back(operand);
        return success();
      })
      .Case<MemRefType>([&](MemRefType memrefType) {
        auto extractStridedMetadataOp =
            rewriter.create<memref::ExtractStridedMetadataOp>(loc, operand);
        // Base ptr.
        callOperands.push_back(extractStridedMetadataOp.getBaseBuffer());
        // Offset.
        callOperands.push_back(extractStridedMetadataOp.getOffset());
        // Strides.
        int stridedOuterDims = stridedOuterDimsAttr
                                   ? stridedOuterDimsAttr.getInt()
                                   : memrefType.getRank();
        auto strides = extractStridedMetadataOp.getStrides();
        for (int i = 0; i < stridedOuterDims; ++i) {
          callOperands.push_back(strides[i]);
        }
        return success();
      })
      .Default([](Type) { return failure(); });
}

//===---------------------------------------------------------------------===//
// VendorKernelSoftmaxOp
//===---------------------------------------------------------------------===//

// These functions should be use in a general interface for all VendorKernel*Op
// TODO: Extract the "lower to function call" to a general function
static FailureOr<func::CallOp> lowerVendorKernelSoftmaxToFunctionCall(
    RewriterBase &rewriter, IREE::Codegen::VendorKernelSoftmaxOp op,
    StringRef fnName, IntegerAttr stridedOuterDimsAttr) {
  // Create the function type based on the operands and results.
  SmallVector<Type> callArgumentTypes;
  for (auto microKernelOpOperandType : op->getOperandTypes()) {
    if (failed(getCallOpType(rewriter.getContext(), microKernelOpOperandType,
                             stridedOuterDimsAttr, callArgumentTypes))) {
      return rewriter.notifyMatchFailure(
          op, llvm::formatv("failed to lower operand type {0}",
                            microKernelOpOperandType));
    }
  }
  SmallVector<Type> callResultTypes;
  for (auto resultType : op->getResultTypes()) {
    if (llvm::isa<ShapedType>(resultType)) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower a `ShapedType` return value to function call");
    }
    if (failed(getCallOpType(rewriter.getContext(), resultType,
                             stridedOuterDimsAttr, callResultTypes))) {
      return rewriter.notifyMatchFailure(
          op, llvm::formatv("failed to lower result type {0}", resultType));
    }
  }

  // Get the operands for the function call.
  SmallVector<Value> callOperands;
  for (auto operand : op->getOperands()) {
    if (failed(lowerToCallOperands(op->getLoc(), rewriter, operand,
                                   stridedOuterDimsAttr, callOperands))) {
      return rewriter.notifyMatchFailure(
          op, "failed to lower operands to function call operands");
    }
  }
  ArrayRef<NamedAttribute> fnDefAttrs = {};
  if (auto specifiedfnDefAttrs = op.getFnDefAttrs()) {
    fnDefAttrs = specifiedfnDefAttrs->getValue();
  }
  return createFunctionCall(rewriter, op, fnName, callArgumentTypes,
                            callResultTypes, callOperands, fnDefAttrs);
}

MutableOperandRange VendorKernelSoftmaxOp::getDpsInitsMutable() {
  return getOutputsMutable();
}

FailureOr<func::CallOp>
VendorKernelSoftmaxOp::lowerToFunctionCall(RewriterBase &rewriter) {
  return lowerVendorKernelSoftmaxToFunctionCall(rewriter, *this, getUKernelFnName(),
                                           getStridedOuterDimsAttr());
}

SmallVector<Range> VendorKernelSoftmaxOp::getIterationDomain(OpBuilder &builder) {
  int64_t operandRank = getInputOperandRank();
  SmallVector<Range> loopBounds(operandRank);
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = getInput();
  for (auto dim : llvm::seq<int64_t>(0, operandRank)) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, source, dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType> VendorKernelSoftmaxOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iteratorTypes(getInputOperandRank(),
                                                 utils::IteratorType::parallel);
  iteratorTypes[getInputOperandRank() - 1] = utils::IteratorType::reduction;
  return iteratorTypes;
}

FailureOr<TilingResult>
VendorKernelSoftmaxOp::getTiledImplementation(OpBuilder &builder,
                                  ArrayRef<OpFoldResult> offsets,
                                  ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getInputOperandRank();
  auto oneAttr = builder.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> strides(rank, oneAttr);
  SmallVector<Value> tiledOperands;

  tiledOperands.emplace_back(
      getSlice(builder, getLoc(), getInput(), offsets, sizes, strides));
  tiledOperands.emplace_back(
      getSlice(builder, getLoc(), getOutput(), offsets, sizes, strides));
  // FIXME(yun): It should check the attribute "(%size: index)" exists.
  auto softmaxOp = dyn_cast<VendorKernelSoftmaxOp>(getOperation());
  auto otherOperands = softmaxOp.getOtherOperands();

  for (auto otherOpernad : otherOperands) {
    tiledOperands.push_back(otherOpernad);
  }

  SmallVector<Type, 4> resultTypes;
  if (hasTensorSemantics())
    resultTypes.push_back(tiledOperands[1].getType());
  Operation *tiledOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{{tiledOp}, SmallVector<Value>(tiledOp->getResults())};
}

LogicalResult VendorKernelSoftmaxOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber == 0) {
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return success();
  }
  return failure();
}

} // namespace mlir::iree_compiler::IREE::Codegen

namespace mlir::iree_compiler {

//===---------------------------------------------------------------------===//
// Register bufferization interface.
//===---------------------------------------------------------------------===//

namespace {

template <typename OpTy>
struct VendorKernelOpsBufferizationInterface
    : public bufferization::DstBufferizableOpInterfaceExternalModel<
          VendorKernelOpsBufferizationInterface<OpTy>, OpTy> {
  LogicalResult
  bufferize(Operation *op, RewriterBase &rewriter,
            const bufferization::BufferizationOptions &options) const {
    // TODO: Handle operations with regions if needed.
    if (op->getNumRegions() != 0) {
      op->emitOpError(
          "unhandled bufferization of micro kernel op with regions");
    }
    SmallVector<Value> bufferOpOperands;

    // Replace all `tensor` operands with corresponding `memref` operands.
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      // For `tensor` type operands, replace with `memref` type operand.
      if (llvm::isa<RankedTensorType>(operand.getType())) {
        FailureOr<Value> memrefOperand = getBuffer(rewriter, operand, options);
        if (failed(memrefOperand)) {
          return op->emitOpError(
              llvm::formatv("failed to bufferize operand {0} ", index));
        }
        bufferOpOperands.push_back(memrefOperand.value());
        continue;
      }

      // For all other operand types, just use the same value.
      bufferOpOperands.push_back(operand);
    }

    // Ignore all result types that are tensor types.
    SmallVector<Type> resultTypes;
    for (auto resultType : op->getResultTypes()) {
      if (llvm::isa<RankedTensorType>(resultType))
        continue;
      resultTypes.push_back(resultType);
    }

    auto bufferOp = rewriter.create<OpTy>(op->getLoc(), resultTypes,
                                          bufferOpOperands, op->getAttrs());
    ValueRange dpsInits =
        cast<DestinationStyleOpInterface>(bufferOp.getOperation())
            .getDpsInits();
    bufferization::replaceOpWithBufferizedValues(rewriter, op, dpsInits);
    return success();
  }
};

template <typename... Ops>
struct RegisterVendorKernelOpsBufferizationInterface {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<VendorKernelOpsBufferizationInterface<Ops>>(
         *context),
     ...);
  }
};

} // namespace

void registerVendorKernelBufferizationInterface(DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *context, IREE::Codegen::IREECodegenDialect *dialect) {
        RegisterVendorKernelOpsBufferizationInterface<
#define GET_OP_LIST
#include "iree/compiler/Codegen/Dialect/VendorKernelOps.cpp.inc"
            >::registerOpInterface(context);
      });
}

} // namespace mlir::iree_compiler
