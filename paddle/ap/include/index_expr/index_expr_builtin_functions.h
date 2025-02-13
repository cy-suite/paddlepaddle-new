// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/ap/include/adt/adt.h"
#include "paddle/ap/include/axpr/builtin_functions.h"
#include "paddle/ap/include/index_expr/index_expr_util.h"
#include "paddle/ap/include/index_expr/valid_index_expr_builder.h"
#include "paddle/ap/include/index_expr/value.h"

namespace ap::index_expr {

using adt::Maybe;
using adt::Result;

template <typename Val>
Result<Val> MakePtrGetItem(const Val&, const std::vector<Val>& args);

template <typename Val>
Result<Val> MakeIndexExprBroadcastMask(const Val&,
                                       const std::vector<Val>& args);

template <typename Val>
Result<Val> MakeSlice(const Val&, const std::vector<Val>& args);

template <typename Val>
Result<Val> MakeIndexExprSlice(const Val&, const std::vector<Val>& args);

template <typename Val>
Result<Val> MakeIndexExprAffine(const Val&, const std::vector<Val>& args);

template <typename Val>
Result<Val> MakeDisjointUnion(const Val&, const std::vector<Val>& args);

template <typename Val>
Result<Val> MakeIndexTupleExprPermute(const Val&, const std::vector<Val>& args);

template <typename Val>
Result<Val> MakeIndexTupleExprReshape(const Val&, const std::vector<Val>& args);

template <typename Val>
Result<Val> MakeIndexTupleExprTransform(axpr::InterpreterBase<Val>* interpreter,
                                        const Val& obj,
                                        const std::vector<Val>& args);

template <typename Val>
Result<Val> MakeOpIndexTupleExprSignature(const Val&,
                                          const std::vector<Val>& args);

template <typename Val>
Result<Val> MakeInIndexTupleExprSignature(const Val&,
                                          const std::vector<Val>& args);

template <typename Val>
Result<Val> MakeOutIndexTupleExprSignature(const Val&,
                                           const std::vector<Val>& args);

template <typename T, typename Val>
inline Maybe<T> TryGetImplIndexExprValue(const Val& val) {
  const auto& ret = val.template TryGet<T>();
  if (ret.template HasOkValue()) {
    return ret.GetOkValue();
  }
  return adt::Nothing{};
}

template <typename Val>
inline adt::Result<symbol::DimExpr> TryGetDimExpr(const Val& val) {
  using RetT = adt::Result<symbol::DimExpr>;
  return val.Match([](int64_t c) -> RetT { return symbol::DimExpr{c}; },
                   [](const axpr::BuiltinClassInstance<Val>& instance) -> RetT {
                     return instance.template TryGet<symbol::DimExpr>();
                   },
                   [&](const auto&) -> RetT {
                     return adt::errors::TypeError{
                         "TryGetDimExpr() failed. argument 1 should an int or "
                         "DimExpr (not " +
                         axpr::GetTypeName(val) + ")"};
                   });
}

template <typename Val>
inline Maybe<int64_t> TryGetInt64(const Val& val) {
  return val.Match(
      [](int64_t c) -> Maybe<int64_t> { return c; },
      [](const symbol::DimExpr& dim_expr) -> Maybe<int64_t> {
        return dim_expr.Match(
            [](const int64_t c) -> Maybe<int64_t> { return c; },
            [](const auto&) -> Maybe<int64_t> { return adt::Nothing{}; });
      },
      [&](const auto&) -> Maybe<int64_t> { return adt::Nothing{}; });
}

template <typename Val>
Result<Val> MakePtrGetItem(const Val&, const std::vector<Val>& args) {
  if (args.size() != 3) {
    return adt::errors::TypeError{
        std::string("PtrGetItem takes 3 arguments but ") +
        std::to_string(args.size()) + "were given."};
  }
  const auto& opt_arg1 = TryGetImplIndexExprValue<IndexTupleExpr>(args.at(1));
  ADT_LET_CONST_REF(dim_expr, TryGetDimExpr(args.at(2)));
  return std::visit(::common::Overloaded{
                        [&](const std::string& ptr_var_name,
                            const IndexTupleExpr& indexes_expr) -> Result<Val> {
                          return PtrGetItem{
                              ptr_var_name,
                              std::make_shared<IndexTupleExpr>(indexes_expr),
                              dim_expr};
                        },
                        [&](const auto&, const auto&) -> Result<Val> {
                          return adt::errors::InvalidArgumentError{
                              "wrong argument type for PtrGetItem"};
                        }},
                    args.at(0).variant(),
                    opt_arg1.variant());
}

namespace detail {

template <typename Val, typename T>
Result<Val> ConvertResult(const T& result) {
  return result.Match([](const auto& impl) -> Result<Val> { return impl; });
}

}  // namespace detail

template <typename Val>
Result<Val> MakeIndexExprBroadcastMask(const Val&,
                                       const std::vector<Val>& args) {
  if (args.size() != 2) {
    return adt::errors::TypeError{
        std::string("IndexExprBroadcastMask takes 2 arguments but ") +
        std::to_string(args.size()) + "were given."};
  }
  ADT_LET_CONST_REF(dim_expr, TryGetDimExpr(args.at(0)));
  const auto& opt_arg1 = TryGetImplIndexExprValue<IndexExpr>(args.at(1));
  ValidIndexExprBuilder builder{};
  const auto& pattern_match = ::common::Overloaded{
      [&](const IndexExpr& index_expr) -> Result<Val> {
        return detail::ConvertResult<Val>(
            builder.BroadcastMask(dim_expr, index_expr));
      },
      [&](const auto&) -> Result<Val> {
        return adt::errors::InvalidArgumentError{
            "wrong argument type for IndexExprBroadcastMask"};
      }};
  return std::visit(pattern_match, opt_arg1.variant());
}

template <typename Val>
Result<Val> MakeSlice(const Val&, const std::vector<Val>& args) {
  if (args.size() != 3) {
    return adt::errors::TypeError{std::string("Slice takes 3 arguments but ") +
                                  std::to_string(args.size()) + "were given."};
  }
  ADT_LET_CONST_REF(start, TryGetDimExpr(args.at(0)));
  ADT_LET_CONST_REF(stop, TryGetDimExpr(args.at(1)));
  ADT_LET_CONST_REF(step, TryGetDimExpr(args.at(1)));
  return Val{Slice{start, stop, step}};
}

template <typename Val>
Result<Val> MakeIndexExprSlice(const Val&, const std::vector<Val>& args) {
  if (args.size() != 3) {
    return adt::errors::TypeError{
        std::string("IndexExprSlice takes 3 arguments but ") +
        std::to_string(args.size()) + "were given."};
  }
  const auto& opt_slice = TryGetImplIndexExprValue<Slice>(args.at(0));
  ADT_LET_CONST_REF(range, TryGetDimExpr(args.at(1)));
  const auto& opt_index_expr = TryGetImplIndexExprValue<IndexExpr>(args.at(2));
  const auto& pattern_match = ::common::Overloaded{
      [](const Slice& slice, const IndexExpr& expr) -> Result<Val> {
        ValidIndexExprBuilder builder{};
        return detail::ConvertResult<Val>(builder.Slice(slice, range, expr));
      },
      [](const auto&, const auto&) -> Result<Val> {
        return adt::errors::InvalidArgumentError{
            "wrong argument type for IndexExprSlice"};
      }};
  return std::visit(
      pattern_match, opt_slice.variant(), opt_index_expr.variant());
}

template <typename Val>
Result<Val> MakeIndexExprAffine(const Val&, const std::vector<Val>& args) {
  if (args.size() != 3) {
    return adt::errors::TypeError{
        std::string("IndexExprAffine takes 3 arguments but ") +
        std::to_string(args.size()) + "were given."};
  }
  const auto& opt_slice = TryGetImplIndexExprValue<Slice>(args.at(0));
  ADT_LET_CONST_REF(range, TryGetDimExpr(args.at(1)));
  const auto& opt_index_expr = TryGetImplIndexExprValue<IndexExpr>(args.at(2));
  return std::visit(
      ::common::Overloaded{
          [](const Slice& slice, const IndexExpr& index_expr) -> Result<Val> {
            ValidIndexExprBuilder builder{};
            return detail::ConvertResult<Val>(
                builder.Affine(slice, range, index_expr));
          },
          [](const auto&, const auto&) -> Result<Val> {
            return adt::errors::InvalidArgumentError{
                "wrong argument type for IndexExprAffine"};
          }},
      opt_slice.variant(),
      opt_index_expr.variant());
}

template <typename Val>
Result<Val> MakeDisjointUnion(const Val&, const std::vector<Val>& args) {
  const auto& opt_lhs = TryGetImplIndexExprValue<IndexExpr>(args.at(1));
  const auto& opt_rhs = TryGetImplIndexExprValue<IndexExpr>(args.at(1));
  return std::visit(
      ::common::Overloaded{
          [](const IndexExpr& lhs, const IndexExpr& rhs) -> Result<Val> {
            ValidIndexExprBuilder builder{};
            return detail::ConvertResult<Val>(builder.DisjointUnion(lhs, rhs));
          },
          [](const auto&, const auto&) -> Result<Val> {
            return adt::errors::InvalidArgumentError{
                "wrong argument type for DisjointUnion"};
          }},
      opt_lhs.variant(),
      opt_rhs.variant());
}

template <typename Val>
inline Maybe<adt::List<int64_t>> TryGetInt64List(const Val& val) {
  return val.Match(
      [](const adt::List<Val>& l) -> Maybe<adt::List<int64_t>> {
        adt::List<int64_t> ret;
        ret->reserve(l->size());
        for (const auto& elt : *l) {
          const auto& opt_int = TryGetInt64(elt);
          if (!opt_int.template Has<int64_t>()) {
            return adt::Nothing{};
          }
          ret->push_back(opt_int.template Get<int64_t>());
        }
        return ret;
      },
      [](const auto&) -> Maybe<adt::List<int64_t>> { return adt::Nothing{}; });
}

template <typename Val>
inline adt::Result<adt::List<symbol::DimExpr>> TryGetDimExprList(
    const Val& val) {
  using RetT = adt::Result<adt::List<symbol::DimExpr>>;
  ADT_LET_CONST_REF(l, val.template CastTo<adt::List<Val>>());
  adt::List<symbol::DimExpr> ret;
  ret->reserve(l->size());
  for (const auto& elt : *l) {
    ADT_LET_CONST_REF(int_val, TryGetDimExpr(elt));
    ret->push_back(int_val);
  }
  return ret;
}

template <typename Val>
Result<Val> MakeIndexTupleExprPermute(const Val&,
                                      const std::vector<Val>& args) {
  if (args.size() != 2) {
    return adt::errors::TypeError{
        std::string("IndexTupleExprPermute takes 2 arguments but ") +
        std::to_string(args.size()) + "were given."};
  }
  const auto& opt_perms = TryGetInt64List(args.at(0));
  const auto& opt_expr = TryGetImplIndexExprValue<IndexTupleExpr>(args.at(1));
  ValidIndexExprBuilder builder{};
  return std::visit(
      ::common::Overloaded{
          [&](const adt::List<int64_t>& perms,
              const IndexTupleExpr& expr) -> Result<Val> {
            return detail::ConvertResult<Val>(builder.Permute(perms, expr));
          },
          [](const auto&, const auto&) -> Result<Val> {
            return adt::errors::InvalidArgumentError{
                "wrong argument type for IndexTupleExprPermute"};
          }},
      opt_perms.variant(),
      opt_expr.variant());
}

template <typename Val>
Result<Val> MakeIndexTupleExprReshape(const Val&,
                                      const std::vector<Val>& args) {
  if (args.size() != 2) {
    return adt::errors::TypeError{
        std::string("IndexTupleExprReshape takes 2 arguments but ") +
        std::to_string(args.size()) + "were given."};
  }
  ADT_LET_CONST_REF(shape, TryGetDimExprList(args.at(0)));
  const auto& opt_expr = TryGetImplIndexExprValue<IndexTupleExpr>(args.at(1));
  ValidIndexExprBuilder builder{};
  return std::visit(
      ::common::Overloaded{
          [&](const IndexTupleExpr& expr) -> Result<Val> {
            return detail::ConvertResult<Val>(builder.Reshape(shape, expr));
          },
          [](const auto&) -> Result<Val> {
            return adt::errors::InvalidArgumentError{
                "wrong argument type for IndexTupleExprReshape"};
          }},
      opt_expr.variant());
}

template <typename Val>
Result<Val> MakeIndexTupleExprTransform(axpr::InterpreterBase<Val>* interpreter,
                                        const Val&,
                                        const std::vector<Val>& args) {
  if (args.size() < 1) {
    return adt::errors::TypeError{
        "IndexTupleExprTransform takes at least 1 argument but 0 were given."};
  }
  const auto& opt_expr = TryGetImplIndexExprValue<IndexTupleExpr>(args.at(0));
  if (!opt_expr.template Has<IndexTupleExpr>()) {
    return adt::errors::TypeError{
        "The first argument of IndexTupleExprTransform must be a "
        "IndexTupleExpr."};
  }
  const auto& indexes_expr = opt_expr.template Get<IndexTupleExpr>();
  const auto& opt_rank = IndexTupleExprGetRank(indexes_expr);
  if (!opt_rank.template Has<int64_t>()) {
    return adt::errors::TypeError{
        "The first argument of IndexTupleExprTransform must be a ranked "
        "IndexTupleExpr."};
  }
  const auto& opt_dim_exprs = IndexTupleExprGetRanges(indexes_expr);
  if (!opt_dim_exprs.template Has<adt::List<symbol::DimExpr>>()) {
    return adt::errors::RuntimeError{
        "error occurred where calling IndexTupleExprGetDims"};
  }
  const auto& dim_exprs =
      opt_dim_exprs.template Get<adt::List<symbol::DimExpr>>();
  if (opt_rank.template Get<int64_t>() != args.size() - 1) {
    return adt::errors::TypeError{
        "The rank of first argument must equal to number of lambdas."};
  }
  adt::List<IndexExpr> transform_index_exprs;
  transform_index_exprs->reserve(args.size() - 1);
  for (int i = 1; i < args.size(); ++i) {
    const auto& opt_closure = args.at(i).template TryGet<axpr::Closure<Val>>();
    ADT_RETURN_IF_ERR(opt_closure);
    const auto& closure = opt_closure.GetOkValue();

    if (closure->lambda->args.size() != 1) {
      return adt::errors::TypeError{std::string("Argument ") +
                                    std::to_string(i) +
                                    " is not a single-argumented closure."};
    }
    int idx = i - 1;
    IndexExprDomain domain{dim_exprs->at(idx)};
    const auto& ret_lambda_call =
        interpreter->InterpretCall(closure, {Val{domain}});
    ADT_RETURN_IF_ERR(ret_lambda_call);
    const auto& ret_index_expr =
        TryGetImplIndexExprValue<IndexExpr>(ret_lambda_call.GetOkValue());
    if (!ret_index_expr.template Has<IndexExpr>()) {
      return adt::errors::TypeError{std::string("closure of argument") +
                                    std::to_string(i) +
                                    " does not return a IndexExpr."};
    }
    transform_index_exprs->push_back(ret_index_expr.template Get<IndexExpr>());
  }
  ValidIndexExprBuilder builder{};
  ADT_LET_CONST_REF(ret,
                    detail::ConvertResult<Val>(builder.Transform(
                        transform_index_exprs, indexes_expr)));
  return ret;
}

template <typename Val>
Result<Val> MakeOpIndexTupleExprSignature(const Val&,
                                          const std::vector<Val>& args) {
  if (args.size() != 2) {
    return adt::errors::TypeError{
        std::string("OpIndexTupleExprSignature takes 2 arguments but ") +
        std::to_string(args.size()) + "were given."};
  }
  const auto& in_sig = args.at(0);
  const auto& opt_in = in_sig.template TryGet<InIndexTupleExprSignature>();
  ADT_RETURN_IF_ERR(opt_in);
  const auto& in = opt_in.GetOkValue();
  const auto& out_sig = args.at(1);
  const auto& opt_out = out_sig.template TryGet<OutIndexTupleExprSignature>();
  ADT_RETURN_IF_ERR(opt_out);
  const auto& out = opt_out.GetOkValue();
  return OpIndexTupleExprSignature{in, out};
}

template <typename Val>
Result<Val> MakeInIndexTupleExprSignature(const Val&,
                                          const std::vector<Val>& args) {
  adt::List<IndexTupleExpr> indexes_exprs;
  indexes_exprs->reserve(args.size());
  for (const auto& arg : args) {
    const auto& maybe_indexes_expr =
        TryGetImplIndexExprValue<IndexTupleExpr>(arg);
    if (!maybe_indexes_expr.template Has<IndexTupleExpr>()) {
      return adt::errors::InvalidArgumentError{
          "only arguments of `IndexTupleExpr` type is valid for "
          "InIndexTupleExprSignature"};
    }
    indexes_exprs->push_back(maybe_indexes_expr.template Get<IndexTupleExpr>());
  }
  return InIndexTupleExprSignature{indexes_exprs};
}

template <typename Val>
Result<Val> MakeOutIndexTupleExprSignature(const Val&,
                                           const std::vector<Val>& args) {
  adt::List<IndexTupleExpr> indexes_exprs;
  indexes_exprs->reserve(args.size());
  for (const auto& arg : args) {
    const auto& maybe_indexes_expr =
        TryGetImplIndexExprValue<IndexTupleExpr>(arg);
    if (!maybe_indexes_expr.template Has<IndexTupleExpr>()) {
      return adt::errors::InvalidArgumentError{
          "only arguments of `IndexTupleExpr` type is valid for "
          "OutIndexTupleExprSignature"};
    }
    indexes_exprs->push_back(maybe_indexes_expr.template Get<IndexTupleExpr>());
  }
  return OutIndexTupleExprSignature{indexes_exprs};
}

}  // namespace ap::index_expr
