/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
#ifndef __MULTILABEL_MARGIN_LOSS_H__
#define __MULTILABEL_MARGIN_LOSS_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../multilabel_margin_loss_tiling_data.h"
#include "../multilabel_margin_loss_tiling_key.h"
#include <type_traits>

namespace NsMultilabelMarginLoss {
using namespace AscendC;

constexpr int32_t RED_NONE = 0;
constexpr int32_t RED_MEAN = 1;
constexpr int32_t RED_SUM = 2;

// T = x/self dtype(float/half/bf16);IsTgtT = is_target 输出 dtype。
// is_target 是 target 派生的 0/1 掩码,与 self 无关:GE 图路径为 int32(对齐 A2),
// aclnn 路径为 float(免 int32->float Cast)。int32/float 同 4 字节,buffer 尺寸不变。
template <typename T, typename IsTgtT>
class KernelMultilabelMarginLoss {
private:
    TPipe pipe;

    TQue<TPosition::VECIN, 1> inputQueue;
    TQue<TPosition::VECIN, 1> targetQueue;
    TQue<TPosition::VECIN, 1> partialsInQueue;
    TQue<TPosition::VECOUT, 1> isTargetOutQueue;
    TQue<TPosition::VECOUT, 1> workspaceOutQueue;
    TQue<TPosition::VECOUT, 1> outputQueue;

    TBuf<TPosition::VECCALC> xRowBuf;
    TBuf<TPosition::VECCALC> isPosBuf;
    TBuf<TPosition::VECCALC> reduceBuf;  // accVec: per-row accumulator over target labels (float)
    TBuf<TPosition::VECCALC> workBuf;    // ReduceSum work buffer
    TBuf<TPosition::VECCALC> tmpBuf;     // per-label margin scratch (float)
    TBuf<TPosition::VECCALC> maskBuf;    // non-target select mask (uint8 bitmask)
    TBuf<TPosition::VECCALC> posMaskBuf; // margin>0 select mask (uint8 bitmask); strict >0 excludes nan (torch z>0)
    TBuf<TPosition::VECCALC> outCastBuf;
    TBuf<TPosition::VECCALC> rowLossBuf;   // this core's row losses (float), staged before atomic add
    TBuf<TPosition::VECCALC> gatherBuf;    // core 0: full float workspace read-back (<= N floats)
    TBuf<TPosition::VECCALC> gatherOutBuf; // core 0: cast-to-T output vector (<= N elems)

    GlobalTensor<T> inputGm;
    GlobalTensor<int32_t> targetGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<IsTgtT> isTargetGm;
    GlobalTensor<float> workspaceGm;

    uint32_t N;
    uint32_t C;
    uint32_t basePerCore;
    uint32_t pivot;
    uint32_t usedCoreNum;
    int32_t reduction;
    uint32_t myRows;
    uint32_t myStartRow;
    uint32_t programId;

public:
    __aicore__ inline KernelMultilabelMarginLoss() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR target, GM_ADDR y, GM_ADDR isTarget, GM_ADDR workspace,
                                const MultilabelMarginLossTilingData* tilingData)
    {
        this->N = tilingData->N;
        this->C = tilingData->C;
        this->basePerCore = tilingData->basePerCore;
        this->pivot = tilingData->pivot;
        this->usedCoreNum = tilingData->usedCoreNum;
        this->reduction = tilingData->reduction;
        this->programId = static_cast<uint32_t>(GetBlockIdx());

        this->myRows = basePerCore + (this->programId < pivot ? 1u : 0u);
        this->myStartRow = this->programId * basePerCore + (this->programId < pivot ? this->programId : pivot);

        InitGlobalBuffers(input, target, y, isTarget, workspace);
        InitLocalBuffers();
    }

    // Multi-core-safe output write.
    // Each y element (per-row for reduction=none, the single scalar for mean/sum) is produced by
    // exactly one core. Writing it directly with a sub-32B DataCopyPad races across cores, because
    // several cores land in the same 32B GM block and the block-granular RMW clobbers neighbours.
    // Fix (same pattern as l2_loss): accumulate into a zero-initialised FLOAT workspace via atomic
    // add (atomic RMW is race-free; add-to-zero == value), then core 0 casts to T and writes the
    // whole y tensor contiguously (single writer -> no race, and float staging avoids fp16/bf16
    // atomic which is unsupported).
    __aicore__ inline void Process()
    {
        uint32_t wsElems = (this->reduction == RED_NONE) ? this->N : 1u;

        if (this->programId == 0u && wsElems > 0u) {
            InitGlobalMemory(workspaceGm, wsElems, 0.0f);
        }
        SyncAll();

        if (this->reduction == RED_NONE) {
            if (this->myRows > 0u) {
                LocalTensor<float> lossVec = rowLossBuf.Get<float>();
                for (uint32_t r = 0; r < this->myRows; r++) {
                    lossVec.SetValue(r, ProcessRow(this->myStartRow + r));
                }
                PipeBarrier<HardEvent::S_MTE3>();
                SetAtomicAdd<float>();
                DataCopyExtParams cpWs{1, this->myRows * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                DataCopyPad(workspaceGm[this->myStartRow], lossVec, cpWs);
                SetAtomicNone();
            }
        } else {
            float coreSum = 0.0f;
            for (uint32_t r = 0; r < this->myRows; r++) {
                coreSum += ProcessRow(this->myStartRow + r);
            }
            // Fold mean's 1/N into each core's contribution so sum-of-partials == mean; FinalizeOutput
            // then only casts+writes (sum(coreSum_c / N) == total / N).
            float coreVal = (this->reduction == RED_MEAN && this->N != 0u) ?
                                (coreSum / static_cast<float>(static_cast<int32_t>(this->N))) :
                                coreSum;
            LocalTensor<float> one = rowLossBuf.Get<float>();
            one.SetValue(0, coreVal);
            PipeBarrier<HardEvent::S_MTE3>();
            SetAtomicAdd<float>();
            DataCopyExtParams cpWs{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(workspaceGm[0], one, cpWs);
            SetAtomicNone();
        }

        SyncAll();

        if (this->programId == 0u) {
            FinalizeOutput(wsElems);
        }
    }

private:
    // Per-row byte size: max of 32B-aligned pad length and 16-element-aligned cast length.
    template <typename U>
    __aicore__ inline uint32_t RowBytes()
    {
        uint32_t padBytes = ((C * sizeof(U) + 31u) / 32u) * 32u;
        uint32_t castBytes = ((C + 15u) / 16u) * 16u * sizeof(U);
        return (padBytes > castBytes) ? padBytes : castBytes;
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR input, GM_ADDR target, GM_ADDR y, GM_ADDR isTarget,
                                             GM_ADDR workspace)
    {
        uint64_t outputElems = (reduction == RED_NONE) ? static_cast<uint64_t>(N) : 1ULL;
        uint64_t nc = static_cast<uint64_t>(N) * static_cast<uint64_t>(C);
        inputGm.SetGlobalBuffer((__gm__ T*)input, nc);
        targetGm.SetGlobalBuffer((__gm__ int32_t*)target, nc);
        outputGm.SetGlobalBuffer((__gm__ T*)y, outputElems);
        isTargetGm.SetGlobalBuffer((__gm__ IsTgtT*)isTarget, nc);
        // Float accumulation workspace: N slots for reduction=none (per-row loss), else 1 (scalar).
        uint64_t wsElems = (reduction == RED_NONE) ? static_cast<uint64_t>(N) : 1ULL;
        if (wsElems == 0ULL) {
            wsElems = 1ULL;
        }
        workspaceGm.SetGlobalBuffer((__gm__ float*)workspace, wsElems);
    }

    __aicore__ inline void InitLocalBuffers()
    {
        uint32_t inputRowBytes = RowBytes<T>();
        uint32_t intRowBytes = RowBytes<int32_t>();
        uint32_t fRowBytes = RowBytes<float>();
        // VF 工作 buffer 按向量寄存器宽度(VLF=VECTOR_REG_WIDTH/4)对齐: 内层 full-VL load
        // DataCopy(reg, addr+i*VL) 末块读满 VL 个元素, buffer 须 >= ceil(C/VL)*VL 才不越界。
        // 这是向量化访问的正当内存布局(同 cross_entropy_loss), 非"补 pad 算垃圾"规避——
        // 尾块无效 lane 由 UpdateMask 屏蔽, 不参与计算, 归约只读 [0,C)。VLF 取自权威设备常量。
        constexpr uint32_t VLF = VECTOR_REG_WIDTH / sizeof(float);
        uint32_t vfRowBytes = (((this->C + VLF - 1u) / VLF) * VLF) * sizeof(float);
        if (vfRowBytes < fRowBytes)
            vfRowBytes = fRowBytes;
        if (vfRowBytes < 32u)
            vfRowBytes = 32u;

        uint32_t partialsBytes = ((usedCoreNum * sizeof(float) + 31u) / 32u) * 32u;
        if (partialsBytes < 32u)
            partialsBytes = 32u;
        uint32_t scalarBytes = 32u;

        pipe.InitBuffer(inputQueue, 1, inputRowBytes);
        pipe.InitBuffer(targetQueue, 1, intRowBytes);
        pipe.InitBuffer(partialsInQueue, 1, partialsBytes);
        pipe.InitBuffer(isTargetOutQueue, 1, RowBytes<IsTgtT>());
        pipe.InitBuffer(workspaceOutQueue, 1, scalarBytes);
        pipe.InitBuffer(outputQueue, 1, scalarBytes);

        pipe.InitBuffer(xRowBuf, vfRowBytes);
        pipe.InitBuffer(isPosBuf, vfRowBytes);
        pipe.InitBuffer(reduceBuf, vfRowBytes);
        pipe.InitBuffer(workBuf, fRowBytes);
        pipe.InitBuffer(tmpBuf, fRowBytes);
        pipe.InitBuffer(maskBuf, fRowBytes);
        pipe.InitBuffer(posMaskBuf, fRowBytes);
        pipe.InitBuffer(outCastBuf, scalarBytes);

        // This core stages its (basePerCore+1) row losses before one atomic add to the workspace.
        uint32_t rowLossBytes = (((this->basePerCore + 1u + 7u) / 8u) * 8u) * sizeof(float);
        if (rowLossBytes < 32u)
            rowLossBytes = 32u;
        // Core 0 reads back up to N float slots and casts them to N T elements for the final write.
        uint32_t nFloatBytes = (((this->N + 7u) / 8u) * 8u) * sizeof(float);
        if (nFloatBytes < 32u)
            nFloatBytes = 32u;
        uint32_t nTBytes = (((this->N + 15u) / 16u) * 16u) * sizeof(T);
        if (nTBytes < 32u)
            nTBytes = 32u;
        pipe.InitBuffer(rowLossBuf, rowLossBytes);
        pipe.InitBuffer(gatherBuf, nFloatBytes);
        pipe.InitBuffer(gatherOutBuf, nTBytes);
    }

    __aicore__ inline void CastInputToFloat(LocalTensor<float>& dst, LocalTensor<T>& src, uint32_t cnt)
    {
        if constexpr (std::is_same<T, float>::value) {
            uint32_t cnt8 = ((cnt + 7u) / 8u) * 8u;
            DataCopy(dst, src, cnt8);
        } else {
            Cast(dst, src, RoundMode::CAST_NONE, cnt);
        }
    }

    __aicore__ inline void StoreScalarAsInput(LocalTensor<T>& outLocal, float value)
    {
        if constexpr (std::is_same<T, float>::value) {
            outLocal.SetValue(0, value);
        } else {
            LocalTensor<float> stage = outCastBuf.Get<float>();
            stage.SetValue(0, value);
            PipeBarrier<HardEvent::S_V>();
            if constexpr (std::is_same<T, bfloat16_t>::value) {
                Cast(outLocal, stage, RoundMode::CAST_RINT, 1);
            } else {
                Cast(outLocal, stage, RoundMode::CAST_NONE, 1);
            }
            PipeBarrier<HardEvent::V_S>();
        }
    }

    // Single-event pipe barrier (e.g. HardEvent::V_S) to order scalar/vector accesses.
    template <HardEvent evt>
    __aicore__ inline void PipeBarrier()
    {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<evt>(ev);
        WaitFlag<evt>(ev);
    }

    // Copy one row of input and target from GM into local tensors (already DeQue'd).
    __aicore__ inline void CopyInRow(uint32_t row, uint32_t cnt, LocalTensor<T>& xRowIn, LocalTensor<int32_t>& tgtIn)
    {
        LocalTensor<T> inputLocal = inputQueue.AllocTensor<T>();
        LocalTensor<int32_t> targetLocal = targetQueue.AllocTensor<int32_t>();

        uint64_t rowOff = static_cast<uint64_t>(row) * static_cast<uint64_t>(cnt);

        // Explicit cast: cnt * sizeof(...) is unsigned long; cast to uint32_t avoids the
        // brace-init narrowing (-Wc++11-narrowing) on DataCopyExtParams.blockLen.
        DataCopyExtParams cpInExt{1, static_cast<uint32_t>(cnt * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padInExt{false, 0, 0, 0};
        DataCopyPad(inputLocal, inputGm[rowOff], cpInExt, padInExt);

        DataCopyExtParams cpTgtExt{1, static_cast<uint32_t>(cnt * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padTgtExt{false, 0, 0, 0};
        DataCopyPad(targetLocal, targetGm[rowOff], cpTgtExt, padTgtExt);

        inputQueue.EnQue(inputLocal);
        targetQueue.EnQue(targetLocal);
        xRowIn = inputQueue.DeQue<T>();
        tgtIn = targetQueue.DeQue<int32_t>();
    }

    // Build is_pos from the target row and copy out the is_target row to GM.
    __aicore__ inline void BuildMasks(uint32_t row, uint32_t cnt, LocalTensor<int32_t>& tgtIn,
                                      LocalTensor<float>& isPos)
    {
        Duplicate(isPos, 0.0f, cnt);
        LocalTensor<IsTgtT> isTargetLocal = isTargetOutQueue.AllocTensor<IsTgtT>();
        Duplicate(isTargetLocal, static_cast<IsTgtT>(0), cnt);

        // V->S: scalar GetValue/SetValue below depends on Duplicate to isPos.
        PipeBarrier<HardEvent::V_S>();

        // Walk target with -1 sentinel break (PyTorch MultiLabelMarginLoss semantics).
        // tt is the class index; guard against out-of-range labels before using it as offset.
        for (uint32_t t = 0; t < cnt; t++) {
            int32_t tt = tgtIn.GetValue(t);
            if (tt == -1)
                break;
            if (tt < 0 || static_cast<uint32_t>(tt) >= cnt)
                continue;
            isPos.SetValue(static_cast<uint32_t>(tt), 1.0f);
            isTargetLocal.SetValue(static_cast<uint32_t>(tt), static_cast<IsTgtT>(1));
        }

        // CopyOut is_target row to GM (every reduction mode).
        isTargetOutQueue.EnQue(isTargetLocal);
        LocalTensor<IsTgtT> isTargetDeq = isTargetOutQueue.DeQue<IsTgtT>();
        uint64_t rowOff = static_cast<uint64_t>(row) * static_cast<uint64_t>(cnt);
        DataCopyExtParams cpIsTgt{1, static_cast<uint32_t>(cnt * sizeof(IsTgtT)), 0, 0, 0};
        DataCopyPad(isTargetGm[rowOff], isTargetDeq, cpIsTgt);
        isTargetOutQueue.FreeTensor(isTargetDeq);
    }

    // Row loss = sum_{k in target labels} sum_{i not in target} max(0, 1 - x[k] + x[i]), matching PyTorch
    // MultiLabelMarginLoss. Outer scalar loop over valid target labels (data-dependent length, -1 sentinel
    // break, out-of-range guard); inner work fully VECTORISED over all C classes:
    //   margin[i] = relu((1 - x[k]) + x[i])   via Adds + CompareScalar(>0) + Select (strict >0 = torch z>0, nan-safe)
    //   drop target positions (i in T) with Select on the non-target mask, NOT a multiply by (1 - isPos):
    //     x[i] = +/-inf at a target position would give inf*0 = NaN and corrupt the sum; Select replaces
    //     the value (no arithmetic), so target slots become an exact 0 and non-target inf/nan propagate
    //     as torch's IEEE result.
    // Accumulate each label's masked margins into accVec, then one ReduceSum per row (sum_k sum_i == sum_i sum_k).
    // arch35 regbase(MicroAPI VF)实现: 外层 target 标量循环(数据依赖,-1 哨兵 break + 越界守卫),
    // 内层按 C 用 RegTensor 硬件向量循环(VF, 尾块 UpdateMask)。语义与 A2 完全一致:
    //   margin[i] = relu((1 - x[k]) + x[i]) 严格 >0 select(nan-safe,对齐 torch `if(z>0)`),
    //   非目标位用 Select 屏蔽(非乘法,避免 target 位 inf*0=NaN)。逐 k 累加进 UB 的 accVec,
    //   最后跨块 masked-add 汇到 sumReg 再 ReduceSum(sum_k sum_i == sum_i sum_k)。
    __aicore__ inline float AccumulateRowLoss(uint32_t cnt, LocalTensor<float>& xRow, LocalTensor<int32_t>& tgtIn,
                                              LocalTensor<float>& isPos)
    {
        if (cnt == 0u) {
            return 0.0f;
        }
        using namespace AscendC::MicroAPI;
        LocalTensor<float> accVec = reduceBuf.Get<float>();
        auto accAddr = (__ubuf__ float*)accVec.GetPhyAddr();
        auto xAddr = (__ubuf__ float*)xRow.GetPhyAddr();
        auto posAddr = (__ubuf__ float*)isPos.GetPhyAddr();

        // VF 必须包在 __VEC_SCOPE__ 内(否则后端 "Do not know how to split the result")。
        // 尾块用 UpdateMask 逐迭代处理(只算有效 lane, 不算 padding 垃圾); 工作 buffer 已按向量寄存器
        // 宽度(cAlign, host tiling 计算)对齐, 使 full-VL load 不越界——正式对齐, 非 kernel 侧补 pad 规避。
        constexpr uint16_t VL = VECTOR_REG_WIDTH / sizeof(float);
        uint16_t repeatTimes = static_cast<uint16_t>((cnt + VL - 1u) / VL);

        // accVec[0..cnt) = 0
        {
            uint32_t sreg = cnt;
            __VEC_SCOPE__
            {
                RegTensor<float> z;
                MaskReg preg;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = UpdateMask<float>(sreg);
                    Duplicate(z, 0.0f, preg);
                    DataCopy(accAddr + i * VL, z, preg);
                }
            }
        }
        PipeBarrier<HardEvent::V_S>(); // V->S: 下面读 x[k] 标量依赖 Cast 产出的 xRow

        // 外层 target 标量循环(数据依赖, -1 哨兵 break + 越界守卫); 内层按 C 向量化累加进 accVec。
        for (uint32_t t = 0; t < cnt; t++) {
            int32_t tt = tgtIn.GetValue(t);
            if (tt == -1)
                break;
            if (tt < 0 || static_cast<uint32_t>(tt) >= cnt)
                continue;
            float s = 1.0f - xRow.GetValue(static_cast<uint32_t>(tt));
            uint32_t sreg = cnt;
            __VEC_SCOPE__
            {
                RegTensor<float> xr, posr, accr, tmp, zero;
                MaskReg preg, posM, tgtM;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = UpdateMask<float>(sreg);
                    DataCopy(xr, xAddr + i * VL);
                    DataCopy(posr, posAddr + i * VL);
                    DataCopy(accr, accAddr + i * VL);
                    Duplicate(zero, 0.0f, preg);
                    Adds(tmp, xr, s, preg);                                   // (1 - x[k]) + x[i]
                    CompareScalar<float, CMPMODE::GT>(posM, tmp, 0.0f, preg); // tmp > 0 (严格)
                    Select(tmp, tmp, zero, posM); // relu 严格 >0 (nan/负 -> 0, +inf 保留)
                    CompareScalar<float, CMPMODE::GT>(tgtM, posr, 0.5f, preg); // isPos > 0.5 (目标位)
                    Select(tmp, zero, tmp, tgtM);                              // 目标位 -> 0, 非目标位保留 tmp
                    Add(accr, accr, tmp, preg);
                    DataCopy(accAddr + i * VL, accr, preg);
                }
            }
        }
        PipeBarrier<HardEvent::V_S>();

        // 行损失 = sum_i accVec[i] (只读有效 [0,cnt))。
        float total = 0.0f;
        for (uint32_t i = 0; i < cnt; i++) {
            total += accVec.GetValue(i);
        }
        return total;
    }

    __aicore__ inline float ProcessRow(uint32_t row)
    {
        const uint32_t cnt = this->C;

        LocalTensor<T> xRowIn;
        LocalTensor<int32_t> tgtIn;
        CopyInRow(row, cnt, xRowIn, tgtIn);

        LocalTensor<float> xRow = xRowBuf.Get<float>();
        LocalTensor<float> isPos = isPosBuf.Get<float>();

        CastInputToFloat(xRow, xRowIn, cnt);
        BuildMasks(row, cnt, tgtIn, isPos);

        float rowLoss = AccumulateRowLoss(cnt, xRow, tgtIn, isPos);
        rowLoss = (this->C == 0u) ? 0.0f : (rowLoss / static_cast<float>(static_cast<int32_t>(this->C)));

        inputQueue.FreeTensor(xRowIn);
        targetQueue.FreeTensor(tgtIn);
        return rowLoss;
    }

    // Core 0 only: read the accumulated float workspace, apply mean division, cast to T, and write
    // the whole y tensor in one contiguous copy (single writer -> no multi-core race).
    // wsElems == N for reduction=none (per-row losses), or 1 for mean/sum (the reduced scalar).
    __aicore__ inline void FinalizeOutput(uint32_t wsElems)
    {
        if (wsElems == 0u) {
            return;
        }
        // Invalidate core 0's cached view of the workspace it zero-initialised, so the read below
        // sees the values other cores atomic-added (stride 8 floats = 32B covers any cache line).
        for (uint32_t off = 0; off < wsElems; off += 8u) {
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(workspaceGm[off]);
        }
        LocalTensor<float> acc = gatherBuf.Get<float>();
        DataCopyExtParams cpIn{1, wsElems * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padIn{false, 0, 0, 0};
        DataCopyPad(acc, workspaceGm[0], cpIn, padIn);
        PipeBarrier<HardEvent::MTE2_V>();

        LocalTensor<T> outVec = gatherOutBuf.Get<T>();
        if constexpr (std::is_same<T, float>::value) {
            Adds(outVec, acc, 0.0f, wsElems);
        } else if constexpr (std::is_same<T, bfloat16_t>::value) {
            Cast(outVec, acc, RoundMode::CAST_RINT, wsElems);
        } else {
            Cast(outVec, acc, RoundMode::CAST_NONE, wsElems);
        }
        PipeBarrier<HardEvent::V_MTE3>();

        DataCopyExtParams cpOut{1, wsElems * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm[0], outVec, cpOut);
    }
};

} // namespace NsMultilabelMarginLoss
#endif
