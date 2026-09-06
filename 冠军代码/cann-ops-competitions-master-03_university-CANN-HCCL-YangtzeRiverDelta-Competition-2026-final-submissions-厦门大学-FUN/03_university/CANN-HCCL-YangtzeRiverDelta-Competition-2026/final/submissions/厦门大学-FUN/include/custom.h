/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Integrated Broadcast solution with a V2-native 4x1 Pipeline15 large-message path.
 */
#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <algorithm>
#include <vector>
#include <hccl/hccl_res.h>
#include <hccl/hccl_types.h>
#include "binary_stream.h"
#include "common.h"

// -------------------- common resource --------------------
struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE]{};
    uint32_t channelCount = 0;
};

/**
 * Unified serialized EngineCtx. Each topology/algorithm uses only the fields it
 * needs. Base contexts own the unique channel set; root/algorithm contexts copy
 * those handles and add only their kernel handles.
 */
struct AlgResourceCtx {
    ThreadHandle workerThread{};
    std::vector<CcuKernelHandle> ccuKernels;

    std::vector<ChannelHandle> intraChannels;
    std::vector<uint32_t> intraPeers;
    uint32_t intraLayer = INVALID_VALUE_RANKID;
    std::vector<ChannelHandle> interChannels;
    std::vector<uint32_t> interPeers;
    uint32_t interLayer = INVALID_VALUE_RANKID;

    std::vector<ChannelHandle> channels;
    std::vector<uint32_t> peers;
    uint32_t netLayer = INVALID_VALUE_RANKID;

    std::vector<char> Serialize()
    {
        BinaryStream stream;
        stream << workerThread;
        stream << ccuKernels;
        stream << intraChannels;
        stream << intraPeers;
        stream << intraLayer;
        stream << interChannels;
        stream << interPeers;
        stream << interLayer;
        stream << channels;
        stream << peers;
        stream << netLayer;
        std::vector<char> result;
        stream.Dump(result);
        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream stream(data);
        stream >> workerThread;
        stream >> ccuKernels;
        stream >> intraChannels;
        stream >> intraPeers;
        stream >> intraLayer;
        stream >> interChannels;
        stream >> interPeers;
        stream >> interLayer;
        stream >> channels;
        stream >> peers;
        stream >> netLayer;
    }
};

// -------------------- 2x8 --------------------
constexpr uint32_t BCAST_2X8_RANK_SIZE = 16;
constexpr uint32_t BCAST_2X8_SERVER0_BASE = 0;
constexpr uint32_t BCAST_2X8_SERVER0_RANK_NUM = 8;
constexpr uint32_t BCAST_2X8_SERVER1_BASE = 8;
constexpr uint32_t BCAST_2X8_SERVER1_RANK_NUM = 8;
constexpr uint32_t BCAST_2X8_OWNER_NUM = BCAST_2X8_RANK_SIZE - 1;
constexpr uint32_t BCAST_2X8_PIPELINE_ROUND_NUM = 2;

enum Broadcast2x8ChannelGroup : uint32_t {
    BCAST_2X8_GROUP_INTRA = 0,
    BCAST_2X8_GROUP_INTER = 1
};

enum Broadcast2x8SmallKernelIndex : uint32_t {
    BCAST_2X8_SMALL_SPECIALIZED_INTRA = 0,
    BCAST_2X8_SMALL_SPECIALIZED_INTER = 1,
    BCAST_2X8_SMALL_SPECIALIZED_KERNEL_NUM = 2
};

enum Broadcast2x8SmallKernelRole : uint32_t {
    BCAST_2X8_ROLE_SENDER = 0,
    BCAST_2X8_ROLE_RECEIVER = 1,
    BCAST_2X8_ROLE_IDLE = 2
};

struct Broadcast2x8SmallSpecializedKernelArg : public CcuKernelArgBase {
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    uint32_t rootRank = INVALID_VALUE_RANKID;
    uint32_t serverId = 0;
    uint32_t groupType = BCAST_2X8_GROUP_INTRA;
    uint32_t role = BCAST_2X8_ROLE_IDLE;
    uint32_t netLayer = INVALID_VALUE_RANKID;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    Broadcast2x8SmallSpecializedKernelArg()
    {
        std::fill_n(peerRanks, MAX_RANK_SIZE, INVALID_VALUE_RANKID);
    }
};

enum Broadcast2x8CompactPipelineKernelIndex : uint32_t {
    BCAST_2X8_PIPELINE_INTRA = 0,
    BCAST_2X8_PIPELINE_INTER = 1,
    BCAST_2X8_PIPELINE_KERNEL_NUM = 2
};

struct Broadcast2x8OwnerPipelineKernelArg : public CcuKernelArgBase {
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    uint32_t rootRank = INVALID_VALUE_RANKID;
    uint32_t serverId = 0;
    uint32_t groupType = BCAST_2X8_GROUP_INTRA;
    uint32_t netLayer = INVALID_VALUE_RANKID;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t channelIndexByRank[MAX_RANK_SIZE]{};
    Broadcast2x8OwnerPipelineKernelArg()
    {
        std::fill_n(peerRanks, MAX_RANK_SIZE, INVALID_VALUE_RANKID);
        std::fill_n(channelIndexByRank, MAX_RANK_SIZE, INVALID_VALUE_RANKID);
    }
};

// -------------------- 4x1 --------------------
constexpr uint32_t BCAST_4X1_RANK_SIZE = 4;

// 4x1 small: root-specialized compact Direct Star (unchanged V7 path).
enum Broadcast4x1SmallKernelIndex : uint32_t {
    BCAST_4X1_SMALL_ROOT_SPECIALIZED = 0,
    BCAST_4X1_SMALL_ROOT_SPECIALIZED_KERNEL_NUM = 1
};

enum Broadcast4x1SmallKernelRole : uint32_t {
    BCAST_4X1_ROLE_SENDER = 0,
    BCAST_4X1_ROLE_RECEIVER = 1
};

struct Broadcast4x1SmallSpecializedKernelArg : public CcuKernelArgBase {
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    uint32_t rootRank = INVALID_VALUE_RANKID;
    uint32_t role = BCAST_4X1_ROLE_RECEIVER;
    uint32_t netLayer = INVALID_VALUE_RANKID;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    Broadcast4x1SmallSpecializedKernelArg()
    {
        std::fill_n(peerRanks, MAX_RANK_SIZE, INVALID_VALUE_RANKID);
    }
};

// 4x1 large: validated V2 one-way chain expanded from 8 to 15 chunks.
// The chain is root -> (root+1)%4 -> (root+2)%4 -> (root+3)%4.
constexpr uint32_t BCAST_4X1_PIPELINE_CHUNK_NUM = 15;
constexpr uint32_t BCAST_4X1_PIPELINE_STAGE_NUM =
    BCAST_4X1_PIPELINE_CHUNK_NUM + 3U;

enum Broadcast4x1LargeKernelIndex : uint32_t {
    BCAST_4X1_LARGE_PIPELINE15 = 0,
    BCAST_4X1_LARGE_KERNEL_NUM = 1
};

struct Broadcast4x1LargePipeline15KernelArg : public CcuKernelArgBase {
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t channelIndexByRank[MAX_RANK_SIZE]{};

    Broadcast4x1LargePipeline15KernelArg()
    {
        std::fill_n(peerRanks, MAX_RANK_SIZE, INVALID_VALUE_RANKID);
        std::fill_n(channelIndexByRank, MAX_RANK_SIZE, INVALID_VALUE_RANKID);
    }
};

// -------------------- 8+4 --------------------
constexpr uint32_t BCAST_8P4_RANK_SIZE = 12;
constexpr uint32_t BCAST_8P4_SERVER0_BASE = 0;
constexpr uint32_t BCAST_8P4_SERVER0_RANK_NUM = 8;
constexpr uint32_t BCAST_8P4_SERVER1_BASE = 8;
constexpr uint32_t BCAST_8P4_SERVER1_RANK_NUM = 4;
constexpr uint32_t BCAST_8P4_OWNER_NUM = BCAST_8P4_RANK_SIZE - 1;
constexpr uint32_t BCAST_8P4_PIPELINE_ROUND_NUM = 2;

enum Broadcast8p4ChannelGroup : uint32_t {
    BCAST_8P4_GROUP_INTRA = 0,
    BCAST_8P4_GROUP_INTER = 1
};

enum Broadcast8p4SmallKernelIndex : uint32_t {
    BCAST_8P4_SMALL_SPECIALIZED_INTRA = 0,
    BCAST_8P4_SMALL_SPECIALIZED_INTER = 1,
    BCAST_8P4_SMALL_SPECIALIZED_KERNEL_NUM = 2
};

enum Broadcast8p4SmallKernelRole : uint32_t {
    BCAST_8P4_ROLE_SENDER = 0,
    BCAST_8P4_ROLE_RECEIVER = 1,
    BCAST_8P4_ROLE_IDLE = 2
};

struct Broadcast8p4SmallSpecializedKernelArg : public CcuKernelArgBase {
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    uint32_t rootRank = INVALID_VALUE_RANKID;
    uint32_t serverId = 0;
    uint32_t groupType = BCAST_8P4_GROUP_INTRA;
    uint32_t role = BCAST_8P4_ROLE_IDLE;
    uint32_t netLayer = INVALID_VALUE_RANKID;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    Broadcast8p4SmallSpecializedKernelArg()
    {
        std::fill_n(peerRanks, MAX_RANK_SIZE, INVALID_VALUE_RANKID);
    }
};

enum Broadcast8p4CompactPipelineKernelIndex : uint32_t {
    BCAST_8P4_PIPELINE_INTRA = 0,
    BCAST_8P4_PIPELINE_INTER = 1,
    BCAST_8P4_PIPELINE_KERNEL_NUM = 2
};

struct Broadcast8p4OwnerPipelineKernelArg : public CcuKernelArgBase {
    uint32_t rankId = INVALID_VALUE_RANKID;
    uint32_t rankSize = 0;
    uint32_t rootRank = INVALID_VALUE_RANKID;
    uint32_t serverId = 0;
    uint32_t groupType = BCAST_8P4_GROUP_INTRA;
    uint32_t netLayer = INVALID_VALUE_RANKID;
    uint32_t peerRanks[MAX_RANK_SIZE]{};
    uint32_t channelIndexByRank[MAX_RANK_SIZE]{};
    Broadcast8p4OwnerPipelineKernelArg()
    {
        std::fill_n(peerRanks, MAX_RANK_SIZE, INVALID_VALUE_RANKID);
        std::fill_n(channelIndexByRank, MAX_RANK_SIZE, INVALID_VALUE_RANKID);
    }
};

#endif // OPS_HCCL_CUSTOM_H
