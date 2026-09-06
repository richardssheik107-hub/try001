# ReduceScatter CCU 实现设计

## 1. 硬件语义与目标

本实现面向 Ascend 950 的 `4×1`、`2×8` 和 `8+4` 通信域，支持 `float32 + sum` ReduceScatter。

Ascend 950 的计算 Die 与 IO Die 分离，CCU 引擎位于 IO Die。因此代码中的 `ENDPOINT_ATTR_DIE_ID`、`HcommCcuKernelRegister` 的 `dieId`，以及双 Thread 的任务归属，全部表示 **IO Die**，不是计算 Die。

实现遵守以下约束：

- 数据搬运、局部规约和双 IO Die 结果合并均由 CCU 完成。
- 每个远端 Rank 只申请一个 Channel。
- 一个 CCU Kernel 只能绑定同一个 IO Die 的 Channel。
- 单次 CCU Read、Copy 或 Reduce 不超过 256 MiB。
- 不使用任何 `Hcomm*WithNotifyOnThread` 接口。
- 相同输入多次执行时采用固定的分组和规约顺序，保证结果确定性。

## 2. 原运行错误与修复原则

原错误：

```text
GetDieIdByChannels failed, the dies of channels are not same
```

原因是一个 Kernel 同时绑定了属于两个 IO Die 的 Channel，并且注册时将 `dieId` 固定为 0。Channel 本身可以申请成功，但 `HcommCcuKernelRegister` 会检查本 Kernel 使用的全部 Channel 是否属于注册参数指定的同一个 IO Die，因此在 Kernel 注册阶段失败。

修复后的资源组织如下：

1. 对每个 Channel 的本地 Endpoint 查询 `ENDPOINT_ATTR_DIE_ID`。
2. 按 IO Die 0、IO Die 1分别保存 Channel 和对应的全局 Rank ID。
3. 每个活跃 IO Die 只注册一个统一 ReduceScatter Kernel，注册参数使用真实 IO Die ID。
4. 每个 Kernel 只访问自己 IO Die 分组内的 Channel。
5. 两个 IO Die 各自产生局部和，最后由 IO Die 0上的无 Channel CCU Kernel 合并。

这样消除了跨 IO Die 混绑 Channel 的注册错误。

## 3. Kernel 与 Thread 布局

资源首次创建时最多注册 3 个 Kernel Handle：

| Kernel | 注册位置 | 用途 |
| --- | --- | --- |
| IO Die 0 ReduceScatter | IO Die 0 | 单 Tile或大数据双 Buffer局部和 |
| IO Die 1 ReduceScatter | IO Die 1 | 单 Tile或大数据双 Buffer局部和 |
| Combine | IO Die 0 | 双 IO Die时合并两份局部和 |

单 IO Die拓扑只注册一个 ReduceScatter Kernel，不注册 Combine。这样相比旧设计的5个 Kernel，双 IO Die最多为3个，单 IO Die仅为1个，降低了 CCU指令空间和寄存器资源压力。

两个 IO Die 都活跃时，Thread 与 IO Die 的对应关系沿用 950 官方双 IO Die 模板：

```text
threads[0] / 当前用户 Stream封装的主 Thread -> IO Die 0 Kernel
threads[1] / 从 Thread                         -> IO Die 1 Kernel
```

如果某个拓扑退化为只有一个 IO Die 有 Channel，则唯一活跃 IO Die 使用当前用户 Stream封装的主 Thread，不会让主 Thread 空闲而只下发从 Thread。

两个 IO Die 都活跃时，主 Thread 和从 Thread 各有一个普通 Thread Notify；只有一个 IO Die 活跃时只保留主 Thread。执行双 IO Die Kernel 前后分别使用：

- `HcommThreadNotifyRecordOnThread`
- `HcommThreadNotifyWaitOnThread`

没有使用 `HcommWriteWithNotifyOnThread`、`HcommReadWithNotifyOnThread` 等任何 `Hcomm*WithNotifyOnThread` 接口。

CCU Kernel 内的 `WriteVariableWithNotify` 是 CCU 地址/Token 交换原语，不属于被禁用的 Thread 数据面接口。

## 4. Channel 分组和 Rank 分组

Host 为每个远端 Rank 选择一条 `COMM_PROTOCOL_UBC_CTP` 链路：

1. 查询 L0，覆盖 Server 内 Mesh。
2. 查询 L1，覆盖 Server 间 Clos。
3. 获取所选链路本地 Endpoint 的 IO Die ID。
4. 将 Channel 放入对应的 `ioDieGroups[ioDieId]`。

Kernel 参数同时保存：

- `channels[]`：当前 IO Die 可访问的 Channel。
- `peerRanks[]`：每条 Channel 对应的全局 Rank。
- `memberRanks[]`：当前 IO Die 负责规约的来源 Rank。

本 Rank 的本地输入只加入一个 IO Die 的 `memberRanks`。优先加入 Channel 数较少的一侧，以平衡两个 IO Die 的来源数量。每组 `memberRanks` 按 Rank ID 排序，局部规约顺序固定。

最终输出固定由 IO Die 0写入（若 IO Die 0不活跃则由唯一活跃 IO Die写入），因此 Combine Kernel始终在同一个主 Thread 的 IO Die 0任务之后执行。

## 5. 小数据路径

当单 Rank 输出片 `recvCount * sizeof(float) <= 1 MiB` 时，统一 Kernel 将 `tileBytes` 设为完整输出片，只执行一个 Tile：

1. 当前 IO Die 上的所有 Channel 交换输入地址和 Token。
2. 同时发起当前 IO Die 所负责远端 Rank 的 Read；本 Rank 数据使用 LocalCopy。
3. 所有来源数据写入 CCL Buffer 的独立槽位。
4. 按当前 IO Die 的 `memberRanks` 固定顺序执行 LocalReduce，生成该 IO Die 的局部和。
5. 两个 IO Die Kernel 完成后，由 Combine Kernel 将第二份局部和加到最终输出。

小数据每个 IO Die 只启动一次统一 Kernel，避免 Host 侧拆 Tile，也不再为小数据额外注册 Direct Kernel。

## 6. 大数据双 Buffer 路径

大数据仍保留通信与规约重叠的双 Buffer：

```text
IO Die d scratch:
bank[0][member 0 ... memberCount-1]
bank[1][member 0 ... memberCount-1]
```

每个 IO Die 独立执行：

1. 将 Tile 0 的全部来源并发预取到 Bank 0。
2. Bank 0 到齐后，立即将下一 Tile 预取到 Bank 1。
3. Bank 1 通信期间，按固定 Rank 顺序规约 Bank 0。
4. 两个 Bank 交替，直到完整 Tile 处理完成。
5. 4B 等尾片复用 Bank 0、Read Event和地址寄存器，在同一个 Kernel 内继续处理，不额外创建资源或启动 Kernel。

双 IO Die 的两个统一 Kernel并发运行。它们完成后，Combine Kernel合并当前窗口的局部和，然后复用窗口空间继续处理下一窗口。这样 512MB 输出片不要求 CCL Buffer 同时保存 512MB 的第二份完整结果。

### 6.1 CCU静态资源复用

`Variable`、`Event`、`LocalAddr` 和 `RemoteAddr` 在 Kernel注册阶段创建后，即使离开 C++局部作用域也不会释放对应硬件资源。旧实现的循环分支和尾片路径反复构造这些对象，且每个 IO Die分别注册 Direct/Pipeline两份类似资源，可能在 `HcommCcuKernelRegister` 或 `HcommCcuKernelRegisterEnd` 返回 `CCU_E_UNAVAIL(7)`。

当前实现参考 HCOMM/HCCL CCU样例，将下列资源在统一 Kernel顶层只创建一次并复用：

- 两组 Bank Slot。
- 两个 Read Event和一个 Reduce Event。
- 一个本地输入地址、一个输出地址。
- 每条 Channel一个远端输入地址。
- `sourceOffset`、`reduceOutputOffset`、`scratchOffset` 等循环变量。

统一 Kernel只加载实际使用的10个64位任务参数，未保留无效的窗口长度寄存器，低于 CCU任务描述符最多13个参数的限制。

尾片直接复用 Bank 0与 Read Event 0，不再创建 `tailSlots`、`tailEvents`。注册失败时会分别打印 RegisterStart、具体 IO Die Kernel、Combine Kernel或 RegisterEnd的返回码，便于继续定位硬件资源瓶颈。

## 7. 为什么不在两个 IO Die Kernel 内直接用命名 Event 同步

`ccu::EventRecord(const char *)` 和 `ccu::EventWait(const char *)` 对应 LocalNotify，只支持同一 IO Die 内不同 Kernel 的同步，不支持跨 IO Die。

因此本实现不再用命名 Event 进行跨 IO Die 握手。跨 IO Die 完成依赖由主/从 Thread Notify 建立：

```text
主 Thread: 前同步 -> IO Die 0 Kernel -> 等待从 Thread -> Combine Kernel
从 Thread: 等待主 Thread -> IO Die 1 Kernel -> 通知主 Thread
```

Combine Kernel 提交在主 Thread 的等待之后，所以启动时两个 IO Die 的局部和都已完成，不存在跨 IO Die LocalNotify 死锁风险。

## 8. CCL Buffer 布局与动态 Tile

设：

- `B`：单 Rank 输出片字节数。
- `N`：Rank 数。
- `T`：实际 Tile 字节数。
- `M0 + M1 = N`：两个 IO Die 的成员总数。

双 IO Die 大数据路径按窗口处理，CCL Buffer 占用为：

```text
2 * T * (M0 + M1) + W
= 2 * T * N + W
```

其中 `W` 是当前窗口的第二份 IO Die 局部和空间，且 `W <= 256 MiB`；它不是完整输出片 `B`。每个窗口完成后立即合并并复用该区域。程序实际采用 `W` 优先、`T` 后收缩的分配方式，因此静态公式始终满足 CCL Buffer 边界。

代码先按拓扑选择 Tile 上限：

- 4 Rank：43 MiB
- 12 Rank：15 MiB
- 16 Rank：11 MiB

然后优先为局部和窗口保留 `min(B, 256 MiB)`，再根据运行时实际 CCL Buffer 大小收缩 Tile：

```text
T <= floor((cclBufferSize - min(B, 256 MiB)) / (2 * N))
```

最后将 `T` 向下对齐到 4字节。窗口容量取剩余 CCL Buffer 空间、未处理输出和 256 MiB 三者最小值。512MB 与 400MB+4B 通常只需约两个窗口，同时每次 Combine 不超过 256 MiB，避免产生过多 Host 下发。

为保持统一 Kernel的固定资源布局，小数据在 CCL Buffer中也预留两组 Bank，但只实际执行 Bank 0上的单 Tile。其占用仍远小于400 MB。

## 9. 同步流程

一次双 IO Die调用包含以下同步：

1. 主 Thread Record，从 Thread Wait，保证两个 IO Die 任务属于同一次调用。
2. 每个 IO Die Kernel 在自己的 Channel 上交换输入地址和 Token。
3. 每个 IO Die Kernel 完成读取和规约后，在自己的 Channel 上执行后同步，保证远端输入 Buffer 生命周期。
4. 主 Thread Wait，从 Thread Record，保证 IO Die 1局部和完成。
5. 主 Thread 启动 Combine Kernel合并当前窗口；若还有剩余数据，复用 scratch 和局部和区域处理下一窗口。

Tile 之间没有 Rank 间 Barrier，只使用 Kernel 内本地 Event 管理 Read、Copy 和 Reduce 依赖。

## 10. 确定性

确定性来自以下固定规则：

- Channel 到全局 Rank 的映射在资源创建后固定。
- 每个 IO Die 的 `memberRanks` 按 Rank ID 排序。
- 每个 IO Die 内按固定顺序规约。
- 两个 IO Die 的局部和始终由同一个 Combine Kernel 按固定顺序相加。

拓扑划分改变了浮点加法的括号结构，但同一通信域和相同输入下不会随链路完成先后改变，因此多次执行结果一致。

## 11. 静态检查结论

- `HcommCcuKernelRegister` 的 IO Die ID与 Kernel 使用的 Channel 归属一致。
- IO Die 0和 IO Die 1分别使用主、从 Thread 下发，符合官方双 IO Die 模板。
- 已移除不合法的跨 IO Die命名 Event同步。
- 未使用任何 `Hcomm*WithNotifyOnThread` 接口。
- 每个活跃 IO Die只注册一个统一局部和 Kernel；双 IO Die额外注册一个 Combine，最多3个 Kernel。
- 循环和尾片路径复用固定的 CCU Variable、Event和地址资源，避免局部对象析构不释放导致的注册资源累积。
- 双 Buffer 地址、完整 Tile、尾片和局部和区域均受运行时 CCL Buffer 边界检查保护。
- 本次只做代码静态检查，没有执行编译或运行测试。
