# ReduceScatter A/B 粗粒度混合流水

## 1. 目标

决赛拓扑中，单卡 Layer-1 Clos 带宽约为单条 Layer-0 Mesh 链路的 `4V`。
纯 A2A 在 2×8 上的主要下界为：

```text
Mesh: 56S / 56V = S/V
Clos: 64S / 32V = 2S/V
```

这里 `S` 是每个 Rank 的输出 slice 大小。混合算法把 slice 只切成两个连续大块：

```text
A = AlignDown(xS, 4 KiB)   // Server 内预规约后再过 Clos
B = S - A                  // 直接 A2A
```

`x` 由 `RoutePlan` 给出：

```text
2×8:        A比例为4/11
8+4, 8侧:  发往4卡侧target的A比例为4/9；为本侧target接收的A比例为2/7
8+4, 4侧:  发往8卡侧target的A比例为2/7；为本侧target接收的A比例为4/9
4×1:       A比例为0，走纯 Direct CMR/A2A
```

比例的分母只用于计算字节数，并不表示网络上真的发送 `S/11` 小包。例如512 MiB、
16 Rank时：

```text
S = 32 MiB
A ≈ 11.63 MiB
B ≈ 20.37 MiB
```

## 2. Mission布局

Layer-0和Layer-1的网络设备位于不同IO Die。每个活跃Die注册两个Mission：

```text
Mesh Die Mission 0: Mesh通信
Mesh Die Mission 1: remote-A、self-A、local-B规约

Clos Die Mission 0: remote-B通信、A partial发布
Clos Die Mission 1: remote-B规约
```

Local Mission不持有Channel，只操作本地HBM，因此Clos Die的Mission 1可以在
Mission 0继续使用Clos网络时规约上一块remote-B。

## 3. 2×8的大任务队列

设当前Rank负责为一个远端target生成A partial。

### 3.1 Mesh队列

```text
M0: 取得本Server所有source对远端target的A贡献
R0: remote-A Reduce，结果写outbound-A

M1: 取得本Server所有source对本Rank的A贡献
R1: self-A Reduce，结果直接写output[0:A]

M2: 取得本Server所有source对本Rank的B贡献
R2: local-B Reduce，结果写output[A:S]
```

M0/M1/M2是9～20 MiB量级的大任务。通信Mission把不同source异步读入独立slot，
统一等待完成；计算Mission在上一任务就绪后立即Reduce：

```text
Mesh Comm:    M0 -------- M1 -------- M2.0 -------- M2.1
Mesh Reduce:       R0 -------- R1 --------- R2.0 -------- R2.1
```

这里的`.0/.1`只是B区间的Tile编号，不是新的算法阶段，也不是把A/B比例继续按分母
切分。CCL Buffer放不下所有source的完整B双Bank时，代码会把B切成少量大块：

```text
M2.0: local-B的第0个大块
M2.1: local-B的第1个大块
```

例如2×8、512 MiB输入时，每Rank输出32 MiB：

```text
A ≈ 11.63 MiB
B ≈ 20.37 MiB

M2.0 ≈ 11.63 MiB
M2.1 ≈  8.74 MiB
```

因此网络粒度仍然是数MiB到十几MiB，而不是`S/11≈2.91 MiB`的11个小包。

### 3.2 Clos队列

```text
C0: 取得远端Server所有source对本Rank的B贡献
CR: remote-B Reduce
C1: 将已经生成的outbound-A发布到目标Rank
W : output[0:A] += inbox-A
```

Clos通信和Clos本地规约同样使用双Bank：

```text
Clos Comm:    C0.0 -------- C0.1 ---------------- C1
Clos Reduce:       CR0 ---------- CR1
Mesh Reduce:             local-B + remote-B -------- W
```

同理，`C0.0/C0.1`是remote-B的第0/1个大块。M2和C0使用相同的B字节区间，
只是分别通过Mesh和Clos取得本地Server与远端Server的贡献，最后按Tile合并。

## 4. 2×8带宽排布

取 `A=4S/11`、`B=7S/11`。

每条Mesh链路承担：

```text
remote-A + self-A + local-B
= A + A + B
= 15S/11
```

Clos侧承担：

```text
remote-B A2A: 8B / 4V = 14S/11V
A partial:       A / 4V =  1S/11V
合计:                      15S/11V
```

理想时间线：

```text
时间       0          4/11       8/11              14/11  15/11

Mesh网络   | remote A | self A    | local B                |
Clos网络   | remote B................................| A发布 |
```

remote-A在`4/11`附近已经生成，而Clos到`14/11`附近才开始发布A，因此A的本地
规约通常可以隐藏在remote-B的大块传输后面。最终只暴露少量尾部Reduce和固定任务开销。

## 5. 8+4非对称拓扑

### 5.1 为什么发送和接收比例不同

`RoutePlan(srcServerSize, dstServerSize)`描述的是一个有方向的过程：

```text
src Server上的source
    → 为dst Server上的输出target生成partial
```

因此，一个Server的“发送A比例”和“接收A比例”来自两次方向相反的计算。这里说的
只是预规约块A的长度，不是该Server跨机总发送/接收字节数；剩余的`B=S-A`仍由各个
source直接发送：

```text
8卡侧发送A: RoutePlan(8, 4) = 4/9
8卡侧接收A: RoutePlan(4, 8) = 2/7

4卡侧发送A: RoutePlan(4, 8) = 2/7
4卡侧接收A: RoutePlan(8, 4) = 4/9
```

以8卡侧Rank 0为例：

```text
Rank 0自己的输出target位于8卡Server：
    4卡Server替它预规约前2/7，因此Rank 0接收A=2/7。

如果Rank 0还负责一个位于4卡Server的target：
    Rank 0收集8卡Server对该target的前4/9贡献，
    规约后发送一个A=4/9的partial。
```

这两个A属于两个不同的输出target，所以长度不需要相等。对于任意一个确定target，
它的本地Server partial和远端Server partial长度仍然严格相同：

```text
target在8卡侧: 两个Server都计算该target的前2/7
target在4卡侧: 两个Server都计算该target的前4/9
```

### 5.2 8卡Server到4卡Server

令8卡Server为source、4卡Server为target，取`x=4/9`。

8卡Server的Mesh流量包括：

```text
本Server 8个本地target的正常ReduceScatter流量:
    8 × 7S = 56S

替4个远端target预规约A产生的额外Mesh流量:
    4 × 7 × (4S/9) = 112S/9

合计:
    56S + 112S/9 = 616S/9
```

8卡Full-Mesh的总带宽是`8×7V=56V`，理想平均时间为：

```text
tMesh(8→4) = (616S/9) / 56V
             = 11S/9V
```

跨Server时，每个4卡侧target接收：

```text
一个预规约A partial:       4S/9
8个source直接发送B: 8 × 5S/9

每target合计: 44S/9
4个target合计: 176S/9
```

跨切面有效聚合带宽受4卡侧限制，为`4×4V=16V`：

```text
tClos(8→4) = (176S/9) / 16V
             = 11S/9V
```

所以`4/9`正好把这个方向的Mesh与Clos平均负载平衡到`11S/9V`：

```text
8卡侧Mesh: | 本地target流量 S | 远端A额外流量 2S/9 |
8→4 Clos:  | direct-B 10S/9   | A partial S/9     |
            0                                      11S/9V
```

8张卡只负责4个远端target，实际每Rank负责的target数`q`为0或1。负责target的Rank
对应Mesh链路负载会高于平均值，但该方向的理论平均下界仍是`11S/9V`；整个8+4拓扑
最终还会被下面更慢的`4→8`方向限制。

### 5.3 4卡Server到8卡Server

反方向令4卡Server为source、8卡Server为target，取`x=2/7`。

4卡Server的Mesh流量：

```text
4个本地target的正常流量:
    4 × 3S = 12S

替8个远端target预规约A:
    8 × 3 × (2S/7) = 48S/7

合计:
    12S + 48S/7 = 132S/7
```

4卡Full-Mesh总带宽为`4×3V=12V`：

```text
tMesh(4→8) = (132S/7) / 12V
             = 11S/7V
```

每个8卡侧target接收：

```text
一个预规约A partial:       2S/7
4个source直接发送B: 4 × 5S/7

每target合计: 22S/7
8个target合计: 176S/7
```

Clos聚合带宽仍受4卡Server的4个端口限制，为`16V`：

```text
tClos(4→8) = (176S/7) / 16V
             = 11S/7V
```

4张卡平均每张负责两个8卡侧target，因此每条Mesh链路的负载可以均匀排成：

```text
4卡侧Mesh: | 本地target流量 S | 两个远端A 4S/7 |
4→8 Clos:  | direct-B 10S/7   | 两个A S/7     |
            0                                      11S/7V
```

### 5.4 8+4的整体下界

在当前双向链路模型下，两个方向分别核算，整体由较慢方向决定：

```text
8→4: 11S/9V ≈ 1.222S/V
4→8: 11S/7V ≈ 1.571S/V

t8+4 >= max(11/9, 11/7)S/V
      = 11S/7V
```

所以8+4无法像2×8一样做到完全对称。`4/9`和`2/7`不是互相矛盾，而是分别把
两个方向各自的Mesh和Clos负载平衡；最终瓶颈是卡数更少、每张卡要替两个远端target
做预规约的4卡Server。

## 6. Scratch布局

两个网络层各使用双Bank，每个Bank为每个source保留一个slot：

```text
Mesh bank[2][local source]
Clos bank[2][remote source]
outbound-A[target count]
inbox-A[1]
```

Mesh Bank先承载M0/M1，随后复用给local-B；Clos Bank承载remote-B。不存在旧实现中
“完整MRCW scratch + 完整Direct scratch”同时常驻的问题。

对称2×8、512 MiB用例约为：

```text
Mesh双Bank: 2 × 8 × 11.63 MiB ≈ 186 MiB
Clos双Bank: 2 × 8 × 11.63 MiB ≈ 186 MiB
outbound-A + inbox-A:              ≈ 23 MiB
总计:                              ≈ 395 MiB
```

8+4的8卡侧容量最紧。实现先为Mesh A预留Bank，再利用剩余CCL Buffer计算B Tile：

```text
BTile = AlignDown(min(meshStride,
        remaining / (2 × remoteSourceCount)), 4 KiB)
```

512 MiB时该侧B Tile约8 MiB，仍保持大粒度传输。

## 7. 同步语义

所有额外Thread满足：

```text
第一个任务: Wait主Thread启动通知
最后一个任务: Record通知主Thread完成
```

Bank复用采用单生产者、单消费者配对：

```text
Comm写Bank
→ DATA_READY Record
→ Reduce Wait
→ Reduce
→ BANK_REUSE Record
→ Comm Wait后复用Bank
```

同一Bank的新一轮`DATA_READY Record`必须经过上一轮`BANK_REUSE Wait`，因此不会出现
两个Record提前写同一个Notify、只唤醒一个Wait的问题。

A partial的跨Rank依赖为：

```text
PUBLISH Write完成
→ Channel NotifyRecord
→ 目标Rank NotifyWait
→ output-A += inbox-A
```

最后两个通信Mission执行一次地址同步作为跨Rank生命周期屏障，防止下一次算子调用
提前复用input、output或scratch。

## 8. 确定性

通信完成顺序不参与浮点加法顺序。每个Server partial始终按固定source rank顺序规约，
最后再按固定顺序执行：

```text
output = local-server partial + remote-server partial
```

A和B路径均遵循该顺序，因此相同输入下结果确定。

## 9. 当前边界

- 小于1 MiB的输出slice使用Direct路径，避免混合调度的固定Launch/Notify开销。
- 4×1没有Server内Mesh预规约，使用Direct路径。
- 当前使用HBM source slots和LocalReduce；尚未改为CcuBuffer + LoopGroup的片上4 KiB微流水。
- 下一步优化应优先实测粗粒度混合流水，再考虑把每个大任务内部改成多Loop并发的
  CcuBuffer流式8路规约。
