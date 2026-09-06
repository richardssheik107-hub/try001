# CCU Buffer适配台账

仅记录上下文恢复所需的结论与实验差分；算法历史见`工作台账.md`。

## 基线与硬约束

- 基线：`f4c64bd clean up reduce scatter execution structure`，S9大数据路径。
- HCCL-VM物理能力为每Die 1536个4 KiB MS，但比赛默认`CCU_SCHED`只给每Die通信域资源池128个连续MS。
- 同一Die所有已注册Kernel共享这128个MS；两个Die资源独立，不能跨Die使用。
- 8路Buffer Reduce每条lane占8个MS，因此当前模式上限是16条lane，共128 MS。
- 网络传输必须继续保持MiB级大块；CCU Buffer只替换数据已经落入本地HBM后的规约。
- `400 MiB + 4 B`必须单独处理4字节尾块，禁止向上对齐越过slot边界。

## 转换优先级

1. `REDUCE_MESH_SLOTS_TO_DST`
   - 8卡Server：`output/self + 7 slots`，8路。
   - 4卡Server：`output/self + 3 slots`，4路。
   - 不改变通信、scratch布局、流水或同步，是首个实验。
2. `FOLD_SLOTS_IN_PLACE`
   - Clos B传输仍先大块落入3/4/8个连续HBM slot，再以4 KiB微块做多路规约。
3. 4x1融合
   - 将`3 slots -> slot0`和`output += slot0`合并成`output + 3 slots -> output`四路规约。
   - 可删除一次中间partial写回和一次Local Kernel launch，预计价值最高，但会改变Mode与调度，后做。

暂不转换：

- `READ_REDUCE_CLOS_PARTIALS`：4 KiB远端Read会破坏网络粒度。
- 普通二路`REDUCE_RANGE`：单独经Buffer搬入搬出未必优于HBM LocalReduce。
- 网络Channel循环和Notify循环：LoopGroup可能损害并行通信，收益很小。

## 资源生命周期

- Buffer/Event在Kernel注册/翻译时申请一次，后续所有Launch复用，不会按调用次数累加。
- LoopGroup并行方案仍是最终性能方向，但当前先用单`CCU_WHILE`诊断Buffer Reduce本身。

## 实验B0：Mesh多路Buffer Reduce（首次提交RE，已修复参数ABI）

- 改动：仅将`REDUCE_MESH_SLOTS_TO_DST`的3/7次串行HBM `LocalReduce`替换为4/8路CCU Buffer Reduce。
- 初版并行度8 lane；4路占32 Buffer、8路占64 Buffer，仅在`remoteLocalMemberCount != 0`的Local Kernel注册。
- Host额外计算并传入LoopGroup的`addrOffset/loopParam/parallelParam/residual`，含Buffer分支的Mesh Local Handle固定为10参数，仍低于13参数限制。
- 不变：所有通信Kernel、网络粒度、Bank、Record/Wait、Clos Fold、4x1及ReadReduce。
- 首次结果：10/11/12/16/17/18 RE，`argsNum=6, loaded=10`。原因是同一Kernel Handle使用过的所有`LoadArg` 构成固定ABI，不能按运行时Mode混用6/10参数。
- 修复：大数据Mesh Local的所有Mode统一10参，非Buffer Mode后4参置0；SmallA2A注册时不翻译Buffer分支，Clos和4x1继续6参。
- 第二次结果：小数据通过，两种复杂拓扑大数据Checker Error 105；最后一个LoopGroup位于Mesh Local图尾，Checker无法确定exit boundary。
- 修复：在Buffer Mode的所有LoopGroup之后增加成对的本地Event Record/Wait，作为同队列尾边界；不改数据且不留未消费Record。
- 第三次结果：四个复杂拓扑大数据只返回`no Checker Success and no error`。
- 第四次：lane从8增至16，结果与第三次完全相同。因此排除“Loop迭代次数太多”；16 lane还会使16个lane Event加通用Event超过每Die 16 CKE，不再使用。
- 当前第五次诊断版：完全删除LoopGroup，改为单`CCU_WHILE`串行处理4 KiB块；只申请4/8个Buffer并复用1个原有Event；Mesh Local ABI从10参收敛为8参。
- 第五次结果：仍为相同RE，说明只删LoopGroup不足以解决。
- 参考`Final-pro`已Pass的小数据CCU Buffer实现：它为每个来源分配一段连续MS，用各段首Buffer组成`LocalReduce`输入列表，并以整段容量作为单次tile。
- 当前第六次版本复制该形状：固定申请128个连续MS；8路每源16块/每tile 64 KiB，4路每源32块/每tile 128 KiB；复用1个Event，无LoopGroup，Kernel内`CCU_WHILE`处理多个tile，ABI保持8参。
- 第六次首次提交在首个Mesh `COPY_RANGE`就RE：`argsNum=6, loaded=8`，尚未执行Buffer规约。原因是ExecOp用未序列化的`localServerRanks.size()`判断ABI，反序列化后它为0。已改为注册时同源且已序列化的`localMemberCounts[meshDie]`。
- ABI修复后四个复杂拓扑大数据仍静默RE：`no Checker Success and no error`。与`Final-pro`再对比发现，它的`MS_AUTO_TILE`每次Launch只做一次Copy→Reduce→Copy，没有运行时CCU循环；我们一次Launch有数百轮`CCU_WHILE`。
- 当前第七次诊断版：完全删除Buffer Mode内循环，Host按64/128 KiB分成多次Local Kernel Launch；每次Kernel形状与`Final-pro`MS分支一致。性能不是本轮目标。
- 第七次结果：仍静默RE。这只能排除“唯一原因是Kernel内长循环”；Host拆分后总Copy/Reduce数不变且CCU Graph数更多，仍可能是Checker总图规模超限。
- 当前第八次诊断版：每个原Mesh Reduce只让首64 KiB（8路）或128 KiB（4路）走一次Buffer Reduce，剩余全部恢复S9的HBM顺序LocalReduce；无Loop、无Host多Launch。
- 判据：若Pass，说明Buffer与Local Mission正常，全量转换是规模问题；若仍静默RE，下一步把Buffer Mode移到Mesh Comm Mission，对齐`Final-pro`的Mission位置与注册形状。
- 注意：当前仓库文档声称Buffer API的`len <= 4096`，但`Final-pro`已评测Pass的实现明确使用连续MS大tile。本实验优先复制评测验证过的行为。
- 当前`bash Final/build.sh`与`git diff --check`通过，待提交功能验证。

## 实验B1：严格按官方Demo改为16-lane LoopGroup

- 前提：B0“每段只做一次Buffer规约，其余退回HBM”已Pass，证明Mesh Local Mission、128个连续MS和4/8路Buffer Reduce本身可用；全量化的障碍在循环/图形状。
- 本轮仅替换`REDUCE_MESH_SLOTS_TO_DST`，不改Clos Fold、4x1、网络传输、Bank布局和Record/Wait。
- 资源形状与`ccu_reduce_scatter_mesh1d_demo.h`一致：16 lane、每lane 8个MS，共128 MS；16个Event；每个Loop处理4 KiB。
- 主体波次使用单Loop的LoopGroup，`bufferOffset=8`、`eventOffset=1`；尾部使用两个Loop表示p与n部分。Host传入`addrOffset/loopParam/parallelParam/residual`四元组。
- ABI：包含Mesh Buffer Mode的Local Handle固定10参；Small、Clos、4x1仍为6参。LoopGroup后保留本地Event Record/Wait作为Checker V3退出边界。
- `400 MiB + 4 B`由goSize拆为完整4 KiB块与4字节尾块，不向上越界对齐。
- 当前`bash Final/build.sh`与`git diff --check`通过，未commit，待评测功能结果。
- 首次B1结果：四个复杂拓扑大数据点仍静默RE，`no op[0] Checker Success and no error`。
- 对照Checker V3的`BuildLoopParallelGroupV3`发现：LoopGroup必须同时有同queue的入口和出口边界。官方Demo前面的Read/EventWait自然提供入口；我们的Local Buffer Mode直接以LoopGroup开始，之前只补了出口。
- 当前B1.1单变量修复：在首个LoopGroup前增加一对已消费的本地Event Record/Wait作入口边界，尾部边界保留；不改Buffer、Loop、ABI、规约或调度。
- B1.1结果：512 MiB仍静默RE；400 MiB+4B已能展开LoopGroup并给出真实错误。2x8出现HBM地址越界；8+4明确报告尾部Loop对同一output区间并发读/写冲突。入口边界修复有效，但暴露了尾部结构问题。
- 根因：官方Demo的`p+n`两Loop可并行，因为输入和输出地址分离；我们是`output + slots -> output`原地规约，`p`写回与`n`读output不能并行。Checker合并两Loop时还会使第二Loop的地址基准无法按我们预期保持。
- 当前B1.2：尾部拆为两个顺序的单LoopGroup：先处理`p`字节残片，Event边隔后再展开`n`个4 KiB块。主64 KiB波次不变；Loop 0/1的MS最高均不超过127。

## 实验B1.3：Mesh Buffer的self输入与输出分离

- B1.2的四个复杂拓扑大数据点仍静默RE：`no op[0] Checker Success and no error`。
- 官方ReduceScatter/AllReduce/Reduce Buffer实现均使用独立self source和destination；当前`output + remote slots -> output`的原地alias与该形状不一致。
- B1.3改为`original self input + remote slots -> destination`，但保留原有self-copy，用于单独验证地址alias是否为checker问题根因。
- Mesh Local Handle固定为12参：原10参加`selfSource/selfSourceToken`；同一Handle的其他Mode传空占位。Small、Clos和4×1仍为6参。
- 不改通信、Bank、Notify、LoopGroup、尾部拆分或调度流水。

## 实验B2：专用MixedRoute Mesh Buffer Local Kernel

- B1.3四个复杂拓扑大数据点仍静默RE，说明单独分离self input/output不足以解决图展开问题。
- 新的`Final-pro` 4×1大数据实现使用专用`CcuReduceScatterFadeLoopKernel`，因此B2先将受跟踪代码恢复到`f4c64bd`，删除B0/B1叠加补丁，再单独实验Kernel形状。
- MixedRoute的Mesh Local Mission改为注册专用`CcuReduceScatterMeshBufferKernel`；Clos Local、4×1、Small和所有Comm Mission保持S9。Mission数量不变。
- 专用Kernel只保留现有流水所需的`COPY_RANGE`、`REDUCE_RANGE`和Mesh Buffer Reduce，固定11参ABI。
- 资源形状复制已跑通的4×1实现：16 lane、每lane 8 MS、总128 MS、16个block Event。8输入每路1 MS/每lane 4 KiB；4输入每路2个连续MS/每lane 8 KiB。
- 主体m使用单LoopGroup；尾n+p恢复官方同一LoopGroup的双Loop形状；仅在末尾复用Event 0增加一对Record/Wait作Checker出口边界。
- Kernel使用独立`slotStride`参数，使Host必要时可将超出单次Loop计数上限的规约分段，同时保持原完整slot间距寻址。
- 保留原self-copy、Mesh/Clos Bank、Notify、网络传输、A/B比例和任务顺序，未运行`hccl_vm`。
- B2首次评测：四个复杂拓扑大数据点仍静默RE，`no op[0] Checker Success and no error`。
- 资源复核：Event/CKE是按IO Die建池并由该Die上Mission共同分配，但普通`Event`与`Array<Event>`分别使用普通CKE和block CKE。Mesh Comm仅1个普通Event，Mesh Local使用16个block Event，分别不超过32/16额度；Launch不会重复申请。

## 实验B2.1：Mesh Buffer四输入/八输入完全静态化

- 将原动态`CcuReduceScatterMeshBufferKernel`拆成两个注册入口：`CcuReduceScatterMeshBuffer4Kernel`和`CcuReduceScatterMeshBuffer8Kernel`。
- Host在注册期按`localMemberCounts[meshDie]`选择固定形状；Kernel不再从`remoteLocalMemberCount`计算4/8路。
- `inputCount`、`msPerInput`、`sliceBytes`、Event mask及所有源地址数组均为模板实例的编译期常量；`loopSrc/sources`改为固定`std::array`。
- 两个Kernel仍使用相同11参ABI；`COPY_RANGE/REDUCE_RANGE`保留在同一Local Mission，因为每Die仅能注册两个Mission。
- 不改通信Mission、Event/MS数量、LoopGroup、地址、Bank、Notify或Host调度。
- `git diff --check`与`bash build.sh`通过，未运行`hccl_vm`，待提交功能验证。
- B2.1结果：四个复杂拓扑大数据点仍静默RE，排除“唯一原因是4/8路形状在KernelArg中动态决定”。

## 实验B2.2：仅8+4四卡侧逐字对齐4x1 Loop Kernel

- 只在MixedRoute且`localMemberCounts[meshDie] == 4`时注册Buffer Kernel；因此仅覆盖8+4的四卡Server。2x8和8+4八卡侧恢复S9串行HBM Reduce。
- Buffer Kernel收缩为固定4路、10参ABI，删除`slotStride`和`REDUCE_RANGE` Mode；三个remote slot直接使用`bytes`作间距，与已Pass的4x1 Kernel一致。
- 每次Mesh Buffer Launch不再做Loop计数上限分段；8+4当前tile均低于单次Loop编码上限。
- B路最后`output += Clos partial`从Mesh Local Mission转移到Clos Local Mission；增加Mesh→Clos完成边，Bank reuse和最终完成边也改由Clos Local发出。
- 诊断目标：若8+4通过，证明4路Loop在MixedRoute可用，问题集中于8路形状或Mesh Local的复合Mode/依赖图；若仍失败，差异就主要剩下MixedRoute的Bank/跨Die调度图。
- `git diff --check`与`bash build.sh`通过，未运行`hccl_vm`，待提交功能验证。
- B2.2功能评测Pass。这证明MixedRoute本身、Mesh Local Mission位置、128 MS/16 Event、多次Loop Launch和Bank调度都不必然导致Checker失败。
- 已验证可用的精确形状是：固定4路、10参、slot间距=`bytes`、Mesh Local只有`COPY_RANGE + BUFFER_REDUCE`，最终二路合并由Clos Local完成。
- 该结果不能证明8路Buffer Reduce不可用：本轮8+4的8卡侧与2x8全部使用S9 HBM Reduce，只有4卡侧使用Buffer。
- 官方HCCL `REDUCE_SCATTER_GROUP_REDUCE_MAX_PIECE_CNT = 8`，且官方8路Loop每lane正是8个MS、每输入1个4 KiB MS，因此8路在原语支持范围内。

## 实验B2.3：仅新增固定8路Kernel

- 保留B2.2已Pass的全部结构：10参ABI、Mesh Local仅`COPY_RANGE + BUFFER_REDUCE`、slot间距=`bytes`、最终二路合并在Clos Local。
- 唯一实验变量：新增`CcuReduceScatterMeshBuffer8Kernel`。固定8输入，每lane使用8个MS、每输入1个4 KiB MS；仍为16 lane、总128 MS。
- 注册期按`localMemberCounts[meshDie]`选择4路或8路Kernel。现在2x8的两侧以及8+4的8卡侧均使用固定8路Buffer Loop。
- `slotStride`是相邻远端slot起始地址的字节间距；早期为Host子段拆分时保持原完tile间距而引入。当前每tile可一次Loop处理，因此slot间距恒等于`bytes`，不再需要第11参。
- `git diff --check`与`bash build.sh`通过，未运行`hccl_vm`，待提交功能验证。
- B2.3结果：四个复杂拓扑大数据点恢复静默RE，`no op[0] Checker Success and no error`。8+4中只要8卡侧启用8路Kernel就会使整个集合通信失败，因此失败已锁定到8路Loop的具体资源/展开形状。
- 当前8路每输入1 MS=4 KiB，每4 KiB输出展开为8次输入Copy+1次Reduce+1次输出Copy；已Pass的4路每输入2 MS=8 KiB，每8 KiB仅4+1+1个数据任务。按每字节任务数估算，8路Checker展开图约为4路的`(10/4 KiB)/(6/8 KiB) = 3.33x`。
- 下一候选应保留直接8路Reduce，但将并行lane数减少、每输入连续MS增加：例如8 lane×2 MS/输入，或4 lane×4 MS/输入，两者总MS仍为128，但单Loop处理8/16 KiB，显著减少Checker展开节点。

## 实验B2.4：8路改为8 lane×2 MS/输入

- 4路保持B2.2已Pass形状：16 lane、2 MS/输入、8 MS/lane、128 MS总91cf。
- 8路从B2.3的16 lane×1 MS/输入改为8 lane×2 MS/输入：每lane 16 MS，总91cf仍128 MS，单lane处理8 KiB。
- LoopGroup的`laneCount/msInterleave/msPerInput/sliceBytes`都按固定输入数在注册实例中变为编译期常量；8路克隆lane数减半，单字节Checker展开任务约减半。
- ABI、Mode、slot间距、Clos最终合并和所有Host调度不变。
- `git diff --check`与`bash build.sh`通过，未运行`hccl_vm`，待提交功能验证。
- B2.4结果：8+4的400 MiB+4B通过；8+4的512 MiB以及2x8两个大点仍静默RE。因此固定8路原语和`8 lane×2 MS`资源形状本身合法，失败与整个集合通信的展开任务量相关，不是Launch时重复申请128 MS。
- 按输入规模和参与8路规约的Rank数排序，这些点的Checker图规模预期为`8+4 400M < 2x8 400M < 8+4 512M < 2x8 512M`，与评测通过/失败边界一致。

## 实验B2.5：8路改为4 lane×4 MS/输入

- 4路保持已Pass的`16 lane×2 MS/输入`。
- 8路从B2.4的`8 lane×2 MS/输入`改为`4 lane×4 MS/输入`：每lane占32 MS，总量仍为128 MS，每lane处理16 KiB，每波仍输出64 KiB。
- 唯一实验变量是8路的lane数和每输入连续MS数；ABI、Mode、LoopGroup总波宽、Host调度、Bank与Clos最终合并不变。
- 相比B2.4，8路的LoopGroup clone数和单字节Checker展开任务理论上再减半。若四个大点全部Pass，则基本确认B2.3/B2.4的静默RE是Checker图规模阈值。
- B2.5结果：8+4的512 MiB和400 MiB+4B全部Pass；2x8两点的Checker图生成、单任务检查和内存冲突检查全部通过，仅语义检查报告B区间`expected 16 sources but got 8`。这证明`4 lane×4 MS`已解决8路Checker图规模问题。
- 2x8剩余错误与Buffer无关：最终`output += Clos partial`被调度到Clos Local Mission，但对称拓扑把该Kernel静态裁为`LocalKernelRole::CLOS`，而旧`REDUCE_RANGE`只在GENERIC/MESH中生成，因此Launch成为no-op，output只包含本Server的8个来源。

## 实验B3：修复Clos最终合并 + Clos slot改为Buffer Fold

- 保留B2.5的Mesh固定形状：4路`16 lane×2 MS/输入`，8路`4 lane×4 MS/输入`。
- 每Die仍只注册通信Mission 0和Local Mission 1，没有新增Mission。Clos Local Mission按`directMemberCount`注册固定4路或8路Buffer Kernel。
- Clos的`FOLD_SLOTS_IN_PLACE`从HBM串行`slot0 += slot1...`改为CCU Buffer四/八路LoopGroup，输入为连续Clos slots，结果仍写回slot0，不改Bank、Pass、网络粒度和Host流水。
- Clos固定Kernel同时保留`READ_REDUCE_CLOS_PARTIALS`与`REDUCE_RANGE`：前者继续大块远端ReadReduce A partial，后者修复2x8/8+4的`output += Clos partial`最终二路合并。
- Mesh/Clos Buffer Loop使用block Event数组；Clos ReadReduce和最终二路合并使用独立普通Event，不混用两类资源。所有Buffer Local Handle固定10参ABI。
- 本轮同时验证：2x8语义修复，以及Clos四/八路Buffer Fold是否能在完整MixedRoute中通过Checker。
- B3结果：2x8和8+4的400 MiB+4B通过，说明2x8最终二路合并已修复，Clos四/八路Buffer Fold的语义也正确；两种拓扑的512 MiB均静默RE。
- 相比B2.5，新增的唯一大规模图来源是Clos Buffer Fold。因此400M通过而512M失败继续符合Checker展开图阈值，不是两Die各申请128 MS互相冲突。

## 实验B3.1：只加粗Clos Buffer Fold粒度

- Mesh保持B2.5已验证形状：4路`16 lane×2 MS/输入`，8路`4 lane×4 MS/输入`。
- Clos 4路从`16 lane×2 MS/输入`改为`8 lane×4 MS/输入`；Clos 8路从`4 lane×4 MS/输入`改为`2 lane×8 MS/输入`。四种形状都恰好占128 MS。
- Clos每lane处理的连续数据量翻倍，lane/clone数减半；每波总输出仍为64 KiB，所以Host的Bank、Pass、地址和调度全部不变。
- 预期Clos Buffer Fold的微块任务数约减半。若512 MiB恢复Pass，则B3静默RE可确认为总Checker图规模超限。
- `git diff --check`与`bash build.sh`通过，未运行`hccl_vm`，待功能评测。
- B3.1结果：全部测试点Pass。因此可固化的结论是：Mesh/Clos两个Die可同时各使用128 MS；固定4/8路Buffer Loop、Clos A ReadReduce、Clos slot Buffer Fold和最终二路合并可在同一MixedRoute中共存。B3的512 MiB静默RE确实由微块展开图过大引起，而非算法、地址、Event或MS资源冲突。
- 当前已Pass基线形状：Mesh 4路`16×2 MS`、Mesh 8路`4×4 MS`、Clos 4路`8×4 MS`、Clos 8路`2×8 MS`。后续性能实验必须以此版本为可回退基线。

## 实验B4：统一Mesh/Clos的粗粒度Buffer几何

- 基线：B3.1 Pass commit `4865517`。
- Clos保持B3.1形状：4路`8 lane×4 MS/输入`，8路`2 lane×8 MS/输入`。
- 唯一性能变量：Mesh 4路从`16×2 MS`加粗为`8×4 MS`，Mesh 8路从`4×4 MS`加粗为`2×8 MS`。四个固定Kernel现在根据输入数共用同一资源几何。
- 每Die仍占128 MS，每波仍处理64 KiB，ABI、Mode、Mission、Bank、Pass、Notify、地址和网络粒度不变。Mesh侧lane clone和单字节微任务数减半。
- 目标：测量更粗的Buffer Copy/Reduce是否减少LoopGroup调度开销并改善六个大数据点；若劣化，直接回退到`4865517`的Mesh细粒度形状。
- B4已提交为commit `07077ea`，功能/性能评测仍在等待。

## 实验B5：最终三路算法家族大合并

- 基线：B4 commit `07077ea`。本轮目标是先形成最终代码结构，不尝试通过一次性能结果拆分各点贡献。
- 小数据完整迁移`Final-pro`已验证的MS Auto Tile方案：
  - 注册期将self分配到成员较少且不超过8路的IO Die；每Die不超过128 MS。
  - 通信Kernel直接将本Rank输出slice的各source tile读入连续MS，一次多路Buffer Reduce后写出die partial。
  - 2x8和4x1将地址/生命周期同步融入MS launch；8+4保留`Final-pro`已测更快的外置同步。
  - 双Die只用一个专用`SmallCombineKernel`完成两个die partial的普通二路合并。
- 2x8与8+4大数据完整保留B4的MixedRoute、Bank、Pass、Notify和Mesh/Clos CCU Buffer规约，未引入`Final-pro`的S9 HBM旧路径。
- 4x1保留12:3:1三轮Fade-out通信与三个常驻远端slot；Local Mission从每轮`slot fold + output add`两个HBM Launch融合为`output/self + 3 slots -> output`一个四路Buffer LoopGroup Launch。
- 4x1首次合并严格保留`Final-pro`已Pass的`16 lane×2 MS/输入`，不和B4的粗粒度实验混合。
- 已删除原`CcuReduceScatterLocalKernel`五Mode通用入口以及`LocalKernelRole`；当前Local Kernel只剩小数据二路Combine、4x1固定四路Fade Buffer和MixedRoute Mesh/Clos固定4/8路入口。
- 小数据路径的`MsAuto*`/`MS_AUTO_TILE`已统一重命名为`SmallBuffer*`/`SMALL_BUFFER_TILE`，避免将MS资源名误读为算法名。
- 已完全删除旧`SmallA2A`/`SMALL_FUSED_A2A`备选路径；小数据Buffer计划不可用时直接返回`HCCL_E_NOT_SUPPORT`，不再静默回退。
- 下一轮仅在B5功能Pass后实验lane数：4路和8路均改为4 lane，作为独立性能/资源变量。

## 实验B6：MixedRoute回退S9，4x1远端直读MS

- 性能动机：B4相对S9明显劣化。2x8由`1.20 ms / 930 us`退化为`1.56 ms / 1.23 ms`，8+4由`1.68 ms / 1.30 ms`退化为`1.97 ms / 1.56 ms`。这只能否定“远端先落HBM slot，再搬入MS规约”的实现，不能否定CCU Buffer本身。
- 小数据保留`Final-pro`的`SmallBufferTile`方案，不改变已合并的小数据算法。
- 2x8与8+4大数据恢复S9的六参数HBM Local Kernel与原调度：Mesh slot串行规约、Clos slot串行fold、最终二路合并均恢复到原Mission和Notify依赖，不再使用大数据LoopGroup Buffer Kernel。
- 4x1仍使用12:3:1三段边界，但删除三个远端HBM slot、scratch token和Local Mission。通信Mission直接执行`self HBM + 3个远端HBM -> 4组MS -> 四路LocalReduce -> output HBM`。
- 4x1首版固定8 lane，每lane四个4 KiB MS，共32 MS；一个wave输出32 KiB。Kernel使用软件`CCU_WHILE`重复wave，Host只传wave数。
- 每段不足32 KiB的尾部按至多4 KiB单独Launch；因此400 MiB+4B的最后4字节精确规约，不向上对齐越界。
- 地址交换和最终跨Rank barrier仍在同一个4x1通信Kernel中；每次Launch保持7参数ABI。
- 更新Engine Context tag，避免复用B4/B5已缓存的旧Kernel Handle和资源布局。
- `git diff --check`与`bash build.sh`通过；按约定未运行本地`hccl_vm`，等待评测功能结果。
- B6首版评测结果：4x1大数据静默RE，`checker failed: no op[0] Checker Success and no error`。首版软件`CCU_WHILE`仅用`8 lane × 4 input × 1 MS=32 MS`，单路Read 4 KiB、每wave输出32 KiB。
- B6.1只调整直读MS几何，不改12:3:1、软件循环、地址同步或数据路径：改为`16 lane × 4 input × 2 MS=128 MS`，正好用满每Die默认MS预算。
- B6.1每次self Copy/远端Read/规约长度为8 KiB，每wave输出128 KiB；相比首版，软件循环次数降为四分之一，估算Checker数据任务总量约减半。
- 尾部Launch上限同步从4 KiB改为8 KiB，最终4字节仍精确处理；Context tag升级为`fade_direct_ms_4x1_v2`，避免复用首版Kernel资源布局。
- B6.1两个4x1大数据点仍静默RE。随后核对官方API发现：`Read/LocalCopy/LocalReduce`只要操作数包含单个`CcuBuffer`，`len`均不得超过4096字节；连续申请两个MS并不授权对第一个句柄执行8 KiB操作。B6.1的8 KiB长度属于未定义行为，不能据此判断任务图阈值。
- B6.2保持16 lane与软件`CCU_WHILE`，仅将每输入恢复为1 MS：总占用64 MS、单次操作4 KiB、每wave输出64 KiB。该版本用于隔离验证B6.1是否因CcuBuffer长度越界失败；Context tag升级为`fade_direct_ms_4x1_v3`。
