# Triple水木——决赛代码归档

## 团队信息

- 赛区：2026 HCCL通信库创新大赛粤港澳赛区
- 队伍：Triple水木
- 归档学校：深圳河套学院&香港中文大学（深圳）
- 参赛成员：
  - Menphina（深圳河套学院）
  - qq_40734045（深圳河套学院）
  - dubai712（香港中文大学（深圳）、深圳河套学院）
- 联系方式：
  - Menphina：`wangyuansen@outlook.com`
  - qq_40734045：`zhaoyumiao99@gmail.com`
  - dubai712：`1006529762@qq.com`

## 作品简介

本作品面向决赛 ReduceScatter CCU 通信算子，在 Host 侧完成资源规划与参数下发，在 CCU Kernel 中实现通信、规约和流水执行。方案围绕多 IO Die、切片布局、Buffer 复用及通信计算重叠进行优化。

## 技术方案

- Host 侧完成拓扑识别、Channel/CCU 资源申请和 Kernel 参数组织；
- CCU Kernel 实现本地拷贝、远端读写、Reduce 及同步编排；
- 针对不同通信规模采用分片和多 Mission 流水；
- 通过 CCU Buffer 台账约束地址生命周期与复用关系；
- 设计决策和流水演进记录随代码一并归档。

## 环境与运行

- CANN Toolkit：9.1.0 配套赛事版本；
- 硬件：赛事指定昇腾 CCU 环境；
- 构建：进入 `code/hccl_reducescatter_ccu` 后执行 `bash build.sh`；
- Debug 构建：执行 `bash build.sh --debug`。

详细环境变量和构建步骤见工程内 `README.md`。

## 性能与验证

算法设计、流水方案及各阶段验证记录分别保存在 `REDUCE_SCATTER_CCU_DESIGN.md`、`pipeline.md` 和工作台账中。本归档 README 不重新填写未经核验的成绩数据。

## 归档内容

本目录归档决赛 ReduceScatter CCU 算子工程：

```text
code/hccl_reducescatter_ccu/
├── include/
├── op_host/
├── op_kernel_ccu/
├── CMakeLists.txt
├── build.sh
├── README.md
├── REDUCE_SCATTER_CCU_DESIGN.md
└── pipeline.md
```

代码来源：`https://gitcode.com/Menphina/ReduceScatter` 的 `Final` 目录。

构建和环境配置说明请阅读 `code/hccl_reducescatter_ccu/README.md`；算法设计与流水线说明分别位于 `REDUCE_SCATTER_CCU_DESIGN.md` 和 `pipeline.md`。

## 提交记录

- Menphina：`adf3c933fd94d5f655b049e2754e81ef4de858a8`
- qq_40734045：`ac7c910dfefe9aab77cedc87968e2269e028634b`
- dubai712：
  - `981ec8b5a00c4a1ff6ffec157185d706f0444ec8`
  - `d4cfc3adfd9ec370f70d439e97424fa0c5b1ea40`
  - `c29e3627b5561460e526270ac79e4f3f9bc80d25`

本次归档不包含编译产物、运行日志或访问凭据。
