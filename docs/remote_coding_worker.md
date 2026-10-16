# SSH 远程编码 Worker

日期：2026-10-15。此功能实现“编码在独立 Linux VM，研究、数据、训练和评测仍在本机”。
接入代码不等于 VM 已部署或真实模型已验证；部署需要用户提供受信任的 SSH 目标。

## 执行路径

```text
本机对话 / ResearchDesignAgent
  -> 计划与编码合同
  -> CodexCodingAgent 创建源码工作区
  -> SshCodexExecutor 通过 SSH 发送源码与任务 JSON
  -> 远端 Python Worker 检查 CLI / 登录状态 / workspace-write 沙箱
  -> 远端 Codex CLI 编码
  -> 返回变更源码、删除清单、CLI 事件与用量
  -> 本机校验路径、变更范围、基线一致性和独立 acceptance commands
  -> 验证通过后应用到本任务 model_pool
  -> 原有本机实验执行器训练、评测、分析与迭代
```

研究/对话/辅助模型配置不变。编码模型仍由 `ALGO_CODING_MODEL` 指定；为空时沿用
本机 Codex 顶层配置的模型快照，再显式传给远端。不是偷偷换成远端默认模型。
记录同时保留请求模型和 CLI 实际回报模型，未回报时不声称确认了服务端模型身份。

## 远端准备

由管理员准备一台允许 Codex Linux 沙箱运行的独立 VM。编码 Worker 本身不要求 GPU；
实际训练仍在当前 GPU 环境。使用专用普通账户，不使用共享管理员账户。

远端需要：

- Python >= 3.11、SSH 服务、兼容的 Codex CLI。
- 本机和 VM 保持时钟同步，绝对过期时间用于拒绝延迟送达的过期任务。
- Worker 账户自身的 Codex 认证及模型服务网络访问。不会从本机复制认证文件。
- 支持 `workspace-write` 沙箱。先验证，再接入；本实现不会自动关闭沙箱。
- 专用工作目录，例如 `/home/algo/coding-jobs`，属于 Worker 账户，权限 0700。
  不放在共享目录或 `/tmp` 下，不通过软链接指向其他位置。
- 足够的临时空间。需要强制 CPU/内存/总磁盘限制时，在 VM/账户/cgroup 层设置。

在远端以该账户执行：

```bash
python3 --version
codex --version
codex login status
codex sandbox linux /bin/true
mkdir -m 700 /home/algo/coding-jobs
```

最后一条仅用于尚不存在的专用目录；不要递归修改已有业务目录的权限。
`login status` 失败时，由账户所有者完成认证，不把密码、私钥或 token 发到聊天中。

不需要在 VM 安装完整 algo-agent。控制端通过 SSH 传递 stdlib-only Worker 程序。
源码检查需要的第三方库可预先安装在指定 Python 环境；GPU 训练数据和依赖不自动迁移。

## SSH 信任和配置

在本机配置一个可非交互使用的 SSH 别名，例如：

```sshconfig
Host algo-code-vm
    HostName <vm-address>
    User algo
    Port 22
    IdentityFile ~/.ssh/algo_worker
    IdentitiesOnly yes
```

HostName 和密钥路径需由实际部署决定。管理员先通过可信渠道核对远端 host key，
再登记到 known_hosts；不能盲目信任扫描得到的 key。
执行器强制 BatchMode、StrictHostKeyChecking，禁用 agent/X11/端口转发及 LocalCommand，
不交互索要密码，不自动接纳未知 host key，也不复用带转发的 SSH master。

修改平台 `.env` 中相应字段，其他模型配置和凭据保持原样：

```dotenv
ALGO_CODING_TRANSPORT=ssh
ALGO_CODEX_ALLOW_REAL=1
ALGO_CODING_SSH_HOST=algo-code-vm
ALGO_CODING_SSH_ROOT=/home/algo/coding-jobs
ALGO_CODING_SSH_PYTHON=/usr/bin/python3
ALGO_CODING_SSH_CODEX=/usr/local/bin/codex
```

可执行文件必须使用远端实际路径。非交互 SSH 的 PATH 可能与登录终端不同，优先填写
绝对路径。支持 SSH 配置中的 ProxyJump；本功能没有另建 HTTP 服务端口。
显式填写 `ALGO_CODING_MODEL` 可避免依赖本机模型快照；不需要复制本机 `.codex`。

`.env` 已有值优先于同名进程环境变量，这是平台原有加载行为。
本次接入没有替用户修改真实 `.env` 或 SSH 配置，也没有创建或付费开通 VM。

## 预检和启用

```bash
algo-agent --root /mnt/data/yangt/codes/QAA/algo_agent --json preflight --probe-coding --strict
```

源码方式等价：

```bash
PYTHONPATH=runtimes/agent_runtime/src:. python3 -m algo_agent.cli.main --json preflight --probe-coding --strict
```

- 不带 `--probe-coding` 时只报告配置。SSH 模式中的 `coding_binary_found` 指本机 ssh，
  不代表远端已有 Codex。
- 带该选项会通过 SSH 检查远端 Python/Linux、Codex 版本、登录状态和沙箱。
  `coding_probe.available=true` 才表示这些检查通过；不调用付费模型。
- `live_verified` 仍为 false：登录状态不是模型生成/网络连通性的完整验证，
  更不是算法效果证明。
- `--strict` 同时检查研究与检索配置。真实自动任务在 intake 阶段会自动执行远端预检；
  不通过则在研究开始前阻塞，保留具体原因。
- 控制服务已启动时需重启该服务，使修改后的 `.env` 被加载；仅编辑文件不会热切换
  已经运行中的任务。

下一步使用独立测试任务做一次很小的真实编码，确认新文件、CLI 事件和本机验收成功，
再开展 IQA 训练。不要把早期已停止的 IQA 任务的旧结果当成新结果。

## 传输、回写和取消

- 仅发送 `model_pool` 中的文本源码、只读插件上下文、编码合同和必要的修复反馈。
  不同步整个仓库、原始图片、标签集、权重、状态数据库、`.env`、`.ssh`、`.codex`。
  这是文件层过滤；源码或需求正文中本来就写入的敏感内容仍需自行审查。
- 源码最多 2000 个文件、合计 16 MiB；请求最多 32 MiB，CLI 日志最多 8 MiB。
  不接受绝对路径、`..`、隐藏/凭据文件、软链接、硬链接或二进制源码。
  超出限制显式报错，不静默截断代码。当前传输文本内容，不同步可执行权限位。
- 远端每次调用使用唯一任务目录和独立进程组；返回时只取变更文件。
  远端和本机都检查修改范围，只允许合同中的 `model_pool/<method>`，
  不允许修改插件或 `coding_contract.json`。
- 回传内容不会直接覆盖正式算法目录。先确认本地工作区没有漂移，应用到临时工作区，
  再由原有 CodexCodingAgent 执行独立验收，最后才应用到任务模型池。
- 取消会中断本地 SSH，并尝试发出按 job ID 定位的远端取消；
  进程身份检查避免只凭复用 PID 误杀。SSH 信号触发清理。
- 网络完全不可达时，远端取消不能保证立即生效；远端自身仍有墙钟截止时间。
  单次远端调用硬上限为 24 小时，且不超过本任务剩余预算。
  取消联络的响应超时为 8 秒，进程清理另有短暂宽限，不等待模型自行结束。
- 正常返回和受控失败后删除本次远端临时目录，本机保留编码事件和结果。
  VM 崩溃或 SIGKILL 可能遗留目录，应由管理员清理；没有实现跨 VM 崩溃恢复、
  远端排队系统、断线后的作业重附着或 exactly-once 保证。

## 安全与能力边界

远端一直使用 `--sandbox workspace-write` 和 `approval_policy="never"`，
从不回退到 `danger-full-access`。auto-process 免人工审批不等于免隔离。
清理针对本次进程组；主动脱离该进程组的进程需由 VM/cgroup 层管理。

VM 是专用编码环境，但本地训练/验收仍是原来的本机进程，并非新的安全沙箱；
不要据此把不可信第三方任务开放给多租户用户。文件、日志传输限额也不是 CPU/内存/
总磁盘硬配额。需要多租户或强资源隔离时，应在独立实例/作业调度层落实。

测试 `tests/test_ssh_coding.py` 使用假的 Codex CLI 和本机子进程，验证传输、
取消、截止时间、路径安全、范围限制和本地独立验收。它不证明真实 SSH、
远端内核沙箱、Codex 认证生成或 IQA 达标。真实部署必须补上述预检和小型编码实测。

2026-10-15 全量平台回归结果：**305 passed in 129.34s**。

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=runtimes/agent_runtime/src:. python3 -m pytest
```
