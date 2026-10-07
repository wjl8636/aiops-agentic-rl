# OOD: TLS 证书过期

**故障是什么**：checkout 调用 payment 网关所用的 TLS 证书临近/到达过期时间，握手在证书校验阶段失败（`x509: certificate has expired`），失败比例随时钟推进从个位数百分比爬升到 100%。

**诊断 Agent 该看什么信号**：证书本身的 `notAfter` 时间（`openssl x509 -in <cert> -noout -enddate` 或直接 `openssl s_client -connect host:443 | openssl x509 -noout -dates`）；错误文本里的 `x509`/`certificate expired`/`unable to verify the first certificate` 关键字；失败率曲线的形态（随时间单调爬升，不是随流量或发版跳变，也不是持续打满）；确认没有配套的代码发版记录、没有对应 flagd 开关变更。

**为什么是 OOD**：现有 4 个 skill playbook（`dependency-error`/`memory-oom`/`queue-backlog`/`resource-cpu`）分别覆盖 5xx 依赖失败、内存单调上升、队列积压、CPU 打满，起手式查询清单里全都没有"查证书有效期"这一步——证书过期的错误特征（TLS 层握手失败、错误里带 x509 关键字）和普通应用层 5xx 依赖故障看起来相似但根因排查路径完全不同（前者查证书文件/证书链，后者查 Jaeger trace 定位调用链哪一跳）。

**Kind 枚举张力**：这条 fixture 把最接近的 `kind` 标成 `"config"`（写在 `alert.json` 的 `scenario_hint` 里，没有塞进真实 alert JSON 本身，因为真实 schema 里告警本来就不带 `kind` 字段——`kind` 只是 `Diagnosis` 输出字段）。但证书过期本质上既不是"配置项取值错误"也不是现有 4 类里任何一类，是一类需要新增枚举值（如 `"credential_expiry"` 或 `"security"`）才能准确表达的根因，这是本项目 OOD 集合暴露出的一个真实设计缺口，建议后续扩展 `Diagnosis.kind` 枚举而不是永远用 `config` 兜底。

**如何注入复现（sketch，留给后续接真实环境时补全）**：
1. 用 `openssl req -x509 -newkey rsa:2048 -days 1 -nodes -keyout key.pem -out cert.pem` 生成一张已经/即将过期的自签名证书（把 `-days` 设成负数或很小的正数，或者生成后用 `faketime`/直接改系统时钟把有效期"推过去"）。
2. 在 docker-compose 里给 payment（或 checkout 前置的一个 nginx/envoy sidecar）挂载这张证书替换掉原来的证书，重启该容器。
3. 观察 checkout → payment 的调用开始报 TLS 握手失败，错误率曲线随证书过期时刻推移爬升；可选：用 `openssl s_client` 直接对目标端口发起连接，观察返回的证书链和过期错误。
4. k8s 环境下等价做法：把证书放进一个 Secret，挂载到目标 Pod，改 Secret 内容后 `kubectl rollout restart` 让新证书生效。
