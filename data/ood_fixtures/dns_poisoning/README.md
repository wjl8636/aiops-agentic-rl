# OOD: DNS 污染

**故障是什么**：调用方（recommendation）对被调用方（product-catalog）的域名/Service 名解析出错误或不可达的 IP，导致连接超时/被拒绝，但被调用方进程本身完全健康——根因发生在名字解析这一层，不在被调用服务的代码或资源上。

**诊断 Agent 该看什么信号**：在调用方容器/Pod 内做 `nslookup`/`dig product-catalog.<ns>.svc.cluster.local`（k8s）或 `getent hosts product-catalog`（docker），把解析结果和 `kubectl get endpoints product-catalog`（k8s）/ `docker network inspect` 里的真实地址做比对，看是否命中一个不在 Endpoints 列表里的 IP；同时确认 product-catalog 自身 `/healthz`、CPU/内存/QPS 全部正常，错误只出现在调用方发起连接这一步，Jaeger trace 甚至可能看不到 span（连接都没建立起来）。

**为什么是 OOD**：`dependency-error` skill 的起手式查询清单假设"能连上、返回 5xx/超时"是应用层问题，用 Jaeger 定位错在调用链哪一跳；但 DNS 污染场景里连接可能压根建立不起来（TCP connect 直接失败或连到错误对端），Jaeger 没有对应 span，`kubectl describe`/`docker stats` 也看不出被调用方有任何异常——现有 4 个 playbook 都没有"先查 DNS 解析结果是否正确"这一步，是一类全新的排查起点。

**Kind 枚举张力**：这里把最接近的 `kind` 标成 `"dependency"`（同样只写在 `scenario_hint` 里，不塞进真实 alert JSON），因为表现上确实是"调用下游失败"；但真实根因和现有 `dependency` 场景（如 s1 的 `productCatalogFailure` flagd 开关）完全不同——一个是应用层故障注入，一个是网络基础设施层的名字解析污染，诊断路径（查证书/查 DNS vs 查 Jaeger trace/查 flagd 配置）几乎不重叠，用同一个 `kind` 值会让 route 判定和 reward 里的"信号覆盖"配置（Jaeger 权重更高）用错权重。

**如何注入复现（sketch，留给后续接真实环境时补全）**：
1. docker 环境：在调用方容器里通过 `--add-host` 或直接改容器内 `/etc/hosts`，把 `product-catalog` 映射到一个错误/不可达的 IP（比如另一个服务的 IP，或者 `127.0.0.1:1`），重启该容器让新的 hosts 生效。
2. k8s 环境：起一个假的 DNS responder（比如自定义 CoreDNS `Corefile` 里加一条 rewrite 规则，或者部署一个最小化的 dnsmasq/自制 UDP 53 responder）对 `product-catalog.*.svc.cluster.local` 返回错误 IP，通过 NetworkPolicy 或 Pod 的 `dnsConfig` 把该 Pod 的 DNS 指向这个假 responder。
3. 观察 recommendation → product-catalog 的调用开始超时/连接被拒绝，同时 product-catalog 自身指标保持健康，验证"调用方侧 DNS 解析异常"这一特征能被区分出来。
