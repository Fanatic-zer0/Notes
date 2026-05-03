# API Gateway Patterns in Kubernetes

## Overview

An **API Gateway** is the single entry point for all client requests. It acts as a **reverse proxy** and **traffic orchestrator** that sits between external clients and internal microservices.

### Core Responsibilities

1. **Routing** - Direct requests to appropriate backend services based on path, headers, or other criteria
2. **Authentication & Authorization** - Validate identity (JWT, OAuth2, API keys) and permissions before forwarding
3. **Rate Limiting** - Prevent abuse by limiting requests per user/IP/API key
4. **Load Balancing** - Distribute traffic across multiple service instances
5. **SSL/TLS Termination** - Handle HTTPS encryption/decryption at gateway edge
6. **Observability** - Centralized logging, distributed tracing, metrics collection
7. **Request/Response Transformation** - Modify headers, bodies, or formats
8. **Circuit Breaking** - Fail fast when downstream services are unhealthy
9. **Caching** - Cache responses to reduce backend load
10. **API Versioning** - Manage multiple API versions (`/v1/`, `/v2/`)

### Why Use an API Gateway?

**Without API Gateway (Direct Service Access):**
```
Mobile App ──┐
Web App ─────┼──→ user-service (needs auth)
IoT Device ──┘    order-service (needs auth)
                  payment-service (needs auth)
                  ↓
Problem: Auth logic duplicated across all services
         No centralized rate limiting
         Direct exposure of internal architecture
```

**With API Gateway:**
```
Mobile App ──┐
Web App ─────┼──→ API Gateway (single auth point)
IoT Device ──┘           ↓
                    Route to appropriate service
                    ↓
                user-service (no auth needed)
                order-service (no auth needed)
                payment-service (no auth needed)
```

## API Gateway vs Ingress

| Feature | Ingress | API Gateway |
|---------|---------|-------------|
| Layer | L7 HTTP | L7 HTTP + L4 TCP |
| Auth | Basic (nginx annotations) | Advanced (JWT, OAuth2, API keys) |
| Rate Limiting | Basic | Advanced (per user, per API key) |
| Transformation | Limited | Full req/res transformation |
| gRPC | Limited | Full support |
| WebSockets | Limited | Full support |
| API Versioning | Manual | Built-in |
| Developer Portal | No | Yes (some) |
| Analytics | No | Yes |

## Architecture Patterns

### Pattern 1: Single API Gateway

```
Internet
    │
    ├─→ Cloud LoadBalancer
    │
    ├─→ API Gateway (Kong / AWS API GW / Traefik)
    │   - Auth (JWT validation)
    │   - Rate limiting
    │   - SSL termination
    │   - Routing rules
    │
    ├─→ Microservices
    │   - user-service
    │   - order-service
    │   - payment-service
    └─→ ...
```

### Pattern 2: Edge + Internal Gateway

```
Internet
    │
    ├─→ Edge API Gateway (external facing)
    │   - SSL, rate limiting, DDoS protection
    │   - Auth for external APIs
    │
    └─→ Internal API Gateway (service mesh)
        - Internal auth (mTLS)
        - Service-to-service routing
        - Circuit breaking
```

### Pattern 3: BFF (Backend for Frontend)

```
Mobile App ──────→ Mobile BFF Gateway (optimized responses)
Web Browser ─────→ Web BFF Gateway (server-side rendering optimizations)
IoT Device ──────→ IoT BFF Gateway (lightweight, MQTT support)
                        │
                   Shared Services
                   - user-service
                   - product-service
                   - order-service
```

## Kubernetes API Gateway — Architecture Overview

### Component Map: Who Does What

```mermaid
flowchart LR
    subgraph Clients["External Clients"]
        Web[Web App / Browser]
        Mobile[Mobile App]
        API[API Consumer]
    end

    subgraph Edge["Edge Layer"]
        DNS[DNS\nRoute53 / Cloud DNS]
        LB[Cloud Load Balancer\nAWS ALB / GCP LB / NLB]
    end

    subgraph ControlPlane["Kubernetes Control Plane"]
        APIServer[kube-apiserver]
        etcd[etcd]
        GWController[Gateway Controller\ne.g. Kong KIC / Traefik / Envoy]
    end

    subgraph GatewayConfig["Gateway API Resources"]
        GWClass[GatewayClass\ndefines implementation]
        GW[Gateway\nlistener + TLS config]
        HTTPRoute[HTTPRoute / GRPCRoute\nrouting rules per team]
    end

    subgraph GatewayPods["Gateway Data Plane (Pods)"]
        GWPod1[Gateway Pod\nnginx + Lua / Envoy proxy]
        GWPod2[Gateway Pod\nnginx + Lua / Envoy proxy]
    end

    subgraph KubeProxy["Node Dataplane"]
        KProxy[kube-proxy\niptables / IPVS rules]
    end

    subgraph Services["Backend Services"]
        SvcA[user-service\nClusterIP]
        SvcB[order-service\nClusterIP]
        SvcC[payment-service\nClusterIP]
    end

    subgraph Pods["Backend Pods"]
        PodA1[user-pod-1]
        PodA2[user-pod-2]
        PodB1[order-pod-1]
        PodC1[payment-pod-1]
    end

    Web & Mobile & API --> DNS
    DNS --> LB
    LB --> GWPod1 & GWPod2

    GWController -- "LIST/WATCH GatewayClass\nGateway, HTTPRoute" --> APIServer
    APIServer <--> etcd
    GatewayConfig -- "stored in" --> APIServer
    GWController -- "translates rules to\nproxy config (nginx.conf / xDS)" --> GWPod1 & GWPod2

    GWPod1 & GWPod2 -- "route to ClusterIP" --> SvcA & SvcB & SvcC
    KProxy -- "programs DNAT rules for\nClusterIP → Pod IPs" --> SvcA & SvcB & SvcC
    SvcA --> PodA1 & PodA2
    SvcB --> PodB1
    SvcC --> PodC1
```

### How Each Component Contributes

| Component | Plane | Role in API Gateway Flow |
| --- | --- | --- |
| `GatewayClass` | Config | Names the implementation (e.g. `konghq.com/kic`). Cluster-admin scoped. |
| `Gateway` | Config | Declares listeners (port 443, TLS cert). Binds to which namespaces can attach routes. |
| `HTTPRoute` | Config | Team-owned routing rules: host, path, headers, weights, filters. Attached to a Gateway. |
| Gateway Controller | Control Plane | Watches GatewayClass/Gateway/HTTPRoute. Translates them into proxy config and reloads data plane. |
| Gateway Pods | Data Plane | The actual reverse-proxy process. Performs authn, rate limiting, routing, TLS termination per request. |
| `kube-apiserver` | Control Plane | Persists all Gateway API objects. Controller and tools interact only through it. |
| `etcd` | Control Plane | Stores Gateway API objects durably. Single source of truth. |
| `kube-proxy` | Node | Programs iptables/IPVS DNAT rules so Service ClusterIPs forward to live Pod IPs. |
| EndpointSlice controller | Control Plane | Keeps EndpointSlices up to date as Pods start/stop. Gateway and kube-proxy consume these. |
| Cloud Load Balancer | Edge | Terminates external TCP/HTTPS, distributes across Gateway Pod replicas. |

### Gateway Controller — Reconciliation Loop

The controller is the bridge between Kubernetes config objects and the running proxy process:

```mermaid
flowchart TD
    Start([Controller starts]) --> List["LIST GatewayClass\nGateway\nHTTPRoute\nEndpointSlice"]
    List --> Cache[Populate local\ninformer cache]
    Cache --> Watch["WATCH for changes\n(long-lived HTTP stream)"]
    Watch --> Event{Event type?}

    Event -->|GatewayClass created\nor updated| GWC[Validate controllerName\nmatches this controller]
    GWC --> SetStatus1[PATCH GatewayClass/status\nAccepted=True]
    SetStatus1 --> Watch

    Event -->|Gateway created\nor updated| GWE[Resolve TLS certs\nfrom Secrets\nValidate listeners]
    GWE --> SetStatus2[PATCH Gateway/status\nReady=True\nassign external IP]
    SetStatus2 --> Watch

    Event -->|HTTPRoute created\nor updated| RT[Validate parentRef\nresolves to allowed Gateway\nCheck namespace policy]
    RT --> Translate[Translate rules to\nproxy config\nnginx.conf / xDS snapshot]
    Translate --> Reload[Push config to\nGateway Pods\nvia xDS API or ConfigMap]
    Reload --> SetStatus3[PATCH HTTPRoute/status\nResolvedRefs=True]
    SetStatus3 --> Watch

    Event -->|EndpointSlice changed| EP[Recompute upstream\nendpoint list]
    EP --> UpdateLB[Update load balancer\nendpoint table in proxy]
    UpdateLB --> Watch

    Event -->|HTTPRoute deleted| Del[Remove route from\nproxy config\nReload]
    Del --> Watch
```

## Request Path: `/v1/app` to Pod — Network Hops Explained

> **Short answer:** The request lands on a **Gateway Pod** (not the Service).  
> A Kubernetes Service is not a process — it is a set of iptables/IPVS rules programmed by `kube-proxy`.  
> Every "connection to a Service ClusterIP" is transparently DNAT'd by those rules to a real Pod IP before a single byte is received by any process.

### Full Network Path with Real IPs

```mermaid
flowchart TD
    Client["🖥️ Client Browser / Mobile App\nGET https://api.example.com/v1/app\nAuthorization: Bearer <JWT>"]

    DNS["🌐 DNS\napi.example.com\n→ 203.0.113.100"]

    CloudLB["☁️ Cloud Load Balancer\nPublic IP: 203.0.113.100:443\nHealth-checks Gateway Pods every 5s\nSelects healthy backend by round-robin"]

    GWSvc["📋 Kong Gateway Service\ntype: LoadBalancer\nExternal IP: 203.0.113.100\nClusterIP: 10.96.50.10:443\n─────────────────\nNOT a process — just routing rules\nkube-proxy programs DNAT for this IP"]

    KProxy1["⚙️ kube-proxy on node-1\niptables DNAT rule:\n10.96.50.10:443 → 10.244.0.5:8443\n(selected Gateway Pod)"]

    GWPod["🔷 Kong Gateway Pod\nReal IP: 10.244.0.5:8443\n─────────────────\nActual process that handles the request:\n① TLS termination (decrypt HTTPS)\n② Rate Limit plugin — check counter\n③ JWT Auth plugin — verify token\n④ Request Transformer — add X-User-* headers,\n   strip /v1 prefix → /app\n⑤ Route match: /app → app-service\n⑥ Pick endpoint from EndpointSlice"]

    AppSvc["📋 app-service\ntype: ClusterIP\nClusterIP: 10.96.80.100:8080\n─────────────────\nNOT a process — just routing rules\nkube-proxy programs DNAT for this IP"]

    KProxy2["⚙️ kube-proxy on node-2\niptables DNAT rule:\n10.96.80.100:8080 → 10.244.2.15:8080\n(selected backend Pod)"]

    AppPod["🟢 app-service Pod\nReal IP: 10.244.2.15:8080\n─────────────────\nActual process receiving request:\nGET /app HTTP/1.1\nX-User-ID: user-123\nX-User-Roles: premium\nX-Request-ID: uuid\n(Authorization header removed)"]

    DB["🗄️ Database\n10.244.3.50:5432"]

    Client -->|"1 — DNS query"| DNS
    DNS -->|"2 — returns LB public IP"| Client
    Client -->|"3 — TCP+TLS to 203.0.113.100:443"| CloudLB
    CloudLB -->|"4 — forwards to Kong Service\n(still public IP layer)"| GWSvc
    GWSvc -->|"5 — kube-proxy DNAT\nClusterIP → Pod IP"| KProxy1
    KProxy1 -->|"6 — packet delivered to\nGateway Pod real IP"| GWPod
    GWPod -->|"7 — new TCP connection\nto app-service ClusterIP"| AppSvc
    AppSvc -->|"8 — kube-proxy DNAT\nClusterIP → Pod IP"| KProxy2
    KProxy2 -->|"9 — packet delivered to\nbackend Pod real IP"| AppPod
    AppPod -->|"10 — DB query"| DB
    DB -->|"11 — result"| AppPod
    AppPod -->|"12 — HTTP 200 JSON"| GWPod
    GWPod -->|"13 — add latency headers,\nemit metrics, re-encrypt TLS"| CloudLB
    CloudLB -->|"14 — HTTPS 200 to client"| Client
```

### The Two DNAT Hops Explained

Every HTTP call in Kubernetes that goes through a Service goes through a DNAT rewrite by `kube-proxy`. This happens **twice** in an API Gateway flow:

```
Hop 1 — Client → Gateway
────────────────────────────────────────────────────────────────
Cloud LB sends TCP to: 203.0.113.100:443         (public IP)
  ↓  cloud routes to node
Node receives packet destined for: 10.96.50.10:443 (ClusterIP)
  ↓  kube-proxy iptables DNAT
Packet rewritten to:  10.244.0.5:8443             (Kong Pod IP)
  ↓  routed via CNI overlay to node-1
Kong Pod process receives the connection ✅

Hop 2 — Gateway → Backend
────────────────────────────────────────────────────────────────
Kong opens new TCP to: 10.96.80.100:8080          (ClusterIP)
  ↓  kube-proxy iptables DNAT (on whichever node Kong is on)
Packet rewritten to:  10.244.2.15:8080            (app Pod IP)
  ↓  routed via CNI overlay to node-2
app-service Pod process receives the connection ✅
```

> **Key insight:** A Service ClusterIP never appears in `ss -tnp` or `netstat` output inside any Pod.  
> By the time the packet reaches a network interface, the destination is always a real Pod IP.

### What Each Layer Owns

```mermaid
flowchart LR
    subgraph L4["L4 — Transport (TCP)"]
        direction TB
        CloudLB2["Cloud LB\nTerminates external TCP\nLoad-balances across nodes"]
        KProxy3["kube-proxy\nDNATs Service VIPs\nto Pod IPs via iptables/IPVS"]
    end

    subgraph L7["L7 — Application (HTTP)"]
        direction TB
        GWPod2["Kong / Envoy Pod\n• TLS termination\n• Auth + Rate limit\n• Routing decision\n• Header rewrite\n• Observability"]
        AppPod2["Backend Pod\n• Business logic\n• DB queries\n• Response body"]
    end

    subgraph Config["Config Plane (Kubernetes API)"]
        direction TB
        GWClass2["GatewayClass\nGateway\nHTTPRoute\n→ tell the controller what rules exist"]
        GWCtrl["Gateway Controller\n→ translates rules into\nnginx.conf / xDS config\n→ pushes to Gateway Pods"]
        EPSlice["EndpointSlice controller\n→ tracks ready Pod IPs\n→ consumed by kube-proxy\nand Gateway Pods"]
    end

    Config --> L7
    Config --> L4
    L4 --> L7
```

### Concrete Example: `GET /v1/app` Step by Step

```
1.  Client:        GET https://api.example.com/v1/app
                   Authorization: Bearer eyJ...

2.  DNS:           api.example.com → 203.0.113.100

3.  Cloud LB:      Receives TCP:443
                   Health-checks show kong-pod-1 and kong-pod-2 are healthy
                   Selects kong-pod-1 (round-robin)
                   Forwards to node-1 port 31080 (NodePort) or directly via
                   target group to pod IP

4.  kube-proxy     Sees dst=10.96.50.10:443 (Kong Service ClusterIP)
    (node-1):      Matches iptables rule:
                     -A KUBE-SVC-KONG -m statistic --mode random \
                       -j KUBE-SEP-KONGPOD1   ← 50% chance
                     -A KUBE-SVC-KONG -j KUBE-SEP-KONGPOD2            ← 50% chance
                   DNAT: 10.96.50.10:443 → 10.244.0.5:8443

5.  Kong Pod       Receives TLS connection on 10.244.0.5:8443
    (10.244.0.5):  ① Decrypt TLS
                   ② Rate Limit: counter=46/100 ✅
                   ③ JWT Auth:   decode, verify RS256, exp valid ✅
                   ④ Transform:  strip /v1 → path becomes /app
                                 add X-User-ID: user-123
                                 remove Authorization header
                   ⑤ Route:      host=api.example.com + path=/app
                                 → matched HTTPRoute rule
                                 → backend: app-service:8080
                   ⑥ Endpoint:   reads EndpointSlice
                                 available: [10.244.2.15, 10.244.2.16, 10.244.3.10]
                                 selects:   10.244.2.15 (round-robin)

6.  kube-proxy     Kong opens TCP to app-service ClusterIP: 10.96.80.100:8080
    (node-1):      DNAT: 10.96.80.100:8080 → 10.244.2.15:8080

7.  CNI overlay:   Routes packet from node-1 to node-2 (where 10.244.2.15 lives)
                   e.g. VXLAN encap → decap on node-2 → delivered to pod veth

8.  app Pod        Receives: GET /app HTTP/1.1
    (10.244.2.15): Headers:
                     X-User-ID: user-123
                     X-User-Roles: premium
                     X-Request-ID: 7f8d9e2a
                     X-Forwarded-For: 203.0.113.50
                   No Authorization header (stripped by gateway) ✅
                   Process request → DB query → build response

9.  Response       app Pod → kube-proxy (reverse NAT) → Kong Pod
    path:          Kong adds:
                     X-Kong-Upstream-Latency: 38
                     X-Kong-Proxy-Latency: 4
                     Access-Control-Allow-Origin: https://app.example.com
                   Kong re-encrypts TLS → Cloud LB → Client

10. Client         HTTP/1.1 200 OK
    receives:      X-RateLimit-Remaining: 54
                   X-Request-ID: 7f8d9e2a
                   { "data": [...] }
                   Total: ~48ms
```

---

## API Gateway Internals

### 1. Config Translation: HTTPRoute → Proxy Config

The controller continuously watches the Kubernetes API and translates declarative objects into the proxy's native config format. No config change touches the proxy directly — everything goes through the controller's translation engine.

```mermaid
flowchart TD
    subgraph K8sObjects["Kubernetes Config Objects (stored in etcd)"]
        GWClass["GatewayClass\ncontrollerName: konghq.com/kic"]
        GW["Gateway\nlisteners:\n  port: 443, HTTPS\n  tls: Secret/gateway-tls\n  allowedRoutes: ns selector"]
        RouteA["HTTPRoute — team-a\nhost: api.example.com\n/v1/users → user-service:8080\n/v1/users/profile → user-svc-v2:8080\nfilter: strip /v1 prefix"]
        RouteB["HTTPRoute — team-b\nhost: api.example.com\n/v1/orders → order-service:8080\ncanary weight: v1=90 v2=10"]
        EPSlice["EndpointSlice\nuser-service:\n  10.244.1.10:8080 ready=true\n  10.244.2.15:8080 ready=true\n  10.244.3.10:8080 ready=false ← excluded"]
        TLSSecret["Secret/gateway-tls\ntls.crt: <PEM cert chain>\ntls.key: <PEM private key>"]
    end

    subgraph Controller["Gateway Controller (e.g. Kong KIC)"]
        Watch["Informer / Reflector\nLIST+WATCH all Gateway API types\n+ EndpointSlices + Secrets"]
        Validate["Validation\nparentRef resolves to allowed Gateway?\nNamespace has gateway-access label?\nBackend Service exists?"]
        Translate["Translation Engine\nHTTPRoute rules → upstream + route objects\nTLS Secret → cert PEM bytes\nEndpointSlice ready addrs → upstream targets\nFilters → plugin config"]
        Diff["Diff Engine\nCompare desired config vs current proxy state\nOnly push deltas — avoid unnecessary reloads"]
        StatusPatch["PATCH status back to API server\nHTTPRoute: ResolvedRefs=True\nGateway: Ready=True, externalIP assigned"]
    end

    subgraph ProxyConfig["Kong Proxy State (via Admin API / xDS)"]
        UpstreamObj["Upstream\nname: user-service.production\nalgorithm: round-robin\ntargets:\n  10.244.1.10:8080 weight=100\n  10.244.2.15:8080 weight=100"]
        RouteObj["Route\nhosts: [api.example.com]\npaths: [/v1/users]\nstrip_path: true\nplugins: rate-limit, jwt, req-xfm"]
        ServiceObj["Service\nname: user-service\nprotocol: http\nupstream: user-service.production"]
        CertObj["Certificate\ncert: <PEM>\nkey: <PEM>\nsnis: [api.example.com]"]
    end

    K8sObjects --> Watch
    Watch --> Validate
    Validate --> Translate
    Translate --> Diff
    Diff -->|"POST/PATCH Admin API\nor xDS snapshot push"| ProxyConfig
    Diff --> StatusPatch
```

**Translation table — what each K8s object becomes in the proxy:**

| K8s Object | Proxy Equivalent | Key decisions made |
| --- | --- | --- |
| `GatewayClass` | Proxy process config, admin API settings | Validates `controllerName` matches this controller only |
| `Gateway` listener | Proxy listen port + TLS cert binding | Reads Secret, decodes PEM, maps SNI → cert |
| `HTTPRoute` rule | Route: host + path + header match conditions | Evaluates match specificity order, most specific wins |
| `HTTPRoute` filter | Plugin config attached to route | `RequestHeaderModifier` → request-transformer plugin config |
| `HTTPRoute` backendRef | Upstream target group | Resolves Service → EndpointSlice → live Pod IPs with `ready=true` |
| `HTTPRoute` weights | Upstream target weights | `v1 weight=90, v2 weight=10` → two upstreams + weighted balancer |
| `EndpointSlice` address | Upstream target | Only `conditions.ready=true` addresses included |
| `Secret` (TLS) | Certificate object + SNI mapping | PEM decode, validates cert chain expiry |

---

### 2. Internal Request Processing Pipeline

Inside the gateway pod, every request travels through a **phase-based pipeline**. Each phase is a hook point where plugins can read, modify, or short-circuit the request. Phases run in a fixed order — a plugin cannot run in `log` phase before `access` phase.

```mermaid
flowchart TD
    TCP["TCP Accept\nTLS ClientHello received"]

    subgraph Pipeline["Phase Pipeline — runs per request"]
        Ph_cert["Phase: certificate\nSNI lookup → select TLS cert\nOptional: verify client mTLS cert\nPlugin hook: custom cert selection logic"]

        Ph_rewrite["Phase: rewrite\nEarly path/method rewriting\nBefore route matching\nPlugin hook: path normalization, redirects"]

        Ph_access["Phase: access\n★ Most plugins run here ★\n• Rate limiting (check + increment counter)\n• JWT / API key authentication\n• IP allow/deny\n• Request size limits\n• Request validation\nPlugin hook: return 4xx to short-circuit"]

        Ph_balancer["Phase: balancer\nUpstream target selection\nHealth-aware load balancing\nRetry on connection failure\nPlugin hook: custom LB algorithm"]

        Ph_upstream["Forward request to selected Pod IP\nUse pooled TCP connection if available\nApply upstream timeout"]

        Ph_header["Phase: header_filter\nModify response headers before body\nAdd CORS, HSTS, security headers\nRemove internal headers (Server, X-Powered-By)\nPlugin hook: runs before body is streamed"]

        Ph_body["Phase: body_filter\nResponse body transformation\nGzip compression / decompression\nResponse caching (store in memory/Redis)\nPlugin hook: can buffer or stream body"]

        Ph_log["Phase: log\nNon-blocking — runs AFTER response sent\nWrite access log entry\nEmit Prometheus counter/histogram\nSend OpenTelemetry span to collector\nAudit trail write"]
    end

    Reject["Return 4xx / 5xx immediately\nRequest never reaches upstream\n(rate limit, auth failure, etc.)"]

    TCP --> Ph_cert --> Ph_rewrite --> Ph_access
    Ph_access -->|"plugin short-circuits"| Reject
    Ph_access -->|"all plugins passed"| Ph_balancer
    Ph_balancer --> Ph_upstream
    Ph_upstream --> Ph_header --> Ph_body --> Ph_log
```

**Why phase placement of plugins matters:**

| Plugin | Phase | Reason |
| --- | --- | --- |
| Rate limiting | `access` | Must reject before touching upstream — saves backend resources |
| JWT auth | `access` | Must reject unauthenticated requests before forwarding |
| Request ID injection | `rewrite` | Must be set before route matching so it appears in logs |
| CORS headers | `header_filter` | Added to response, not request — runs after backend replies |
| Gzip compression | `body_filter` | Operates on response body stream |
| Prometheus metrics | `log` | Non-blocking — does not add latency to the client response |

---

### 3. Upstream Connection Pool

The gateway does **not** open a new TCP connection to a backend Pod for every request. It maintains a **per-target connection pool** — connections are reused across requests.

```mermaid
flowchart LR
    subgraph GWPod["Kong Gateway Pod (10.244.0.5)"]
        Req["Incoming request\nrouted to user-service"]
        Pool1{{"Connection pool\n10.244.1.10:8080\nidle: 3 / max: 60"}}
        Pool2{{"Connection pool\n10.244.2.15:8080\nidle: 0 / max: 60"}}
        Pool3{{"Connection pool\n10.244.3.10:8080\nhealth: FAILED\nCircuit: OPEN"}}
        NewConn["Open new TCP connection\n(3-way handshake ~1ms)"]
        ReuseConn["Reuse idle keepalive\nconnection (0ms overhead)"]
        CB["Circuit breaker\nOPEN: fail immediately\nHALF-OPEN: allow 1 probe\nCLOSED: normal operation"]
    end

    subgraph Backends["Backend Pods"]
        P1["10.244.1.10:8080"]
        P2["10.244.2.15:8080"]
        P3["10.244.3.10:8080\n(unhealthy)"]
    end

    Req -->|"round-robin selects P1"| Pool1
    Pool1 -->|"idle conn available"| ReuseConn
    ReuseConn --> P1

    Req -->|"round-robin selects P2"| Pool2
    Pool2 -->|"pool empty, not full"| NewConn
    NewConn --> P2

    Req -->|"round-robin selects P3"| Pool3
    Pool3 --> CB
    CB -->|"OPEN — skip P3, retry next target"| Req

    P1 -->|"response → return conn to pool"| Pool1
    P2 -->|"response → return conn to pool"| Pool2
```

**Circuit breaker state machine:**

```
         ┌─────────────────────────────────────────────────────────┐
         │                                                         │
    CLOSED (normal)                                               │
    All requests pass through                                     │
         │                                                         │
         │  N consecutive failures (e.g. 5 timeouts or 5xx)       │
         ▼                                                         │
    OPEN (fail fast)                                              │
    Reject immediately — no connection attempt                    │
    Wait T seconds (e.g. 30s)                                    │
         │                                                         │
         │  After timeout                                          │
         ▼                                                         │
    HALF-OPEN (probe)                                             │
    Allow 1 request through as a health probe                    │
         │                          │                              │
    Probe succeeds            Probe fails                         │
         │                          │                              │
         └──────── CLOSED ◄─────────┘                              │
                                    └────────────── OPEN ──────────┘
```

---

### 4. Endpoint Discovery: How the Gateway Tracks Live Pod IPs

The gateway never queries DNS per request. It maintains a live table of ready Pod IPs sourced from **EndpointSlices**, updated in near-real-time as Pods start and stop passing their readiness probes.

```mermaid
sequenceDiagram
    participant Kubelet as kubelet
    participant API as kube-apiserver
    participant ESCtrl as EndpointSlice controller
    participant KIC as Gateway Controller
    participant Kong as Kong Proxy

    Note over Kubelet: Pod readinessProbe passes
    Kubelet->>API: PATCH Pod/status\nconditions: Ready=True

    API-->>ESCtrl: WATCH event: Pod MODIFIED
    ESCtrl->>API: PATCH EndpointSlice\nadd 10.244.2.15:8080\nconditions.ready=true

    API-->>KIC: WATCH event: EndpointSlice MODIFIED
    KIC->>KIC: Recompute upstream targets\nuser-service.production:\n  10.244.1.10:8080 ✅\n  10.244.2.15:8080 ✅ ← new\n  10.244.3.10:8080 ❌ ready=false

    KIC->>Kong: Admin API POST\n/upstreams/user-service.production/targets\n{ "target": "10.244.2.15:8080", "weight": 100 }
    Note over Kong: New target added\nNext round-robin cycle includes it

    Note over Kubelet: Pod readinessProbe fails
    Kubelet->>API: PATCH Pod/status\nconditions: Ready=False

    API-->>ESCtrl: WATCH event: Pod MODIFIED
    ESCtrl->>API: PATCH EndpointSlice\n10.244.2.15:8080 conditions.ready=false

    API-->>KIC: WATCH event: EndpointSlice MODIFIED
    KIC->>Kong: Admin API DELETE\n/upstreams/user-service.production/targets/10.244.2.15:8080
    Note over Kong: Target removed\nIn-flight requests complete\nNew requests skip this pod
```

**Two layers of Pod health — both must pass:**

| Layer | Mechanism | Enforced by | Effect on traffic |
| --- | --- | --- | --- |
| Kubernetes readiness | readinessProbe (HTTP/TCP/exec) | kubelet → EndpointSlice controller | Removes pod from EndpointSlice → controller removes from upstream |
| Gateway active health | Periodic HTTP/TCP probe from gateway | Gateway upstream health checker | Removes target from connection pool independently |
| Gateway passive health | Count 5xx / timeouts on live traffic | Circuit breaker per target | Opens circuit, stops sending, probes to recover |

A Pod can fail only at one layer — for example it passes readiness but its HTTP responses are returning 500s. The gateway's passive health check catches this and opens the circuit even though the EndpointSlice still lists it as ready.

---

### 5. TLS Termination Internals

The gateway handles all TLS negotiation with external clients. Backends typically receive plain HTTP within the cluster network.

```mermaid
flowchart TD
    Client["Client sends TLS ClientHello\nSNI extension: api.example.com\nSupported: TLS 1.3, 1.2\nCipherSuites: AES-256-GCM-SHA384\nAES-128-GCM-SHA256, ChaCha20-Poly1305"]

    SNI["Gateway TLS stack\nExtract SNI from ClientHello\nbefore decryption — SNI is plaintext"]

    CertLookup["Certificate store\n(built from Kubernetes Secrets)\n─────────────────────────────\napi.example.com       → cert-A (exact match ✅)\n*.internal.example.com → cert-B (wildcard)\nDefault fallback       → cert-C"]

    Handshake["TLS 1.3 Handshake\n① Server → Certificate + CertificateVerify\n② Client verifies cert chain against trust store\n③ ECDH key exchange (X25519)\n④ Both sides derive session keys\n⑤ Finished messages — encrypted channel established"]

    HTTP["Plaintext HTTP/1.1 or HTTP/2\nGateway can now read headers, path, body"]

    BackendHTTP["Plain HTTP to backend Pod\nNo TLS overhead within cluster\nNetwork trust boundary: cluster network\nBackend receives: http://10.244.2.15:8080"]

    BackendMTLS["mTLS to backend Pod\nGateway presents its own client cert\nBackend verifies gateway identity\nUse case: zero-trust, PCI-DSS, regulated workloads"]

    CertRotate["cert-manager issues new cert\nStores in Secret/gateway-tls\nController WATCHes Secret → pushes new cert\nKong hot-reloads — no dropped connections ✅"]

    Client --> SNI
    SNI --> CertLookup
    CertLookup -->|"exact SNI match → cert-A"| Handshake
    Handshake --> HTTP
    HTTP --> BackendHTTP
    HTTP --> BackendMTLS
    CertRotate -.->|"cert auto-rotation"| CertLookup
```

---

### 6. Request and Response Transformation — Before vs After

What arrives from the client vs what the backend Pod actually receives, and what the backend sends vs what the client gets back:

```
REQUEST: Client → Gateway → Backend Pod
────────────────────────────────────────────────────────────────────
INBOUND (from client)                   OUTBOUND (to backend)
────────────────────────────────────────────────────────────────────
GET /v1/app/users?page=2                GET /app/users?page=2&source=gw
Host: api.example.com                   Host: user-service
Authorization: Bearer eyJ...            (removed — gateway strips token)
X-User-Agent: MobileApp/2.1             X-User-Agent: MobileApp/2.1
                                        X-User-ID: user-123         ← from JWT sub claim
                                        X-User-Email: a@example.com ← from JWT email claim
                                        X-User-Roles: user,premium  ← from JWT roles claim
                                        X-Request-ID: 7f8d-9e2a     ← generated UUID
                                        X-Forwarded-For: 203.0.113.50  ← original client IP
                                        X-Forwarded-Proto: https    ← original scheme
                                        X-Kong-Request-Start: 1683648000.123

RESPONSE: Backend Pod → Gateway → Client
────────────────────────────────────────────────────────────────────
INBOUND (from backend)                  OUTBOUND (to client)
────────────────────────────────────────────────────────────────────
HTTP/1.1 200 OK                         HTTP/1.1 200 OK
Content-Type: application/json          Content-Type: application/json
Server: Express                         Server: (stripped — don't expose backend tech)
X-Powered-By: Express                   X-Powered-By: (stripped)
                                        X-RateLimit-Limit: 100      ← added by rate plugin
                                        X-RateLimit-Remaining: 54   ← added by rate plugin
                                        X-Kong-Upstream-Latency: 38 ← ms to backend
                                        X-Kong-Proxy-Latency: 4     ← ms in gateway
                                        Access-Control-Allow-Origin: https://app.example.com
                                        Strict-Transport-Security: max-age=31536000
                                        X-Request-ID: 7f8d-9e2a     ← echo for client tracing
```

**Why the gateway strips `Authorization` before forwarding:**  
The JWT contains the user's identity and scopes. Once the gateway has verified it and injected the claims as `X-User-*` headers, the raw token is no longer needed by the backend — and forwarding it would be a security risk if a backend were ever compromised or misconfigured to log request headers.

**Why the gateway strips `Server` and `X-Powered-By` from responses:**  
These headers reveal backend technology (`Express`, `nginx/1.18.0`, `Python/3.11`). Removing them at the gateway prevents fingerprinting by attackers without requiring every backend service to configure header removal individually.

---

## Gateway API (Kubernetes Standard)

**Kubernetes Gateway API** is the next-generation Ingress, standardizing advanced routing.

### Resources

```
GatewayClass  (Infrastructure)
     │
     └─→ Gateway (Entrypoint)
              │
              ├─→ HTTPRoute (HTTP routing)
              ├─→ GRPCRoute (gRPC routing)
              ├─→ TCPRoute (TCP routing)
              └─→ TLSRoute (TLS routing)
```

### Example: Multi-Team Routing

```yaml
# cluster-admin creates GatewayClass
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: kong
spec:
  controllerName: konghq.com/kic-gateway-controller
---
# cluster-admin creates Gateway (shared infrastructure)
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: shared-gateway
  namespace: infra
spec:
  gatewayClassName: kong
  listeners:
  - name: https
    port: 443
    protocol: HTTPS
    tls:
      mode: Terminate
      certificateRefs:
      - kind: Secret
        name: gateway-tls
    allowedRoutes:
      namespaces:
        from: Selector
        selector:
          matchLabels:
            gateway-access: allowed  # Which namespaces can attach
---
# Team A attaches their routes (in team-a namespace)
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: users-route
  namespace: team-a
spec:
  parentRefs:
  - name: shared-gateway
    namespace: infra
    sectionName: https
  hostnames:
  - "api.example.com"
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /users
    backendRefs:
    - name: user-service
      port: 8080
  - matches:
    - path:
        type: PathPrefix
        value: /users/profile
    filters:
    - type: RequestHeaderModifier
      requestHeaderModifier:
        add:
        - name: X-User-Service-Version
          value: "v2"
    backendRefs:
    - name: user-service-v2
      port: 8080
      weight: 100
---
# Team B attaches their routes (in team-b namespace)
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: orders-route
  namespace: team-b
spec:
  parentRefs:
  - name: shared-gateway
    namespace: infra
    sectionName: https
  hostnames:
  - "api.example.com"
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /orders
    backendRefs:
    - name: order-service
      port: 8080
```

### Traffic Weighting (Canary Deployments)

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: payments-canary
spec:
  parentRefs:
  - name: shared-gateway
    namespace: infra
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /payments
    backendRefs:
    - name: payment-service-v1    # Stable version
      port: 8080
      weight: 90                   # 90% of traffic
    - name: payment-service-v2    # Canary version
      port: 8080
      weight: 10                   # 10% of traffic (testing)
```

#### **Traffic Distribution Flow**

```
100 Requests arrive at /payments endpoint

┌──────────────────────────────────────────────────────────────┐
│ GATEWAY LOAD BALANCING WITH WEIGHTS                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ Total weight: 90 + 10 = 100                                 │
│                                                              │
│ For each request, gateway generates random number [0-99]:  │
│                                                              │
│ Request 1: random = 42                                      │
│   ┌─ IF random < 90 (0-89):                                │
│   │    ✅ Route to payment-service-v1                      │
│   │    Selected: ✅                                         │
│   └─ ELSE (90-99):                                          │
│       ❌ Route to payment-service-v2                        │
│                                                              │
│ Request 2: random = 95                                      │
│   ┌─ IF random < 90:                                        │
│   │    ❌ Route to payment-service-v1                       │
│   └─ ELSE (90-99):                                          │
│       ✅ Route to payment-service-v2 (Canary!)             │
│       Selected: ✅                                          │
│                                                              │
│ Over 100 requests:                                          │
│   ~90 requests → payment-service-v1 (stable)               │
│   ~10 requests → payment-service-v2 (canary)               │
│                                                              │
│ Backend Selection After Weight Decision:                   │
│   payment-service-v1 endpoints:                             │
│     • 10.244.1.10:8080 (pod 1)  ┐                          │
│     • 10.244.1.11:8080 (pod 2)  ├─ Round-robin within v1   │
│     • 10.244.1.12:8080 (pod 3)  ┘                          │
│                                                              │
│   payment-service-v2 endpoints:                             │
│     • 10.244.2.20:8080 (pod 1)  ┐                          │
│     • 10.244.2.21:8080 (pod 2)  ┘─ Round-robin within v2   │
└──────────────────────────────────────────────────────────────┘

Progressive Canary Rollout Strategy:
  
  Day 1:  v1=95%, v2=5%   (Deploy canary, minimal risk)
  Day 2:  v1=90%, v2=10%  (If no errors, increase)
  Day 3:  v1=70%, v2=30%  (Monitor metrics)
  Day 4:  v1=50%, v2=50%  (Equal distribution)
  Day 5:  v1=20%, v2=80%  (Almost complete)
  Day 6:  v1=0%,  v2=100% (Full rollout, remove v1)
```

#### **Canary Validation Flow with Rollback**

```
┌────────────────────────────────────────────────────────────────┐
│ MONITORING CANARY DEPLOYMENT                                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ Metrics collected for payment-service-v2 (10% traffic):       │
│   • Error rate: 0.5% (acceptable < 1%)  ✅                    │
│   • P95 latency: 120ms (acceptable < 200ms)  ✅               │
│   • Success rate: 99.5%  ✅                                    │
│   • CPU usage: 45% (normal)  ✅                               │
│                                                                │
│ Decision: ✅ Healthy → Increase to 30%                        │
│                                                                │
│ ─────────────────────────────────────────────────────────────  │
│                                                                │
│ After increasing to 30%:                                       │
│   • Error rate: 5% (threshold exceeded!)  ❌                  │
│   • P95 latency: 850ms (too slow!)  ❌                        │
│   • Database connection errors appearing                      │
│                                                                │
│ Decision: ❌ UNHEALTHY → ROLLBACK IMMEDIATELY                 │
│                                                                │
│ Automated Rollback Actions:                                   │
│ 1. Update HTTPRoute:                                           │
│    weight: v1=100%, v2=0%  (stop sending to canary)          │
│                                                                │
│ 2. Alert sent to team:                                         │
│    "Canary rollback triggered: Error rate exceeded threshold" │
│                                                                │
│ 3. Scale down v2 deployment:                                   │
│    kubectl scale deployment payment-service-v2 --replicas=0   │
│                                                                │
│ 4. Investigate logs from failed canary pods                   │
└────────────────────────────────────────────────────────────────┘
```

### Header-Based Routing

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: feature-flag-routing
spec:
  rules:
  # Rule 1: Beta users get new service (priority evaluated first)
  - matches:
    - headers:
      - name: X-User-Group
        value: beta
      - name: X-Feature-Flag
        value: "new-ui"
    backendRefs:
    - name: api-service-v2
      port: 8080
  
  # Rule 2: Internal requests (from other services)
  - matches:
    - headers:
      - name: X-Internal-Request
        value: "true"
    filters:
    - type: RequestHeaderModifier
      requestHeaderModifier:
        add:
        - name: X-Priority
          value: "high"
    backendRefs:
    - name: api-service-internal
      port: 8080
  
  # Rule 3: Default route (no header match)
  - backendRefs:
    - name: api-service-v1
      port: 8080
```

#### **Routing Decision Flow**

```
Incoming Request: GET /api/users
Headers:
  X-User-Group: beta
  X-Feature-Flag: new-ui
  
┌─────────────────────────────────────────────────────┐
│ GATEWAY ROUTE MATCHING LOGIC                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│ 1. Evaluate Rule 1 (beta + feature flag):         │
│    ┌─ Match header X-User-Group: beta             │
│    │   Request has: "beta" ✅                      │
│    │                                                │
│    └─ Match header X-Feature-Flag: "new-ui"       │
│       Request has: "new-ui" ✅                     │
│                                                     │
│    Both conditions match!                          │
│    ✅ Route to: api-service-v2:8080                │
│    ❌ Skip remaining rules                         │
│                                                     │
│ 2. (Not evaluated - already matched)              │
│                                                     │
│ 3. (Not evaluated - already matched)              │
└─────────────────────────────────────────────────────┘
         │
         └─→ Forward to api-service-v2

Alternative scenario - Regular user:
Request headers: (no special headers)
  
  1. Rule 1: ❌ No X-User-Group header → Skip
  2. Rule 2: ❌ No X-Internal-Request header → Skip
  3. Rule 3 (Default): ✅ No match conditions → MATCH
     → Route to api-service-v1:8080
```

## Kong Gateway in Kubernetes

### Installation

```bash
# Install Kong with Helm
helm install kong kong/ingress --namespace kong --create-namespace \
  --set controller.ingressController.installCRDs=false
```

### Kong Ingress Controller

```yaml
# Classic Ingress (with Kong annotations)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api-ingress
  annotations:
    konghq.com/plugins: rate-limiting,jwt-auth,request-transformer
    konghq.com/strip-path: "true"
spec:
  ingressClassName: kong
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /v1/users
        pathType: Prefix
        backend:
          service:
            name: user-service
            port:
              number: 8080
```

### Kong Plugins

```yaml
# Rate Limiting Plugin
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: rate-limiting
plugin: rate-limiting
config:
  minute: 100           # 100 requests per minute
  hour: 1000            # 1000 per hour
  policy: local
  error_code: 429
  error_message: "Rate limit exceeded"
---
# JWT Authentication Plugin
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: jwt-auth
plugin: jwt
config:
  key_claim_name: kid
  claims_to_verify:
  - exp
  - nbf
---
# Request Transformation Plugin
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: request-transformer
plugin: request-transformer
config:
  add:
    headers:
    - "X-Request-ID: $(uuid)"
    - "X-Gateway-Version: 1.0"
  remove:
    headers:
    - "X-Internal-Token"
---
# Apply plugin to Ingress
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  annotations:
    konghq.com/plugins: rate-limiting,jwt-auth,request-transformer
```

## AWS API Gateway + EKS

### Architecture

```
Internet
    │
    ├─→ Route53: api.example.com
    │
    ├─→ AWS API Gateway (HTTP API / REST API)
    │   - JWT Authorizer (AWS Cognito or custom)
    │   - Rate limiting (throttling)
    │   - Usage plans (API keys)
    │   - Request/response mapping
    │
    ├─→ VPC Link (private connectivity)
    │
    └─→ NLB in VPC
        │
        └─→ EKS Services (NodePort or ALB)
```

### VPC Link Integration

```yaml
# In EKS: Expose service via NLB
apiVersion: v1
kind: Service
metadata:
  name: user-service-nlb
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-internal: "true"
spec:
  type: LoadBalancer
  ports:
  - port: 443
    targetPort: 8080
  selector:
    app: user-service
```

```hcl
# Terraform: AWS API Gateway HTTP API
resource "aws_apigatewayv2_api" "main" {
  name          = "production-api"
  protocol_type = "HTTP"
  
  cors_configuration {
    allow_origins = ["https://app.example.com"]
    allow_methods = ["GET", "POST", "PUT", "DELETE"]
    allow_headers = ["Content-Type", "Authorization"]
    max_age       = 300
  }
}

resource "aws_apigatewayv2_vpc_link" "eks" {
  name               = "eks-vpc-link"
  subnet_ids         = var.private_subnet_ids
  security_group_ids = [aws_security_group.vpc_link.id]
}

resource "aws_apigatewayv2_integration" "user_service" {
  api_id             = aws_apigatewayv2_api.main.id
  integration_type   = "HTTP_PROXY"
  integration_uri    = aws_lb_listener.user_service.arn
  connection_type    = "VPC_LINK"
  connection_id      = aws_apigatewayv2_vpc_link.eks.id
  integration_method = "ANY"
}

resource "aws_apigatewayv2_route" "users" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "ANY /users/{proxy+}"
  target    = "integrations/${aws_apigatewayv2_integration.user_service.id}"
  
  authorization_type = "JWT"
  authorizer_id      = aws_apigatewayv2_authorizer.cognito.id
}
```

## Deep Dive: Complete Request Flow (Kong API Gateway)

### End-to-End Request Sequence

```mermaid
sequenceDiagram
    actor Client as Client\n(Web / Mobile / API)
    participant DNS as DNS
    participant LB as Cloud Load Balancer
    participant GWPod as Gateway Pod\n(Kong / Envoy)
    participant RateLimit as Rate Limit Plugin
    participant JWTAuth as JWT Auth Plugin
    participant ReqXfm as Request Transformer
    participant Router as Route Matcher
    participant KProxy as kube-proxy\n(iptables / IPVS)
    participant Svc as user-service\n(ClusterIP)
    participant Pod as user-service Pod

    Client->>DNS: Resolve api.example.com
    DNS-->>Client: 203.0.113.100 (LB IP)
    Client->>LB: HTTPS GET /v1/users\nAuthorization: Bearer <JWT>
    Note over LB: Health-check Gateway Pods\nSelect healthy replica
    LB->>GWPod: Forward TCP stream\nto Kong pod

    Note over GWPod: TLS termination\nDecrypt HTTPS → HTTP

    GWPod->>RateLimit: Run plugin (priority 901)
    Note over RateLimit: Check counter for client IP\nin shared memory
    alt Counter within limit
        RateLimit-->>GWPod: ✅ Allow — counter=46/100
    else Limit exceeded
        RateLimit-->>Client: ❌ 429 Too Many Requests\nRetry-After: 42
    end

    GWPod->>JWTAuth: Run plugin (priority 1005)
    Note over JWTAuth: Decode JWT header\nFetch public key from JWKS\nVerify RS256 signature\nValidate exp / nbf claims
    alt Token valid
        JWTAuth-->>GWPod: ✅ Authenticated\nsub=user-123, roles=premium
    else Token invalid / expired
        JWTAuth-->>Client: ❌ 401 Unauthorized\nWWW-Authenticate: Bearer error=invalid_token
    end

    GWPod->>ReqXfm: Run plugin (priority 800)
    Note over ReqXfm: Add headers from JWT claims:\nX-User-ID: user-123\nX-User-Roles: user,premium\nX-Request-ID: uuid\nStrip /v1 prefix from path\nRemove Authorization header
    ReqXfm-->>GWPod: ✅ Request transformed

    GWPod->>Router: Match route rules
    Note over Router: Match host: api.example.com ✅\nMatch path: /users ✅\nCheck canary header → none\nSelect: user-service-v1\nWeight LB: pick pod endpoint
    Router-->>GWPod: Backend: 10.244.2.15:8080

    GWPod->>KProxy: Connect to ClusterIP\n10.96.80.45:8080
    Note over KProxy: iptables DNAT rule:\nClusterIP:8080 → PodIP:8080\nSelected: 10.244.2.15:8080
    KProxy->>Pod: Forward HTTP request\n(transformed headers)

    Note over Pod: Extract X-User-ID from header\nQuery database (35ms)\nBuild JSON response
    Pod-->>KProxy: HTTP 200 OK JSON
    KProxy-->>GWPod: Response forwarded

    Note over GWPod: Measure upstream latency: 40ms\nAdd CORS headers\nAdd X-Kong-*-Latency headers\nEmit Prometheus metrics\nWrite access log
    GWPod-->>LB: HTTP 200 OK + headers
    LB-->>Client: HTTPS 200 OK\nX-RateLimit-Remaining: 54\nX-Request-ID: uuid\nTotal: ~50ms
```

### Request Flow: Step-by-Step Breakdown

```
Client: GET https://api.example.com/v1/users
Authorization: Bearer eyJhbGciOiJSUzI1NiJ9...
X-User-Agent: MobileApp/2.1
```

---

#### **Phase 1: DNS Resolution & Network Entry**

```
┌─────────────────────────────────────────────────────────┐
│ 1. DNS RESOLUTION                                       │
│    Client queries: api.example.com                      │
│    DNS returns: 203.0.113.100 (LoadBalancer IP)        │
│    Protocol: HTTPS (port 443)                           │
└─────────────────────────────────────────────────────────┘
         │
         ├─→ Client initiates TCP handshake
         ├─→ TLS handshake (client verifies server cert)
         └─→ HTTP request sent over encrypted connection
```

**Component: Cloud LoadBalancer (e.g., AWS ALB/NLB, GCP LB)**
- **Responsibility:** Distribute incoming traffic across Kong gateway pods
- **Decision Point:** Select healthy Kong pod based on algorithm (round-robin, least connections)
- **Health Check:** Pings Kong `/health` endpoint every 5s
- **Condition:** If all Kong pods unhealthy → Return 503 Service Unavailable

---

#### **Phase 2: Kong Gateway Ingress**

```
┌─────────────────────────────────────────────────────────┐
│ 2. LOAD BALANCER → KONG SERVICE                        │
│    LoadBalancer selects pod: kong-gateway-5f7d8-abc123  │
│    Forwards to: Kong Service ClusterIP 10.96.50.10:443 │
│    kube-proxy routes to selected pod                    │
└─────────────────────────────────────────────────────────┘
         │
         ├─→ Arrives at Kong pod (nginx + Lua runtime)
         └─→ Request enters Kong's plugin execution chain
```

**Component: Kubernetes Service (kong-gateway)**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: kong-gateway
spec:
  type: LoadBalancer
  ports:
  - name: https
    port: 443
    targetPort: 8443  # Kong's HTTPS listener
  selector:
    app: kong
```

---

#### **Phase 3: Plugin Execution Chain**

Kong executes plugins in **priority order** (higher priority = runs first).

```
┌─────────────────────────────────────────────────────────────────────┐
│ PLUGIN 1: RATE LIMITING (Priority: 901)                            │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Extract client identifier:                                      │
│    • Option A: IP address = 203.0.113.50                          │
│    • Option B: Header X-API-Key = "key_abc123" (if present)       │
│                                                                     │
│ 2. Check rate limit counter (stored in Kong's shared memory):     │
│    Key: "ratelimit:203.0.113.50:minute"                           │
│    Current: 45 requests                                            │
│    Limit: 100 requests/minute                                      │
│                                                                     │
│ 3. Decision:                                                        │
│    ┌─ IF counter <= limit:                                         │
│    │    ✅ Increment counter → 46                                  │
│    │    ✅ Add headers:                                            │
│    │       X-RateLimit-Limit: 100                                 │
│    │       X-RateLimit-Remaining: 54                              │
│    │       X-RateLimit-Reset: 1683650400                          │
│    │    ✅ ALLOW → Continue to next plugin                        │
│    │                                                                │
│    └─ IF counter > limit:                                          │
│       ❌ REJECT → Return HTTP 429 Too Many Requests                │
│       Response body: {"message": "Rate limit exceeded"}            │
│       Headers: Retry-After: 42                                     │
│       ❌ STOP processing (request never reaches backend)           │
└─────────────────────────────────────────────────────────────────────┘
         │ (Allowed, continue...)
         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PLUGIN 2: JWT AUTHENTICATION (Priority: 1005)                      │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Extract JWT token:                                              │
│    Authorization: Bearer <token>                                   │
│    Token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWI..."      │
│                                                                     │
│ 2. Decode JWT header:                                              │
│    {                                                                │
│      "alg": "RS256",                                               │
│      "typ": "JWT",                                                 │
│      "kid": "key-2023-05"  ← Key ID                               │
│    }                                                                │
│                                                                     │
│ 3. Fetch public key:                                               │
│    • Kong queries configured JWKS endpoint:                        │
│      https://auth.example.com/.well-known/jwks.json               │
│    • Finds key with kid="key-2023-05"                             │
│    • Caches key for 1 hour                                         │
│                                                                     │
│ 4. Verify signature:                                               │
│    ✅ Signature valid using RS256 + public key                     │
│                                                                     │
│ 5. Decode JWT payload:                                             │
│    {                                                                │
│      "sub": "user-123",         ← Subject (user ID)               │
│      "email": "alice@example.com",                                │
│      "roles": ["user", "premium"],                                │
│      "exp": 1683650000,         ← Expiration timestamp            │
│      "nbf": 1683646400,         ← Not before                      │
│      "iat": 1683646400          ← Issued at                       │
│    }                                                                │
│                                                                     │
│ 6. Validate claims:                                                │
│    ┌─ Check expiration:                                            │
│    │   Current time: 1683648000                                    │
│    │   exp: 1683650000                                             │
│    │   ✅ exp > current_time → VALID                              │
│    │   (If expired: ❌ Return 401 Unauthorized)                    │
│    │                                                                │
│    ├─ Check "not before":                                          │
│    │   ✅ nbf <= current_time → VALID                             │
│    │                                                                │
│    └─ Check required claims (configured in plugin):                │
│       ✅ "sub" claim exists                                        │
│       ✅ "roles" claim exists                                      │
│                                                                     │
│ 7. Store JWT data in Kong's context (for later plugins):          │
│    kong.ctx.shared.authenticated_credential = {                    │
│      sub: "user-123",                                              │
│      email: "alice@example.com",                                  │
│      roles: ["user", "premium"]                                   │
│    }                                                                │
│                                                                     │
│ 8. Decision:                                                        │
│    ✅ Authentication successful → Continue to next plugin          │
│    (If any check fails: ❌ Return 401 Unauthorized + WWW-Authenticate header) │
└─────────────────────────────────────────────────────────────────────┘
         │ (Authenticated, continue...)
         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PLUGIN 3: REQUEST TRANSFORMER (Priority: 800)                      │
├─────────────────────────────────────────────────────────────────────┤
│ Purpose: Modify request before forwarding to backend               │
│                                                                     │
│ 1. Add headers:                                                     │
│    ✅ X-User-ID: user-123 (from JWT sub claim)                     │
│    ✅ X-User-Email: alice@example.com                              │
│    ✅ X-User-Roles: user,premium                                   │
│    ✅ X-Request-ID: 7f8d9e2a-4b3c-4d5e-8f9a-1b2c3d4e5f6a (UUID)   │
│    ✅ X-Forwarded-For: 203.0.113.50 (original client IP)          │
│    ✅ X-Kong-Request-Start: 1683648000.123                         │
│                                                                     │
│ 2. Remove headers (security - don't leak to backend):             │
│    ❌ Authorization: Bearer ... (remove sensitive token)           │
│                                                                     │
│ 3. Modify path (if configured):                                    │
│    Original: /v1/users                                             │
│    ✅ Strip "/v1" prefix → /users                                  │
│    (Backend doesn't need to know about versioning)                │
│                                                                     │
│ 4. Add query parameters (if needed):                               │
│    ✅ ?source=api-gateway                                          │
└─────────────────────────────────────────────────────────────────────┘
         │ (Transformed, continue...)
         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PLUGIN 4: ROUTE MATCHING & BACKEND SELECTION                       │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Match route rules (configured in Ingress/HTTPRoute):           │
│    • Host: api.example.com ✅                                      │
│    • Path: /users ✅ (after transformation)                        │
│    • Method: GET ✅                                                 │
│                                                                     │
│ 2. Check canary routing rules:                                     │
│    ┌─ IF header "X-User-Group: beta" exists:                       │
│    │    → Route to user-service-v2:8080 (10% traffic)             │
│    └─ ELSE:                                                         │
│         → Route to user-service-v1:8080 (90% traffic)             │
│                                                                     │
│    Decision: No beta header → user-service-v1                     │
│                                                                     │
│ 3. Resolve backend service:                                        │
│    Service name: user-service                                      │
│    Namespace: production                                           │
│    ClusterIP: 10.96.80.45                                         │
│    Port: 8080                                                      │
│                                                                     │
│ 4. Get backend endpoints (from kube-proxy/iptables):              │
│    Endpoints:                                                       │
│    • 10.244.1.10:8080 (pod user-service-5f7d8-abc)               │
│    • 10.244.2.15:8080 (pod user-service-5f7d8-def)               │
│    • 10.244.3.20:8080 (pod user-service-5f7d8-ghi)               │
│                                                                     │
│ 5. Load balancing (Kong uses round-robin by default):             │
│    Last used: 10.244.1.10                                          │
│    ✅ Select next: 10.244.2.15:8080                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

#### **Phase 4: Forwarding to Backend Service**

```
┌─────────────────────────────────────────────────────────────────────┐
│ 4. KONG → BACKEND SERVICE (user-service)                           │
├─────────────────────────────────────────────────────────────────────┤
│ Kong opens new HTTP connection to backend:                         │
│                                                                     │
│ GET /users HTTP/1.1                                                │
│ Host: user-service:8080                                            │
│ X-User-ID: user-123                                                │
│ X-User-Email: alice@example.com                                    │
│ X-User-Roles: user,premium                                         │
│ X-Request-ID: 7f8d9e2a-4b3c-4d5e-8f9a-1b2c3d4e5f6a               │
│ X-Forwarded-For: 203.0.113.50                                     │
│ X-Kong-Request-Start: 1683648000.123                              │
│                                                                     │
│ → Sent to pod IP: 10.244.2.15:8080                                │
│                                                                     │
│ Timeout configured: 60 seconds                                     │
│ ┌─ IF backend doesn't respond within 60s:                          │
│ │    ❌ Kong returns 504 Gateway Timeout                           │
│ │    Circuit breaker may trip after N consecutive failures         │
│ └─                                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

**Component: user-service Pod**
```
┌─────────────────────────────────────────────────────────────────────┐
│ 5. BACKEND SERVICE PROCESSING                                      │
├─────────────────────────────────────────────────────────────────────┤
│ user-service receives request:                                     │
│                                                                     │
│ 1. Extract user context from headers:                              │
│    userID = request.headers["X-User-ID"]  // "user-123"           │
│    roles = request.headers["X-User-Roles"]  // "user,premium"     │
│                                                                     │
│ 2. Check permissions (already authenticated by gateway):           │
│    ✅ No need to verify JWT again                                  │
│    ✅ Trust gateway-added headers                                  │
│                                                                     │
│ 3. Query database:                                                  │
│    SELECT * FROM users WHERE id = 'user-123'                       │
│    Query time: 35ms                                                 │
│                                                                     │
│ 4. Build response:                                                  │
│    {                                                                │
│      "id": "user-123",                                             │
│      "name": "Alice Johnson",                                      │
│      "email": "alice@example.com",                                │
│      "premium": true                                               │
│    }                                                                │
│                                                                     │
│ 5. Return to Kong:                                                  │
│    HTTP 200 OK                                                      │
│    Content-Type: application/json                                  │
│    Response time: 40ms (5ms app + 35ms DB)                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

#### **Phase 5: Response Processing**

```
┌─────────────────────────────────────────────────────────────────────┐
│ 6. KONG RECEIVES BACKEND RESPONSE                                  │
├─────────────────────────────────────────────────────────────────────┤
│ Response from user-service received after 40ms                     │
│                                                                     │
│ 1. Calculate latencies:                                            │
│    Kong request start: 1683648000.123                              │
│    Backend response: 1683648000.163                                │
│    Upstream latency: 40ms                                           │
│    Total latency: 45ms (includes plugin processing)               │
│                                                                     │
│ 2. Response plugins execute (reverse order):                       │
│    • Response transformer: Add CORS headers                        │
│    • Logging plugin: Log request/response metrics                 │
│                                                                     │
│ 3. Add observability headers:                                      │
│    ✅ X-Kong-Upstream-Latency: 40                                  │
│    ✅ X-Kong-Proxy-Latency: 5                                      │
│    ✅ X-Kong-Response-Latency: 45                                  │
│    ✅ X-Request-ID: 7f8d9e2a-4b3c-4d5e-8f9a-1b2c3d4e5f6a         │
│                                                                     │
│ 4. Add CORS headers (if configured):                               │
│    ✅ Access-Control-Allow-Origin: https://app.example.com         │
│    ✅ Access-Control-Allow-Methods: GET, POST, PUT, DELETE         │
│                                                                     │
│ 5. Emit metrics and logs:                                          │
│    • Prometheus: http_request_duration_seconds{route="/users"}=0.045 │
│    • Access log:                                                    │
│      203.0.113.50 [2023-05-09T10:00:00Z] "GET /v1/users" 200 45ms │
│                                                                     │
│ 6. Forward response to client through TLS connection              │
└─────────────────────────────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 7. CLIENT RECEIVES RESPONSE                                        │
├─────────────────────────────────────────────────────────────────────┤
│ HTTP/1.1 200 OK                                                     │
│ Content-Type: application/json                                      │
│ X-Request-ID: 7f8d9e2a-4b3c-4d5e-8f9a-1b2c3d4e5f6a               │
│ X-RateLimit-Remaining: 54                                          │
│ X-Kong-Response-Latency: 45                                        │
│                                                                     │
│ {                                                                   │
│   "id": "user-123",                                                │
│   "name": "Alice Johnson",                                         │
│   "email": "alice@example.com",                                   │
│   "premium": true                                                  │
│ }                                                                   │
│                                                                     │
│ Total end-to-end latency: ~50ms                                    │
│ • DNS: 2ms                                                          │
│ • TLS handshake: 3ms                                               │
│ • Kong processing: 45ms                                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Error Handling Flows

#### **Scenario 1: Rate Limit Exceeded**

```
Client Request (46th request in same minute)
     │
     ├─→ LoadBalancer → Kong
     │
     ├─→ Plugin 1: Rate Limiting
     │       Counter: 100/100 (limit reached)
     │       ❌ REJECT
     │
     └─→ Immediate Response to Client:
              HTTP 429 Too Many Requests
              Retry-After: 42
              X-RateLimit-Limit: 100
              X-RateLimit-Remaining: 0
              X-RateLimit-Reset: 1683650400
              
              {"message": "API rate limit exceeded"}

Backend service never contacted ✅ (saves resources)
```

#### **Scenario 2: Invalid JWT Token**

```
Client Request with expired JWT
     │
     ├─→ LoadBalancer → Kong
     │
     ├─→ Plugin 1: Rate Limiting ✅ (passes)
     │
     ├─→ Plugin 2: JWT Auth
     │       Decode token
     │       Check expiration:
     │         exp: 1683640000 (already expired)
     │         current: 1683648000
     │       ❌ REJECT (token expired)
     │
     └─→ Response to Client:
              HTTP 401 Unauthorized
              WWW-Authenticate: Bearer error="invalid_token",
                error_description="Token is expired"
              
              {"message": "Token expired"}

Backend service never contacted ✅
```

#### **Scenario 3: Backend Service Timeout**

```
Client Request
     │
     ├─→ LoadBalancer → Kong
     ├─→ Plugin Chain ✅ (all pass)
     │
     ├─→ Forward to user-service
     │       Kong waits... 60 seconds timeout configured
     │       
     │       ┌─ Backend is slow/stuck
     │       │  30s... 40s... 50s... 60s...
     │       │  ❌ Timeout reached
     │       │
     │       └─→ Kong closes connection
     │
     ├─→ Circuit Breaker Check:
     │       Consecutive failures: 4
     │       Threshold: 5
     │       Status: Still CLOSED (allowing requests)
     │       (If 5th failure: Circuit OPENS → fail fast for 30s)
     │
     └─→ Response to Client:
              HTTP 504 Gateway Timeout
              X-Kong-Upstream-Latency: 60000
              
              {"message": "Upstream service timeout"}
```

#### **Scenario 4: Backend Service Unavailable**

```
Client Request
     │
     ├─→ LoadBalancer → Kong
     ├─→ Plugin Chain ✅
     │
     ├─→ Route to user-service
     │       Attempt connection to: 10.244.2.15:8080
     │       ❌ Connection refused (pod crashed)
     │
     ├─→ Retry with next endpoint:
     │       Attempt: 10.244.1.10:8080
     │       ❌ Connection refused
     │
     ├─→ Retry with next endpoint:
     │       Attempt: 10.244.3.20:8080
     │       ❌ Connection refused
     │
     ├─→ All endpoints failed
     │
     └─→ Response to Client:
              HTTP 503 Service Unavailable
              Retry-After: 30
              
              {"message": "Service temporarily unavailable"}
```

## Observability

### Distributed Tracing

```yaml
# Kong + OpenTelemetry
apiVersion: configuration.konghq.com/v1
kind: KongPlugin
metadata:
  name: opentelemetry
plugin: opentelemetry
config:
  endpoint: http://otel-collector:4317
  resource_attributes:
    service.name: api-gateway
  propagation:
    extract: [w3c, b3, jaeger]
    inject: [w3c]
```

**Trace flow:**
```
Client Request → Kong (trace start, generates trace-id)
     │              └─→ traceId: abc123, spanId: 001
     ↓
user-service (receives trace headers, creates child span)
     │              └─→ traceId: abc123, spanId: 002
     ↓
database call (database span)
     │              └─→ traceId: abc123, spanId: 003
     ↓
Jaeger/Zipkin visualizes complete trace
```

## Best Practices

1. **Centralize Auth** at the gateway, don't replicate in every service
2. **Rate limit per user** not just per IP (use API keys or JWT claims)
3. **Version your APIs** (`/v1/`, `/v2/`) at the gateway level
4. **Circuit breaking** - fail fast when downstream services are slow
5. **Request timeouts** - always set timeouts at gateway
6. **API Documentation** - Use Swagger/OpenAPI spec with gateway
7. **Monitor latency at each hop** - Gateway, service, database separately

---

**Next:** [Admission Controllers & Webhooks](13-admission-controllers.md)
