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
