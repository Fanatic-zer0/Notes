# DNS Resolution in Kubernetes

## Overview

Kubernetes uses **CoreDNS** (or kube-dns in older versions) to provide service discovery through DNS. Every pod automatically gets DNS resolution capabilities to find services by name.

## Architecture

```
Pod wants to reach "backend-service"
     │
     ├─→ 1. DNS query to nameserver (10.96.0.10)
     │       Configured in pod's /etc/resolv.conf
     │
     ├─→ 2. CoreDNS pod receives query
     │       backend-service.default.svc.cluster.local
     │
     ├─→ 3. CoreDNS looks up in cluster database
     │       Queries kube-apiserver for Service info
     │
     ├─→ 4. Returns Service ClusterIP or Pod IPs
     │       10.96.20.30 (for ClusterIP service)
     │       or [10.244.1.5, 10.244.2.8] (for headless)
     │
     └─→ 5. Pod connects to returned IP
              Uses kube-proxy for load balancing
```

## CoreDNS Deployment

```yaml
apiVersion: v1
kind: Service
metadata:
  name: kube-dns
  namespace: kube-system
spec:
  clusterIP: 10.96.0.10  # Fixed IP, used by all pods
  selector:
    k8s-app: kube-dns
  ports:
  - name: dns
    port: 53
    protocol: UDP
  - name: dns-tcp
    port: 53
    protocol: TCP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coredns
  namespace: kube-system
spec:
  replicas: 2  # HA setup
  selector:
    matchLabels:
      k8s-app: kube-dns
  template:
    metadata:
      labels:
        k8s-app: kube-dns
    spec:
      containers:
      - name: coredns
        image: coredns/coredns:1.10.1
        args: [ "-conf", "/etc/coredns/Corefile" ]
        volumeMounts:
        - name: config-volume
          mountPath: /etc/coredns
      volumes:
      - name: config-volume
        configMap:
          name: coredns
```

## DNS Names in Kubernetes

### Service DNS Names

**Pattern:** `<service-name>.<namespace>.svc.<cluster-domain>`

| Short Name | Fully Qualified Domain Name (FQDN) | Resolves To |
|------------|-------------------------------------|-------------|
| `backend-service` | `backend-service.default.svc.cluster.local` | Service ClusterIP |
| `backend-service.default` | `backend-service.default.svc.cluster.local` | Service ClusterIP |
| `backend-service.default.svc` | `backend-service.default.svc.cluster.local` | Service ClusterIP |

**From same namespace:**
```bash
curl http://backend-service:8080  # Works!
```

**From different namespace:**
```bash
curl http://backend-service.production:8080  # Must specify namespace
```

### Pod DNS Names

**Pattern:** `<pod-ip-with-dashes>.<namespace>.pod.<cluster-domain>`

For pod with IP `10.244.1.5`:
```
10-244-1-5.default.pod.cluster.local
```

**Note:** Pod DNS is rarely used directly. Services are preferred.

### StatefulSet Pod DNS

**Pattern:** `<pod-name>.<headless-service-name>.<namespace>.svc.<cluster-domain>`

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mysql
spec:
  clusterIP: None  # Headless
  selector:
    app: mysql
  ports:
  - port: 3306
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mysql
spec:
  serviceName: mysql  # Links to headless service
  replicas: 3
  selector:
    matchLabels:
      app: mysql
  template:
    metadata:
      labels:
        app: mysql
    spec:
      containers:
      - name: mysql
        image: mysql:8.0
```

**Stable DNS names:**
- `mysql-0.mysql.default.svc.cluster.local`
- `mysql-1.mysql.default.svc.cluster.local`
- `mysql-2.mysql.default.svc.cluster.local`

These persist even if pods are rescheduled!

## DNS Resolution Flow

### Step-by-Step Example

```
┌────────────────────────────────────────────────────────────────┐
│ Pod: frontend-pod-abc123                                       │
│ Container executes: curl http://api-service:8080/users        │
└───────────────────┬────────────────────────────────────────────┘
                    │ 1. Check /etc/resolv.conf
                    ↓
┌────────────────────────────────────────────────────────────────┐
│ /etc/resolv.conf (auto-injected by kubelet)                    │
│   nameserver 10.96.0.10                                        │
│   search default.svc.cluster.local svc.cluster.local cluster.local │
│   options ndots:5                                              │
└───────────────────┬────────────────────────────────────────────┘
                    │ 2. DNS query construction
                    ↓
┌────────────────────────────────────────────────────────────────┐
│ DNS Query Logic (ndots:5)                                      │
│ "api-service" has 0 dots (< 5), so try search domains first:  │
│   Try 1: api-service.default.svc.cluster.local                │
│   Try 2: api-service.svc.cluster.local                        │
│   Try 3: api-service.cluster.local                            │
│   Try 4: api-service (absolute)                               │
└───────────────────┬────────────────────────────────────────────┘
                    │ 3. First query
                    ↓
┌────────────────────────────────────────────────────────────────┐
│ CoreDNS Pod (10.96.0.10:53)                                    │
│ Query: api-service.default.svc.cluster.local A?               │
└───────────────────┬────────────────────────────────────────────┘
                    │ 4. CoreDNS plugin chain
                    ↓
┌────────────────────────────────────────────────────────────────┐
│ CoreDNS Corefile Processing                                    │
│   ┌────────────────────────────────────────────────────────┐  │
│   │ kubernetes plugin                                      │  │
│   │ - Matches *.cluster.local                             │  │
│   │ - Queries kube-apiserver                              │  │
│   │ - GET /api/v1/namespaces/default/services/api-service │  │
│   └────────────────────────────────────────────────────────┘  │
└───────────────────┬────────────────────────────────────────────┘
                    │ 5. API Server response
                    ↓
┌────────────────────────────────────────────────────────────────┐
│ Service Object (from etcd)                                     │
│   Name: api-service                                            │
│   Namespace: default                                           │
│   ClusterIP: 10.96.20.30                                       │
│   Type: ClusterIP                                              │
└───────────────────┬────────────────────────────────────────────┘
                    │ 6. DNS response
                    ↓
┌────────────────────────────────────────────────────────────────┐
│ DNS Response to Pod                                            │
│   api-service.default.svc.cluster.local. 30 IN A 10.96.20.30  │
└───────────────────┬────────────────────────────────────────────┘
                    │ 7. Connection established
                    ↓
                Pod connects to 10.96.20.30:8080
```

## CoreDNS Corefile

**ConfigMap:** `coredns` in `kube-system` namespace

```
.:53 {
    errors
    health {
       lameduck 5s
    }
    ready
    kubernetes cluster.local in-addr.arpa ip6.arpa {
       pods insecure
       fallthrough in-addr.arpa ip6.arpa
       ttl 30
    }
    prometheus :9153
    forward . /etc/resolv.conf {
       max_concurrent 1000
    }
    cache 30
    loop
    reload
    loadbalance
}
```

### Plugin Explanations

| Plugin | Purpose |
|--------|---------|
| `errors` | Log errors |
| `health` | Health check endpoint at :8080/health |
| `ready` | Readiness check at :8181/ready |
| `kubernetes` | K8s service discovery (main plugin) |
| `prometheus` | Metrics at :9153/metrics |
| `forward` | Forward external queries to upstream DNS |
| `cache` | Cache responses (TTL 30s) |
| `loop` | Detect forwarding loops |
| `reload` | Auto-reload on Corefile changes |
| `loadbalance` | Round-robin A/AAAA records |

## ndots Configuration

**Problem:** Short names trigger many DNS queries.

`curl http://api-service:8080` with `ndots:5` generates:
1. api-service.default.svc.cluster.local (SUCCESS)
2. ~~api-service.svc.cluster.local~~
3. ~~api-service.cluster.local~~
4. ~~api-service~~

**Solution 1:** Use FQDN with trailing dot
```bash
curl http://api-service.default.svc.cluster.local.:8080
# Only 1 DNS query!
```

**Solution 2:** Reduce ndots (pod-level)
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: frontend-pod
spec:
  dnsConfig:
    options:
    - name: ndots
      value: "2"  # Default is 5
  containers:
  - name: app
    image: myapp:latest
```

## Headless Service DNS

**Headless service** (clusterIP: None) returns pod IPs directly.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: api-headless
spec:
  clusterIP: None
  selector:
    app: api
  ports:
  - port: 8080
```

**DNS Query:**
```bash
nslookup api-headless.default.svc.cluster.local

# Returns ALL pod IPs:
Name:    api-headless.default.svc.cluster.local
Address: 10.244.1.5
Address: 10.244.2.8
Address: 10.244.3.12
```

**Use Cases:**
- StatefulSet stable network IDs
- Custom client-side load balancing
- Service meshes (Istio/Linkerd)
- Databases with replication

## External Name Service

**Map external services** to internal DNS names.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: external-db
spec:
  type: ExternalName
  externalName: db.external.com
```

**DNS Query:**
```bash
nslookup external-db.default.svc.cluster.local

# Returns CNAME:
external-db.default.svc.cluster.local
  canonical name = db.external.com
```

**Use Cases:**
- Migration from external to internal services
- Multi-cloud service references
- Development/production environment switching

## Custom DNS Configuration

### Custom Nameservers (Pod-level)
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: custom-dns-pod
spec:
  dnsPolicy: None  # Override default
  dnsConfig:
    nameservers:
    - 8.8.8.8
    - 8.8.4.4
    searches:
    - custom.local
    options:
    - name: ndots
      value: "2"
  containers:
  - name: app
    image: myapp:latest
```

### DNS Policies

| Policy | Behavior |
|--------|----------|
| `Default` | Inherits node's DNS config |
| `ClusterFirst` | K8s DNS first, fallback to node (default) |
| `ClusterFirstWithHostNet` | For pods with hostNetwork: true |
| `None` | Custom DNS config only |

## DNS Caching

### CoreDNS Cache (Cluster-level)
```
cache 30  # Cache for 30 seconds
```

### NodeLocal DNSCache (Optional)

**Problem:** All DNS queries go to CoreDNS pods (network overhead)

**Solution:** Run DNS cache on every node

```
┌─────────────────────────────────────────────────────────┐
│ Node                                                    │
│   ┌─────────────┐                                       │
│   │ Pod         │                                       │
│   │ nameserver: │                                       │
│   │ 169.254.20.10 ─────┐                               │
│   └─────────────┘       │                               │
│                         ↓                                │
│   ┌─────────────────────────────────┐                   │
│   │ NodeLocal DNSCache (169.254.20.10) │               │
│   │ - Local cache on each node      │                   │
│   │ - Falls back to CoreDNS         │                   │
│   └─────────────────────────────────┘                   │
│                         │                                │
└─────────────────────────┼────────────────────────────────┘
                          ↓
                    CoreDNS (10.96.0.10)
```

**Benefits:**
- Reduced latency (~1ms vs ~5ms)
- Lower CoreDNS load
- Better availability

**Deploy:**
```bash
kubectl apply -f https://k8s.io/examples/admin/dns/nodelocaldns.yaml
```

## DNS Performance Optimization

### 1. Use FQDNs with Trailing Dots
```go
// Bad (5 DNS queries with ndots:5)
http.Get("http://api-service:8080")

// Good (1 DNS query)
http.Get("http://api-service.default.svc.cluster.local.:8080")
```

### 2. Application-Level Caching
```go
// Cache DNS results in your app
resolver := &net.Resolver{
    PreferGo: true,
    Dial: func(ctx context.Context, network, address string) (net.Conn, error) {
        // Custom caching logic
    },
}
```

### 3. Reduce ndots
```yaml
dnsConfig:
  options:
  - name: ndots
    value: "1"  # For apps that mostly use FQDNs
```

### 4. Deploy NodeLocal DNSCache
See above section.

## Troubleshooting DNS

### Common Issues

**1. Service Not Resolving**
```bash
# Check CoreDNS pods
kubectl get pods -n kube-system -l k8s-app=kube-dns

# Check CoreDNS logs
kubectl logs -n kube-system -l k8s-app=kube-dns

# Verify service exists
kubectl get service api-service
```

**2. Test from Debug Pod**
```bash
kubectl run -it --rm debug --image=busybox --restart=Never -- sh

# Inside pod
nslookup kubernetes.default
nslookup api-service.default.svc.cluster.local
cat /etc/resolv.conf
```

**3. Check CoreDNS Service**
```bash
kubectl get service -n kube-system kube-dns
# Should have ClusterIP (usually 10.96.0.10)
```

**4. Verify DNS Resolution**
```bash
# From a pod
kubectl exec -it frontend-pod -- nslookup api-service

# Should return Service ClusterIP
```

### DNS Query Tracing

Enable CoreDNS log plugin:
```
.:53 {
    log  # Add this line
    errors
    ...
}
```

```bash
# Reload CoreDNS
kubectl rollout restart -n kube-system deployment/coredns

# Watch logs
kubectl logs -n kube-system -l k8s-app=kube-dns -f
```

## DNS Metrics

**CoreDNS Prometheus Metrics:**
- `coredns_dns_requests_total` - Total DNS requests
- `coredns_dns_responses_total` - Total responses by rcode
- `coredns_cache_hits_total` - Cache hit count
- `coredns_forward_requests_total` - Forwarded queries

```bash
# Query CoreDNS metrics
kubectl port-forward -n kube-system svc/kube-dns 9153:9153
curl http://localhost:9153/metrics
```

---

**Next:** [Network Policies](05-network-policies.md)
