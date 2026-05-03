# DNS and CoreDNS in Kubernetes

## Overview

Every Kubernetes pod needs to discover services without hardcoding IP addresses. **CoreDNS** provides cluster-internal DNS, allowing pods to find each other by name. This guide covers DNS fundamentals and CoreDNS configuration.

## 1. DNS Fundamentals

### What is DNS?

DNS (Domain Name System) translates human-readable names to IP addresses.

```
Application: connect to "my-service"
                  │
                  ▼ DNS Query
            DNS Server
                  │
                  ▼ DNS Response
            10.96.100.50 ← ClusterIP
```

### DNS Query Flow

```
Pod (app process)
    │
    │ 1. App calls getaddrinfo("my-service")
    ▼
/etc/resolv.conf
    │
    │ 2. Reads nameserver and search domains
    │    nameserver 10.96.0.10 (CoreDNS ClusterIP)
    │    search default.svc.cluster.local svc.cluster.local cluster.local
    ▼
CoreDNS (10.96.0.10:53)
    │
    │ 3. Try: my-service.default.svc.cluster.local
    │    Found! Return 10.96.100.50
    ▼
Pod receives: 10.96.100.50
```

### DNS Record Types

| Type | Purpose | Example |
|------|---------|---------|
| A | IPv4 address | my-svc → 10.96.100.50 |
| AAAA | IPv6 address | my-svc → fd00::1 |
| CNAME | Alias to another name | www → my-svc |
| SRV | Service location (port+host) | _http._tcp.my-svc → host:80 |
| PTR | Reverse DNS (IP→name) | 10.96.100.50 → my-svc.default... |
| NS | Name server | cluster.local → coredns-server |

### DNS in Linux

```bash
# Pod's /etc/resolv.conf (auto-configured by kubelet)
cat /etc/resolv.conf

# OUTPUT:
nameserver 10.96.0.10
search default.svc.cluster.local svc.cluster.local cluster.local
options ndots:5

# nameserver: CoreDNS ClusterIP
# search: Appended when name has < 5 dots
# ndots:5: Names with <5 dots get search domains appended first
```

### The ndots:5 Behavior

```
Query: "my-service"  (0 dots < 5)
  1. Try: my-service.default.svc.cluster.local → FOUND → 10.96.100.50

Query: "my-service.default"  (1 dot < 5)
  1. Try: my-service.default.default.svc.cluster.local → NXDOMAIN
  2. Try: my-service.default.svc.cluster.local → FOUND → 10.96.100.50

Query: "httpbin.org"  (1 dot < 5)
  1. Try: httpbin.org.default.svc.cluster.local → NXDOMAIN
  2. Try: httpbin.org.svc.cluster.local → NXDOMAIN
  3. Try: httpbin.org.cluster.local → NXDOMAIN
  4. Try: httpbin.org → FOUND → 54.23.x.x (internet)
```

**Performance impact:** External lookups cause 3 extra DNS queries. To avoid:
```
curl http://httpbin.org.  # Trailing dot = absolute name (no search appended)
```

## 2. Kubernetes DNS Name Formats

### Service DNS Names

```
Format: <service-name>.<namespace>.svc.<cluster-domain>

Examples:
  my-service.default.svc.cluster.local
  mysql.database.svc.cluster.local
  redis.cache.svc.cluster.local

Short forms (within same namespace):
  my-service                              → works within same namespace
  my-service.default                      → works from any namespace
  my-service.default.svc                 → explicit SVC
  my-service.default.svc.cluster.local   → fully qualified (FQDN)
```

### Pod DNS Names

```
Format: <pod-ip-dashes>.<namespace>.pod.<cluster-domain>

Example: Pod with IP 10.244.1.5 in namespace "default"
  10-244-1-5.default.pod.cluster.local

# Test from another pod
nslookup 10-244-1-5.default.pod.cluster.local
```

### StatefulSet Pod DNS

```
Format: <pod-name>.<service-name>.<namespace>.svc.<cluster-domain>

Example: StatefulSet "mysql" in "database" namespace:
  mysql-0.mysql.database.svc.cluster.local
  mysql-1.mysql.database.svc.cluster.local
  mysql-2.mysql.database.svc.cluster.local

# This enables stable network identities!
```

### Headless Service DNS

```yaml
# Headless service (clusterIP: None) returns pod IPs directly
apiVersion: v1
kind: Service
metadata:
  name: mysql-headless
spec:
  clusterIP: None  # Headless!
  selector:
    app: mysql
```

```bash
# Regular service DNS
nslookup my-service.default.svc.cluster.local
# Returns: 10.96.100.50 (single ClusterIP)

# Headless service DNS
nslookup mysql-headless.default.svc.cluster.local
# Returns: 10.244.1.5, 10.244.1.6, 10.244.1.7 (all pod IPs!)
```

### SRV Records

```bash
# Get service port information via SRV
dig SRV _http._tcp.my-service.default.svc.cluster.local

# OUTPUT:
# my-service.default.svc.cluster.local. 5 IN SRV 10 100 80 my-service.default.svc.cluster.local.
#
# Priority=10, Weight=100, Port=80, Target=hostname
```

## 3. CoreDNS

### What is CoreDNS?

CoreDNS is a flexible, extensible DNS server written in Go. It replaced kube-dns in Kubernetes 1.12.

**Architecture:**
```
DNS Request (UDP/TCP port 53)
         │
         ▼
    CoreDNS Process
         │
    Corefile (config)
         │
    ┌────┴──────────────────────────────────┐
    │ Plugin Chain (processes request)       │
    │  errors → log → health → ready        │
    │  → kubernetes → forward → cache       │
    └────────────────────────────────────────┘
         │
    ┌────┴────────────────────────────────┐
    │                                     │
    ▼                                     ▼
Kubernetes API              External DNS
(cluster.local queries)     (everything else)
```

### CoreDNS Deployment

```bash
# CoreDNS runs as a Deployment
kubectl get deployment coredns -n kube-system
# NAME      READY   UP-TO-DATE   AVAILABLE   AGE
# coredns   2/2     2            2           30d

# CoreDNS ClusterIP (constant for all clusters)
kubectl get svc kube-dns -n kube-system
# NAME       TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)
# kube-dns   ClusterIP   10.96.0.10   <none>        53/UDP,53/TCP
```

### Corefile - CoreDNS Configuration

```bash
# View current Corefile
kubectl get configmap coredns -n kube-system -o yaml
```

**Default Corefile:**
```
.:53 {
    errors                          # Log errors to stdout
    health {                        # Health endpoint
       lameduck 5s
    }
    ready                           # Readiness endpoint
    kubernetes cluster.local in-addr.arpa ip6.arpa {  # K8s plugin
       pods insecure                # Enable pod DNS
       fallthrough in-addr.arpa ip6.arpa
       ttl 30
    }
    prometheus :9153                # Metrics endpoint
    forward . /etc/resolv.conf {   # Forward non-k8s to host DNS
       max_concurrent 1000
    }
    cache 30                        # Cache responses for 30s
    loop                            # Detect forwarding loops
    reload                          # Auto-reload Corefile changes
    loadbalance                     # Randomize answer order
}
```

### CoreDNS Plugin Chain Explained

```
Query: my-service.default.svc.cluster.local

1. errors plugin:
   Sets up error logging for chain

2. health plugin:
   /health endpoint for liveness probe

3. ready plugin:
   /ready endpoint for readiness probe

4. kubernetes plugin:
   Is query for cluster.local?  YES
   → Query Kubernetes API for services/pods
   → Found: my-service in default namespace → 10.96.100.50
   → Return A record

Query: api.example.com

4. kubernetes plugin:
   Is query for cluster.local?  NO
   → fallthrough to next plugin

5. forward plugin:
   Forward to /etc/resolv.conf nameservers
   → Forward to 8.8.8.8 (or host's nameserver)
   → Return external IP
```

## 4. Custom DNS Configuration

### Custom DNS for Specific Domains

```yaml
# Override DNS for specific domains
# kubectl edit configmap coredns -n kube-system
apiVersion: v1
kind: ConfigMap
metadata:
  name: coredns
  namespace: kube-system
data:
  Corefile: |
    .:53 {
        errors
        health
        kubernetes cluster.local in-addr.arpa ip6.arpa {
           pods insecure
           fallthrough in-addr.arpa ip6.arpa
        }
        # Route internal.company.com to corporate DNS
        forward internal.company.com 10.100.0.1 10.100.0.2
        
        # Route example.com to specific server
        forward example.com 192.168.10.100
        
        # Default: forward to upstream
        forward . /etc/resolv.conf
        cache 30
        loop
        reload
        loadbalance
    }
```

### Per-Pod DNS Config

```yaml
# Override DNS settings for a specific pod
apiVersion: v1
kind: Pod
metadata:
  name: custom-dns-pod
spec:
  dnsPolicy: "None"  # Don't use default
  dnsConfig:
    nameservers:
      - 10.96.0.10    # CoreDNS (still use for k8s)
      - 8.8.8.8       # Google DNS fallback
    searches:
      - default.svc.cluster.local
      - svc.cluster.local
      - cluster.local
      - company.internal  # Add custom search domain!
    options:
      - name: ndots
        value: "2"   # Reduce extra queries for external names
  containers:
  - name: app
    image: nginx
```

### DNS Policies

| Policy | Behavior |
|--------|---------|
| `ClusterFirst` (default) | Cluster DNS, forward unknown to host DNS |
| `Default` | Use node's DNS (/etc/resolv.conf only) |
| `None` | Use dnsConfig only |
| `ClusterFirstWithHostNet` | ClusterFirst for pods with hostNetwork: true |

```yaml
# Pod with hostNetwork needs special policy
spec:
  hostNetwork: true          # Uses host network namespace
  dnsPolicy: ClusterFirstWithHostNet  # Still use CoreDNS
```

## 5. DNS Debugging

### Check Pod's DNS Config

```bash
# View resolv.conf in pod
kubectl exec -it my-pod -- cat /etc/resolv.conf

# Expected output:
# nameserver 10.96.0.10
# search default.svc.cluster.local svc.cluster.local cluster.local
# options ndots:5
```

### Test DNS Resolution

```bash
# Basic resolution
kubectl exec -it debug-pod -- nslookup kubernetes.default.svc.cluster.local

# With dig (more detail)
kubectl exec -it debug-pod -- dig my-service.default.svc.cluster.local

# Test SRV record
kubectl exec -it debug-pod -- dig SRV _http._tcp.my-service.default.svc.cluster.local

# Reverse DNS
kubectl exec -it debug-pod -- dig -x 10.96.100.50
```

### Using a Debug Pod

```bash
# Deploy DNS debug pod
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: dnsutils
spec:
  containers:
  - name: dnsutils
    image: registry.k8s.io/e2e-test-images/jessie-dnsutils:1.3
    command:
      - sleep
      - "3600"
EOF

kubectl exec -it dnsutils -- nslookup kubernetes
kubectl exec -it dnsutils -- nslookup my-service
kubectl exec -it dnsutils -- nslookup google.com
```

### Check CoreDNS Logs

```bash
# View CoreDNS logs
kubectl logs -n kube-system -l k8s-app=kube-dns

# Enable verbose logging (debug)
kubectl edit configmap coredns -n kube-system
# Add: log  (before or after errors)
```

### CoreDNS Metrics

```bash
# Port-forward CoreDNS metrics
kubectl port-forward -n kube-system pod/coredns-xxxx 9153:9153

# View metrics
curl http://localhost:9153/metrics | grep coredns_

# Key metrics:
# coredns_dns_requests_total          - Total requests
# coredns_dns_responses_total         - Responses by rcode
# coredns_forward_requests_total      - Forwarded requests
# coredns_cache_hits_total            - Cache hits
# coredns_cache_misses_total          - Cache misses
```

## 6. Common DNS Issues

### Issue 1: NXDOMAIN for Valid Service

```bash
# Check service exists
kubectl get svc my-service -n default

# Check endpoints
kubectl get endpoints my-service -n default

# If endpoints are empty, pod labels don't match selector!
kubectl get pods -l app=my-app
kubectl describe svc my-service | grep Selector
```

### Issue 2: DNS Resolution Timeout

```bash
# Check CoreDNS pod status
kubectl get pods -n kube-system -l k8s-app=kube-dns
# Should show 2+ Running pods

# Check resource usage
kubectl top pod -n kube-system -l k8s-app=kube-dns
# If CPU is pegged → CoreDNS overloaded

# Scale up CoreDNS
kubectl scale deployment coredns -n kube-system --replicas=4
```

### Issue 3: External DNS Slow

```bash
# Check forward plugin config
# If forwarding to slow upstream DNS → latency

# Solution: Use more reliable upstream
kubectl edit configmap coredns -n kube-system
# Change: forward . /etc/resolv.conf
# To:     forward . 8.8.8.8 8.8.4.4
```

### Issue 4: ndots Causing Extra Lookups

```bash
# Problem: Querying external services is slow due to 3 failed lookups
# curl http://api.example.com → tries 3 cluster searches first

# Solution 1: Use FQDN with trailing dot
curl http://api.example.com.

# Solution 2: Reduce ndots in pod spec
dnsConfig:
  options:
  - name: ndots
    value: "2"  # Only append search if <2 dots (catches most k8s names)
```

## 7. NodeLocal DNSCache

**Purpose:** Cache DNS responses at each node to reduce CoreDNS load.

```
Without NodeLocal DNSCache:
  Pod → CoreDNS Pod (network hop!)

With NodeLocal DNSCache:
  Pod → Node's local cache (same node, no network!)
       → Only cache misses reach CoreDNS
```

```bash
# Check if NodeLocal DNSCache is deployed
kubectl get daemonset nodelocaldns -n kube-system

# NodeLocal DNS listens on link-local address
# 169.254.20.10 (non-routable, safe)
```

## 8. DNS for Custom Resources

```yaml
# ExternalName service creates CNAME record
apiVersion: v1
kind: Service
metadata:
  name: external-db
spec:
  type: ExternalName
  externalName: db.company.internal  # Actual hostname
```

```bash
# Inside cluster:
nslookup external-db.default.svc.cluster.local
# Returns CNAME: db.company.internal
# Then resolves via external DNS
```

## 9. Reference: DNS Name Patterns

```
# Service in same namespace
http://my-service

# Service in different namespace
http://my-service.other-namespace

# Fully qualified
http://my-service.other-namespace.svc.cluster.local

# StatefulSet pod (direct)
http://pod-0.my-statefulset.default.svc.cluster.local

# Headless service (returns all pod IPs)
dig my-headless-svc.default.svc.cluster.local

# SRV record (gets port too)
dig SRV _grpc._tcp.my-service.default.svc.cluster.local
```

## Next Steps

✅ DNS query flow and record types  
✅ Kubernetes DNS name formats (services, pods, StatefulSets)  
✅ CoreDNS configuration and plugins  
✅ Custom DNS policies and debugging  

**Move to:** [05-containerization-deep-dive.md](05-containerization-deep-dive.md) to understand container images, layers, and runtimes.
