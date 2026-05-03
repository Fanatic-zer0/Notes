# Netfilter, iptables, and IPVS for Kubernetes

## Overview

When you create a Kubernetes Service, traffic doesn't magically reach your pods. **kube-proxy** implements service routing using either iptables or IPVS rules in the Linux kernel's **netfilter** subsystem. This guide explains exactly how.

## 1. Netfilter Framework

**What It Is:** The Linux kernel framework for packet filtering, NAT, and packet mangling.

### Netfilter Hooks

Packets flow through the kernel and trigger hooks at specific points:

```
Network Interface
      │
      │ RECEIVE
      ▼
┌─────────────┐
│  PREROUTING │◄── Hook 1 (NF_IP_PRE_ROUTING)
└──────┬──────┘
       │
       ├─── Is it for this host?
       │
   YES │                    NO
       │                    │
       ▼                    ▼
┌──────────────┐    ┌───────────────┐
│    INPUT     │    │   FORWARD     │
│ (to process) │    │ (to route on) │
└──────┬───────┘    └───────┬───────┘
       │                    │
       ▼                    ▼
   Process             ┌────────────┐
       │               │ POSTROUTING│
       │               └────────────┘
       │                    │
       ▼                    ▼
  ┌──────────┐         Network
  │  OUTPUT  │         Interface
  └────┬─────┘
       │
       ▼
┌─────────────┐
│ POSTROUTING │◄── Hook 5 (NF_IP_POST_ROUTING)
└─────────────┘
       │
       ▼
  Network Interface (SEND)
```

### Tables and Chains

Netfilter has **tables** (purposes) and **chains** (hooks):

```
Tables:
  filter  → Packet filtering (accept/drop)
  nat     → Address translation (DNAT/SNAT/MASQUERADE)
  mangle  → Packet modification (TTL, TOS, marks)
  raw     → Skip connection tracking
  security→ SELinux contexts

Chains (in each table):
  PREROUTING   → Arriving packets before routing decision
  INPUT        → Packets destined for local process
  FORWARD      → Packets being routed through
  OUTPUT       → Packets from local process
  POSTROUTING  → Packets leaving system
```

## 2. iptables - The User Space Tool

### Basic Commands

```bash
# List all rules
sudo iptables -L -n -v

# List with line numbers
sudo iptables -L -n --line-numbers

# List specific table
sudo iptables -t nat -L -n -v

# List NAT table (where Kubernetes magic happens!)
sudo iptables -t nat -L -n | less
```

### iptables Rule Structure

```
iptables -t TABLE -A CHAIN -m MODULE --options -j TARGET

Examples:
# Drop packets to port 80
iptables -A INPUT -p tcp --dport 80 -j DROP

# Allow established connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Redirect to different IP/port (DNAT)
iptables -t nat -A PREROUTING -d 10.96.0.1 -p tcp --dport 443 -j DNAT --to-destination 10.244.1.5:8443
```

**Targets:**
| Target | Action |
|--------|--------|
| ACCEPT | Allow packet through |
| DROP | Silently discard |
| REJECT | Discard with error |
| DNAT | Destination NAT (change destination IP:port) |
| SNAT | Source NAT (change source IP) |
| MASQUERADE | SNAT using interface IP |
| RETURN | Return to calling chain |
| LOG | Log to kernel log |
| `chain-name` | Jump to custom chain |

## 3. How kube-proxy Uses iptables for Services

### The Chain Hierarchy

kube-proxy creates a complex chain of rules:

```
PREROUTING/OUTPUT
       │
       ▼
  KUBE-SERVICES
       │
       ├──► KUBE-SVC-<hash> (per Service)
       │           │
       │           ├──► KUBE-SEP-<hash> (per Endpoint/Pod)
       │           │
       │           └──► KUBE-SEP-<hash>
       │
       └──► KUBE-NODEPORTS (for NodePort services)
```

### Step-by-Step: Service Traffic Flow

**Scenario:** Pod → Service IP → Backend Pod

```bash
# 1. View Kubernetes services
kubectl get svc
# NAME         TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)
# my-service   ClusterIP   10.96.100.50   <none>        80/TCP

# 2. Look at iptables rules for this service
sudo iptables -t nat -L KUBE-SERVICES -n | grep 10.96.100.50
# OUTPUT:
# target     prot opt source     destination
# KUBE-SVC-XXXXXXXX  tcp  --  0.0.0.0/0   10.96.100.50 tcp dpt:80
```

**Tracing the chain:**
```bash
# 3. Follow the chain
sudo iptables -t nat -L KUBE-SVC-XXXXXXXX -n --line-numbers
# OUTPUT:
# 1 KUBE-MARK-MASQ  tcp  --  !10.244.0.0/16  10.96.100.50  tcp dpt:80
# 2 KUBE-SEP-AAA    all  --  0.0.0.0/0       0.0.0.0/0     statistic mode random probability 0.33
# 3 KUBE-SEP-BBB    all  --  0.0.0.0/0       0.0.0.0/0     statistic mode random probability 0.50
# 4 KUBE-SEP-CCC    all  --  0.0.0.0/0       0.0.0.0/0
# (3 pods = 33.3%, 50%, 100% probability = equal distribution!)

# 4. Each KUBE-SEP is an endpoint (pod)
sudo iptables -t nat -L KUBE-SEP-AAA -n
# OUTPUT:
# KUBE-MARK-MASQ  all  --  10.244.1.5  0.0.0.0/0   (mark for masquerade)
# DNAT  tcp  --  0.0.0.0/0  0.0.0.0/0  tcp to:10.244.1.5:8080
```

**Visual flow:**
```
Pod A sends to 10.96.100.50:80
         │
         ▼ iptables NAT (PREROUTING)
    KUBE-SERVICES
         │
         ▼ Match: dst=10.96.100.50 port=80
    KUBE-SVC-XXXXXXXX
         │
         ├─ 33% → KUBE-SEP-AAA → DNAT to 10.244.1.5:8080
         ├─ 33% → KUBE-SEP-BBB → DNAT to 10.244.1.6:8080
         └─ 33% → KUBE-SEP-CCC → DNAT to 10.244.1.7:8080
         
Packet now destined for 10.244.1.5:8080 (actual pod)
```

### NodePort Rules

```bash
# NodePort service creates rules in KUBE-NODEPORTS chain
kubectl get svc my-service
# PORT(S): 80:31234/TCP

# iptables rule
sudo iptables -t nat -L KUBE-NODEPORTS -n | grep 31234
# KUBE-SVC-XXXXX  tcp  --  0.0.0.0/0  0.0.0.0/0  tcp dpt:31234
# Reuses same SVC chain!
```

### Connection Tracking (conntrack)

**Why it matters:** DNAT is stateful. Return packets need SNAT.

```bash
# View connection tracking table
conntrack -L | grep 10.96.100.50

# OUTPUT:
# tcp 6 86399 ESTABLISHED src=10.244.0.5 dst=10.96.100.50 sport=42000 dport=80
#                          src=10.244.1.5 dst=10.244.0.5  sport=8080  dport=42000
#                          [ASSURED] use=1
#
# Forward:  Pod A → Service IP
# Reply:    Backend Pod → Pod A (reverse DNAT applied automatically!)
```

## 4. IPVS - The Scalable Alternative

### Why IPVS?

**Problem with iptables:**
- O(n) rule lookup - scales linearly with services
- At 10,000 services → very large rule sets
- Rule updates lock iptables briefly

**IPVS advantages:**
- O(1) hash-based lookup
- Faster at scale (5000+ services)
- More load-balancing algorithms
- Kernel-native load balancer

### IPVS Load Balancing Algorithms

| Algorithm | Description | Use Case |
|-----------|-------------|---------|
| rr | Round Robin | Default, equal weight pods |
| lc | Least Connection | Long-lived connections |
| dh | Destination Hash | Session affinity |
| sh | Source Hash | Session affinity |
| sed | Shortest Expected Delay | Low latency |
| nq | Never Queue | Real-time |

### Enabling IPVS Mode

```bash
# Check current mode
kubectl get configmap kube-proxy -n kube-system -o yaml | grep mode

# Edit to enable IPVS
kubectl edit configmap kube-proxy -n kube-system
# Change: mode: ""  →  mode: "ipvs"

# Restart kube-proxy
kubectl rollout restart daemonset kube-proxy -n kube-system
```

### IPVS in Action

```bash
# View IPVS virtual services
sudo ipvsadm -ln

# OUTPUT:
# IP Virtual Server version 1.2.1 (size=4096)
# Prot LocalAddress:Port Scheduler Flags
#   -> RemoteAddress:Port Forward Weight ActiveConn InActConn
# TCP  10.96.100.50:80 rr
#   -> 10.244.1.5:8080 Masq    1      0          0
#   -> 10.244.1.6:8080 Masq    1      0          0
#   -> 10.244.1.7:8080 Masq    1      0          0
```

**Compare with iptables:**
```
iptables:
  Service with 3 pods = 7+ rules (chains, SEPs, masquerade)
  Service with 1000 pods = thousands of rules

IPVS:
  Service with 3 pods = 1 virtual service + 3 real servers
  Service with 1000 pods = 1 virtual service + 1000 real servers
  Hash table lookup = O(1) always
```

## 5. MASQUERADE and SNAT

### Why Pods Need MASQUERADE

When pod traffic leaves a node to reach external internet:

```
Pod (10.244.1.5) → External (8.8.8.8)
       │
       ▼ POSTROUTING
  MASQUERADE (src IP = pod IP → node IP)
       │
       ▼
Node sends: src=192.168.1.10 dst=8.8.8.8
(Pod IP hidden, Node IP visible)
       │
       ▼ Response arrives at node
  conntrack reverses MASQUERADE
       │
       ▼
Pod receives: src=8.8.8.8 dst=10.244.1.5
```

**iptables rule:**
```bash
sudo iptables -t nat -L POSTROUTING -n | grep MASQUERADE
# KUBE-POSTROUTING  all  --  0.0.0.0/0  0.0.0.0/0   /* kubernetes postrouting rules */

sudo iptables -t nat -L KUBE-POSTROUTING -n
# MASQUERADE  all  --  0.0.0.0/0  0.0.0.0/0  mark match 0x4000/0x4000
```

## 6. Packet Tracing and Debugging

### Watch iptables Rule Matches

```bash
# Enable logging (temporary debugging only!)
sudo iptables -t nat -I PREROUTING 1 -d 10.96.100.50 -j LOG --log-prefix "KUBE-SVC: "

# Watch logs
sudo dmesg -w | grep "KUBE-SVC"

# Remove after debugging
sudo iptables -t nat -D PREROUTING 1
```

### Trace Connection to Service

```bash
# Use conntrack to trace
sudo conntrack -E  # Event mode (watch connections)

# In another terminal
kubectl exec -it debug-pod -- curl http://my-service:80

# conntrack output:
# [NEW] tcp 6 src=10.244.0.5 dst=10.96.100.50 sport=12345 dport=80
# [UPDATE] ESTABLISHED ...
# [UPDATE] CLOSE ...
```

### xtables-addons (advanced tracing)

```bash
# Install xtables-addons (Ubuntu)
sudo apt install xtables-addons-common

# Trace a specific packet
sudo iptables -t raw -A PREROUTING -s 10.244.0.5 -j TRACE
sudo iptables -t raw -A OUTPUT -s 10.244.0.5 -j TRACE

# View trace
sudo dmesg | tail -50
```

## 7. NetworkPolicy and iptables

When you create a NetworkPolicy, the CNI plugin adds **drop rules**:

```yaml
# This policy:
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-from-other
spec:
  podSelector:
    matchLabels:
      app: backend
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
```

**Calico iptables result:**
```bash
sudo iptables -L -n | grep calico
# cali-FORWARD     all  -- 0.0.0.0/0  0.0.0.0/0  (Calico hook)
# cali-from-wl-dispatch  (per-workload rules)

# Drop rule for non-matching traffic:
# DROP  all  --  0.0.0.0/0  10.244.1.5  (if source != frontend pod)
```

## 8. kube-proxy Modes Comparison

| Feature | iptables | IPVS | userspace (deprecated) |
|---------|----------|------|----------------------|
| Performance | Good (<5000 svc) | Excellent (any scale) | Poor |
| Load balancing | Random/probability | Multiple algorithms | Round robin |
| Connection affinity | Limited | Full support | No |
| Health-based routing | No | Yes | No |
| Memory | Higher | Lower | N/A |
| Debugging | Easy (iptables commands) | Medium (ipvsadm) | Easy |

### When to Choose IPVS

- Clusters with > 1000 services
- Need advanced load balancing (least connection, source hash)
- Performance-critical workloads
- Need session affinity

## 9. Hands-On Lab

### Lab 1: Trace a Service Request

```bash
# 1. Create test service
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: lab-svc
spec:
  selector:
    app: lab
  ports:
  - port: 80
    targetPort: 8080
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lab-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lab
  template:
    metadata:
      labels:
        app: lab
    spec:
      containers:
      - name: app
        image: nginx
        ports:
        - containerPort: 80
EOF

# 2. Find service ClusterIP
kubectl get svc lab-svc -o jsonpath='{.spec.clusterIP}'

# 3. On a node, trace iptables
sudo iptables -t nat -L KUBE-SERVICES -n | grep <ClusterIP>
sudo iptables -t nat -L KUBE-SVC-<hash> -n

# 4. Send traffic and watch conntrack
sudo conntrack -E &
kubectl exec -it debug-pod -- curl http://lab-svc:80
```

### Lab 2: Compare iptables vs IPVS Performance

```bash
# Create 100 test services
for i in $(seq 1 100); do
  kubectl create service clusterip test-svc-$i --tcp=80:80
done

# Measure rule count
sudo iptables -t nat -L | wc -l

# Switch to IPVS mode and compare
# ipvsadm -ln | wc -l
```

## Next Steps

✅ How netfilter hooks process packets  
✅ How kube-proxy creates iptables chains for Services  
✅ IPVS as a scalable alternative  
✅ NAT and connection tracking  

**Move to:** [04-dns-coredns.md](04-dns-coredns.md) to understand how Kubernetes DNS works and how pods discover services.
