# Networking Fundamentals for Kubernetes

## Overview

Kubernetes networking builds on fundamental IP networking concepts. This guide covers IP addressing, CIDR notation, OSI model layers, and TCP/IP — everything you need to understand how traffic flows in a cluster.

## 1. IP Addressing

### IPv4 Basics

An IPv4 address is a **32-bit number** written in dotted-decimal notation:

```
192    .    168    .     1    .    100
 │           │           │         │
 └─ Octet 1  └─ Octet 2  └─ Octet 3 └─ Octet 4
11000000 . 10101000 . 00000001 . 01100100
```

**Address Classes (historical):**
| Class | Range | Default Mask | Hosts |
|-------|-------|--------------|-------|
| A | 1.0.0.0 - 126.x.x.x | /8 | 16.7M |
| B | 128.0.0.0 - 191.255.x.x | /16 | 65K |
| C | 192.0.0.0 - 223.255.255.x | /24 | 254 |

**Private Ranges (RFC 1918):**
| Range | CIDR | Common Use |
|-------|------|-----------|
| 10.0.0.0 - 10.255.255.255 | 10.0.0.0/8 | Pod CIDR |
| 172.16.0.0 - 172.31.255.255 | 172.16.0.0/12 | Node networks |
| 192.168.0.0 - 192.168.255.255 | 192.168.0.0/16 | Home/office |

**Kubernetes Defaults:**
```
Pod CIDR:     10.244.0.0/16  (Flannel default)
Service CIDR: 10.96.0.0/12   (API server default)
Node subnet:  10.244.x.0/24  (one per node)
```

## 2. CIDR Notation

**CIDR** (Classless Inter-Domain Routing) expresses IP ranges compactly:

```
10.244.0.0/16
     │       │
     │       └── Prefix length (network bits)
     └────────── Network address
```

### Breaking Down CIDR

**Example: 10.244.0.0/16**
```
10.244. 0  . 0
           ↑
Network:   10.244.0.0 → 10.244.255.255
Mask:      255.255.0.0
Hosts:     2^16 = 65536 addresses

Binary:
10.244.0.0:    00001010.11110100.00000000.00000000
Subnet Mask:   11111111.11111111.00000000.00000000
               ←────── /16 ────────→←── Host bits ──→
```

**Example: 10.244.1.0/24 (one node's pod subnet)**
```
Network: 10.244.1.0 → 10.244.1.255
Hosts:   254 usable (minus network/broadcast)
```

### CIDR Quick Reference

| CIDR | Hosts | Example Use |
|------|-------|-------------|
| /8 | 16.7M | Large clusters |
| /16 | 65536 | Pod CIDR default |
| /24 | 254 | Per-node pod subnet |
| /32 | 1 | Single IP (pod IP in route table) |

### Kubernetes CIDR Planning

```
┌──────────────────────────────────────────────────────────┐
│ Cluster CIDR Planning                                    │
├──────────────────────────────────────────────────────────┤
│ Pod CIDR:       10.244.0.0/16  (256 /24 subnets)         │
│                 ↓                                        │
│ Node 1 pods:    10.244.0.0/24  (254 pods)                │
│ Node 2 pods:    10.244.1.0/24  (254 pods)                │
│ Node 3 pods:    10.244.2.0/24  (254 pods)                │
│ ...                                                      │
├──────────────────────────────────────────────────────────┤
│ Service CIDR:   10.96.0.0/12   (1M addresses)            │
│ Node CIDR:      192.168.0.0/24 (physical network)        │
└──────────────────────────────────────────────────────────┘
```

## 3. OSI Model & TCP/IP Stack

### OSI Model (7 Layers)

```
┌─────┬────────────────┬─────────────────────────────────────┐
│ L7  │  Application   │ HTTP, gRPC, WebSocket               │
│ L6  │  Presentation  │ TLS/SSL, encoding                   │
│ L5  │  Session       │ Session management                  │
│ L4  │  Transport     │ TCP, UDP (ports, reliability)       │
│ L3  │  Network       │ IP (routing, addressing)            │
│ L2  │  Data Link     │ Ethernet, MAC addresses             │
│ L1  │  Physical      │ Cables, WiFi                        │
└─────┴────────────────┴─────────────────────────────────────┘
```

### TCP/IP Stack (4 Layers)

```
┌──────────────────┬────────────────────────────────────────┐
│  Application     │ HTTP, DNS, gRPC (L5-L7)                │
│  Transport       │ TCP, UDP (L4)                          │
│  Internet        │ IP, ICMP, ARP (L3)                     │
│  Network Access  │ Ethernet, WiFi (L1-L2)                 │
└──────────────────┴────────────────────────────────────────┘
```

### Kubernetes and the Layers

| Layer | Kubernetes Component |
|-------|---------------------|
| L7 Application | Ingress controllers, API gateway, service mesh |
| L4 Transport | Services (TCP/UDP load balancing) |
| L3 Network | Pod IPs, routing, CNI plugins |
| L2 Data Link | ARP, bridge interfaces (cni0) |

## 4. Layer 3 - The Network Layer

### IP Routing

**How packets find their way:**
```bash
# View routing table on a node
ip route show

# OUTPUT (typical k8s node):
default via 192.168.1.1 dev eth0
10.244.0.0/24 dev cni0 proto kernel scope link src 10.244.0.1
10.244.1.0/24 via 192.168.1.11 dev eth0   # Route to Node 2's pods
10.244.2.0/24 via 192.168.1.12 dev eth0   # Route to Node 3's pods
192.168.1.0/24 dev eth0 proto kernel scope link src 192.168.1.10
```

**Reading the table:**
- `default via 192.168.1.1 dev eth0` → Unknown destinations → gateway
- `10.244.0.0/24 dev cni0` → Local pods, use bridge
- `10.244.1.0/24 via 192.168.1.11` → Pod subnet on Node2 → send there

### ARP (Address Resolution Protocol)

**Purpose:** Resolve IP address → MAC address (L3 → L2).

```bash
# View ARP table
arp -n
# or
ip neigh show

# OUTPUT:
# 192.168.1.1 dev eth0 lladdr aa:bb:cc:dd:ee:ff REACHABLE
# 10.244.0.5 dev cni0 lladdr 42:3c:1e:ab:cd:ef STALE
```

**Process:**
```
Pod A (10.244.0.5) → Pod B (10.244.0.8)
1. Same subnet? Yes (/24)
2. Check ARP cache for 10.244.0.8
3. Not found → ARP broadcast: "Who has 10.244.0.8?"
4. Pod B responds: "I'm at MAC aa:bb:cc..."
5. Cache updated, packet sent
```

## 5. Layer 4 - The Transport Layer

### TCP vs UDP

| Feature | TCP | UDP |
|---------|-----|-----|
| Connection | 3-way handshake | Connectionless |
| Reliability | Guaranteed delivery | Best effort |
| Order | In-order delivery | No ordering |
| Speed | Slower (overhead) | Faster |
| Use Cases | HTTP, databases, control plane | DNS, monitoring metrics |

### TCP 3-Way Handshake

```
Client                        Server
  │                             │
  │──── SYN (seq=x) ──────────►│
  │                             │
  │◄─── SYN-ACK (seq=y,ack=x+1)│
  │                             │
  │──── ACK (ack=y+1) ─────────►│
  │                             │
  │ Connection Established      │
```

**Kubernetes Impact:**
- kubelet → API server connection uses TCP
- Container probes (liveness/readiness) create TCP connections
- Services terminate TCP and can balance across pods

### Ports and Port Ranges

```bash
# View listening ports
ss -tlnp  # or netstat -tlnp

# Common Kubernetes ports:
# 6443  - API Server
# 2379  - etcd
# 2380  - etcd peer
# 10250 - Kubelet API
# 10257 - Controller Manager
# 10259 - Scheduler
# 30000-32767 - NodePort services
```

## 6. Layer 7 - The Application Layer

### HTTP/HTTPS

```
Request:
GET /api/v1/pods HTTP/1.1
Host: kubernetes.default.svc.cluster.local
Authorization: Bearer <token>

Response:
HTTP/1.1 200 OK
Content-Type: application/json

{"apiVersion": "v1", "items": [...]}
```

**Kubernetes uses HTTP for:**
- kubectl → API server (HTTPS)
- Services → Pod communication
- Health checks (readiness/liveness probes)

### gRPC

**Used by:**
- kubelet → CRI (container runtime) via unix socket
- Internal component communication
- Some custom controllers

### Why Layer 7 Matters for Kubernetes

```
Layer 4 Load Balancing (Services):
  Client → Service IP → Pod (round-robin by connection)
  Cannot inspect HTTP content

Layer 7 Load Balancing (Ingress):
  Client → Ingress → Pod (route by path, host, headers)
  /api/* → backend-service
  /web/* → frontend-service
```

## 7. Practical Packet Flow in Kubernetes

### Scenario: Web Request from Browser to Pod

```
Browser (external)
     │
     │ HTTPS to 203.0.113.5:443
     ▼
┌─────────────────────────────────────────┐
│ Cloud Load Balancer (L4)                │
│ 203.0.113.5 → NodePort on any Node      │
└─────────────┬───────────────────────────┘
              │ TCP to NodeIP:30080
              ▼
┌─────────────────────────────────────────┐
│ Node (Physical Machine)                 │
│ iptables: NodePort 30080 → Service IP   │
│           Service IP → Pod IP (DNAT)    │
└─────────────┬───────────────────────────┘
              │ TCP to PodIP:8080
              ▼
┌─────────────────────────────────────────┐
│ Pod (Container)                         │
│ App listening on :8080                  │
│ Returns HTTP response                   │
└─────────────────────────────────────────┘
```

**Layer-by-layer breakdown:**
```
L7: HTTP GET /products
L4: TCP, source port 54321, dest port 8080
L3: Src 10.244.1.5, Dst 10.244.2.8 (after DNAT)
L2: Src MAC (veth), Dst MAC (bridge)
L1: Physical/virtual ethernet
```

## 8. subnetting Practice for Kubernetes

### Calculate Pod Capacity

**Given: Pod CIDR = 10.244.0.0/16, node prefix = /24**

```
Available node subnets: 2^(24-16) = 2^8 = 256 nodes
Pods per node:          2^(32-24) - 2 = 254 pods

Total pod capacity: 256 × 254 = 65,024 pods
```

**Command to verify:**
```bash
# Check current pod CIDR allocation
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.podCIDR}{"\n"}{end}'

# OUTPUT:
# node1    10.244.0.0/24
# node2    10.244.1.0/24
# node3    10.244.2.0/24
```

### Service CIDR Planning

```
Service CIDR: 10.96.0.0/12
Range: 10.96.0.0 → 10.111.255.255
Total IPs: 2^20 = 1,048,576 service IPs

kubectl cluster-info dump | grep -m1 service-cluster-ip-range
# --service-cluster-ip-range=10.96.0.0/12
```

## 9. Network Troubleshooting Commands

### Connectivity Testing

```bash
# Test pod-to-pod
kubectl exec -it <pod> -- ping 10.244.1.8

# Test DNS resolution
kubectl exec -it <pod> -- nslookup kubernetes.default.svc.cluster.local

# Test service connectivity
kubectl exec -it <pod> -- curl -v http://my-service:80

# Test external connectivity
kubectl exec -it <pod> -- curl -v https://httpbin.org/get
```

### Network Inspection

```bash
# View pod networking
kubectl exec -it <pod> -- ip addr   # Interfaces
kubectl exec -it <pod> -- ip route  # Routing table
kubectl exec -it <pod> -- ss -tlnp  # Listening ports

# Node-level inspection
# On node:
ip route show          # Routing table
ip link show           # Interfaces
bridge link show       # Bridge members
ip neigh show          # ARP table

# Capture traffic
tcpdump -i eth0 host 10.244.1.5
tcpdump -i cni0 port 8080
```

### Common Network Issues

| Problem | Symptoms | Debug Steps |
|---------|----------|-------------|
| Pod can't reach service | DNS resolves, curl fails | Check NetworkPolicy, Endpoints |
| DNS not resolving | `nslookup` fails | Check CoreDNS pods, ConfigMap |
| Cross-node pod comm fails | Ping works same node | Check CNI, node routing table |
| Service unreachable | Connection refused | Check port, pod labels match selector |

## 10. IPv6 in Kubernetes

**Dual-Stack Support (Kubernetes 1.21+):**
```yaml
# kube-apiserver config
--service-cluster-ip-range=10.96.0.0/12,fd00::/108
--pod-cluster-cidr=10.244.0.0/16,fd01::/48
```

**Pod gets both IPv4 and IPv6:**
```bash
kubectl exec -it <pod> -- ip addr
# eth0: inet 10.244.1.5/24
#       inet6 fd01::a8:f4:c0:1/120
```

## Next Steps

✅ IP addressing and CIDR notation  
✅ OSI model and where Kubernetes fits  
✅ Layer 3/4 routing and TCP/UDP  
✅ How packets flow through a cluster  

**Move to:** [03-netfilter-iptables.md](03-netfilter-iptables.md) to understand how Services actually work using iptables and IPVS rules.
