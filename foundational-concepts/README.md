# Foundational Concepts for Kubernetes

## Overview

Before diving into Kubernetes, understanding these foundational concepts is **absolutely critical**. Kubernetes builds heavily on Linux kernel features, networking primitives, and containerization technologies. Without this foundation, Kubernetes concepts will seem like magic rather than logical systems built on well-understood technologies.

## Why These Foundations Matter

```
┌─────────────────────────────────────────────────────────────┐
│                    KUBERNETES                               │
│  ┌───────────────────────────────────────────────────────┐ │
│  │           Built On Top Of                             │ │
│  └───────────────────────────────────────────────────────┘ │
│                          │                                  │
│              ┌───────────┼───────────┐                      │
│              │           │           │                      │
│    ┌─────────▼───┐  ┌───▼─────┐  ┌──▼──────────┐          │
│    │   Linux     │  │ Network │  │ Container   │          │
│    │ Primitives  │  │  Stack  │  │ Technology  │          │
│    │             │  │         │  │             │          │
│    │ • Namespaces│  │ • IP/   │  │ • Images    │          │
│    │ • cgroups   │  │   CIDR  │  │ • Layers    │          │
│    │ • veth      │  │ • OSI   │  │ • Runtime   │          │
│    │ • systemd   │  │ • ipt.  │  │ • Registry  │          │
│    └─────────────┘  └─────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Real-World Example: What Happens When You Create a Pod

```
kubectl create -f pod.yaml
         │
         ├─→ API Server receives request (REST API)
         │   Validates YAML syntax
         │
         ├─→ Scheduler assigns to Node
         │
         ├─→ Kubelet on Node starts container
         │   │
         │   ├─→ Creates Linux Namespaces (PID, NET, MNT, UTS, IPC)
         │   ├─→ Applies cgroup limits (CPU, Memory)
         │   ├─→ Creates veth pair (virtual ethernet)
         │   ├─→ Configures iptables rules for Service
         │   ├─→ Pulls container image (understand layers!)
         │   ├─→ Creates overlay filesystem (image layers)
         │   └─→ Starts process in isolated environment
         │
         └─→ Pod is Running!
```

**Without understanding these foundations, you won't know:**
- Why containers are isolated (Namespaces)
- How resource limits work (cgroups)
- How pod networking actually functions (veth, bridges, iptables)
- What happens during `docker pull` (image layers, overlay2)
- How Services route traffic (iptables/IPVS rules)
- How DNS resolution works (CoreDNS, /etc/resolv.conf)

## Learning Path

### 1. Linux Fundamentals (Start Here!)

**File:** [01-linux-fundamentals.md](01-linux-fundamentals.md)

**Topics Covered:**
- **Linux Namespaces** - The foundation of container isolation
  - PID namespace (process isolation)
  - Network namespace (network stack isolation)
  - Mount namespace (filesystem isolation)
  - UTS namespace (hostname isolation)
  - IPC namespace (inter-process communication)
  - User namespace (UID/GID mapping)
- **Control Groups (cgroups)** - Resource limiting and accounting
  - CPU limits and shares
  - Memory limits and OOM killer
  - Block I/O throttling
  - Network bandwidth control
- **Virtual Ethernet (veth)** - Container networking
  - veth pair creation
  - Bridge networking
  - Network namespaces + veth
- **Systemd & Journalctl** - Service management
  - Unit files
  - Service lifecycle
  - Log management

**Time Investment:** 8-12 hours (with hands-on practice)

---

### 2. Networking Mastery

**Files:** 
- [02-networking-fundamentals.md](02-networking-fundamentals.md) - IP, CIDR, OSI Model, TCP/IP
- [03-netfilter-iptables.md](03-netfilter-iptables.md) - iptables, IPVS, packet filtering
- [04-dns-coredns.md](04-dns-coredns.md) - DNS resolution, CoreDNS

**Topics Covered:**
- **IP Addressing & CIDR** - How pod IPs are allocated
- **OSI Model & TCP/IP Stack** - Understanding network layers
- **Layer 3/4 (Network/Transport)** - Pod-to-pod communication
- **Layer 7 (Application)** - Ingress controllers, HTTP routing
- **Netfilter & iptables** - How kube-proxy works
- **IPVS** - Scalable load balancing alternative
- **DNS & CoreDNS** - Service discovery in Kubernetes

**Time Investment:** 12-16 hours

---

### 3. Essential Systems & Tools

**Files:**
- [05-containerization-deep-dive.md](05-containerization-deep-dive.md) - Docker/containerd internals
- [06-yaml-rest-apis.md](06-yaml-rest-apis.md) - YAML & Kubernetes API

**Topics Covered:**
- **Containerization Deep Dive**
  - How images are built (layer by layer)
  - Dockerfile instructions and layers
  - Image manifests and registries
  - Overlay filesystem (overlay2, OverlayFS)
  - Container runtime interface (CRI)
- **YAML Syntax** - Kubernetes manifests
- **REST APIs** - How kubectl communicates with API server

**Time Investment:** 8-10 hours

---

## How to Use This Guide

### For Complete Beginners
```
Day 1-3:   Linux Fundamentals (focus on namespaces, cgroups)
Day 4-6:   Networking Fundamentals (IP, CIDR, OSI model)
Day 7-9:   iptables and packet flow
Day 10-11: DNS and CoreDNS
Day 12-14: Containerization deep dive
Day 15:    YAML and REST APIs
```

### For Developers with Some Linux Experience
```
Day 1:   Linux Namespaces and cgroups (hands-on labs)
Day 2:   Networking (IP/CIDR, OSI) + iptables basics
Day 3:   Container internals (image layers, overlay filesystem)
Day 4:   Review and practice
```

### For System Administrators
```
Focus Areas:
1. How Kubernetes uses iptables (critical for troubleshooting)
2. Network namespace and veth pairs (pod networking)
3. cgroups and resource limits (capacity planning)
4. systemd integration (kubelet as a service)
```

## Hands-On Practice Recommendations

Each section includes **practical labs** and **commands to run**. You should:

1. **Set up a Linux VM** (Ubuntu 22.04 or similar)
2. **Run every command** shown in the guides
3. **Break things intentionally** to understand failure modes
4. **Build your own containers** from scratch (without Docker!)
5. **Trace network packets** with tcpdump
6. **Create manual iptables rules** to understand kube-proxy

## Connection to Kubernetes Concepts

| Foundation Concept | Kubernetes Usage |
|-------------------|------------------|
| **Linux Namespaces** | Pod isolation, container separation |
| **cgroups** | Resource requests/limits, QoS classes |
| **veth pairs** | Pod networking, CNI plugins |
| **iptables** | kube-proxy Service implementation |
| **IPVS** | Alternative kube-proxy mode (scalable) |
| **Overlay filesystem** | Container image layers, efficient storage |
| **DNS** | Service discovery, CoreDNS |
| **Network namespaces** | Pod network isolation |
| **systemd** | kubelet, container runtime services |

## Prerequisites

- **Linux machine** (Ubuntu 22.04 LTS recommended)
- **Root/sudo access** (many commands require privileges)
- **Basic command-line comfort** (ls, cd, grep, pipe)
- **Text editor** (vim, nano, or VS Code)

## What You'll Learn

After completing these foundational concepts, you will:

✅ Understand how pods are isolated using Linux kernel features  
✅ Know exactly how container images work (layers, manifests, registries)  
✅ Be able to troubleshoot network issues in Kubernetes  
✅ Understand how Services route traffic using iptables/IPVS  
✅ Know how DNS resolution works in a cluster  
✅ Be able to read and write complex YAML manifests  
✅ Understand how kubectl communicates with the API server  
✅ Have the foundation to learn advanced Kubernetes topics  

## Files in This Directory

1. **[01-linux-fundamentals.md](01-linux-fundamentals.md)** - Namespaces, cgroups, veth, systemd (Essential!)
2. **[02-networking-fundamentals.md](02-networking-fundamentals.md)** - IP/CIDR, OSI model, TCP/IP
3. **[03-netfilter-iptables.md](03-netfilter-iptables.md)** - iptables, IPVS, packet filtering
4. **[04-dns-coredns.md](04-dns-coredns.md)** - DNS fundamentals, CoreDNS in K8s
5. **[05-containerization-deep-dive.md](05-containerization-deep-dive.md)** - Image layers, build process, overlayfs
6. **[06-yaml-rest-apis.md](06-yaml-rest-apis.md)** - YAML syntax, K8s REST API

## Next Steps

After mastering these foundations:

1. Move to core Kubernetes concepts in the main directory
2. Set up a local cluster (minikube, kind, or k3s)
3. Practice creating pods, deployments, services
4. Experiment with different CNI plugins
5. Explore kube-proxy modes (iptables vs IPVS)

