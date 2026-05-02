# Pod-to-Pod Communication in Kubernetes

## Core Networking Principles

Kubernetes follows a **flat network model** with three fundamental rules:
1. **All pods can communicate** with each other without NAT
2. **All nodes can communicate** with all pods without NAT
3. **A pod sees itself** with the same IP that others see it with

## Communication Flow

### Same Node Communication
```
Pod A (10.244.1.5) → Pod B (10.244.1.8) [Same Node]
     │
     ├─→ veth0 (Pod A's virtual ethernet)
     │
     ├─→ cni0/cbr0 (Node's bridge interface)
     │
     └─→ veth1 (Pod B's virtual ethernet)
```

**Steps:**
1. Pod A sends packet to Pod B's IP (10.244.1.8)
2. Packet exits Pod A's network namespace via **veth pair** (virtual ethernet)
3. Enters the **bridge interface** (cni0/cbr0) on the node
4. Bridge forwards to Pod B's veth pair
5. Packet arrives at Pod B

### Different Node Communication
```
Pod A (10.244.1.5, Node1) → Pod B (10.244.2.8, Node2)
     │
     ├─→ veth0 (Pod A's virtual ethernet)
     │
     ├─→ cni0 (Node1 bridge)
     │
     ├─→ eth0 (Node1 physical network)
     │
     ├─→ [Physical Network/Overlay]
     │
     ├─→ eth0 (Node2 physical network)
     │
     ├─→ cni0 (Node2 bridge)
     │
     └─→ veth1 (Pod B's virtual ethernet)
```

**Steps:**
1. Pod A sends packet to Pod B (10.244.2.8)
2. Packet exits to Node1's bridge
3. Bridge checks **routing table** → destination is on different subnet
4. Packet forwarded to **Node1's eth0** interface
5. **CNI plugin** handles cross-node routing:
   - **Overlay networks** (Flannel VXLAN, Calico IPIP): Encapsulate packet
   - **Direct routing** (Calico BGP, AWS VPC CNI): Use native routing
6. Packet traverses physical network
7. Arrives at Node2, reverse process to reach Pod B

## CNI (Container Network Interface) Plugins

**Role:** CNI plugins implement the networking layer.

### Popular Options

| Plugin | Method | Performance | Use Case |
|--------|--------|-------------|----------|
| **Flannel** | VXLAN overlay | Medium | Simple, cross-platform |
| **Calico** | BGP routing | High | Network policies, no encapsulation |
| **Cilium** | eBPF | Very High | Advanced security, observability |
| **Weave** | Mesh overlay | Medium | Easy setup, encryption |
| **AWS VPC CNI** | Direct ENI | High | AWS-native, no overlay |

## Detailed Packet Flow Example

**Scenario:** Pod in Deployment A → Pod in Deployment B (different nodes)

```
┌─────────────────────────────────────────────────────────────────┐
│ Pod A (nginx-85f7fb6b5c-xyz, IP: 10.244.1.5)                    │
│ Container: curl http://10.244.2.8:80                            │
└───────────────────┬─────────────────────────────────────────────┘
                    │ 1. DNS Resolution (if using service name)
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ CoreDNS: service-b.default.svc.cluster.local → 10.96.10.20      │
│ (Service ClusterIP)                                             │
└───────────────────┬─────────────────────────────────────────────┘
                    │ 2. kube-proxy iptables/IPVS rules
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Service Load Balancing: 10.96.10.20 → 10.244.2.8 (Pod B IP)    │
└───────────────────┬─────────────────────────────────────────────┘
                    │ 3. Packet leaves Pod A namespace
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ NODE 1                                                          │
│  ┌──────────┐                                                   │
│  │ Pod A NS │ → veth pair → cni0 bridge                        │
│  └──────────┘                  ↓                                │
│                          Routing table:                         │
│                          10.244.2.0/24 via Node2 (192.168.1.11) │
└───────────────────┬─────────────────────────────────────────────┘
                    │ 4. CNI encapsulation (if overlay)
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ PHYSICAL NETWORK                                                │
│ VXLAN/IPIP tunnel: Outer header (Node1 → Node2)                │
│                    Inner header (Pod A → Pod B)                │
└───────────────────┬─────────────────────────────────────────────┘
                    │ 5. Packet arrives at Node2
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ NODE 2                                                          │
│                  Decapsulation → cni0 bridge                    │
│                          ↓                                      │
│                  ┌──────────┐                                   │
│                  │ Pod B NS │ ← veth pair                       │
│                  └──────────┘                                   │
│                  IP: 10.244.2.8                                 │
└───────────────────┬─────────────────────────────────────────────┘
                    │ 6. Response (reverse path)
                    ↓
                Pod A receives response
```

## Network Policies

Control traffic between pods using **NetworkPolicy** objects:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
spec:
  podSelector:
    matchLabels:
      app: backend
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080
```

**Effect:** Only pods with label `app=frontend` can reach pods with `app=backend` on port 8080.

## Service Discovery Flow

**When using Services (recommended):**

```
Pod A wants to reach "backend-service"
     │
     ├─→ 1. DNS Query: backend-service.default.svc.cluster.local
     │
     ├─→ 2. CoreDNS returns: 10.96.10.20 (Service ClusterIP)
     │
     ├─→ 3. kube-proxy iptables/IPVS rules intercept
     │
     ├─→ 4. Load balance to one of: [10.244.1.8, 10.244.2.9, 10.244.3.7]
     │
     └─→ 5. Direct pod-to-pod communication (as above)
```

## Key Components

### 1. Network Namespace
- Each pod gets isolated network namespace
- Own IP address, routing table, network devices

### 2. veth Pairs
- Virtual ethernet cable connecting pod to node
- One end in pod namespace, other on node

### 3. Bridge (cni0/cbr0)
- Layer 2 virtual switch on each node
- Connects all pod veth pairs on that node

### 4. kube-proxy
- Implements Service abstraction
- Manages iptables/IPVS rules for load balancing

### 5. CNI Plugin
- Assigns IP addresses to pods
- Sets up routing between nodes
- May create overlay networks

## Performance Considerations

### Latency Components
- Same node: ~0.1ms (bridge forwarding)
- Different nodes (overlay): ~0.5-2ms (encapsulation overhead)
- Different nodes (BGP): ~0.2-0.5ms (native routing)

### Optimization Tips
1. Use **nodeAffinity** to colocate communicating pods
2. Choose **direct routing CNI** (Calico BGP) for low latency
3. Enable **eBPF** data plane (Cilium) for ~40% better performance
4. Use **hostNetwork: true** for extreme performance (loses isolation)

## Troubleshooting

```bash
# Check pod IP and network interface
kubectl exec -it pod-name -- ip addr

# Trace network path
kubectl exec -it pod-name -- traceroute target-ip

# Check DNS resolution
kubectl exec -it pod-name -- nslookup service-name

# View CNI configuration
cat /etc/cni/net.d/10-calico.conflist

# Check routing table on node
ip route show
```

---

**Next:** [Service Types & Load Balancing](02-service-types-loadbalancing.md)
