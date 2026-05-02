# Service Types & Load Balancing in Kubernetes

## Overview

Kubernetes Services provide **stable networking** for ephemeral pods. While pod IPs change on restarts, Service IPs remain constant, providing reliable endpoints for communication.

## Service Types

| Type | Scope | Use Case | IP Range |
|------|-------|----------|----------|
| **ClusterIP** | Internal only | Pod-to-pod within cluster | 10.96.0.0/12 (default) |
| **NodePort** | External access | Development, on-prem | ClusterIP + Node ports |
| **LoadBalancer** | External access | Production cloud | Cloud provider LB |
| **ExternalName** | DNS alias | External service mapping | No IP (CNAME) |

## ClusterIP Service

**Default type** - Creates virtual IP accessible only within the cluster.

### Flow Diagram
```
Pod A (10.244.1.5) wants to reach "backend-service"
     │
     ├─→ 1. DNS: backend-service.default.svc.cluster.local
     │         CoreDNS returns: 10.96.10.20 (ClusterIP)
     │
     ├─→ 2. Send packet to 10.96.10.20:8080
     │
     ├─→ 3. kube-proxy iptables rules intercept
     │         PREROUTING chain: -A KUBE-SERVICES -d 10.96.10.20/32 -p tcp -m tcp --dport 8080 -j KUBE-SVC-XXX
     │
     ├─→ 4. Load balance to backend pods (round-robin)
     │         50% → 10.244.1.8:8080 (backend-pod-1)
     │         50% → 10.244.2.9:8080 (backend-pod-2)
     │
     └─→ 5. Direct pod-to-pod communication (see 01-pod-to-pod-communication.md)
```

### YAML Example
```yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  type: ClusterIP  # Default, can be omitted
  selector:
    app: backend
  ports:
  - protocol: TCP
    port: 8080        # Service port
    targetPort: 8080  # Pod container port
```

### How It Works
1. **Service created** → kube-apiserver stores in etcd
2. **EndpointSlice controller** watches the Service and matching pods by selector
3. **Creates EndpointSlice object** with pod IPs: `[10.244.1.8:8080, 10.244.2.9:8080]`
4. **kube-proxy** (on every node) watches Service/EndpointSlice
5. **Configures iptables/IPVS** rules to forward ClusterIP traffic to pod IPs

> **Note:** Kubernetes 1.21+ uses `EndpointSlice` objects (replacing the older `Endpoints` object). EndpointSlices partition backends across multiple objects for scalability. The old `Endpoints` API still exists for compatibility but EndpointSlice is the data plane truth.

## NodePort Service

**Exposes service** on a static port (30000-32767) on every node's IP.

### Flow Diagram
```
External Client (203.0.113.50)
     │
     ├─→ 1. HTTP request to http://Node1-IP:32080
     │         Node1 IP: 192.168.1.10:32080
     │
     ├─→ 2. Arrives at Node1's eth0 interface
     │
     ├─→ 3. iptables PREROUTING chain
     │         -A KUBE-NODEPORTS -p tcp -m tcp --dport 32080 -j KUBE-SVC-XXX
     │
     ├─→ 4. DNAT (Destination NAT) to ClusterIP
     │         Rewrite: 192.168.1.10:32080 → 10.96.10.20:8080
     │
     ├─→ 5. ClusterIP load balancing (same as above)
     │         Random pod selection: 10.244.2.9:8080 (on Node2)
     │
     ├─→ 6. Pod-to-pod routing (may cross nodes)
     │
     └─→ 7. Response path (reverse SNAT/DNAT)
              10.244.2.9:8080 → 10.96.10.20:8080 → 192.168.1.10:32080 → Client
```

### YAML Example
```yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  type: NodePort
  selector:
    app: backend
  ports:
  - protocol: TCP
    port: 8080        # ClusterIP port
    targetPort: 8080  # Pod port
    nodePort: 32080   # Node port (optional, auto-assigned if omitted)
```

### Key Characteristics
- **Every node** listens on the NodePort (even nodes without matching pods)
- **externalTrafficPolicy: Cluster** (default): Load balances across all pods, may SNAT
- **externalTrafficPolicy: Local**: Only forwards to local pods, preserves client IP

### externalTrafficPolicy Comparison

#### Cluster (Default)
```
Client → Node1:32080 → Any Pod (Node1, Node2, or Node3)
Pros: Even load distribution
Cons: Extra hop, client IP lost (SNAT), higher latency
```

#### Local
```
Client → Node1:32080 → Only Pods on Node1
Pros: No extra hop, client IP preserved, lower latency
Cons: Uneven load if pods not on all nodes, health check complexity
```

```yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  type: NodePort
  externalTrafficPolicy: Local  # Only route to local pods
  selector:
    app: backend
  ports:
  - port: 8080
    nodePort: 32080
```

## LoadBalancer Service

**Cloud-managed** external load balancer (AWS ELB/NLB, GCP LB, Azure LB).

### Flow Diagram
```
External Client (Internet)
     │
     ├─→ 1. DNS: api.example.com → 203.0.113.100 (Cloud LB IP)
     │
     ├─→ 2. Cloud Load Balancer (AWS NLB/ELB)
     │         Backend targets: [Node1:32080, Node2:32080, Node3:32080]
     │         Health checks: TCP :32080 every 10s
     │
     ├─→ 3. Load balances to healthy node
     │         Selected: Node2:32080
     │
     ├─→ 4. NodePort handling (same as above)
     │         Node2:32080 → ClusterIP → Pod
     │
     └─→ 5. Response back through LB
```

### YAML Example
```yaml
apiVersion: v1
kind: Service
metadata:
  name: frontend-service
  annotations:
    # AWS-specific annotations
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"  # Network LB (L4)
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: "tcp"
spec:
  type: LoadBalancer
  selector:
    app: frontend
  ports:
  - protocol: TCP
    port: 80          # LB port
    targetPort: 8080  # Pod port
```

### Cloud Provider Integration

**AWS:**
```yaml
metadata:
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"  # NLB (recommended)
    service.beta.kubernetes.io/aws-load-balancer-ssl-cert: "arn:aws:acm:..."
    service.beta.kubernetes.io/aws-load-balancer-ssl-ports: "443"
```

**GCP:**
```yaml
metadata:
  annotations:
    cloud.google.com/load-balancer-type: "Internal"  # Internal LB
```

**Azure:**
```yaml
metadata:
  annotations:
    service.beta.kubernetes.io/azure-load-balancer-internal: "true"
```

## kube-proxy Modes

### iptables Mode (Default)

**How it works:**
- Creates iptables rules for each Service/Endpoint
- Uses DNAT (Destination NAT) for load balancing
- Random selection per connection

```bash
# View iptables rules for a service
iptables -t nat -L KUBE-SERVICES -n | grep backend-service

# Example rule
-A KUBE-SERVICES -d 10.96.10.20/32 -p tcp -m tcp --dport 8080 -j KUBE-SVC-ABCD1234
-A KUBE-SVC-ABCD1234 -m statistic --mode random --probability 0.50000 -j KUBE-SEP-POD1
-A KUBE-SVC-ABCD1234 -j KUBE-SEP-POD2
```

**Pros:** Mature, stable
**Cons:** Doesn't scale well (>5000 services), no real load balancing

### IPVS Mode

**How it works:**
- Uses Linux IPVS (IP Virtual Server) kernel module
- True load balancing with algorithms: rr, lc, dh, sh, sed, nq
- Better performance for large clusters

```yaml
# kube-proxy ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: kube-proxy
  namespace: kube-system
data:
  config.conf: |
    mode: "ipvs"
    ipvs:
      scheduler: "rr"  # round-robin (default)
```

```bash
# View IPVS rules
ipvsadm -Ln

# Example output
TCP  10.96.10.20:8080 rr
  -> 10.244.1.8:8080      Masq    1      0          0
  -> 10.244.2.9:8080      Masq    1      0          0
```

**Load Balancing Algorithms:**
- **rr** (round-robin): Equal distribution
- **lc** (least connection): Least active connections
- **sh** (source hashing): Same client → same pod
- **dh** (destination hashing): Consistent hashing

**Pros:** Better performance, true load balancing, scales to 10k+ services
**Cons:** Requires kernel modules, more complex debugging

## Session Affinity

**Sticky sessions** - Same client always reaches same pod.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-service
spec:
  selector:
    app: backend
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800  # 3 hours
  ports:
  - port: 8080
```

**How it works:**
- iptables mode: Uses `recent` module to track client IPs
- IPVS mode: Uses `sh` (source hashing) scheduler

## Headless Service

**No ClusterIP** - Returns pod IPs directly for DNS queries.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: database-headless
spec:
  clusterIP: None  # Headless
  selector:
    app: database
  ports:
  - port: 5432
```

**DNS Resolution:**
```bash
# Normal service
nslookup backend-service.default.svc.cluster.local
# Returns: 10.96.10.20 (ClusterIP)

# Headless service
nslookup database-headless.default.svc.cluster.local
# Returns: 10.244.1.8, 10.244.2.9, 10.244.3.7 (All pod IPs)
```

**Use Cases:**
- StatefulSets (stable pod DNS names)
- Custom load balancing in application
- Service meshes (Istio, Linkerd)

## Complete Traffic Flow Example

```
┌─────────────────────────────────────────────────────────────────┐
│ External Client (Internet)                                      │
└───────────────────┬─────────────────────────────────────────────┘
                    │ HTTP GET /api/users
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Cloud Load Balancer (AWS NLB)                                   │
│ Public IP: 203.0.113.100:80                                     │
│ Backends: [Node1:32080, Node2:32080, Node3:32080]              │
└───────────────────┬─────────────────────────────────────────────┘
                    │ Selected: Node2:32080
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Node2 (192.168.1.11)                                            │
│ iptables KUBE-NODEPORTS chain                                   │
│ DNAT: 192.168.1.11:32080 → 10.96.10.20:8080 (ClusterIP)       │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ kube-proxy iptables rules                                       │
│ KUBE-SERVICES: Match 10.96.10.20:8080                          │
│ KUBE-SVC-XXX: Random load balance                              │
│   33% → 10.244.1.8:8080 (backend-pod-1, Node1)                 │
│   33% → 10.244.2.9:8080 (backend-pod-2, Node2) ← Selected      │
│   33% → 10.244.3.7:8080 (backend-pod-3, Node3)                 │
└───────────────────┬─────────────────────────────────────────────┘
                    │ DNAT: 10.96.10.20 → 10.244.2.9
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Pod: backend-pod-2 (10.244.2.9)                                 │
│ Container: nginx listening on :8080                             │
│ Processes request, returns 200 OK                              │
└───────────────────┬─────────────────────────────────────────────┘
                    │ Response path (reverse NAT)
                    ↓
                Back to client via LB
```

## Performance & Best Practices

### Service Performance
- **ClusterIP:** Fastest (local iptables/IPVS, no extra hops)
- **NodePort:** Medium (extra hop to node, then routing)
- **LoadBalancer:** Slowest (cloud LB → node → pod)

### Optimization Tips
1. **Use IPVS** for clusters with >1000 services
2. **externalTrafficPolicy: Local** for latency-sensitive apps
3. **Session affinity** for stateful applications
4. **Headless services** for databases and StatefulSets
5. **Topology-aware routing** via EndpointSlice topology hints (`hints.forZones`) for multi-zone clusters — note: the old `topologyKeys` field was removed in Kubernetes 1.27

### Monitoring
```bash
# Check service endpoint slices (modern)
kubectl get endpointslices -l kubernetes.io/service-name=backend-service

# Legacy endpoints view (still works)
kubectl get endpoints backend-service

# View kube-proxy logs
kubectl logs -n kube-system kube-proxy-xxxxx

# Test service resolution
kubectl run -it --rm debug --image=busybox --restart=Never -- sh
  wget -O- http://backend-service:8080
```

---

**Next:** [Ingress & External Traffic](03-ingress-external-traffic.md)
