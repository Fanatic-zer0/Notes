# Pod Scheduling in Kubernetes

## Overview

The **kube-scheduler** decides which node a pod runs on. This is a multi-step process of **filtering** (which nodes can run this pod) and **scoring** (which node is best).

## Scheduler Architecture

```
Pod created (schedulerName: default-scheduler)
     │
     ├─→ 1. Scheduler watches for unscheduled pods
     │       Pods with .spec.nodeName == ""
     │
     ├─→ 2. Filtering Phase
     │       Eliminate nodes that cannot run this pod
     │
     ├─→ 3. Scoring Phase
     │       Rank remaining nodes by "best fit"
     │
     ├─→ 4. Binding
     │       Assign pod to highest-scoring node
     │       Write .spec.nodeName to pod spec
     │
     └─→ 5. kubelet on selected node starts pod
```

## Filtering (Predicate) Plugins

```
All Nodes: [Node1, Node2, Node3, Node4, Node5]
     │
     ├─→ NodeResourcesFit: Remove nodes with insufficient CPU/Memory
     │       Node4 (out of CPU) → REMOVED
     │
     ├─→ NodeAffinity: Remove nodes not matching affinity rules
     │       Node3 (wrong zone) → REMOVED
     │
     ├─→ TaintToleration: Remove nodes with intolerable taints
     │       Node5 (GPU-only taint) → REMOVED
     │
     ├─→ PodTopologySpread: Remove nodes that would violate spread
     │       (when maxSkew would be exceeded)
     │
     ├─→ VolumeBinding: Remove nodes where PVC can't be bound
     │
     ├─→ NodePorts: Remove nodes where required ports are occupied
     │
     └─→ Feasible Nodes: [Node1, Node2]
```

## Scoring (Priority) Plugins

```
Feasible Nodes: [Node1, Node2]
     │
     ├─→ LeastAllocated: Prefer nodes with less resource usage
     │       Node1: CPU 20%, Mem 30% → Score: 75
     │       Node2: CPU 60%, Mem 70% → Score: 35
     │
     ├─→ SelectorSpread: Prefer nodes with fewer same-label pods
     │       Node1: 0 backend pods → +10
     │       Node2: 2 backend pods → +0
     │
     ├─→ InterPodAffinity: Prefer nodes near affinity targets
     │
     ├─→ ImageLocality: Prefer nodes that already have the image
     │       Node2: image cached → +5
     │
     └─→ Final Scores:
              Node1: 85 (Winner!)
              Node2: 40
              Scheduler binds pod to Node1
```

## Node Selection Methods

### 1. nodeName (Direct Assignment)

```yaml
apiVersion: v1
kind: Pod
spec:
  nodeName: node1  # Bypass scheduler completely
  containers:
  - name: app
    image: myapp:latest
```

> ⚠️ Bypass scheduler - use only for special cases.

### 2. nodeSelector (Simple Label Matching)

```yaml
# Label the node first
kubectl label node node1 disk=ssd

# Pod targets labeled node
apiVersion: v1
kind: Pod
spec:
  nodeSelector:
    disk: ssd
    zone: us-east-1a
  containers:
  - name: app
    image: myapp:latest
```

### 3. Node Affinity (Expressive Rules)

```yaml
spec:
  affinity:
    nodeAffinity:
      # HARD requirement (must match)
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: kubernetes.io/arch
            operator: In
            values:
            - amd64
          - key: kubernetes.io/os
            operator: In
            values:
            - linux
      
      # SOFT preference (try to match, but not required)
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 80  # High priority preference
        preference:
          matchExpressions:
          - key: disk
            operator: In
            values:
            - ssd
      - weight: 20  # Lower priority preference
        preference:
          matchExpressions:
          - key: zone
            operator: In
            values:
            - us-east-1a
```

**Operator types:**
| Operator | Meaning |
|----------|---------|
| `In` | Label value in set |
| `NotIn` | Label value not in set |
| `Exists` | Label key exists |
| `DoesNotExist` | Label key does not exist |
| `Gt` | Label value greater than |
| `Lt` | Label value less than |

### 4. Taints and Tolerations

**Taints** are on nodes (repel pods). **Tolerations** are on pods (allow scheduling on tainted nodes).

**Taint Effects:**

| Effect | Behavior |
|--------|----------|
| `NoSchedule` | Don't schedule pods without toleration |
| `PreferNoSchedule` | Try not to schedule (soft) |
| `NoExecute` | Evict running pods without toleration |

```bash
# Add taint to node
kubectl taint nodes node1 gpu=true:NoSchedule
kubectl taint nodes node2 environment=production:NoExecute

# Remove taint
kubectl taint nodes node1 gpu=true:NoSchedule-
```

```yaml
# Pod with toleration (can be scheduled on GPU node)
spec:
  tolerations:
  - key: "gpu"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
  
  # Tolerate any taint with this key
  - key: "special-node"
    operator: "Exists"
    effect: "NoSchedule"
  
  # Tolerate NoExecute with grace period
  - key: "node.kubernetes.io/not-ready"
    operator: "Exists"
    effect: "NoExecute"
    tolerationSeconds: 300  # Stay 5 min before eviction
```

### Common Node Taints

```yaml
# Control plane nodes are tainted (prevents app workloads)
# node-role.kubernetes.io/control-plane:NoSchedule

# Spot instance eviction
# cloud.google.com/gke-spot:NoSchedule

# Node not ready
# node.kubernetes.io/not-ready:NoExecute

# Node unreachable
# node.kubernetes.io/unreachable:NoExecute

# Disk pressure
# node.kubernetes.io/disk-pressure:NoSchedule

# Memory pressure
# node.kubernetes.io/memory-pressure:NoSchedule
```

## Pod Affinity and Anti-Affinity

### Co-locate Pods (Affinity)

```yaml
# Frontend pods schedule near backend pods
spec:
  affinity:
    podAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - backend
        topologyKey: kubernetes.io/hostname  # Same node
      # or:
      # topologyKey: topology.kubernetes.io/zone  # Same zone
```

### Spread Pods (Anti-Affinity)

```yaml
# No two frontend replicas on same node
spec:
  affinity:
    podAntiAffinity:
      # HARD: Required across nodes
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: app
            operator: In
            values:
            - frontend
        topologyKey: kubernetes.io/hostname
      
      # SOFT: Prefer different zones
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchExpressions:
            - key: app
              operator: In
              values:
              - frontend
          topologyKey: topology.kubernetes.io/zone
```

## Topology Spread Constraints

**Newer, more powerful spread mechanism:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
spec:
  replicas: 6
  template:
    spec:
      topologySpreadConstraints:
      
      # Spread evenly across zones
      - maxSkew: 1           # Max difference between zone counts
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule  # or ScheduleAnyway
        labelSelector:
          matchLabels:
            app: frontend
      
      # Spread evenly across nodes
      - maxSkew: 1
        topologyKey: kubernetes.io/hostname
        whenUnsatisfiable: ScheduleAnyway  # Best effort
        labelSelector:
          matchLabels:
            app: frontend
```

**Result with 3 zones (us-east-1a/b/c):**
```
us-east-1a: [pod1, pod2]
us-east-1b: [pod3, pod4]
us-east-1c: [pod5, pod6]
```

## Priority and Preemption

### PriorityClass

```yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority
value: 1000000
globalDefault: false
description: "High priority for critical workloads"
---
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: low-priority
value: 100
preemptionPolicy: Never  # Can be scheduled but won't evict others
---
# Assign to pod
spec:
  priorityClassName: high-priority
```

### Preemption Flow

```
High-priority pod cannot be scheduled (no space)
     │
     ├─→ Scheduler identifies nodes
     │       "If I evict these low-priority pods, can I fit it?"
     │
     ├─→ Select victim pods (lowest priority first)
     │
     ├─→ Delete victim pods gracefully
     │       terminationGracePeriodSeconds honored
     │
     └─→ Schedule high-priority pod on freed node
```

## Resource Quota Scheduling

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: compute-quota
  namespace: production
spec:
  hard:
    pods: "20"              # Max 20 pods
    requests.cpu: "4"       # Total CPU requests
    requests.memory: 8Gi    # Total memory requests
    limits.cpu: "8"
    limits.memory: 16Gi
```

**Effect on Scheduling:**
- If namespace quota exceeded, pod won't be scheduled
- Pods must specify requests/limits to be admitted

## LimitRange

**Default resource limits** per namespace:

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: resource-defaults
  namespace: production
spec:
  limits:
  - type: Container
    default:          # Default limits
      cpu: 500m
      memory: 512Mi
    defaultRequest:   # Default requests
      cpu: 100m
      memory: 128Mi
    max:              # Max per container
      cpu: "2"
      memory: 2Gi
    min:              # Min per container
      cpu: 50m
      memory: 64Mi
```

## Descheduler

**Re-balance** already running pods:

```yaml
# Descheduler policy
apiVersion: "descheduler/v1alpha1"
kind: "DeschedulerPolicy"
strategies:
  "RemoveDuplicates":
    enabled: true
  "RemovePodsViolatingTopologySpreadConstraint":
    enabled: true
  "LowNodeUtilization":
    enabled: true
    params:
      nodeResourceUtilizationThresholds:
        thresholds:
          cpu: 20
          memory: 20
        targetThresholds:
          cpu: 50
          memory: 50
```

## Complete Scheduling Example

```
Deployment: frontend (3 replicas)
Requirements:
  - Needs SSD disk (nodeAffinity)
  - Cannot run on control-plane nodes (toleration needed)
  - Spread across zones (topology spread)
  - 250m CPU, 256Mi memory required
```

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      # Tolerate control-plane (usually not needed)
      # tolerations:
      # - key: "node-role.kubernetes.io/control-plane"
      #   operator: "Exists"
      #   effect: "NoSchedule"
      
      # Prefer SSD nodes
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - key: disk
                operator: In
                values:
                - ssd
      
      # Spread across zones and nodes
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            app: frontend
      - maxSkew: 1
        topologyKey: kubernetes.io/hostname
        whenUnsatisfiable: ScheduleAnyway
        labelSelector:
          matchLabels:
            app: frontend
      
      containers:
      - name: frontend
        image: frontend:latest
        resources:
          requests:
            cpu: 250m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 512Mi
```

**Scheduling Result:**
```
Node1 (zone: us-east-1a, disk: ssd) → frontend-abc
Node2 (zone: us-east-1b, disk: ssd) → frontend-def
Node3 (zone: us-east-1c, disk: hdd) → frontend-ghi (no ssd, but best available)
```

## Troubleshooting Scheduling

```bash
# Pod stuck in Pending
kubectl describe pod my-pod
# Look for: "Events" section showing why scheduling failed

# Common messages:
# "0/3 nodes are available: 3 Insufficient cpu"
# "0/3 nodes are available: 3 node(s) didn't match Pod's node affinity"
# "0/3 nodes are available: 3 node(s) had taint {...}, that the pod didn't tolerate"

# View node resources
kubectl describe node node1 | grep -A 5 "Allocated resources"

# View node labels
kubectl get nodes --show-labels

# View node taints
kubectl get nodes -o custom-columns=NAME:.metadata.name,TAINTS:.spec.taints

# Check if scheduler is running
kubectl get pods -n kube-system | grep scheduler

# View scheduler logs
kubectl logs -n kube-system kube-scheduler-controlplane
```
