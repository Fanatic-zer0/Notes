# Autoscaling in Kubernetes

## Overview

Kubernetes has three distinct autoscaling mechanisms:

| Scaler | What it scales | Metric-based |
|--------|---------------|--------------|
| **HPA** (Horizontal Pod Autoscaler) | Number of pod replicas | CPU, memory, custom metrics |
| **VPA** (Vertical Pod Autoscaler) | CPU/memory requests per pod | Historical usage |
| **CA** (Cluster Autoscaler) | Number of nodes | Pending pods, unused nodes |

## 1. Horizontal Pod Autoscaler (HPA)

**Scales the NUMBER of pods** based on observed metrics.

### How HPA Works

```
Every 15 seconds (default):
     │
     ├─→ 1. HPA controller queries Metrics API
     │       /apis/metrics.k8s.io/v1beta1/namespaces/default/pods
     │
     ├─→ 2. Calculate desired replicas
     │       desiredReplicas = ceil(currentReplicas * (currentMetricValue / desiredMetricValue))
     │
     ├─→ Example:
     │       Current: 3 pods, CPU usage: 80%
     │       Target: 50% CPU
     │       Desired: ceil(3 * (80/50)) = ceil(4.8) = 5 pods
     │
     ├─→ 3. Update Deployment/ReplicaSet replicas
     │       Scale up: 3 → 5 pods
     │
     └─→ 4. Cooldown period before next scale
              Scale up cooldown: 0s (immediate)
              Scale down cooldown: 5 min (default)
```

### HPA Configuration

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  
  minReplicas: 2    # Always run at least 2
  maxReplicas: 20   # Never exceed 20
  
  metrics:
  
  # CPU (most common)
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # Target 70% CPU
  
  # Memory
  - type: Resource
    resource:
      name: memory
      target:
        type: AverageValue
        averageValue: 500Mi  # 500Mi per pod
  
  # Custom metric (requests/second)
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: 1000  # 1000 req/s per pod
  
  # External metric (SQS queue depth)
  - type: External
    external:
      metric:
        name: sqs_queue_length
        selector:
          matchLabels:
            queue: backend-queue
      target:
        type: AverageValue
        averageValue: 100  # 100 messages per pod
  
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0  # Scale up immediately
      policies:
      - type: Pods
        value: 4               # Add max 4 pods per period
        periodSeconds: 15
      - type: Percent
        value: 100             # Or double pods per period
        periodSeconds: 15
      selectPolicy: Max        # Use whichever adds more pods
    
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
      policies:
      - type: Pods
        value: 1               # Remove max 1 pod per period
        periodSeconds: 60
```

### HPA Scaling Visualization

```
Traffic Pattern:
  08:00 ─────────╮
  09:00          │ Peak traffic
  10:00          │
  11:00 ─────────╯
  12:00 ──────────────── Low traffic

Pod Count (maxReplicas: 10, minReplicas: 2):
  08:00: 2 pods (min)
  08:15: 4 pods (CPU 70%+)
  08:30: 7 pods (CPU spike)
  09:00: 10 pods (max, during peak)
  11:00: 8 pods (traffic dropping)
  11:30: 5 pods (5-min stabilization)
  12:00: 3 pods
  12:30: 2 pods (back to min)
```

### Required: Metrics Server

```bash
# Install metrics-server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Verify
kubectl top pods
kubectl top nodes
```

## 2. Vertical Pod Autoscaler (VPA)

**Adjusts resource requests/limits** for individual pods based on historical usage.

### VPA Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ VPA Components                                                  │
│                                                                  │
│  1. VPA Recommender                                             │
│     - Analyzes historical CPU/memory usage                      │
│     - Produces recommendations                                  │
│     - Does NOT apply changes                                    │
│                                                                  │
│  2. VPA Updater                                                 │
│     - Watches VPA recommendations                               │
│     - Evicts pods that need resource updates                    │
│     - (Pod restarts with new resources)                         │
│                                                                  │
│  3. VPA Admission Plugin                                        │
│     - Mutating webhook                                          │
│     - Modifies resource requests on new/updated pods           │
└─────────────────────────────────────────────────────────────────┘
```

### VPA Configuration

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: backend-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  
  updatePolicy:
    updateMode: "Auto"    # Auto, Off, Initial, or Recreate
    # Auto:    Recommend AND update (evicts pods)
    # Off:     Recommend only (no changes)
    # Initial: Set requests only at creation
    # Recreate: Evict and recreate pods
  
  resourcePolicy:
    containerPolicies:
    - containerName: backend
      minAllowed:
        cpu: 100m
        memory: 50Mi
      maxAllowed:
        cpu: "4"
        memory: 4Gi
      controlledResources: [cpu, memory]
      controlledValues: RequestsAndLimits  # or RequestsOnly
```

### VPA Recommendation Flow

```
VPA Recommender analyzes last 8 days of metrics
     │
     ├─→ Percentile calculation:
     │       CPU target: 90th percentile
     │       Memory target: 90th percentile + safety margin
     │
     ├─→ Example findings:
     │       Pod was configured: requests=100m, limits=500m
     │       Actual usage: P90 CPU=350m, P90 Memory=300Mi
     │
     ├─→ VPA recommendation:
     │       requests.cpu: 350m (was 100m)
     │       requests.memory: 350Mi (was 256Mi)
     │       limits.cpu: 700m (2x requests)
     │       limits.memory: 700Mi (2x requests)
     │
     └─→ Applied on pod restart/eviction
```

```bash
# View VPA recommendations (without applying)
kubectl describe vpa backend-vpa
```

> ⚠️ **Important:** Do NOT use HPA and VPA (Auto mode) on the same pod simultaneously for the same metric. Use VPA `Off` mode for recommendations only when HPA is active.

## 3. Cluster Autoscaler (CA)

**Scales the number of NODES** based on pod scheduling pressure.

### Scale Up Flow

```
Pending Pod (can't be scheduled)
     │
     ├─→ 1. CA watches for pods in Pending state
     │
     ├─→ 2. Simulate: Could this pod fit if we add a node?
     │
     ├─→ 3. Select best node group to expand
     │       (based on pod requirements and node group templates)
     │
     ├─→ 4. Call cloud provider API
     │       AWS: auto-scaling group.setDesiredCapacity(current + 1)
     │       GCP: managed instance group resize
     │
     ├─→ 5. New node joins cluster
     │       ~2-5 minutes (cloud provider dependent)
     │
     ├─→ 6. Node ready
     │       kubelet registers, reports capacity
     │
     └─→ 7. Scheduler places pending pods on new node
```

### Scale Down Flow

```
Node underutilized for 10+ minutes
     │
     ├─→ 1. CA finds nodes with low utilization
     │       CPU < 50% AND Memory < 50% (default thresholds)
     │
     ├─→ 2. Can all pods on this node be moved?
     │       - No system pods (kube-system)?
     │       - No PVs that can't be migrated?
     │       - PodDisruptionBudget allows eviction?
     │
     ├─→ 3. Simulate: Do all pods fit on remaining nodes?
     │
     ├─→ 4. Cordon node (no new pods)
     │
     ├─→ 5. Drain pods gracefully
     │       Respects terminationGracePeriodSeconds
     │       Respects PodDisruptionBudget
     │
     ├─→ 6. Call cloud API to remove node
     │
     └─→ 7. Node terminated
```

### Cluster Autoscaler Config (AWS EKS)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  template:
    spec:
      containers:
      - name: cluster-autoscaler
        image: registry.k8s.io/autoscaling/cluster-autoscaler:v1.28.0
        command:
        - ./cluster-autoscaler
        - --cloud-provider=aws
        - --namespace=kube-system
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/my-cluster
        - --balance-similar-node-groups    # Balance across multiple AZs
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste          # Strategy: least-waste, most-pods, random, price
        - --scale-down-delay-after-add=10m
        - --scale-down-unneeded-time=10m
        - --scale-down-utilization-threshold=0.5  # 50% utilization threshold
```

### Expanders

| Expander | Strategy |
|----------|----------|
| `random` | Randomly select node group |
| `most-pods` | Expand group that can schedule most pending pods |
| `least-waste` | Minimize wasted CPU/memory |
| `price` | Prefer cheapest node group |
| `priority` | User-defined priorities |

## PodDisruptionBudget (PDB)

**Protect pods during voluntary disruptions** (node drain, rolling updates, CA scale-down).

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: backend-pdb
spec:
  # Option 1: Minimum available
  minAvailable: 2     # At least 2 pods must always be running
  
  # Option 2: Maximum unavailable
  # maxUnavailable: 1  # At most 1 pod can be down at a time
  
  # Can also use percentages:
  # minAvailable: "80%"
  # maxUnavailable: "20%"
  
  selector:
    matchLabels:
      app: backend
```

**Effect on Cluster Autoscaler:**
```
CA wants to drain node-3 (has 3 backend pods)
     │
     ├─→ Check PDB: minAvailable=2
     │
     ├─→ Total backend pods: 6
     │       - 2 on node-1
     │       - 1 on node-2
     │       - 3 on node-3 (to be drained)
     │
     ├─→ If all 3 on node-3 evicted: 3 remaining
     │       3 >= 2 (minAvailable) → OK to drain!
     │
     └─→ CA evicts pods one by one, checking PDB each time
```

## KEDA (Event-Driven Autoscaling)

**Advanced HPA** - Scale based on external events/queues.

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: backend-scaler
spec:
  scaleTargetRef:
    name: backend-deployment
  minReplicaCount: 0    # Scale to ZERO when idle!
  maxReplicaCount: 100
  cooldownPeriod: 300
  triggers:
  
  # AWS SQS
  - type: aws-sqs-queue
    metadata:
      queueURL: https://sqs.us-east-1.amazonaws.com/123/my-queue
      queueLength: "50"     # Target 50 msgs per pod
      awsRegion: us-east-1
  
  # Kafka topic
  - type: kafka
    metadata:
      bootstrapServers: kafka:9092
      consumerGroup: my-group
      topic: orders
      lagThreshold: "100"   # 100 unprocessed messages per pod
  
  # Prometheus metric
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: http_requests_total
      threshold: "1000"
      query: sum(rate(http_requests_total{app="backend"}[2m]))
  
  # Cron schedule
  - type: cron
    metadata:
      timezone: America/New_York
      start: "0 9 * * 1-5"   # 9 AM weekdays
      end: "0 17 * * 1-5"    # 5 PM weekdays
      desiredReplicas: "10"  # Scale to 10 during business hours
```

## Multi-Dimensional Autoscaling

**Combining HPA + CA:**

```
Traffic spike detected
     │
     ├─→ HPA detects high CPU
     │       Wants to scale from 5 to 15 pods
     │
     ├─→ HPA updates Deployment replicas: 15
     │
     ├─→ 10 new pods → Pending (not enough node capacity)
     │
     ├─→ CA detects Pending pods
     │       Can't fit 10 pods on existing nodes
     │
     ├─→ CA adds 3 new nodes
     │       Each node can fit 4-5 pods
     │
     ├─→ Pending pods scheduled on new nodes
     │
     └─→ All 15 pods running, traffic handled

Traffic drops:
     │
     ├─→ HPA scales down to 3 pods
     │       Removes 12 pods
     │
     ├─→ 3 nodes now underutilized (10 min timer)
     │
     └─→ CA removes 3 nodes
```

## Complete Production Setup

```yaml
# 1. Metrics Server (required for HPA)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# 2. HPA for application scaling
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 65
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50       # Max 50% increase per minute
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10       # Max 10% decrease per 5 minutes
        periodSeconds: 300

# 3. PDB for rolling update protection
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: backend-pdb
spec:
  maxUnavailable: 1
  selector:
    matchLabels:
      app: backend

# 4. VPA in recommendation mode (when HPA active)
---
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: backend-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  updatePolicy:
    updateMode: "Off"  # Recommendations only, don't auto-update
```

## Troubleshooting

```bash
# HPA not scaling up
kubectl describe hpa backend-hpa
# Look for: "Conditions" section and "Events"

# Check if metrics are available
kubectl get --raw /apis/metrics.k8s.io/v1beta1/namespaces/default/pods

# View HPA events
kubectl get events --field-selector reason=SuccessfulRescale

# CA not adding nodes
kubectl logs -n kube-system -l app=cluster-autoscaler

# View CA status
kubectl describe configmap -n kube-system cluster-autoscaler-status

# VPA recommendations
kubectl describe vpa backend-vpa | grep Recommendation -A 20
```

