# Pod Lifecycle in Kubernetes

## Overview

A pod goes through several phases from creation to termination. Understanding this lifecycle is critical for building resilient applications.

## Pod Phases

| Phase | Meaning |
|-------|---------|
| `Pending` | Accepted by cluster, not yet running (scheduling or image pull) |
| `Running` | Bound to node, at least one container running |
| `Succeeded` | All containers terminated successfully (exit code 0) |
| `Failed` | All containers terminated, at least one with non-zero exit |
| `Unknown` | Pod state cannot be determined (node communication lost) |

## Complete Lifecycle Flow

```
kubectl apply -f pod.yaml
     │
     ├─→ 1. API Server validation & admission
     │         - Admission controllers run (webhooks, policies)
     │         - Pod spec validated
     │         - Stored in etcd
     │
     ├─→ 2. Scheduler assigns pod to node
     │         - Filter: Which nodes meet requirements?
     │         - Score: Which node is best?
     │         - Bind: Assign pod to node
     │
     ├─→ 3. kubelet on node picks up pod
     │         - Watches API server for pods assigned to its node
     │         - Pulls container images (if not cached)
     │
     ├─→ 4. Container Runtime (containerd/CRI-O) creates containers
     │         - Pull image from registry
     │         - Create container sandbox
     │         - Set up networking (CNI)
     │
     ├─→ 5. Init containers run (sequential)
     │         - Run to completion before app containers
     │         - If any fails, restart (respecting restartPolicy)
     │
     ├─→ 6. App containers start
     │         - All containers start in parallel
     │         - postStart lifecycle hook fires
     │
     ├─→ 7. Probes begin
     │         - startupProbe (if defined) - wait for startup
     │         - livenessProbe - is container healthy?
     │         - readinessProbe - is container ready for traffic?
     │
     ├─→ 8. Pod running
     │         - Service routes traffic only when ready
     │
     ├─→ 9. Termination triggered (kubectl delete / crash / eviction)
     │
     └─→ 10. Graceful shutdown
              - preStop lifecycle hook fires
              - SIGTERM sent to container
              - Grace period (default 30s)
              - SIGKILL if still running
```

## Init Containers

**Run before app containers.** Must complete successfully.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: backend-pod
spec:
  initContainers:
  
  # 1. Wait for database
  - name: wait-for-db
    image: busybox
    command: ['sh', '-c', 'until nc -z postgres.default.svc.cluster.local 5432; do echo waiting; sleep 2; done']
  
  # 2. Run database migrations
  - name: run-migrations
    image: myapp:latest
    command: ['./migrate', 'up']
    env:
    - name: DATABASE_URL
      valueFrom:
        secretKeyRef:
          name: db-secret
          key: url
  
  # 3. App container starts AFTER both init containers succeed
  containers:
  - name: backend
    image: myapp:latest
    ports:
    - containerPort: 8080
```

**Init container flow:**
```
wait-for-db (runs until db is up)
     ↓ (success)
run-migrations (runs db migrations)
     ↓ (success)
backend (starts serving traffic)
```

**If any init container fails:**
- Entire pod restarts (based on restartPolicy)
- `Pending` phase until all init containers succeed

## Health Probes

### Three Probe Types

| Probe | Purpose | Effect on Failure |
|-------|---------|------------------|
| `startupProbe` | "Is app done starting up?" | Restart container |
| `livenessProbe` | "Is app still alive?" | Restart container |
| `readinessProbe` | "Is app ready for traffic?" | Remove from Service endpoints |

### Probe Mechanisms

| Mechanism | How It Works |
|-----------|-------------|
| `httpGet` | HTTP GET request, success = 200-399 |
| `tcpSocket` | TCP connection, success = connection established |
| `exec` | Run command, success = exit code 0 |
| `grpc` | gRPC health check protocol |

### Complete Probe Configuration

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: backend
    image: myapp:latest
    
    # startupProbe: Wait for slow-starting app
    startupProbe:
      httpGet:
        path: /health/startup
        port: 8080
      failureThreshold: 30   # 30 * 10s = 5 min startup window
      periodSeconds: 10
    
    # livenessProbe: Detect deadlocks, restart if unhealthy
    livenessProbe:
      httpGet:
        path: /health/live
        port: 8080
        httpHeaders:
        - name: Custom-Header
          value: liveness-check
      initialDelaySeconds: 10  # Wait before first probe
      periodSeconds: 10        # Check every 10s
      timeoutSeconds: 5        # Timeout after 5s
      failureThreshold: 3      # Fail 3 times before restart
      successThreshold: 1      # 1 success to be "alive"
    
    # readinessProbe: Only serve traffic when ready
    readinessProbe:
      httpGet:
        path: /health/ready
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 5
      timeoutSeconds: 3
      failureThreshold: 3
      successThreshold: 2  # 2 successes to be "ready"
```

### Probe Effect on Traffic

```
New Pod starts
     │
     ├─→ startupProbe running
     │   - Pod NOT added to Service endpoints
     │   - livenessProbe/readinessProbe PAUSED
     │
     ├─→ startupProbe SUCCESS
     │   - livenessProbe and readinessProbe begin
     │
     ├─→ readinessProbe SUCCESS
     │   - Pod ADDED to Service endpoints
     │   - Traffic starts flowing to pod
     │
     ├─→ readinessProbe FAILS (e.g., db connection lost)
     │   - Pod REMOVED from Service endpoints
     │   - No traffic (but pod keeps running)
     │
     ├─→ readinessProbe SUCCESS again
     │   - Pod ADDED back to endpoints
     │
     └─→ livenessProbe FAILS 3x
         - Container RESTARTED
```

## Container Restart Policies

| Policy | Behavior |
|--------|---------|
| `Always` | Always restart (default for Deployments) |
| `OnFailure` | Restart only if exit code != 0 |
| `Never` | Never restart |

```yaml
apiVersion: v1
kind: Pod
spec:
  restartPolicy: OnFailure  # Jobs use this
  containers:
  - name: batch-job
    image: my-job:latest
```

## Lifecycle Hooks

### postStart Hook

Runs **immediately after** container starts. If it fails, container is killed.

```yaml
containers:
- name: backend
  image: myapp:latest
  lifecycle:
    postStart:
      exec:
        command: ["/bin/sh", "-c", "echo started > /tmp/started"]
      # Or HTTP:
      # httpGet:
      #   host: my-host
      #   path: /register
      #   port: 8080
```

> ⚠️ **Note:** `postStart` runs concurrently with container's entrypoint. Container entrypoint might run BEFORE postStart completes.

### preStop Hook

Runs **before** container receives SIGTERM. Used for graceful shutdown.

```yaml
containers:
- name: nginx
  image: nginx:latest
  lifecycle:
    preStop:
      exec:
        command: ["/usr/sbin/nginx", "-s", "quit"]
  # Or for HTTP:
  # preStop:
  #   httpGet:
  #     path: /shutdown
  #     port: 8080
```

## Graceful Termination Flow

```
kubectl delete pod backend-pod
     │
     ├─→ 1. Pod phase → Terminating
     │       Added to pod's DeletionTimestamp
     │
     ├─→ 2. Endpoints controller
     │       REMOVES pod from Service endpoints
     │       (Traffic stops flowing to pod)
     │
     ├─→ 3. preStop lifecycle hook executes
     │       (e.g., deregister from discovery, drain connections)
     │
     ├─→ 4. SIGTERM sent to container PID 1
     │       App should start graceful shutdown:
     │       - Stop accepting new connections
     │       - Finish in-flight requests
     │       - Flush buffers, close DB connections
     │
     ├─→ 5. Grace period timer starts (default: 30s)
     │       terminationGracePeriodSeconds: 30
     │
     ├─→ 6. If container exits → ✅ Clean termination
     │
     └─→ 7. If grace period expires → SIGKILL
              Container forcefully killed
```

### Configuring Grace Period

```yaml
apiVersion: v1
kind: Pod
spec:
  terminationGracePeriodSeconds: 60  # Give 60s for graceful shutdown
  containers:
  - name: backend
    image: myapp:latest
```

### Proper Signal Handling (Application Code)

```go
// Go example: Handle SIGTERM gracefully
func main() {
    srv := &http.Server{Addr: ":8080"}
    
    // Start server
    go srv.ListenAndServe()
    
    // Wait for termination signal
    sig := make(chan os.Signal, 1)
    signal.Notify(sig, syscall.SIGTERM, syscall.SIGINT)
    <-sig
    
    // Graceful shutdown (max 30s)
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    if err := srv.Shutdown(ctx); err != nil {
        log.Fatal("Server shutdown failed:", err)
    }
    log.Println("Server gracefully stopped")
}
```

## Container States

| State | Meaning |
|-------|---------|
| `Waiting` | Not yet running (pulling image, init containers) |
| `Running` | Executing normally |
| `Terminated` | Finished (success or failure) |

```bash
# View container state
kubectl get pod my-pod -o jsonpath='{.status.containerStatuses[0].state}'

# View last termination reason
kubectl get pod my-pod -o jsonpath='{.status.containerStatuses[0].lastState}'
```

## Pod Conditions

| Condition | Meaning |
|-----------|---------|
| `PodScheduled` | Pod scheduled to a node |
| `ContainersReady` | All containers ready |
| `Initialized` | Init containers completed |
| `Ready` | Pod ready to serve requests |

```bash
kubectl get pod my-pod -o jsonpath='{.status.conditions}'
```

## Common Pod Issues

### CrashLoopBackOff

```
Container crashes → Restart → Crash → Restart → Wait → ...
Backoff: 10s, 20s, 40s, ... up to 5 minutes
```

**Debug:**
```bash
# View current logs
kubectl logs my-pod

# View previous crash logs
kubectl logs my-pod --previous

# Describe for events
kubectl describe pod my-pod
```

**Common causes:**
- Application error on startup
- Missing environment variables
- Database not reachable
- Missing secrets/configmaps

### ImagePullBackOff

```bash
# Check image name and tag
kubectl describe pod my-pod | grep Image

# Check imagePullSecrets
kubectl get pod my-pod -o jsonpath='{.spec.imagePullSecrets}'
```

### OOMKilled (Out of Memory)

```bash
# Check exit code 137 (SIGKILL for OOM)
kubectl get pod my-pod -o jsonpath='{.status.containerStatuses[0].lastState.terminated.exitCode}'
# Should return 137

# Check memory usage
kubectl top pod my-pod

# View resource limits
kubectl describe pod my-pod | grep -A 3 "Limits:"
```

## Resource Limits & Quality of Service

### Resource Requests and Limits

```yaml
spec:
  containers:
  - name: backend
    image: myapp:latest
    resources:
      requests:
        memory: "256Mi"   # Minimum guaranteed
        cpu: "250m"       # 0.25 CPU cores
      limits:
        memory: "512Mi"   # Maximum allowed
        cpu: "500m"       # 0.5 CPU cores
```

### Quality of Service (QoS) Classes

| Class | Condition | Eviction Priority |
|-------|-----------|------------------|
| `Guaranteed` | requests == limits for all | Last evicted |
| `Burstable` | requests < limits | Middle |
| `BestEffort` | No requests/limits | First evicted |

```bash
# Check pod QoS class
kubectl get pod my-pod -o jsonpath='{.status.qosClass}'
```

## Pod Topology

### Affinity and Anti-Affinity

```yaml
spec:
  affinity:
    # Schedule on nodes with SSD
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: disk-type
            operator: In
            values:
            - ssd
    
    # Spread replicas across zones
    podAntiAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchExpressions:
            - key: app
              operator: In
              values:
              - backend
          topologyKey: topology.kubernetes.io/zone
```

## Sidecar Pattern

**Multiple containers** in one pod sharing network and storage.

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  
  # Main app container
  - name: backend
    image: myapp:latest
    ports:
    - containerPort: 8080
  
  # Log shipping sidecar
  - name: log-shipper
    image: fluentd:latest
    volumeMounts:
    - name: logs
      mountPath: /var/log/app
  
  # Metrics sidecar (Prometheus exporter)
  - name: metrics
    image: prometheus-exporter:latest
    ports:
    - containerPort: 9090
  
  volumes:
  - name: logs
    emptyDir: {}
```

