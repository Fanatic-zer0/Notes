# Pod Lifecycle Issues - Troubleshooting Guide

## Version Scope

Important version-aware notes:
- `kubectl debug` is built in on modern kubectl; the old `kubectl-debug` plugin is no longer the primary recommendation.
- Native sidecar containers are `stable` in Kubernetes `1.33` and enabled by default.
- Some restart-backoff timings can differ if cluster operators enable newer kubelet feature gates such as reduced CrashLoop backoff decay.
- Lifecycle hooks now include `sleep` handlers in addition to `exec` and `httpGet`; older material often omits that.

## 1. CrashLoopBackOff

### Overview
`CrashLoopBackOff` is a Kubernetes status string indicating that a container is stuck in a restart loop and kubelet is applying exponential restart backoff. On most clusters, the default delay starts at `10s`, doubles on repeated failures, and is capped at `300s`.

In Kubernetes `1.33+`, keep in mind that operators can change this behavior with kubelet feature gates and configuration. If `ReduceDefaultCrashLoopBackOffDecay` is enabled, the default backoff can start lower and cap earlier than the long-standing `10s -> 300s` behavior.

**Root Causes:**
- Application crashes due to bugs or misconfigurations
- Missing dependencies (databases, services, config files)
- Out of memory (OOMKilled)
- Incorrect container entrypoint or command
- Application exits immediately (exit code 0 or non-zero)
- Resource constraints

---

### Diagnostic Commands

#### 1. Check Pod Status and Recent Events
```bash
# Get pod status overview
kubectl get pod <pod-name> -n <namespace>

# Detailed pod description with events
kubectl describe pod <pod-name> -n <namespace>

# Watch pod status in real-time
kubectl get pod <pod-name> -n <namespace> -w
```

**What to look for:**
- `Restart Count`: High number indicates persistent issue
- `Last State`: Shows exit code and reason
- `Events`: Shows crash history and reasons

#### 2. Get Logs from Current Container
```bash
# Current container logs
kubectl logs <pod-name> -n <namespace>

# Specific container in multi-container pod
kubectl logs <pod-name> -n <namespace> -c <container-name>

# Follow logs in real-time
kubectl logs <pod-name> -n <namespace> -f

# Get last N lines
kubectl logs <pod-name> -n <namespace> --tail=100
```

#### 3. Get Logs from Previous Crashed Container
```bash
# Previous container logs (CRITICAL for CrashLoopBackOff)
kubectl logs <pod-name> -n <namespace> --previous

# Previous logs for specific container
kubectl logs <pod-name> -n <namespace> -c <container-name> --previous

# Save previous logs to file for analysis
kubectl logs <pod-name> -n <namespace> --previous > crash-logs.txt

# Get timestamps to correlate with events
kubectl logs <pod-name> -n <namespace> --previous --timestamps
```

**Pro Tip:** If the container crashes too quickly, you may miss logs. Use `--previous` immediately after seeing the crash.

---

### Exit Code Analysis

#### Check Container Exit Codes
```bash
# Get exit code from pod status
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.status.containerStatuses[0].lastState.terminated.exitCode}'

# Get complete termination details
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.status.containerStatuses[*].lastState.terminated}'

# Get exit codes for all containers
kubectl get pod <pod-name> -n <namespace> -o json | jq '.status.containerStatuses[] | {name: .name, exitCode: .lastState.terminated.exitCode, reason: .lastState.terminated.reason}'
```

#### Common Exit Codes and Meanings

| Exit Code | Meaning | Likely Cause | Action |
|-----------|---------|--------------|--------|
| **0** | Success | Application exited normally but shouldn't | Check if main process exits too quickly; add `sleep infinity` or proper daemon |
| **1** | General error | Application-specific error | Check application logs for stack traces |
| **2** | Misuse of shell builtin | Command not found, syntax error | Verify entrypoint and command syntax |
| **126** | Command cannot execute | Permission denied | Check file permissions: `chmod +x` on binary |
| **127** | Command not found | Entrypoint/binary doesn't exist | Verify path: `ls -la /path/to/binary` |
| **128+n** | Fatal error signal | Signal n caused termination | See signal table below |
| **130** | Terminated by Ctrl+C | SIGINT (Signal 2) | Usually manual termination |
| **137** | **OOMKilled** | **Out of Memory (SIGKILL)** | **Increase memory limits or fix memory leak** |
| **139** | **Segmentation Fault** | **SIGSEGV (Signal 11)** | **Application bug, core dump needed** |
| **143** | Graceful termination | SIGTERM (Signal 15) | Normal shutdown, check why pod was killed |
| **255** | Exit status out of range | Application returned invalid code | Check application exit handling |

#### Signal to Exit Code Mapping
```bash
# Exit Code = 128 + Signal Number
# Common signals:
# SIGHUP (1)   = 129
# SIGINT (2)   = 130
# SIGQUIT (3)  = 131
# SIGKILL (9)  = 137  ← OOMKilled
# SIGSEGV (11) = 139  ← Segmentation Fault
# SIGTERM (15) = 143
# SIGPIPE (13) = 141
```

---

### Deep-Dive: OOMKilled (Exit Code 137)

#### Identify OOM Issues
```bash
# Check if pod was OOMKilled
kubectl describe pod <pod-name> -n <namespace> | grep -i oom

# Check last termination reason
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.status.containerStatuses[0].lastState.terminated.reason}'

# Get memory limits and requests
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[0].resources}'

# Compare with actual memory usage
kubectl top pod <pod-name> -n <namespace>
```

#### Real-Time Memory Monitoring
```bash
# Monitor memory usage
kubectl top pod <pod-name> -n <namespace> --containers

# Watch memory over time
watch -n 2 'kubectl top pod <pod-name> -n <namespace>'

# Get node-level memory pressure
kubectl describe node <node-name> | grep -A 5 "Memory Pressure"
```

#### Solutions for OOMKilled
```bash
# 1. Increase memory limits
kubectl set resources deployment <deployment-name> -c <container-name> --limits=memory=2Gi

# 2. Check current memory configuration
kubectl get pod <pod-name> -n <namespace> -o yaml | grep -A 5 "resources:"

# Example patch to increase memory
kubectl patch deployment <deployment-name> -n <namespace> --type='json' -p='[
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/resources/limits/memory",
    "value": "2Gi"
  }
]'
```

#### Extract Memory Profile Before OOM
```bash
# For Java applications - enable heap dump on OOM
# Add JVM flags in deployment:
# -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/tmp/heapdump.hprof

# For Go applications - enable pprof
# Import: _ "net/http/pprof"
# Then access: kubectl port-forward <pod> 6060:6060
# curl http://localhost:6060/debug/pprof/heap > heap.out

# For Node.js applications
# --max-old-space-size=2048 --heapsnapshot-signal=SIGUSR2
```

---

### Deep-Dive: Segmentation Fault (Exit Code 139)

#### Identify Segfault
```bash
# Check for segfault in pod events
kubectl describe pod <pod-name> -n <namespace> | grep -i "segmentation\|signal 11\|139"

# Get core dump location (if enabled)
kubectl exec <pod-name> -n <namespace> -- ls -lh /tmp/cores/ /var/crash/
```

#### Enable Core Dumps in Kubernetes
```yaml
# Add to pod spec to enable core dumps
apiVersion: v1
kind: Pod
metadata:
  name: app-with-coredump
spec:
  containers:
  - name: app
    image: myapp:latest
    securityContext:
      # Allow core dumps
      capabilities:
        add: ["SYS_PTRACE"]
    volumeMounts:
    - name: core-dumps
      mountPath: /tmp/cores
    # Set core dump pattern
    command: ["/bin/sh", "-c"]
    args:
    - |
      ulimit -c unlimited
      echo '/tmp/cores/core.%e.%p.%t' > /proc/sys/kernel/core_pattern
      exec /app/binary
  volumes:
  - name: core-dumps
    emptyDir: {}
```

#### Extract Core Dumps
```bash
# List available core dumps
kubectl exec <pod-name> -n <namespace> -- ls -lh /tmp/cores/

# Copy core dump to local machine
kubectl cp <namespace>/<pod-name>:/tmp/cores/core.myapp.1234.1234567890 ./core.dump

# Analyze with gdb (for C/C++ apps)
gdb /path/to/binary ./core.dump
# Inside gdb:
# (gdb) bt          # backtrace
# (gdb) bt full     # full backtrace with variables
# (gdb) info threads # all threads
# (gdb) thread apply all bt # backtrace for all threads
```

#### Debug Segfault Without Core Dumps
```bash
# Run container with debugging tools
kubectl debug <pod-name> -n <namespace> -it --image=nicolaka/netshoot --share-processes

# Attach strace to running process
kubectl exec <pod-name> -n <namespace> -- strace -p <pid> -f -o /tmp/strace.log

# Use an ephemeral debug container
kubectl debug <pod-name> -n <namespace> -it --image=busybox:1.28 --target=<container-name>
```

---

### Advanced Troubleshooting Techniques

#### 1. Prevent Container from Restarting
```bash
# Override entrypoint to keep container alive for debugging
kubectl run debug-pod --image=<your-image> --restart=Never -- /bin/sh -c "sleep infinity"

# Or patch existing deployment
kubectl patch deployment <deployment-name> -n <namespace> --type='json' -p='[
  {
    "op": "add",
    "path": "/spec/template/spec/containers/0/command",
    "value": ["/bin/sh", "-c", "sleep infinity"]
  }
]'

# Then exec into it
kubectl exec -it <pod-name> -n <namespace> -- /bin/sh

# Manually run the original command to see real-time errors
/app/original-entrypoint
```

#### 2. Check for Missing Dependencies
```bash
# Exec into the container (if it stays up long enough)
kubectl exec <pod-name> -n <namespace> -it -- /bin/sh

# Check if required files exist
ls -la /app/
ls -la /etc/config/

# Check environment variables
env | sort

# Test network connectivity to dependencies
nc -zv database-service 5432
curl -v http://api-service:8080/health

# Check DNS resolution
nslookup database-service
dig database-service.namespace.svc.cluster.local
```

#### 3. Enable Verbose Application Logging
```bash
# Set debug log level via environment variable
kubectl set env deployment/<deployment-name> LOG_LEVEL=debug

# Or patch the deployment
kubectl patch deployment <deployment-name> -n <namespace> --type='json' -p='[
  {
    "op": "add",
    "path": "/spec/template/spec/containers/0/env/-",
    "value": {"name": "DEBUG", "value": "true"}
  }
]'
```

#### 4. Check Resource Limits and Requests
```bash
# View current resource configuration
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[*].resources}' | jq

# Check if node has sufficient resources
kubectl describe node <node-name> | grep -A 10 "Allocated resources"

# See actual resource usage vs limits
kubectl top pod <pod-name> -n <namespace> --containers
```

#### 5. Analyze Container Startup Sequence
```bash
# Get detailed container state history
kubectl get pod <pod-name> -n <namespace> -o json | jq '.status.containerStatuses[] | {
  name: .name,
  restartCount: .restartCount,
  lastState: .lastState,
  ready: .ready,
  started: .started
}'

# Check for init container failures
kubectl get pod <pod-name> -n <namespace> -o json | jq '.status.initContainerStatuses[]'

# Get timeline of pod events
kubectl get events -n <namespace> --field-selector involvedObject.name=<pod-name> --sort-by='.lastTimestamp'
```

---

### Best Practices and Prevention

#### 1. Health Checks Configuration
```yaml
# Proper liveness and readiness probes
apiVersion: v1
kind: Pod
metadata:
  name: healthy-pod
spec:
  containers:
  - name: app
    image: myapp:latest
    livenessProbe:
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 30    # Wait for app to start
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3         # Allow 3 failures before restart
    readinessProbe:
      httpGet:
        path: /ready
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 5
      successThreshold: 1
```

#### 2. Resource Limits Best Practices
```yaml
resources:
  requests:
    memory: "256Mi"      # Guaranteed minimum
    cpu: "250m"
  limits:
    memory: "512Mi"      # Maximum allowed (OOM kill at this point)
    cpu: "1000m"         # Throttled if exceeded
```

**Guidelines:**
- Set `requests` = average usage
- Set `limits` = peak usage + 20% buffer
- For memory: `limits` should be 1.5-2x `requests`
- Monitor actual usage with `kubectl top pod` for 7 days before setting limits

#### 3. Startup Configuration
```yaml
# Use startup probe for slow-starting containers
startupProbe:
  httpGet:
    path: /healthz
    port: 8080
  initialDelaySeconds: 0
  periodSeconds: 10
  failureThreshold: 30      # Allow up to 300s for startup (30 × 10s)
```

#### 4. Graceful Shutdown Handling
```yaml
lifecycle:
  preStop:
    exec:
      command: ["/bin/sh", "-c", "sleep 15"]  # Allow time for connection draining
# Increase termination grace period
terminationGracePeriodSeconds: 60
```

---

### Quick Reference Checklist

When encountering CrashLoopBackOff:

- [ ] `kubectl describe pod` - Check events and last state
- [ ] `kubectl logs --previous` - Get logs from crashed container
- [ ] Check exit code - 137 (OOM), 139 (segfault), 127 (not found)
- [ ] `kubectl top pod` - Monitor memory/CPU usage
- [ ] Verify image and tag are correct
- [ ] Check environment variables and secrets/configmaps exist
- [ ] Test entrypoint/command syntax
- [ ] Verify network connectivity to dependencies
- [ ] Review resource limits and requests
- [ ] Check liveness probe configuration
- [ ] Examine application logs for stack traces
- [ ] Consider using `kubectl debug` for live troubleshooting

---

### Common Scenarios and Solutions

#### Scenario 1: Java Application OOMKilled
```bash
# Problem: Exit code 137, Java heap exhausted
# Solution 1: Increase memory limits
kubectl set resources deployment myapp --limits=memory=4Gi

# Solution 2: Tune JVM heap size (set to 75% of container limit)
kubectl set env deployment/myapp JAVA_OPTS='-Xmx3g -Xms3g'

# Solution 3: Enable heap dump on OOM
kubectl set env deployment/myapp JAVA_OPTS='-XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/tmp/heapdump.hprof'
```

#### Scenario 2: Application Can't Connect to Database
```bash
# Problem: Logs show "connection refused" or "unknown host"
# Debug steps:
kubectl exec <pod-name> -- nslookup mysql-service
kubectl exec <pod-name> -- nc -zv mysql-service 3306
kubectl exec <pod-name> -- env | grep -i db

# Common fixes:
# - Wrong service name in connection string
# - Database not ready when app starts (add retry logic)
# - Network policy blocking traffic
# - Wrong namespace (use FQDN: mysql-service.database.svc.cluster.local)
```

#### Scenario 3: Missing ConfigMap or Secret
```bash
# Problem: Container crashes with "file not found" or "environment variable missing"
# Check if ConfigMap/Secret exists:
kubectl get configmap <configmap-name> -n <namespace>
kubectl get secret <secret-name> -n <namespace>

# Verify the pod references the correct name:
kubectl get pod <pod-name> -n <namespace> -o yaml | grep -A 10 "configMapRef\|secretRef"

# Check mounted volume:
kubectl exec <pod-name> -- ls -la /etc/config/
```

#### Scenario 4: Binary Permission Denied (Exit Code 126)
```bash
# Problem: "permission denied" when starting container
# Fix: Rebuild image with executable permissions
# In Dockerfile:
# COPY --chmod=755 app /app/app
# Or:
# RUN chmod +x /app/app

# Temporary workaround:
kubectl patch deployment <name> --type='json' -p='[
  {
    "op": "add",
    "path": "/spec/template/spec/containers/0/command",
    "value": ["/bin/sh", "-c", "chmod +x /app/binary && /app/binary"]
  }
]'
```

---

### Tools and Utilities

#### Built-In `kubectl debug`
```bash
# `kubectl debug` is built into modern kubectl versions
# Use a copied debug pod with shared process namespace
kubectl debug <pod-name> -it --image=nicolaka/netshoot --share-processes --copy-to=debug-pod
```

**Version note:** on Kubernetes `1.33+`, prefer the built-in `kubectl debug` workflows over the historical `kubectl-debug` Krew plugin unless your environment specifically standardizes on that plugin.

#### Stern - Multi-Pod Log Tailing
```bash
# Install stern
brew install stern

# Tail logs from all pods in deployment
stern <deployment-name> -n <namespace>

# Include previous logs
stern <deployment-name> -n <namespace> --since 1h --include-previous
```

#### kubetail - Aggregate Logs
```bash
# Install kubetail
brew tap johanhaleby/kubetail && brew install kubetail

# Tail all pods with label
kubetail -l app=myapp -n <namespace>
```

---

## 2. OOMKilled

### Overview
`OOMKilled` means the Linux kernel killed the container because it exceeded its memory limit. In Kubernetes this typically shows up as exit code `137`, which maps to `SIGKILL`.

This is not just “the pod used a lot of RAM.” It specifically means memory usage crossed a hard boundary. The most common causes are:
- Container memory limit is set too low for the workload
- Application has a memory leak
- Heap size is larger than the container limit
- Sudden startup spikes exceed steady-state memory assumptions
- Node-level memory pressure contributes to eviction or kills
- Sidecars and init behavior were ignored when sizing total pod memory

---

### Fast Triage

Run these first to confirm the problem and identify whether it is container-scoped or node-scoped.

```bash
# Confirm OOMKilled reason and exit code
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.status.containerStatuses[*].lastState.terminated.reason}{"\n"}{.status.containerStatuses[*].lastState.terminated.exitCode}{"\n"}'

# Describe pod to inspect restart history and events
kubectl describe pod <pod-name> -n <namespace>

# Check memory requests and limits on all containers
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.containers[] | {name: .name, requests: .resources.requests, limits: .resources.limits}'

# Check live pod and container usage
kubectl top pod <pod-name> -n <namespace> --containers

# Check if the node is under memory pressure
kubectl describe node <node-name> | grep -A 8 -i 'memory pressure\|allocated resources'
```

**What to verify immediately:**
- `reason: OOMKilled`
- `exitCode: 137`
- Which container was killed in a multi-container pod
- Whether the container limit is close to observed peak usage
- Whether the node itself is under pressure

---

### How OOMKilled Happens

Kubernetes enforces memory `limits` as a hard ceiling. When a process crosses that limit, the kernel OOM killer terminates it. Unlike CPU, memory is not throttled. A container can run near its limit for hours and then die during one spike.

Important distinction:
- `requests.memory` affects scheduling
- `limits.memory` affects whether the kernel kills the container

That means a pod can schedule successfully but still fail immediately if the real working set is larger than the configured limit.

---

### Check Limits, Requests, and Actual Usage

#### Inspect Current Resource Configuration
```bash
# View requests and limits for every container
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.containers[] | {
  name: .name,
  requests: .resources.requests,
  limits: .resources.limits
}'

# Inspect deployment template instead of a live pod
kubectl get deployment <deployment-name> -n <namespace> -o json | jq '.spec.template.spec.containers[] | {
  name: .name,
  requests: .resources.requests,
  limits: .resources.limits
}'

# Quick yaml view
kubectl get deployment <deployment-name> -n <namespace> -o yaml | grep -A 8 'resources:'
```

#### Compare to Live Consumption
```bash
# Pod-level view
kubectl top pod <pod-name> -n <namespace>

# Container-level breakdown
kubectl top pod <pod-name> -n <namespace> --containers

# Namespace-wide sort by memory usage
kubectl top pod -n <namespace> --containers | sort -k4 -h
```

**Tip:** `kubectl top` shows current usage, not peak usage. If the pod dies during short spikes, current usage may look normal after restart. Pair this with restart timestamps and application-level metrics.

---

### Identify Whether Startup Spikes Are the Problem

Many OOM events happen only during bootstrap: cache warmup, JIT compilation, schema loading, large config parsing, or sidecar initialization.

```bash
# Watch restarts while checking usage repeatedly
kubectl get pod <pod-name> -n <namespace> -w

# Capture container start and termination timestamps
kubectl get pod <pod-name> -n <namespace> -o json | jq '.status.containerStatuses[] | {
  name: .name,
  restartCount: .restartCount,
  state: .state,
  lastState: .lastState
}'

# Review recent events in order
kubectl get events -n <namespace> --field-selector involvedObject.name=<pod-name> --sort-by='.lastTimestamp'
```

**Best approach:** if the container only OOMs on startup, add a `startupProbe`, reduce startup parallelism, or temporarily raise limits while profiling the initialization path.

---

### Capture Heap Dumps and Memory Profiles Before the Container Dies

The right technique depends on the runtime. The goal is to write diagnostic output to a mounted volume so it survives long enough to copy out.

#### Java
```bash
# Add JVM flags for heap dump on OOM
kubectl set env deployment/<deployment-name> -n <namespace> \
  JAVA_TOOL_OPTIONS='-XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/dumps/heapdump.hprof'
```

```yaml
# Mount a writable volume for dumps
volumeMounts:
- name: heap-dumps
  mountPath: /dumps
volumes:
- name: heap-dumps
  emptyDir: {}
```

```bash
# Copy heap dump after restart
kubectl cp <namespace>/<pod-name>:/dumps/heapdump.hprof ./heapdump.hprof
```

**Best practice:** set `-Xmx` explicitly below the container limit. If the pod limit is `2Gi`, do not let the JVM assume it can consume the full `2Gi`; leave headroom for metaspace, threads, buffers, and libc.

#### Go
```bash
# If the app exposes pprof, capture heap profile before failure
kubectl port-forward pod/<pod-name> 6060:6060 -n <namespace>
curl http://127.0.0.1:6060/debug/pprof/heap > heap.out

# Capture goroutine dump if leak is suspected
curl http://127.0.0.1:6060/debug/pprof/goroutine?debug=2 > goroutines.txt
```

**Tip:** if the process dies too fast, run a debug replica with a higher limit and the same traffic pattern so profiling can complete.

#### Node.js
```bash
# Constrain old-space size below container limit
kubectl set env deployment/<deployment-name> -n <namespace> \
  NODE_OPTIONS='--max-old-space-size=1536 --heapsnapshot-signal=SIGUSR2'
```

```bash
# Trigger heap snapshot if the container stays alive long enough
kubectl exec <pod-name> -n <namespace> -- kill -USR2 1
```

#### Python
```bash
# Common approach: enable tracemalloc in app startup
python -X tracemalloc app.py

# For gunicorn-based apps, inspect worker behavior and max requests
gunicorn app:app --workers 4 --max-requests 1000 --max-requests-jitter 100
```

**Operational tip:** for Python, memory growth is often easier to diagnose by recycling workers and comparing RSS growth over time rather than waiting for a single fatal dump.

---

### Use a Debug Pod or Temporary Limit Increase to Investigate

If the application dies too quickly to inspect, stabilize it first.

```bash
# Temporarily raise memory limits on the deployment
kubectl set resources deployment <deployment-name> -n <namespace> \
  --containers=<container-name> --requests=memory=1Gi --limits=memory=2Gi

# Rollout and watch
kubectl rollout status deployment/<deployment-name> -n <namespace>
kubectl get pods -n <namespace> -w
```

```bash
# Create a separate debug pod from the same image
kubectl run oom-debug --image=<image> -n <namespace> --restart=Never -- /bin/sh -c 'sleep infinity'

# Inspect environment and mounted files
kubectl exec -it oom-debug -n <namespace> -- /bin/sh
```

**Best approach:** do not leave the temporary higher limit in place without measuring. Use it to capture evidence, then right-size the application or deployment settings.

---

### Node-Level Versus Container-Level Memory Problems

A container OOM and node memory pressure are related but not identical.

#### Container-level OOM
- Container exceeds its own `limits.memory`
- Pod often restarts with `OOMKilled`
- Exit code usually `137`

#### Node-level pressure
- Node is short on free memory overall
- Kubelet may evict pods based on QoS and pressure
- Events may mention `Evicted` rather than only `OOMKilled`

```bash
# Inspect node allocatable versus current allocations
kubectl describe node <node-name>

# Find recent eviction-related events
kubectl get events -A --sort-by='.lastTimestamp' | grep -i 'evict\|memory pressure\|oom'
```

**Tip:** if many unrelated pods restart on the same node, stop focusing only on one deployment. The node may be overcommitted.

---

### Common Root Causes and Fixes

#### 1. JVM Heap Too Large for Container
```bash
# Example: 2Gi limit, keep heap smaller than limit
kubectl set env deployment/<deployment-name> -n <namespace> \
  JAVA_TOOL_OPTIONS='-Xms1g -Xmx1536m -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/dumps/heapdump.hprof'
```

Fix:
- Set `-Xmx` explicitly
- Leave 20-30% memory headroom outside the heap
- Review direct buffers, thread count, metaspace, and native memory

#### 2. Memory Leak in Application
```bash
# Compare restart timing and memory trend from APM/metrics
kubectl top pod <pod-name> -n <namespace> --containers
```

Fix:
- Capture heap/profile snapshots at multiple times
- Look for unbounded caches, request accumulation, goroutine leaks, or worker churn
- Reproduce under representative load, not just idle startup

#### 3. Sidecars Were Not Included in Sizing
```bash
# Check all containers in the pod, not just the main app
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.containers[].name'
kubectl top pod <pod-name> -n <namespace> --containers
```

Fix:
- Sum app container + service mesh sidecar + log forwarder + agent overhead
- Adjust pod-level expectations accordingly

#### 4. Large Startup Spike
Fix:
- Add `startupProbe`
- Delay heavy background jobs until after readiness
- Warm caches gradually
- Increase limit only if the spike is legitimate and expected

---

### Prevention and Sizing Guidance

#### Resource Sizing Rules of Thumb
- Set `requests.memory` close to normal working-set usage
- Set `limits.memory` above observed peak, with headroom
- For runtimes with managed heaps, size the heap below the container limit
- Revisit limits after major releases, dependency changes, or traffic growth

#### Helpful Probe Configuration
```yaml
startupProbe:
  httpGet:
    path: /healthz
    port: 8080
  periodSeconds: 10
  failureThreshold: 30

livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
```

This prevents Kubernetes from killing a slow-starting process before it finishes initialization, which is often misdiagnosed as pure memory failure.

---

### Quick Reference Checklist

When troubleshooting `OOMKilled`:

- [ ] Confirm `reason=OOMKilled` and `exitCode=137`
- [ ] Identify which container was killed
- [ ] Compare `limits.memory` to actual container usage
- [ ] Check whether the failure happens only during startup
- [ ] Inspect node memory pressure and eviction events
- [ ] Capture heap dump, pprof heap, or runtime-specific profile
- [ ] Set runtime heap below container memory limit
- [ ] Account for sidecars and native memory overhead
- [ ] Raise limits temporarily only to collect evidence
- [ ] Fix leaks or reduce memory spikes before finalizing new limits

---

### Practical Tips and Tricks

- Use a writable mounted path for dumps. Writing to a read-only image filesystem usually fails silently or disappears on restart.
- If `kubectl top` is empty, verify Metrics Server is installed before assuming usage is zero.
- For bursty workloads, current memory usage is a weak signal; pair it with APM, Prometheus, or cgroup metrics.
- If a pod only fails on one node, compare kernel version, allocatable memory, and colocated workloads.
- If you increase limits and the app still OOMs, stop tuning Kubernetes first. That usually indicates an application-level leak or a bad heap/runtime configuration.

---

## 3. Pod Stuck in Terminating

### Overview
A pod stuck in `Terminating` has received a delete request, but Kubernetes has not finished cleanup. That usually means one of the shutdown or deletion steps is blocked:
- The container is still running and has not exited within `terminationGracePeriodSeconds`
- A `preStop` hook is hanging
- A finalizer is waiting on some controller that never completes
- The kubelet on the node is unhealthy or unreachable
- A CSI volume or network cleanup step is blocked
- The pod belongs to a StatefulSet or controller where force deletion can cause duplicate identity or storage issues

The right fix depends on what is actually blocking termination. Force deletion is useful, but it is not always the safest first move.

---

### Fast Triage

Start by confirming whether the block is inside the pod lifecycle, on the node, or in the API object metadata.

```bash
# Check current status, node, and deletion timestamp
kubectl get pod <pod-name> -n <namespace> -o wide

# Inspect full pod details including events and finalizers
kubectl get pod <pod-name> -n <namespace> -o yaml

# Describe the pod for recent lifecycle events
kubectl describe pod <pod-name> -n <namespace>

# See whether the pod still has running containers
kubectl get pod <pod-name> -n <namespace> -o json | jq '{
  phase: .status.phase,
  deletionTimestamp: .metadata.deletionTimestamp,
  finalizers: .metadata.finalizers,
  containerStatuses: .status.containerStatuses,
  initContainerStatuses: .status.initContainerStatuses
}'
```

**What to look for immediately:**
- `deletionTimestamp` is set but the pod remains present
- `metadata.finalizers` is non-empty
- Events mention volume detach, unmount, CNI cleanup, or hook failure
- The node hosting the pod is `NotReady` or unreachable
- The pod is part of a StatefulSet or uses persistent storage

---

### How Pod Termination Works

When you delete a pod, Kubernetes does not instantly remove it. The normal flow is:
1. API server sets `deletionTimestamp`
2. Kubelet sends `SIGTERM` to containers
3. `preStop` hook runs if defined
4. Kubernetes waits up to `terminationGracePeriodSeconds`
5. Remaining processes get `SIGKILL`
6. Volumes, networking, and finalizers are cleaned up
7. Pod object is removed from the API

If any of those steps stalls, the pod can remain in `Terminating` for minutes or indefinitely.

---

### Most Common Causes

#### 1. Long or Stuck `preStop` Hook
```bash
# Inspect lifecycle hooks in the pod spec
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.containers[] | {name: .name, lifecycle: .lifecycle}'
```

Symptoms:
- Pod stays in `Terminating` close to or beyond grace period
- Application logs stop, but pod object does not disappear
- Hook runs a command that never exits or waits on unreachable downstream systems

Best approach:
- Keep `preStop` hooks short and deterministic
- Avoid network calls that can hang indefinitely
- Use app-level graceful shutdown logic instead of long shell scripts where possible

#### 2. Finalizers Blocking Deletion
```bash
# Print only finalizers
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.metadata.finalizers}{"\n"}'
```

Symptoms:
- Containers may already be gone, but pod object remains
- No new lifecycle progress happens
- A custom controller or admission workflow stopped reconciling

Common examples:
- Service mesh injection or cleanup controllers
- Custom operators
- Storage-related or policy controllers

#### 3. Kubelet or Node Problems
```bash
# Check node health
kubectl get node <node-name>
kubectl describe node <node-name>
```

Symptoms:
- Pod is stuck on a `NotReady` node
- `kubectl delete pod` returns, but the object never fully clears
- Multiple pods on the same node are stuck terminating

#### 4. Volume Unmount or CSI Cleanup Stuck
```bash
# Inspect pod volumes and PVCs
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.volumes'
kubectl get pvc -n <namespace>
```

Symptoms:
- Events mention mount/unmount, attach/detach, or CSI driver errors
- Persistent volume remains attached to the node
- Stateful workloads are most affected

---

### Safe Diagnostic Commands

#### Check Deletion Timestamp, Grace Period, and Finalizers
```bash
kubectl get pod <pod-name> -n <namespace> -o json | jq '{
  name: .metadata.name,
  deletionTimestamp: .metadata.deletionTimestamp,
  gracePeriodSeconds: .metadata.deletionGracePeriodSeconds,
  finalizers: .metadata.finalizers,
  nodeName: .spec.nodeName
}'
```

#### Inspect Events in Time Order
```bash
kubectl get events -n <namespace> \
  --field-selector involvedObject.name=<pod-name> \
  --sort-by='.lastTimestamp'
```

#### Check if the Process Is Still Running
```bash
# If the pod is still reachable, inspect processes before force deleting
kubectl exec <pod-name> -n <namespace> -- ps aux

# Check whether the main process is still alive
kubectl exec <pod-name> -n <namespace> -- ps -ef
```

#### Confirm Node Reachability
```bash
kubectl get pod <pod-name> -n <namespace> -o wide
kubectl get node <node-name>
kubectl describe node <node-name> | grep -A 10 -i 'ready\|network unavailable\|memory pressure\|disk pressure'
```

---

### Force Delete the Pod

If the pod is already dead or the node is gone, force deletion removes the API object immediately instead of waiting for normal confirmation.

```bash
# Standard force delete
kubectl delete pod <pod-name> -n <namespace> --grace-period=0 --force
```

What this does:
- Skips graceful shutdown timing
- Removes the pod object from the API server quickly
- Does not guarantee the process is actually gone on the node if the kubelet is unhealthy

**Use force delete when:**
- The node is unreachable or permanently lost
- The container is already gone but the API object is stuck
- You have confirmed no critical cleanup step needs to finish

**Be careful with force delete when:**
- The pod writes to persistent volumes
- The pod is part of a StatefulSet
- The workload uses unique network identity or leader election
- You suspect the process may still be running on a partially reachable node

---

### Remove Stuck Finalizers

If a finalizer is the only thing blocking deletion, remove it explicitly after verifying the owning controller is not going to finish on its own.

#### Check Finalizers
```bash
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.metadata.finalizers}{"\n"}'
```

#### Remove All Finalizers from the Pod
```bash
kubectl patch pod <pod-name> -n <namespace> --type='json' -p='[
  {
    "op": "remove",
    "path": "/metadata/finalizers"
  }
]'
```

#### Alternative Merge Patch
```bash
kubectl patch pod <pod-name> -n <namespace> -p '{"metadata":{"finalizers":null}}'
```

**Best approach:** inspect which controller owns the finalizer before removing it. If the finalizer belongs to a storage, backup, or policy controller, blind removal can leave external resources orphaned.

---

### StatefulSet and Persistent Volume Caution

For StatefulSet pods, force deletion can be riskier than for stateless Deployments.

Why:
- Pod identity is stable and reused, for example `app-0`
- Attached storage may still be mounted on the old node
- A replacement pod may come up before the old process is truly dead
- Some databases can suffer split-brain or filesystem corruption if two instances use the same identity or volume

Check ownership before deleting:
```bash
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.metadata.ownerReferences[*].kind}{"\n"}'
kubectl get pvc -n <namespace>
kubectl describe pvc <pvc-name> -n <namespace>
```

**Best approach:** for StatefulSets, confirm the old node is fenced or the process is definitely gone before force deleting.

---

### Troubleshoot `preStop` Hook Problems

#### Inspect Current Hook Configuration
```bash
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.containers[] | {
  name: .name,
  preStop: .lifecycle.preStop,
  terminationMessagePath: .terminationMessagePath
}'
```

#### Check Application Logs Around Shutdown
```bash
kubectl logs <pod-name> -n <namespace> --timestamps

# If the container already restarted before deletion attempts
kubectl logs <pod-name> -n <namespace> --previous --timestamps
```

Typical issues:
- Hook waits on a socket that is already closed
- Hook depends on DNS or a service that is unavailable during teardown
- Hook uses a shell command that never returns
- Hook duration is longer than `terminationGracePeriodSeconds`

Fixes:
- Move complex shutdown into the application process
- Add command timeouts inside the hook
- Shorten the hook and increase grace period only if truly needed

---

### Troubleshoot Node-Unreachable Cases

If the node is down, the API object may outlive the actual process state.

```bash
# See pods still assigned to the node
kubectl get pods -A -o wide --field-selector spec.nodeName=<node-name>

# Inspect node conditions
kubectl describe node <node-name>
```

If the node will not return:
- Force delete stuck pods
- Cordon and drain surviving unhealthy nodes when possible
- Investigate the underlying node failure separately

If the node may still be partially alive:
- Avoid immediately replacing stateful workloads on shared volumes
- Confirm the old process is gone through infrastructure tooling if available

---

### Troubleshoot Volume and CSI Cleanup Problems

Persistent storage issues are a common reason a pod lingers in `Terminating`.

```bash
# Check PVC and PV state
kubectl get pvc <pvc-name> -n <namespace>
kubectl get pv <pv-name>

# Inspect volume attachments if your cluster exposes them
kubectl get volumeattachments

# Review CSI-related events
kubectl get events -A --sort-by='.lastTimestamp' | grep -i 'csi\|attach\|detach\|unmount\|mount'
```

Typical problems:
- Storage backend is slow or unavailable
- Attach/detach controller is stuck
- Node plugin for the CSI driver is unhealthy
- Volume is still attached to a failed node

Best approach:
- Resolve the storage controller issue first when possible
- Do not repeatedly force delete replacement pods if the same volume is still blocked

---

### Best Practices and Prevention

#### 1. Keep Graceful Shutdown Simple
```yaml
terminationGracePeriodSeconds: 30
lifecycle:
  preStop:
    exec:
      command: ["/bin/sh", "-c", "sleep 5"]
```

Use `preStop` only for short, deterministic cleanup. Prefer the main application to trap `SIGTERM` and shut down cleanly.

#### 2. Avoid Long-Running Hooks That Depend on Other Services
- Do not call external APIs without timeouts
- Do not block on database shutdown checks forever
- Do not assume service discovery still works during teardown

#### 3. Watch Finalizers Added by Custom Controllers
```bash
kubectl get pod <pod-name> -n <namespace> -o yaml | grep -A 3 finalizers
```

If custom operators add finalizers, make sure their controllers are highly available and observable.

#### 4. Be Careful With Force Delete in Stateful Workloads
- Verify fencing first
- Verify storage detach status
- Verify no second writer will start on the same disk

---

### Quick Reference Checklist

When a pod is stuck in `Terminating`:

- [ ] Check `deletionTimestamp` and current node assignment
- [ ] Review events for hook, storage, or network cleanup failures
- [ ] Inspect `metadata.finalizers`
- [ ] Confirm whether the process is still running
- [ ] Check whether the node is `NotReady` or unreachable
- [ ] Use `--grace-period=0 --force` only after understanding the risk
- [ ] Remove finalizers only if the controller is stuck or gone
- [ ] Be extra careful with StatefulSets and persistent volumes
- [ ] Verify replacement pods are not racing old instances on shared storage

---

### Practical Tips and Tricks

- A pod disappearing from the API after force delete does not prove the old process died on the node.
- If many pods are stuck terminating on one node, stop debugging at the pod level and inspect kubelet, runtime, and CSI health on that node.
- For stateless Deployments, force delete is usually low risk once you confirm there is no critical cleanup step.
- For StatefulSets, treat force delete as a fencing decision, not just a cleanup shortcut.
- If finalizers keep coming back on recreated pods, fix the responsible controller instead of repeatedly patching pod objects.

---

## 4. Init Container Failures

### Overview
Init containers run before the main application containers. Kubernetes executes them one by one, in order, and does not start the main container until every init container succeeds.

That means a failure in any init container can block the entire pod even when the main image, command, and probes are correct.

Common reasons init containers fail:
- Missing binaries or bad shell commands in the init image
- ConfigMap, Secret, or volume content not mounted as expected
- Network dependency not reachable during bootstrap
- Permission problems while writing to shared volumes
- Wrong image architecture or incompatible base image
- Init script exits non-zero after a validation or migration step

---

### How Init Containers Behave

Important lifecycle rules:
- Init containers run sequentially, not in parallel
- Each init container must complete successfully before the next starts
- Main containers remain in `Waiting` until all init containers finish
- A failing init container can restart repeatedly and keep the pod in `Init:CrashLoopBackOff` or similar states

Because init containers are short-lived, their logs are often more important than live inspection.

---

### Fast Triage

Start by identifying which init container is failing and whether it ever completed once.

```bash
# Show pod phase and init status
kubectl get pod <pod-name> -n <namespace>

# Describe pod for init-specific events
kubectl describe pod <pod-name> -n <namespace>

# Show init container states, reasons, and restart counts
kubectl get pod <pod-name> -n <namespace> -o json | jq '.status.initContainerStatuses[] | {
  name: .name,
  state: .state,
  lastState: .lastState,
  restartCount: .restartCount,
  ready: .ready
}'
```

**What to look for immediately:**
- Which init container is currently failing
- Whether the failure is `Waiting`, `Terminated`, or `CrashLoopBackOff`
- The exit code and termination reason
- Whether the pod is blocked before any app container starts

---

### Check Init Logs Separately

This is the first thing many people miss: init container logs are separate from main container logs.

```bash
# Get logs from a specific init container
kubectl logs <pod-name> -n <namespace> -c <init-container-name>

# If the init container restarted, check previous logs too
kubectl logs <pod-name> -n <namespace> -c <init-container-name> --previous

# Include timestamps for ordering
kubectl logs <pod-name> -n <namespace> -c <init-container-name> --timestamps
```

**Best approach:** always use `-c <init-container-name>` explicitly. If the pod has both init and app containers, the default `kubectl logs` target is often not the one you need.

---

### Inspect Exit Codes and Reasons

```bash
# Exit code of each init container
kubectl get pod <pod-name> -n <namespace> -o json | jq '.status.initContainerStatuses[] | {
  name: .name,
  exitCode: .lastState.terminated.exitCode,
  reason: .lastState.terminated.reason,
  message: .lastState.terminated.message
}'
```

Common patterns:
- `127`: command or script not found
- `126`: permission denied on script or binary
- `1`: generic script failure
- `137`: init container hit memory limit and was OOMKilled
- `ImagePullBackOff`: image could not be pulled before execution even started

If the init container never reaches `Terminated`, inspect `state.waiting.reason` instead.

---

### Why `exec` Into Init Containers Often Fails

Users often try to run `kubectl exec` into an init container, but this only works if that init container is still actively running.

If it exits quickly or keeps restarting, `exec` is unreliable because:
- The init container may already be terminated
- Kubernetes only allows exec into a running container
- The next restart may happen too quickly to catch interactively

#### Try Exec If the Init Container Stays Alive Long Enough
```bash
kubectl exec -it <pod-name> -n <namespace> -c <init-container-name> -- /bin/sh
```

If that fails, the better approach is usually to reproduce the same image and command in a debug pod.

---

### Reproduce the Init Container Manually

When the init container exits too fast, create a separate pod using the same image and run the failing script manually.

```bash
# Create a debug pod from the same image
kubectl run init-debug -n <namespace> --image=<init-image> --restart=Never -- /bin/sh -c 'sleep infinity'

# Exec into it
kubectl exec -it init-debug -n <namespace> -- /bin/sh
```

Inside the debug pod, verify:
- The failing script exists
- The script has execute permission
- Required environment variables are present
- Mounted files or config content exist
- Network dependencies resolve and accept connections

**Best approach:** copy the init container command and run it manually step by step instead of re-running the whole pod repeatedly.

---

### Check Init Container Configuration

```bash
# Inspect all init container definitions
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.initContainers[] | {
  name: .name,
  image: .image,
  command: .command,
  args: .args,
  env: .env,
  volumeMounts: .volumeMounts,
  resources: .resources
}'
```

Things to verify:
- `image` tag is correct
- `command` and `args` match the image contents
- Volume mount paths exist and are writable where expected
- Resource limits are not too low for migrations or setup work
- Environment variables from Secrets or ConfigMaps resolve correctly

---

### Debug Common Failure Types

#### 1. Missing Script or Binary
```bash
# Common error pattern: exit 127
kubectl logs <pod-name> -n <namespace> -c <init-container-name>
```

Typical causes:
- Script path is wrong
- File was copied to a different location in the image
- Shell path is wrong, for example `/bin/bash` in an image that only has `/bin/sh`

Fixes:
- Verify file paths in the image
- Use `command` and `args` that match the image base
- Prefer absolute paths in init commands

#### 2. Permission Denied on Script or Shared Volume
```bash
# Look for exit 126 or filesystem permission errors
kubectl logs <pod-name> -n <namespace> -c <init-container-name> --previous
```

Typical causes:
- Script is not executable
- Container user cannot write to mounted volume
- `securityContext` differs between init and app containers

Fixes:
- Set executable bit in the image build
- Align UID/GID and `fsGroup` if writing shared files
- Verify mount ownership and read-only flags

#### 3. ConfigMap or Secret Missing
```bash
# Check referenced objects
kubectl get configmap <configmap-name> -n <namespace>
kubectl get secret <secret-name> -n <namespace>

# Inspect mounted references in the pod
kubectl get pod <pod-name> -n <namespace> -o yaml | grep -A 12 'configMap:\|secret:'
```

Typical causes:
- Wrong object name
- Wrong namespace assumption
- Required key missing inside the Secret or ConfigMap

Fixes:
- Confirm exact object names
- Validate keys exist
- Check whether env var or mounted file paths match what the script expects

#### 4. Network Dependency Not Ready Yet
```bash
# Test service DNS and reachability from a debug pod
kubectl exec -it init-debug -n <namespace> -- nslookup <service-name>
kubectl exec -it init-debug -n <namespace> -- nc -zv <service-name> <port>
```

Typical causes:
- Database or API dependency is not ready yet
- Wrong service name or port
- NetworkPolicy blocks the traffic

Fixes:
- Add retry logic with backoff in the init script
- Use fully qualified service DNS if needed
- Verify dependency readiness independently

#### 5. Database Migration Job Inside Init Container Fails
```bash
# Check logs carefully for migration errors
kubectl logs <pod-name> -n <namespace> -c <init-container-name> --timestamps
```

Typical causes:
- Schema already locked
- Migration is not idempotent
- Credentials are wrong
- App version and schema version are incompatible

Fixes:
- Make migrations idempotent
- Separate heavy migrations into a Job when appropriate
- Avoid long, risky schema changes inside init containers for high-scale rollouts

#### 6. Init Container OOMKilled
```bash
# Check init resources and last termination reason
kubectl get pod <pod-name> -n <namespace> -o json | jq '.status.initContainerStatuses[] | {
  name: .name,
  reason: .lastState.terminated.reason,
  exitCode: .lastState.terminated.exitCode
}'
```

Fixes:
- Increase memory limits for the init container specifically
- Do not assume app container resources apply to init containers appropriately
- Profile one-time setup tasks like decompression, templating, or migrations

---

### Shared Volume and Handoff Problems

A common pattern is using an init container to write files into a shared `emptyDir` or mounted volume that the main container reads later.

```bash
# Inspect volume mounts on init and app containers
kubectl get pod <pod-name> -n <namespace> -o json | jq '{
  initContainers: [.spec.initContainers[] | {name: .name, volumeMounts: .volumeMounts}],
  containers: [.spec.containers[] | {name: .name, volumeMounts: .volumeMounts}],
  volumes: .spec.volumes
}'
```

Typical problems:
- Init container writes to one path while app reads another
- Files are created as root and app runs as non-root
- Init container expects persistent data but writes to `emptyDir`

Best approach:
- Validate the exact shared path used by both containers
- Align permissions between init and app containers
- Make generated artifacts observable and easy to inspect

---

### Best Practices and Prevention

#### 1. Keep Init Containers Focused
- Use init containers for short bootstrap tasks
- Avoid embedding large, multi-step operational logic in shell scripts
- Move long-running orchestration into Jobs or controllers when appropriate

#### 2. Make Init Scripts Deterministic
```bash
# Good pattern inside scripts
set -euo pipefail
```

Fail early, log clearly, and avoid silent retries with no timeout.

#### 3. Separate Validation From Mutation
- One init container can validate dependencies
- Another can generate files or do setup
- Avoid one opaque script that does everything with no clear failure stage

#### 4. Set Resource Limits Deliberately
- Database migrations, archive extraction, and template rendering can be memory-heavy
- Size init container resources based on the actual setup work, not the steady-state app process

#### 5. Prefer Jobs for Heavy One-Time Work
- Use init containers for per-pod setup
- Use a Job for cluster-wide migrations or long-running bootstrapping tasks

---

### Quick Reference Checklist

When troubleshooting init container failures:

- [ ] Identify which init container is failing
- [ ] Get logs with `kubectl logs -c <init-container-name>`
- [ ] Check `--previous` logs if it restarts quickly
- [ ] Inspect exit code, reason, and restart count
- [ ] Verify image, command, args, and shell path
- [ ] Confirm Secrets, ConfigMaps, and volume mounts exist
- [ ] Reproduce the command in a debug pod if `exec` is not possible
- [ ] Check network reachability to dependencies
- [ ] Verify shared volume paths and permissions
- [ ] Move heavy migrations or global setup to a Job when needed

---

### Practical Tips and Tricks

- If the pod says `PodInitializing`, the app container logs may be empty because the main container never started.
- `kubectl describe pod` is especially useful for init containers because it shows the order and last failure reason clearly.
- If an init container fails once and later succeeds, check whether your bootstrap logic is depending on race conditions or external timing.
- Keep init images small but not too minimal; removing every debugging tool can slow real incident response.
- For repeat failures during rollout, compare the old ReplicaSet template to the new one. Init container regressions are often introduced by small path or env var changes.

---

## 5. ImagePullBackOff

### Overview
`ImagePullBackOff` means Kubernetes tried to pull a container image, failed, and is now retrying with backoff. The root issue is usually one of these:
- Registry authentication failed
- The image name, tag, or digest is wrong
- The registry endpoint is unreachable from the node
- The referenced image exists in one registry account or region, but the cluster is pulling from another
- ECR or other short-lived registry tokens expired
- A tag was moved, deleted, or no longer matches the expected digest

This is a node-side problem first. The kubelet or container runtime on the worker node performs the image pull, not the Kubernetes API server.

---

### Fast Triage

Start with pod events. They usually tell you whether the failure is auth, name resolution, not found, or digest mismatch.

```bash
# Basic pod status
kubectl get pod <pod-name> -n <namespace>

# Detailed events and pull errors
kubectl describe pod <pod-name> -n <namespace>

# Show waiting reasons for all containers
kubectl get pod <pod-name> -n <namespace> -o json | jq '.status.containerStatuses[] | {
  name: .name,
  state: .state,
  lastState: .lastState,
  image: .image,
  imageID: .imageID
}'
```

**What to look for immediately:**
- `ErrImagePull` before it becomes `ImagePullBackOff`
- `pull access denied`
- `no basic auth credentials`
- `manifest unknown`
- `unauthorized`
- `x509`, DNS, timeout, or TLS handshake errors
- `failed to resolve reference` or digest mismatch messages

---

### Understand the Usual Failure Modes

#### 1. Authentication Failure
Typical event messages:
- `pull access denied`
- `unauthorized: authentication required`
- `no basic auth credentials`

This usually means the node or pod does not have valid registry credentials.

#### 2. Image Not Found
Typical event messages:
- `manifest unknown`
- `not found`
- `failed to pull and unpack image`

This usually means the repository, tag, or registry path is wrong.

#### 3. Network or DNS Failure
Typical event messages:
- timeout errors
- temporary failure in name resolution
- TLS or certificate validation failures

This points to node egress, DNS, proxy, firewall, or certificate configuration.

#### 4. Digest Mismatch or Tag Drift
Typical event messages:
- digest does not match
- reference resolves unexpectedly
- image exists under a different manifest than expected

This usually happens when a mutable tag was republished or the deployment expects a digest from a different build.

---

### Check Events and Container Status Carefully

```bash
# Events in timestamp order
kubectl get events -n <namespace> \
  --field-selector involvedObject.name=<pod-name> \
  --sort-by='.lastTimestamp'

# Show image references configured in the pod
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.containers[] | {
  name: .name,
  image: .image,
  imagePullPolicy: .imagePullPolicy
}'
```

**Best approach:** copy the exact error string from `kubectl describe pod`. Most `ImagePullBackOff` incidents are resolved by following the precise registry error rather than guessing from the high-level pod status.

---

### Debug Private Registry Authentication

#### Check Whether the Pod Uses an Image Pull Secret
```bash
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.imagePullSecrets}{"\n"}'

# Check the default service account too
kubectl get serviceaccount default -n <namespace> -o yaml
```

If the pod relies on a service account, make sure the service account has the expected `imagePullSecrets` configured.

#### Inspect the Secret Type and Name
```bash
kubectl get secret <secret-name> -n <namespace>
kubectl get secret <secret-name> -n <namespace> -o yaml
```

For Docker-compatible registries, the secret should usually be type:
- `kubernetes.io/dockerconfigjson`

#### Recreate a Broken Pull Secret
```bash
kubectl create secret docker-registry <secret-name> \
  -n <namespace> \
  --docker-server=<registry-host> \
  --docker-username=<username> \
  --docker-password=<password> \
  --docker-email=<email>
```

#### Attach It to the Service Account
```bash
kubectl patch serviceaccount default -n <namespace> -p '{
  "imagePullSecrets": [
    {"name": "<secret-name>"}
  ]
}'
```

**Tip:** if the pod spec explicitly sets `imagePullSecrets`, that overrides relying on the service account alone for that workload’s configuration clarity.

---

### Debug Amazon ECR Authentication and Token Expiry

ECR auth is a common source of pull failures because tokens are short-lived.

#### Common ECR Problems
- Cluster is pulling from the wrong AWS account
- Cluster is pulling from the wrong region
- The Docker auth secret was generated long ago and expired
- IAM permissions for node role, IRSA, or credential provider are incomplete
- The image exists in ECR, but under a different repository path or tag

#### Verify the ECR Image Reference
```bash
# Example ECR image format
<account-id>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>
```

Check carefully:
- AWS account ID
- Region
- Repository name
- Tag or digest

#### Refresh ECR Credentials Manually
```bash
aws ecr get-login-password --region <region> | docker login \
  --username AWS \
  --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
```

#### Recreate the Kubernetes Pull Secret for ECR
```bash
kubectl create secret docker-registry ecr-pull-secret \
  -n <namespace> \
  --docker-server=<account-id>.dkr.ecr.<region>.amazonaws.com \
  --docker-username=AWS \
  --docker-password="$(aws ecr get-login-password --region <region>)"
```

#### Check IAM Access if Using Node Role or IRSA
If the cluster depends on cloud-native credential resolution instead of a static pull secret, verify:
- Worker node role can access ECR
- IRSA or kubelet credential provider is configured correctly
- The runtime on the node supports the configured ECR credential flow

**Best approach:** if ECR pulls suddenly start failing after working for hours or days, suspect expired credentials first.

---

### Debug Image Name, Tag, and Digest Problems

#### Inspect the Exact Image Reference in the Workload
```bash
kubectl get deployment <deployment-name> -n <namespace> -o json | jq '.spec.template.spec.containers[] | {
  name: .name,
  image: .image
}'
```

Common mistakes:
- Typo in repo or tag
- Using a tag that was never pushed
- Missing registry hostname for a private image
- Using `latest` unintentionally in one environment and pinned tags in another

#### Check Digest-Pinned Images
```bash
# Example digest-pinned format
my-registry.example.com/app@sha256:<digest>
```

If a deployment uses a digest:
- Verify the digest exists in the target registry
- Confirm the digest matches the same registry and architecture variant
- Do not assume a digest from one registry mirror exists in another mirror

#### Tag Versus Digest Mismatch
This often shows up when:
- CI republished the same tag with new content
- A mirror registry lagged behind upstream
- The cluster cached a previous manifest while the deployment now expects another

**Best approach:** for production, prefer immutable tags or digest pinning. Mutable tags make incident response harder because “same tag” may no longer mean “same image.”

---

### Verify the Image Actually Exists

If you have registry CLI access, confirm the repository and tag exist outside Kubernetes.

Examples:
```bash
# Docker Hub or generic registry pull test from a workstation
docker pull <registry>/<repo>:<tag>

# ECR list images
aws ecr list-images --repository-name <repo> --region <region>

# ECR describe a specific image tag
aws ecr describe-images --repository-name <repo> --image-ids imageTag=<tag> --region <region>
```

If the image cannot be pulled outside the cluster either, the problem is likely not Kubernetes-specific.

---

### Check Node-Level Connectivity to the Registry

Because pulls happen on the node, node egress matters.

```bash
# Identify the node hosting the pod
kubectl get pod <pod-name> -n <namespace> -o wide

# Launch a debug pod on the same node if needed
kubectl debug node/<node-name> -it --image=nicolaka/netshoot
```

From a debug environment, validate:
- DNS resolution for the registry hostname
- Outbound connectivity on TCP 443
- Proxy configuration if one is required
- Certificate chain trust for private registries

Useful checks inside a debug environment:
```bash
nslookup <registry-host>
nc -zv <registry-host> 443
curl -vk https://<registry-host>/v2/
```

**Tip:** if only one node shows pull failures, compare node network path, DNS config, proxy settings, and container runtime health against a working node.

---

### Container Runtime and Node Clues

If pod events are vague, the node runtime logs often contain the real pull error.

Possible places to inspect on the node:
- kubelet logs
- containerd logs
- CRI-O logs

Common runtime-side problems:
- Bad registry mirror configuration
- Invalid TLS certs for private registry
- Proxy variables not applied to the runtime service
- Disk pressure blocking image unpacking

At the Kubernetes layer, also check:
```bash
kubectl describe node <node-name>
```

Watch for:
- `DiskPressure`
- image filesystem exhaustion
- runtime not ready conditions

---

### Best Practices and Prevention

#### 1. Prefer Immutable Image References
- Use unique version tags per build
- Use digest pinning for sensitive production workloads
- Avoid reusing `latest` or mutable release tags without strict controls

#### 2. Standardize Registry Auth
- Use service accounts consistently
- Document which namespaces need which pull secrets
- Rotate short-lived credentials before they expire

#### 3. Validate Images in CI Before Deployment
- Confirm the image was pushed successfully
- Confirm the exact tag or digest exists in the target registry
- Confirm the deployment manifest references the same artifact CI built

#### 4. Be Explicit About Multi-Arch Images
- Verify the manifest contains the architecture your nodes run
- Check `arm64` versus `amd64` when one environment works and another fails

#### 5. Monitor Node Disk Usage
- Pulls can fail if the image filesystem is full
- Clean up stale images and monitor runtime storage capacity

---

### Quick Reference Checklist

When troubleshooting `ImagePullBackOff`:

- [ ] Read the exact event message from `kubectl describe pod`
- [ ] Confirm the image name, tag, or digest is correct
- [ ] Verify the registry secret exists and is attached correctly
- [ ] Check whether the issue is auth, not found, DNS, TLS, or timeout
- [ ] Refresh ECR credentials if using short-lived auth
- [ ] Confirm the image exists in the expected registry account and region
- [ ] Validate node connectivity to the registry hostname
- [ ] Check node disk pressure and runtime health
- [ ] Prefer immutable tags or digest pinning to avoid tag drift

---

### Practical Tips and Tricks

- `ErrImagePull` tells you the first real failure; `ImagePullBackOff` is just the retry state after that.
- If the same image works in one namespace but not another, compare service accounts and pull secrets before comparing network paths.
- If only new nodes fail to pull, check bootstrap configuration, IAM role attachment, and runtime registry settings on those nodes.
- If the issue started right after an image push, verify CI actually pushed the architecture variant your cluster needs.
- If events mention digest mismatch, stop using the mutable tag as your source of truth and verify the exact manifest in the registry.

---

## 6. Pod Stuck in ContainerCreating

### Overview
`ContainerCreating` means Kubernetes has scheduled the pod to a node, but the runtime is still preparing the execution environment. The container process has not started yet.

At this stage, Kubernetes may still be:
- Creating the pod sandbox
- Setting up networking through the CNI plugin
- Attaching or mounting volumes
- Resolving referenced Secrets or ConfigMaps
- Pulling images for sidecars or other containers
- Waiting on node runtime readiness

This is why `kubectl logs` is usually not useful yet: there may be no container process running to produce logs.

---

### Fast Triage

Start with events. `ContainerCreating` is a broad state, but the events usually narrow it down immediately.

```bash
# Check pod placement and current state
kubectl get pod <pod-name> -n <namespace> -o wide

# Read detailed lifecycle events
kubectl describe pod <pod-name> -n <namespace>

# Show status and node assignment in JSON
kubectl get pod <pod-name> -n <namespace> -o json | jq '{
  phase: .status.phase,
  conditions: .status.conditions,
  containerStatuses: .status.containerStatuses,
  initContainerStatuses: .status.initContainerStatuses,
  nodeName: .spec.nodeName
}'

# Sort recent events for the pod
kubectl get events -n <namespace> \
  --field-selector involvedObject.name=<pod-name> \
  --sort-by='.lastTimestamp'
```

**What to look for immediately:**
- `FailedCreatePodSandBox`
- `FailedMount`
- `Unable to attach or mount volumes`
- `secret not found`
- `configmap not found`
- CNI plugin errors or sandbox creation failures
- Timeouts talking to CSI or node runtime components
- `PodReadyToStartContainers=False` in pod conditions on Kubernetes `1.33+`

---

### How ContainerCreating Works

Before the first user process starts, Kubernetes has to complete a few infrastructure steps:
1. Schedule the pod to a node
2. Ask the container runtime to create a pod sandbox
3. Configure networking through the CNI plugin
4. Attach and mount volumes
5. Resolve referenced Secrets and ConfigMaps
6. Pull images if needed
7. Start init containers or app containers

If any of those fail, the pod can remain stuck in `ContainerCreating` even though scheduling succeeded.

On Kubernetes `1.33+`, the `PodReadyToStartContainers` condition is also useful here. When present and `False`, it usually means sandbox, networking, storage setup, or other pre-container prerequisites are not complete yet.

---

### Most Common Causes

#### 1. CNI Plugin Failure
Typical event messages:
- `FailedCreatePodSandBox`
- `network plugin is not ready`
- `failed to setup network for sandbox`
- IP allocation errors from the CNI plugin

This means the node runtime could not create the pod network namespace or attach the pod to the cluster network.

#### 2. Volume Mount Timeout
Typical event messages:
- `Unable to attach or mount volumes`
- `timed out waiting for the condition`
- CSI driver attach or mount errors

This usually points to storage backend, CSI node plugin, or node mount problems.

#### 3. Missing Secret or ConfigMap
Typical event messages:
- `secret "..." not found`
- `configmap "..." not found`

This blocks startup before the container process begins because the pod spec cannot be fully materialized.

#### 4. Node Runtime or Sandbox Problems
Typical event messages:
- CRI runtime not ready
- sandbox create failures
- runtime or image filesystem issues

---

### Debug CNI Plugin Failures

If events mention sandbox creation or network setup, shift attention from the workload to the node and CNI stack.

```bash
# Check the exact sandbox error on the pod
kubectl describe pod <pod-name> -n <namespace>

# Inspect node conditions
kubectl get node <node-name>
kubectl describe node <node-name>

# Identify CNI pods, common examples
kubectl get pods -A -o wide | grep -Ei 'calico|cilium|weave|flannel|aws-node|antrea'
```

Common CNI problems:
- IPAM exhausted, no more pod IPs available
- CNI daemonset unhealthy on that node
- Node lost connectivity to the control plane or network backend
- Host routing or iptables rules are broken
- Newly added node missing CNI bootstrap configuration

**Best approach:** if only pods on one node are stuck in `ContainerCreating`, compare the CNI daemonset pod and node health against a working node before changing the workload.

Useful follow-up checks:
```bash
# Check CNI daemonset rollout and health
kubectl get daemonset -A

# Inspect logs for the network plugin pod on the affected node
kubectl logs -n <cni-namespace> <cni-pod-name>
```

---

### Debug Volume Attach and Mount Delays

If events mention `FailedMount` or `Unable to attach or mount volumes`, inspect the storage path end to end.

```bash
# Inspect pod volumes
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.volumes'

# Check PVC state
kubectl get pvc -n <namespace>
kubectl describe pvc <pvc-name> -n <namespace>

# Inspect PV details if applicable
kubectl get pv <pv-name>
kubectl describe pv <pv-name>

# Check volume attachments if supported
kubectl get volumeattachments
```

Typical causes:
- PVC not yet bound
- CSI driver on the node is down
- Volume still attached elsewhere
- Cloud attach API is slow or failing
- Filesystem mount is hanging on the node

**Tip:** if the pod is part of a StatefulSet and uses persistent volumes, do not assume the problem is inside the pod. Check whether the previous instance still holds the volume.

---

### Debug Missing Secret or ConfigMap Dependencies

Referenced objects must exist in the same namespace as the pod unless projected through some other mechanism.

```bash
# Check objects exist
kubectl get secret <secret-name> -n <namespace>
kubectl get configmap <configmap-name> -n <namespace>

# Inspect the references in the pod spec
kubectl get pod <pod-name> -n <namespace> -o yaml | grep -A 15 'secretName:\|configMap:'
```

Common causes:
- Wrong object name
- Object exists in a different namespace
- Key exists in dev but not prod
- Helm or Kustomize rendered a different name than expected

**Best approach:** compare the rendered workload manifest to the actual object names in the namespace. Most Secret and ConfigMap issues are naming mismatches, not runtime failures.

---

### Inspect the Service Account and Projected Volumes

Some pods stall because service account token or projected volume setup is failing.

```bash
# Check service account used by the pod
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.serviceAccountName}{"\n"}'

# Inspect projected volumes
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.volumes[] | select(.projected != null or .secret != null or .configMap != null)'
```

This matters when:
- Service account token projection is misconfigured
- CSI secret store integration is failing
- Pod depends on external secret injection controllers

---

### Check Node-Level Runtime Health

Because `ContainerCreating` happens before the container runs, kubelet and the container runtime matter a lot.

```bash
# Inspect node conditions and capacity
kubectl describe node <node-name>
```

Watch for:
- `NotReady`
- `DiskPressure`
- `MemoryPressure`
- runtime not ready conditions
- image filesystem or ephemeral storage exhaustion

If only one node is affected, compare:
- kubelet health
- runtime health
- CNI daemonset status
- CSI node plugin status
- local disk usage

---

### Pod Sandbox Creation Failures

If the event says `FailedCreatePodSandBox`, Kubernetes could not create the basic runtime environment for the pod.

Common reasons:
- CNI plugin failure
- CRI runtime issue
- Sandbox image pull failure
- DNS or network namespace creation problem
- Node runtime misconfiguration after upgrade or reboot

```bash
# Look for sandbox-specific error text
kubectl describe pod <pod-name> -n <namespace> | grep -A 5 -i 'sandbox\|network plugin\|failed'
```

**Best approach:** treat `FailedCreatePodSandBox` as an infrastructure signal first, not an application bug.

---

### Best Practices and Prevention

#### 1. Keep Secret and ConfigMap References Predictable
- Use stable naming conventions
- Validate rendered manifests in CI
- Avoid manual object renames outside deployment automation

#### 2. Monitor CNI and CSI Daemonsets
- Alert when CNI pods are not ready on a node
- Alert when CSI node plugins are failing or restarting
- Include node-level daemonsets in incident triage, not just application pods

#### 3. Watch Pod IP Capacity and Node Bootstrap
- New nodes need working CNI initialization
- Cluster scaling can expose subnet or IP exhaustion problems
- Track pod density against network limits

#### 4. Monitor Ephemeral Storage and Disk Pressure
- Container sandbox creation and image unpacking need local disk
- Volume mount operations can fail when nodes are under storage pressure

#### 5. Distinguish Infrastructure Delay From Workload Failure
- `ContainerCreating` often means the container process never started
- Avoid spending time on app logs before confirming the sandbox, network, and volumes are ready

---

### Quick Reference Checklist

When a pod is stuck in `ContainerCreating`:

- [ ] Read pod events from `kubectl describe pod`
- [ ] Check for `FailedCreatePodSandBox` or `FailedMount`
- [ ] Confirm the node is healthy and ready
- [ ] Verify the CNI daemonset is healthy on that node
- [ ] Inspect PVC, PV, and volume attachment state
- [ ] Confirm all referenced Secrets and ConfigMaps exist
- [ ] Check projected volumes and service account usage
- [ ] Inspect node disk pressure and runtime readiness
- [ ] Treat sandbox creation failures as node or infrastructure issues first

---

### Practical Tips and Tricks

- If `kubectl logs` says the container is waiting to start, that is expected in `ContainerCreating`; switch to events and node checks.
- If multiple unrelated pods are stuck on one node, stop debugging individual manifests and inspect CNI, CSI, kubelet, and runtime on that node.
- If the same manifest works on other nodes, compare the affected node’s daemonset pods and conditions first.
- Missing Secrets and ConfigMaps often look like runtime issues in dashboards, but they are usually simple spec-to-namespace mismatches.
- For storage-backed workloads, volume attach delays can outlast scheduling success by several minutes; do not assume scheduling means the pod is close to ready.

---

## 7. RunContainerError

### Overview
`RunContainerError` means Kubernetes got far enough to create the container, but the runtime failed when trying to start it. This is later than `ContainerCreating` and earlier than a normal application crash.

In practice, this usually means the image was pulled and the sandbox exists, but the command could not execute correctly.

Common causes:
- Entrypoint or command does not exist in the image
- Script or binary is not executable
- Shell path is wrong for the image base
- The image architecture does not match the node architecture
- Required shared libraries or interpreter are missing
- Security context or filesystem permissions prevent execution

---

### Fast Triage

Start with events and the effective image/command configuration.

```bash
# Check current pod status
kubectl get pod <pod-name> -n <namespace>

# Describe pod for runtime error messages
kubectl describe pod <pod-name> -n <namespace>

# Inspect configured image, command, and args
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.containers[] | {
  name: .name,
  image: .image,
  command: .command,
  args: .args
}'

# Inspect container waiting/terminated reason
kubectl get pod <pod-name> -n <namespace> -o json | jq '.status.containerStatuses[] | {
  name: .name,
  state: .state,
  lastState: .lastState,
  restartCount: .restartCount
}'
```

**What to look for immediately:**
- `exec: no such file or directory`
- `permission denied`
- `exec format error`
- `container has runAsNonRoot and image will run as root`
- missing interpreter such as `/bin/sh` or `/bin/bash`
- missing dynamic linker or shared library errors

---

### How RunContainerError Differs From CrashLoopBackOff

This distinction matters:
- `RunContainerError`: the process never started correctly
- `CrashLoopBackOff`: the process started and then exited or crashed repeatedly

If the container runtime cannot launch the configured command at all, focus on the image contents, runtime settings, and command override first, not application logs.

---

### Debug Entrypoint Not Found

One of the most common causes is a mismatch between the image contents and the command Kubernetes is trying to run.

#### Inspect the Effective Command
```bash
kubectl get deployment <deployment-name> -n <namespace> -o json | jq '.spec.template.spec.containers[] | {
  name: .name,
  image: .image,
  command: .command,
  args: .args
}'
```

Typical mistakes:
- Overriding the image `ENTRYPOINT` with a path that does not exist
- Using `/bin/bash` in minimal images that only include `/bin/sh`
- Referencing a startup script that was never copied into the image
- Using a relative path instead of an absolute binary path

#### Common Error Pattern
```text
exec: "/app/start.sh": stat /app/start.sh: no such file or directory
```

#### Best Approach
- Verify the binary or script exists in the image
- Use absolute paths
- Match the shell to the base image
- Avoid overriding `command` unless necessary

#### Reproduce With a Debug Pod
```bash
kubectl run runcontainer-debug -n <namespace> --image=<image> --restart=Never -- /bin/sh -c 'sleep infinity'
kubectl exec -it runcontainer-debug -n <namespace> -- /bin/sh
```

Inside the debug pod, inspect:
```bash
ls -lah /app
which sh
which bash
cat /etc/os-release
```

---

### Debug Permission Denied on Binary or Script

If the runtime says `permission denied`, the file exists but cannot be executed.

Typical causes:
- Script or binary is missing the execute bit
- Mounted file from ConfigMap is not executable
- Container runs as a user that cannot access the path
- Filesystem is mounted `noexec`

#### Check the File in a Debug Pod
```bash
ls -lah /app/start.sh
stat /app/start.sh
id
mount | grep -i noexec
```

#### Common Fixes
- Set executable permission during the image build
- Avoid executing scripts directly from ConfigMap mounts unless permissions are handled explicitly
- Align `runAsUser`, `runAsGroup`, and `fsGroup` with file ownership

Example Dockerfile fix:
```dockerfile
COPY --chmod=755 start.sh /app/start.sh
```

Or:
```dockerfile
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh
```

---

### Debug Wrong Architecture Images

`exec format error` is a strong signal that the image architecture does not match the node architecture.

Example scenario:
- Image was built for `arm64`
- Cluster nodes are `amd64`
- The binary exists and is executable, but Linux cannot run it on that CPU architecture

#### Check Node Architecture
```bash
kubectl get pod <pod-name> -n <namespace> -o wide
kubectl get node <node-name> -o jsonpath='{.status.nodeInfo.architecture}{"\n"}'
```

#### Check Image Platform Outside the Cluster
```bash
docker manifest inspect <image>
```

Look for whether the image manifest includes the architecture your nodes use.

**Best approach:** build and publish multi-arch images when clusters may run mixed node types, or pin the workload to nodes that match the built image architecture.

---

### Debug Missing Interpreter or Shared Library Problems

Sometimes the target file exists and has execute permission, but the interpreter or dynamic loader it depends on does not.

Common examples:
- Script starts with `#!/bin/bash` but image has no bash
- Binary expects glibc, image only includes musl or minimal runtime libraries
- Dynamic linker path in the binary does not exist

Typical runtime message:
```text
no such file or directory
```

This can be misleading because the file you invoked does exist. The missing file is actually the interpreter or loader behind it.

#### Check the Shebang and Binary Type
```bash
head -n 1 /app/start.sh
file /app/mybinary
ldd /app/mybinary
```

**Tip:** if `file` reports the binary is dynamically linked and `ldd` shows missing libraries, fix the image contents rather than the pod spec.

---

### Debug Security Context Problems

Execution can fail even when the file is present and valid if the security context conflicts with the image.

```bash
# Inspect security context settings
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.containers[] | {
  name: .name,
  securityContext: .securityContext
}'
```

Common issues:
- `runAsNonRoot: true` but image defaults to root without a non-root user configured
- Read-only root filesystem blocks startup writes
- SELinux/AppArmor/seccomp restrictions in hardened environments

Fixes:
- Create a non-root user in the image
- Ensure writable directories are backed by volumes if needed
- Keep the security context aligned with how the image was built

---

### Debug Command Overrides in Kubernetes Manifests

Kubernetes `command` and `args` override the image defaults. Many startup regressions are introduced there rather than in the image itself.

```bash
# Compare image reference and command override in the workload
kubectl get deployment <deployment-name> -n <namespace> -o yaml | grep -A 12 'image:\|command:\|args:'
```

Common mistakes:
- Swapping `command` and `args`
- Passing the whole command as one string when the image expects an argv list
- Removing the image `ENTRYPOINT` unintentionally

Example:
```yaml
command: ["/bin/sh", "-c"]
args: ["/app/start.sh"]
```

This is valid only if `/bin/sh` exists and `/app/start.sh` is executable.

---

### Best Practices and Prevention

#### 1. Keep Startup Paths Explicit and Stable
- Use absolute paths for startup scripts and binaries
- Avoid fragile shell wrappers unless needed
- Keep image `ENTRYPOINT` and Kubernetes overrides aligned

#### 2. Build Images for the Target Platform
- Publish multi-arch manifests when needed
- Verify `arm64` and `amd64` compatibility in CI
- Do not rely on local laptop builds as proof the cluster image is correct

#### 3. Set Execute Permissions at Build Time
- Do not depend on runtime `chmod` hacks for production startup
- Verify copied scripts keep expected permissions

#### 4. Keep Base Image Assumptions Minimal
- If you require `bash`, include it intentionally
- If you need glibc-linked binaries, use a compatible runtime image

#### 5. Test the Final Image Directly
- Run the exact built image locally or in CI with the intended command
- Validate that the binary starts before deploying it to Kubernetes

---

### Quick Reference Checklist

When troubleshooting `RunContainerError`:

- [ ] Read the exact runtime error from `kubectl describe pod`
- [ ] Inspect the configured image, command, and args
- [ ] Verify the target file exists in the image
- [ ] Check execute permissions on the binary or script
- [ ] Confirm the image architecture matches node architecture
- [ ] Check whether `/bin/sh` or `/bin/bash` actually exists
- [ ] Inspect shebangs, dynamic linker, and shared library dependencies
- [ ] Review security context settings for root versus non-root conflicts
- [ ] Reproduce the startup command in a debug pod

---

### Practical Tips and Tricks

- `no such file or directory` does not always mean the script is missing; it often means the interpreter or loader behind it is missing.
- `exec format error` is usually architecture mismatch until proven otherwise.
- If a container starts fine locally but fails in the cluster, compare the cluster node architecture and security context before changing app code.
- Scripts mounted from ConfigMaps are a common source of permission problems because they are not automatically executable in every pattern.
- If the failure began right after a Dockerfile base image change, inspect shell availability and library compatibility first.

---

## 8. Pod Restarts With Exit Code 0

### Overview
An exit code of `0` means the container process ended successfully. In Kubernetes, that can still be a problem if the workload is supposed to stay running.

This usually happens when:
- The container is running a one-shot command instead of a long-lived service
- The main process starts, completes, and exits cleanly
- A shell wrapper launches a child process incorrectly and then exits
- The real server process forks into the background instead of staying in the foreground
- A probe or restart policy causes repeated restarts around a process that is technically succeeding

Exit code `0` is one of the most misleading restart patterns because nothing “crashed,” but the pod still never becomes healthy.

---

### Fast Triage

Start by confirming that the container is exiting normally rather than failing.

```bash
# Basic pod state and restart count
kubectl get pod <pod-name> -n <namespace>

# Describe pod to inspect restart reason and probe events
kubectl describe pod <pod-name> -n <namespace>

# Check termination details and exit code
kubectl get pod <pod-name> -n <namespace> -o json | jq '.status.containerStatuses[] | {
  name: .name,
  restartCount: .restartCount,
  state: .state,
  lastState: .lastState
}'

# Fetch previous logs from the last completed run
kubectl logs <pod-name> -n <namespace> --previous
```

**What to look for immediately:**
- `exitCode: 0`
- `reason: Completed`
- Whether the process logs indicate startup succeeded and then stopped
- Whether liveness probe failures are also present in pod events
- Whether the workload is actually a job-like task deployed as a Deployment or StatefulSet

---

### Why Exit Code 0 Can Still Be Wrong

Kubernetes expects different behavior depending on the workload type:
- A `Deployment` or `StatefulSet` usually expects the container to keep running
- A `Job` expects the container to finish and exit

If a long-lived app exits with `0`, Kubernetes will still restart it under normal pod restart behavior. That creates a loop of “successful” completions that never produce a healthy service.

---

### Most Common Causes

#### 1. One-Shot Process Used in a Long-Lived Workload
Examples:
- Running a migration command in a Deployment
- Using a script that renders config and exits
- Starting a batch worker in a service pod definition

Symptoms:
- Logs show successful completion
- Exit code is `0`
- Pod keeps restarting under a Deployment

#### 2. Wrapper Script Exits Too Early
Typical pattern:
```bash
#!/bin/sh
/app/server &
exit 0
```

Here the wrapper exits cleanly, but the container lifecycle is tied to PID 1, which has already ended.

#### 3. Daemon or Service Forks Into the Background
Many traditional server commands daemonize by default. In containers, the main process must stay in the foreground.

Typical pattern:
- Service launches
- Parent process exits `0`
- Kubernetes sees container complete and restarts it

#### 4. Liveness Probe Makes a Healthy-Looking Startup Appear Unstable
If the process starts and then gets killed by Kubernetes because the liveness probe is too aggressive, the surrounding events can obscure whether the original process exited cleanly or the kubelet terminated it shortly afterward.

---

### Check the Actual Exit Reason and Events

```bash
# Show exit reason and code clearly
kubectl get pod <pod-name> -n <namespace> -o json | jq '.status.containerStatuses[] | {
  name: .name,
  exitCode: .lastState.terminated.exitCode,
  reason: .lastState.terminated.reason,
  startedAt: .lastState.terminated.startedAt,
  finishedAt: .lastState.terminated.finishedAt
}'

# Read timeline of events
kubectl get events -n <namespace> \
  --field-selector involvedObject.name=<pod-name> \
  --sort-by='.lastTimestamp'
```

Interpretation:
- `Completed` with exit code `0`: the process ended normally
- Probe failures followed by restarts: the process may not be exiting on its own
- Very short runtime between `startedAt` and `finishedAt`: command may be a one-shot startup script

---

### Check the Configured Command and Args

Many exit-code-0 loops are caused by the wrong startup command in the manifest.

```bash
kubectl get deployment <deployment-name> -n <namespace> -o json | jq '.spec.template.spec.containers[] | {
  name: .name,
  image: .image,
  command: .command,
  args: .args
}'
```

Common mistakes:
- Running `/bin/sh -c 'do-setup && exit 0'`
- Overriding the image entrypoint with a validation command
- Starting a server with a wrapper that backgrounds the actual process

**Best approach:** inspect the exact startup command and confirm that PID 1 is the long-lived process Kubernetes should manage.

---

### Debug Shell Wrappers and PID 1 Behavior

Shell wrappers are a common source of clean exits.

Bad pattern:
```sh
#!/bin/sh
/app/server &
echo "started"
exit 0
```

Better pattern:
```sh
#!/bin/sh
exec /app/server
```

Why `exec` matters:
- Replaces the shell with the real server process
- Keeps the service as PID 1
- Ensures signals like `SIGTERM` reach the correct process

If you use a startup script, it should normally finish with `exec <real-process>`.

---

### Debug Backgrounding and Daemonization

Traditional Linux services often detach and run in the background, but containers expect foreground execution.

Typical examples:
- `nginx` without `daemon off;`
- custom server started with a `--daemon` flag
- supervisor-style scripts that return immediately

Fixes:
- Run the service in the foreground
- Disable daemon mode
- Use the documented container-friendly startup command

Examples:
```bash
# Good foreground pattern for nginx
nginx -g 'daemon off;'

# Example for other daemons
myserver --foreground
```

---

### Debug Liveness Probe Problems

Sometimes the container does not really have an exit-code-0 lifecycle bug. Instead, it starts normally and then gets restarted because Kubernetes kills it due to a bad liveness probe.

#### Inspect Probe Configuration
```bash
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.containers[] | {
  name: .name,
  livenessProbe: .livenessProbe,
  readinessProbe: .readinessProbe,
  startupProbe: .startupProbe
}'
```

#### Look for Probe Failures in Events
```bash
kubectl describe pod <pod-name> -n <namespace> | grep -A 5 -i 'liveness\|readiness\|startup'
```

Typical issues:
- Liveness probe starts too early
- Probe timeout is too short
- Slow startup path is treated as dead process
- The app binds late or warms up slowly

Fixes:
- Add a `startupProbe` for slow-starting apps
- Increase `initialDelaySeconds`
- Increase `timeoutSeconds`
- Make the liveness probe less aggressive

Example:
```yaml
startupProbe:
  httpGet:
    path: /healthz
    port: 8080
  periodSeconds: 10
  failureThreshold: 30

livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
  initialDelaySeconds: 30
  timeoutSeconds: 5
```

**Best approach:** if a process needs time to initialize, use `startupProbe` instead of stretching liveness until it becomes meaningless.

---

### Check Whether This Should Be a Job Instead

If the process is meant to run once and exit successfully, Kubernetes may be using the wrong workload type.

Use a `Job` when:
- The container performs migrations
- The task generates data and exits
- The work is batch-oriented or finite

Use a `Deployment` when:
- The container should keep serving traffic
- The process should stay alive indefinitely

**Best approach:** do not force one-shot commands into Deployments just to reuse the same manifest pattern. That creates noisy restart loops and misleading alerts.

---

### Reproduce the Startup Command in a Debug Pod

If the behavior is unclear, run the same image interactively and test the startup command directly.

```bash
kubectl run exit0-debug -n <namespace> --image=<image> --restart=Never -- /bin/sh -c 'sleep infinity'
kubectl exec -it exit0-debug -n <namespace> -- /bin/sh
```

Inside the container, verify:
- What the startup script actually does
- Whether it backgrounds the real process
- Whether the server forks or daemonizes
- Whether the script explicitly exits `0`

---

### Best Practices and Prevention

#### 1. Keep the Main Service in the Foreground
- PID 1 should be the real server process
- Avoid scripts that background the application
- Use `exec` in shell wrappers

#### 2. Match Workload Type to Process Behavior
- One-shot task: use a `Job`
- Long-running service: use a `Deployment`, `DaemonSet`, or `StatefulSet`

#### 3. Configure Probes Around Real Startup Time
- Use `startupProbe` for slow initialization
- Keep liveness probes for deadlock or hang detection, not startup gating

#### 4. Review Command Overrides Carefully
- Do not replace the image entrypoint with a setup command unless you intend the container to exit
- Keep setup tasks in init containers or Jobs when appropriate

#### 5. Test the Final Command Path in CI
- Run the built image with the exact command from the manifest
- Verify the process remains alive when it is supposed to

---

### Quick Reference Checklist

When a pod restarts with exit code `0`:

- [ ] Confirm `exitCode=0` and `reason=Completed`
- [ ] Read previous logs from the last run
- [ ] Inspect the configured `command` and `args`
- [ ] Check whether a wrapper script backgrounds the real process
- [ ] Verify the service runs in the foreground
- [ ] Check pod events for liveness probe failures
- [ ] Add or tune `startupProbe` if startup is slow
- [ ] Decide whether the workload should really be a `Job`
- [ ] Reproduce the startup path in a debug pod

---

### Practical Tips and Tricks

- Exit code `0` means success for the process, not correctness for the workload design.
- If logs end with a clean “startup complete” message and then the pod restarts, suspect backgrounding or daemonization first.
- If the command is doing setup and then exiting, move that logic to an init container or Job instead of keeping it in the main container.
- Probe misconfiguration and clean exits can coexist; always inspect both container status and event history.
- PID 1 mistakes are common in shell-based entrypoints and usually easy to fix once you inspect the wrapper.

---

## 9. PostStartHook Failure

### Overview
A `PostStart` hook runs immediately after a container is created. It is part of the container lifecycle and executes after the container process starts, but before Kubernetes considers startup fully complete.

This creates a confusing failure mode:
- The container may appear to have started
- The application may even write some logs
- But the `PostStart` hook can still fail and cause the container to be killed or restarted

That makes `PostStart` issues easy to misread as random startup instability, readiness problems, or application crashes.

Common causes:
- The hook command is wrong or the binary does not exist
- The hook depends on files, sockets, or network endpoints that are not ready yet
- The hook exits non-zero or times out
- The hook writes to a path that is not writable
- The hook duplicates logic that should have been handled in an init container

---

### How PostStart Works

`PostStart` is a lifecycle hook configured in the pod spec. It can run as:
- `exec`: a command inside the container namespace
- `httpGet`: an HTTP call made by kubelet
- `sleep`: a kubelet-managed pause for a specified duration

Important behavior:
- The hook is triggered immediately after container creation and runs concurrently with the container entrypoint
- The container may not transition cleanly to `Running` until the hook completes
- If the hook fails, Kubernetes treats container startup as failed
- Hook output is not surfaced as cleanly as normal application logs

This is why the container can look alive briefly but still never become stable.

---

### Fast Triage

Start with pod events and lifecycle configuration. That is usually where the hook failure shows up most clearly.

```bash
# Check pod status and restart count
kubectl get pod <pod-name> -n <namespace>

# Describe pod to inspect lifecycle events
kubectl describe pod <pod-name> -n <namespace>

# Inspect lifecycle hook configuration
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.containers[] | {
  name: .name,
  lifecycle: .lifecycle,
  command: .command,
  args: .args
}'

# Inspect container state and last termination details
kubectl get pod <pod-name> -n <namespace> -o json | jq '.status.containerStatuses[] | {
  name: .name,
  state: .state,
  lastState: .lastState,
  restartCount: .restartCount
}'
```

**What to look for immediately:**
- Events mentioning `PostStartHookError`
- Container restarts with little or no application output
- The hook command or URL references resources that do not exist yet
- Readiness never succeeds because the container keeps restarting during startup

---

### Recognize the Typical Symptoms

Common patterns when a `PostStart` hook fails:
- Container briefly transitions to `Running` and then restarts
- Pod looks healthy at a glance, but readiness never stabilizes
- Application logs seem incomplete or stop right after startup
- There is no obvious application exception, only hook-related events in `kubectl describe pod`

This is one of the easiest issues to miss if you only check `kubectl logs` and ignore events.

---

### Inspect the Hook Configuration

```bash
# Show the lifecycle block for all containers
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.containers[] | {
  name: .name,
  postStart: .lifecycle.postStart,
  preStop: .lifecycle.preStop
}'
```

Typical examples:

```yaml
lifecycle:
  postStart:
    exec:
      command: ["/bin/sh", "-c", "/app/bootstrap.sh"]
```

```yaml
lifecycle:
  postStart:
    httpGet:
      path: /warmup
      port: 8080
```

Things to verify:
- Does the command exist in the image?
- Does the script have execute permission?
- Is the hook using the correct shell?
- Is the HTTP path valid and reachable at that startup moment?
- Is the hook doing work that should have been in an init container instead?

---

### Read Events for Hook Failures

`kubectl describe pod` is often the best source for `PostStart` issues because hook failures are event-driven.

```bash
kubectl describe pod <pod-name> -n <namespace>
```

Typical event messages:
- `FailedPostStartHook`
- `PostStartHookError`
- `exec: no such file or directory`
- `permission denied`
- HTTP connection refused or timeout

**Best approach:** if you suspect `PostStart`, trust pod events before application logs. Hook failures are more visible there than in the app’s stdout/stderr stream.

---

### Debug Exec PostStart Hooks

`exec` hooks fail for the same reasons normal startup commands fail, plus timing issues unique to startup.

#### Common Exec Hook Problems
- Script or binary missing
- Wrong shell path, such as `/bin/bash` in a minimal image
- Hook script not executable
- Hook tries to write to a read-only filesystem
- Hook assumes a socket, config file, or mount is ready before it really is

#### Reproduce the Command in a Debug Pod
```bash
kubectl run poststart-debug -n <namespace> --image=<image> --restart=Never -- /bin/sh -c 'sleep infinity'
kubectl exec -it poststart-debug -n <namespace> -- /bin/sh
```

Inside the debug pod:
```bash
ls -lah /app
head -n 1 /app/bootstrap.sh
stat /app/bootstrap.sh
/bin/sh -c /app/bootstrap.sh
```

**Tip:** many `PostStart` bugs are really image packaging or permission bugs exposed by hook timing.

---

### Debug HTTP PostStart Hooks

HTTP-based hooks often fail because the app is not actually ready to serve the endpoint yet.

Typical problems:
- The server has not bound the port yet
- The endpoint requires dependencies that are still warming up
- The hook calls `localhost`, but the server is listening on another interface or port
- The hook path performs real work and fails transiently during boot

#### Inspect the Hook Definition
```bash
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.containers[] | {
  name: .name,
  postStart: .lifecycle.postStart
}'
```

#### Check Whether the App Actually Serves the Hook Path
If you can stabilize the container in a debug copy, test:
```bash
curl -v http://127.0.0.1:8080/<path>
```

**Best approach:** `PostStart` is rarely the right tool for waiting on full application readiness. Because the hook runs concurrently with container startup, use `readinessProbe` and `startupProbe` for readiness gating instead.

### Debug Sleep PostStart Hooks

`sleep` handlers are available in modern Kubernetes lifecycle hooks and are sometimes used to intentionally delay startup progression.

Typical problems:
- Sleep duration is masking a real dependency issue
- Hook adds fixed delay but startup time is variable
- Teams use `sleep` as a substitute for real readiness signaling

Example:
```yaml
lifecycle:
  postStart:
    sleep:
      seconds: 10
```

**Best approach:** use `sleep` only for narrow, intentional timing control. If the real requirement is “wait until dependency is usable”, that still belongs in an init container, startup logic, or probes.

---

### Distinguish PostStart Failure From Probe Failure

These are often confused because both happen near startup.

#### PostStart failure
- Happens during container lifecycle startup
- Usually appears in lifecycle hook events
- Can kill the container before normal readiness stabilizes

#### Probe failure
- Happens after startup checks begin
- Usually appears as `Unhealthy` events
- Often means the app started but is not healthy enough yet

```bash
# Look at hook and probe configuration together
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.containers[] | {
  name: .name,
  lifecycle: .lifecycle,
  startupProbe: .startupProbe,
  readinessProbe: .readinessProbe,
  livenessProbe: .livenessProbe
}'
```

If the logic is “wait until the app is ready,” it probably belongs in a probe, not `PostStart`.

---

### When PostStart Logic Should Really Be an Init Container

Move the logic to an init container if it:
- Downloads or generates files before the app should start
- Waits on another service to be reachable
- Performs setup that does not require the main process to be running yet
- Needs clean, separate logs and failure visibility

Keep it in `PostStart` only if it truly depends on the container process already existing.

**Best approach:** if the hook is doing setup work rather than lifecycle coordination, use an init container instead.

---

### Reproduce and Stabilize the Startup Path

If the container restarts too fast, use a debug copy of the image and run the hook logic manually.

```bash
kubectl run hook-debug -n <namespace> --image=<image> --restart=Never -- /bin/sh -c 'sleep infinity'
kubectl exec -it hook-debug -n <namespace> -- /bin/sh
```

Then verify step by step:
- The hook command exists
- The hook command succeeds manually
- Required files and mounts are present
- Required ports are bound when using `httpGet`
- The hook is not racing the application’s real startup sequence

---

### Best Practices and Prevention

#### 1. Keep PostStart Hooks Short and Deterministic
- Avoid complex orchestration logic in `PostStart`
- Avoid long shell scripts with hidden dependencies
- Keep the hook fast enough that startup remains predictable

#### 2. Use the Right Mechanism for the Job
- App readiness: `readinessProbe`
- Slow startup: `startupProbe`
- Pre-run setup: init container
- Graceful shutdown: `preStop`

#### 3. Make Hook Commands Observable
- Use commands that log clearly
- Avoid silent failures inside shell wrappers
- Keep scripts versioned and testable like any other startup path

#### 4. Avoid Network-Dependent Hooks Unless Necessary
- Startup-time networking is often the least stable moment
- If the hook must call HTTP, keep the endpoint lightweight and local

#### 5. Test Hooks in the Final Image
- Validate that the hook command exists in the deployed image
- Run the exact hook command in CI or a pre-prod environment

---

### Quick Reference Checklist

When troubleshooting `PostStartHook` failures:

- [ ] Read pod events for `FailedPostStartHook` or similar messages
- [ ] Inspect the `lifecycle.postStart` configuration
- [ ] Verify the hook command, path, and permissions
- [ ] Check whether the hook depends on files, ports, or services that are not ready yet
- [ ] Distinguish hook failure from probe failure
- [ ] Move setup logic to an init container if appropriate
- [ ] Use probes instead of `PostStart` for readiness gating
- [ ] Reproduce the hook manually in a debug pod

---

### Practical Tips and Tricks

- If the app appears to start and then vanish without clear logs, inspect lifecycle events before assuming the app crashed.
- `PostStart` is a poor substitute for “wait until the service is ready.” That is what `startupProbe` and `readinessProbe` are for.
- HTTP `PostStart` hooks are especially fragile during startup because the app may not yet be listening even though the container exists.
- If a hook script and the main startup path both try to mutate the same files, race conditions become very likely.
- When incidents are hard to reproduce, removing the `PostStart` hook temporarily in a test environment is often the fastest way to isolate whether it is the real failure point.

---

## 10. Long Pod Startup Times

### Overview
Some pods are not broken at all. They are just slow to become ready. The operational problem is that slow startup often looks identical to failure for the first few minutes.

Long startup times usually come from one or more of these layers:
- Slow image pull
- Slow volume attach or mount
- Heavy init-container work
- Application warmup, cache priming, or JIT compilation
- Readiness or startup probes that are too aggressive
- External dependencies that delay true readiness

The key is to separate “container has not started yet” from “container is running but not ready yet” from “container is healthy, but the readiness signal is wrong.”

---

### Fast Triage

Start by building a simple timeline: scheduled, image pulled, volume mounted, container started, ready.

```bash
# Basic state and node placement
kubectl get pod <pod-name> -n <namespace> -o wide

# Full event timeline
kubectl describe pod <pod-name> -n <namespace>

# Sorted events for cleaner chronology
kubectl get events -n <namespace> \
  --field-selector involvedObject.name=<pod-name> \
  --sort-by='.lastTimestamp'

# Check container start and readiness status
kubectl get pod <pod-name> -n <namespace> -o json | jq '{
  phase: .status.phase,
  conditions: .status.conditions,
  initContainerStatuses: .status.initContainerStatuses,
  containerStatuses: .status.containerStatuses
}'
```

**What to look for immediately:**
- Long delay before the first `Pulled` event
- Long delay between `Pulled` and `Started`
- `FailedMount` or attach-related retries
- Containers running but `Ready=false`
- Probe failures during warmup
- Init containers doing large setup work before the app even starts

---

### Build a Startup Timeline

For slow startups, ordering matters more than any single error line.

Typical timeline to reconstruct:
1. Pod scheduled to node
2. Images pulled
3. Volumes attached and mounted
4. Init containers completed
5. Main container started
6. Readiness turned true

If the delay is before step 5, focus on infrastructure or init work.
If the delay is after step 5, focus on application warmup and probes.

---

### Debug Slow Image Pulls

Image pulls are a common hidden source of startup delay, especially with large images, cold nodes, or private registries.

```bash
# Inspect image names and pull policy
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.containers[] | {
  name: .name,
  image: .image,
  imagePullPolicy: .imagePullPolicy
}'

# Watch pull-related events
kubectl describe pod <pod-name> -n <namespace> | grep -A 3 -i 'pull\|image'
```

Common causes:
- Large image layers
- Node cache miss after autoscaling or node replacement
- Slow registry or cross-region pulls
- Private registry auth retries
- `imagePullPolicy: Always` on large images

Best approaches:
- Keep runtime images small
- Use regional or local registry mirrors when possible
- Avoid unnecessary `Always` pulls for immutable versioned images
- Pre-pull very large images on nodes if startup time is critical

**Tip:** if startup is slow only on fresh nodes, suspect cold image cache before suspecting the app.

---

### Debug Slow Volume Attach and Mount

Persistent storage can add substantial delay before the container even starts.

```bash
# Inspect volumes in the pod
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.volumes'

# Check PVC state
kubectl get pvc -n <namespace>
kubectl describe pvc <pvc-name> -n <namespace>

# Check PV and volume attachment state when applicable
kubectl get pv <pv-name>
kubectl get volumeattachments
```

Common causes:
- Cloud disk attach latency
- CSI node plugin delay or restart
- Previous pod still detaching the volume
- StatefulSet rescheduling to another node
- Filesystem checks or mount operations on large volumes

**Best approach:** if the delay is between scheduling and container start, and events mention volume attach or mount, stop looking at app logs and inspect storage first.

---

### Debug Slow Init Containers

Init containers can make a healthy app look slow because the app container has not started yet.

```bash
# Show init container timing and restart data
kubectl get pod <pod-name> -n <namespace> -o json | jq '.status.initContainerStatuses[] | {
  name: .name,
  state: .state,
  lastState: .lastState,
  restartCount: .restartCount
}'

# Read logs from a slow init container
kubectl logs <pod-name> -n <namespace> -c <init-container-name>
```

Typical causes:
- Migrations or schema checks
- Large config rendering or file generation
- Waiting on external services
- Archive extraction or dependency downloads

Fixes:
- Move one-time cluster-wide work into a Job
- Keep init work small and deterministic
- Add explicit logging around each init phase

---

### Debug Application Warmup and Readiness Delays

Sometimes the container starts quickly but takes a long time to become truly ready.

Examples:
- JVM warmup or class loading
- Cache warmup
- Model loading
- Leader election
- Connection pool establishment
- Slow dependency handshakes

```bash
# Check whether the container is running but not ready
kubectl get pod <pod-name> -n <namespace> -o json | jq '.status.containerStatuses[] | {
  name: .name,
  ready: .ready,
  started: .started,
  state: .state
}'

# Follow application logs during warmup
kubectl logs <pod-name> -n <namespace> -f --timestamps
```

**Best approach:** once the container is running, switch from infrastructure debugging to readiness-path debugging. The pod may be fine, but the service is not yet ready to accept traffic.

---

### Debug Readiness and Startup Probes

Mis-tuned probes often turn a slow startup into a restart loop or prolonged unready state.

```bash
# Inspect probe configuration
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.containers[] | {
  name: .name,
  startupProbe: .startupProbe,
  readinessProbe: .readinessProbe,
  livenessProbe: .livenessProbe
}'

# Inspect probe-related events
kubectl describe pod <pod-name> -n <namespace> | grep -A 8 -i 'startup\|readiness\|liveness\|unhealthy'
```

Typical problems:
- No `startupProbe` for slow-starting apps
- `readinessProbe` checks a deep dependency path instead of lightweight readiness
- Probe timeout too short
- Probe begins before the app binds its port

Good pattern:
```yaml
startupProbe:
  httpGet:
    path: /healthz
    port: 8080
  periodSeconds: 10
  failureThreshold: 30

readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  periodSeconds: 5
  timeoutSeconds: 3
```

**Tip:** use `startupProbe` to protect slow initialization and keep `readinessProbe` focused on whether the pod should receive traffic.

---

### Check Node-Specific Effects

If startup is slow only on some nodes, compare the node environment rather than the workload manifest.

```bash
# Compare pod placement
kubectl get pod -n <namespace> -o wide

# Inspect the affected node
kubectl describe node <node-name>
```

Things to compare:
- Network path to registry
- Local image cache state
- CSI and CNI daemonset health
- CPU or disk contention
- Node pressure conditions

Common node-scoped causes:
- New autoscaled nodes pull everything cold
- Busy nodes have slow disk IO
- One node has unhealthy CNI or CSI components

---

### Measure Where the Time Is Going

When startup is consistently slow, quantify each stage instead of calling it “slow startup” generically.

Useful signals:
- Time from scheduling to `Pulled`
- Time from `Pulled` to `Started`
- Time from `Started` to `Ready`
- Init container completion times
- Probe failure count before readiness succeeds

Operationally, this helps separate:
- registry problem
- storage problem
- app warmup problem
- probe configuration problem

---

### Best Practices and Prevention

#### 1. Keep Images Lean
- Reduce image size
- Avoid unnecessary layers and tooling in runtime images
- Prefer registry locality close to the cluster

#### 2. Use Startup Probes for Slow Apps
- Protect long initialization phases from premature liveness failures
- Tune readiness separately from liveness

#### 3. Minimize Heavy Per-Pod Initialization
- Move global setup to Jobs where possible
- Keep init containers focused and repeatable
- Avoid large downloads during pod startup

#### 4. Watch Storage and Node Cold-Start Effects
- Expect slower starts on fresh nodes
- Monitor CSI health and disk attach latency
- Understand StatefulSet volume handoff timing

#### 5. Instrument the Application Startup Path
- Log startup milestones clearly
- Distinguish “process started” from “ready for traffic” inside the app
- Expose lightweight readiness endpoints

---

### Quick Reference Checklist

When troubleshooting long pod startup times:

- [ ] Build the event timeline from scheduled to ready
- [ ] Check whether delay is image pull, volume attach, init work, or app warmup
- [ ] Inspect image size, pull policy, and registry latency
- [ ] Review PVC, PV, and volume attachment events
- [ ] Check init container duration and logs
- [ ] Confirm whether the container is running but not ready
- [ ] Inspect `startupProbe`, `readinessProbe`, and related events
- [ ] Compare behavior across nodes to identify cold-start or node-local issues
- [ ] Add application startup milestones to logs if timing is unclear

---

### Practical Tips and Tricks

- If the first pod on a new node is always slow but later replicas are fast, the delay is often image pull or node warmup, not application behavior.
- A pod can be healthy enough to start but not ready enough for traffic; do not treat those as the same phase.
- Slow readiness is often more useful to investigate than raw pod phase, because `Running` does not mean the service is usable.
- If startup time varies wildly, compare node placement, storage attach events, and external dependency latency before tuning probes blindly.
- When teams say “pods are failing to start,” insist on a timeline. Most incidents in this class are really “pods are starting slowly at one specific stage.”

---

## 11. Sidecar Container Not Starting Before Main Container

### Overview
Many pod designs assume a sidecar will be fully available before the main container starts. That assumption is often wrong.

In traditional Kubernetes pod startup behavior:
- Regular containers in `.spec.containers` are started independently by kubelet
- There is no strict startup ordering guarantee among regular app containers
- A main container can begin before its sidecar is actually ready

This creates common failures such as:
- App starts before log shipper, proxy, or agent is ready
- Main container depends on a local sidecar endpoint that is not listening yet
- Service mesh proxy or auth helper is present but not initialized when the app bootstraps
- Teams expect “container list order” to imply startup ordering, which it does not

Kubernetes-native sidecar container behavior is `stable` in Kubernetes `1.33` and enabled by default. Sidecars defined in `initContainers` with `restartPolicy: Always` should now be treated as a first-class option rather than a niche or emerging pattern.

---

### Fast Triage

Start by confirming whether the workload is using:
- a regular sidecar in `.spec.containers`
- an init-container workaround
- or a native sidecar pattern in newer Kubernetes versions

```bash
# Inspect pod spec for containers and init containers
kubectl get pod <pod-name> -n <namespace> -o json | jq '{
  initContainers: .spec.initContainers,
  containers: .spec.containers
}'

# Inspect status of all containers
kubectl get pod <pod-name> -n <namespace> -o json | jq '{
  initContainerStatuses: .status.initContainerStatuses,
  containerStatuses: .status.containerStatuses
}'

# Describe pod to inspect startup events
kubectl describe pod <pod-name> -n <namespace>
```

**What to look for immediately:**
- Main container starts before the sidecar is `Ready`
- Main container logs show connection refused to `localhost` sidecar ports
- App assumes a proxy, agent, or file producer is already available
- Sidecar is implemented as a regular container when ordering is actually required

---

### Understand the Default Ordering Rules

For standard pods:
- All init containers must complete successfully before app containers start
- After init containers are done, regular containers may start without guaranteed order
- Readiness of one regular container does not automatically delay startup of another regular container

That means this mental model is wrong:
- “If I list sidecar first in the YAML, it starts first and is ready first.”

It may start earlier, but Kubernetes does not guarantee that the main container will wait for it to become usable.

---

### Common Failure Patterns

#### 1. Main Container Depends on a Local Sidecar Port
Examples:
- App sends traffic through `localhost:15001` for a mesh proxy
- App expects a local auth or secrets agent on `127.0.0.1`
- App logs to a local collector sidecar at startup

Symptoms:
- App fails on boot with `connection refused`
- Restarts stop once the sidecar finally becomes available
- Intermittent startup behavior appears across replicas

#### 2. Sidecar Produces Files the App Needs
Examples:
- Sidecar fetches secrets or certs
- Sidecar renders config into a shared volume
- Main app reads those files immediately on boot

Symptoms:
- App starts before files exist
- Config or cert file errors appear in main container logs

#### 3. Service Mesh or Proxy Initialization Lag
Symptoms:
- Main app starts before iptables, proxy bootstrap, or cert provisioning is complete
- Readiness or outbound calls fail briefly during startup

---

### Debug the Main-to-Sidecar Dependency

First identify what the main container is actually waiting on.

```bash
# Inspect shared volumes and mount points
kubectl get pod <pod-name> -n <namespace> -o json | jq '{
  initContainers: [.spec.initContainers[]? | {name: .name, volumeMounts: .volumeMounts}],
  containers: [.spec.containers[] | {name: .name, volumeMounts: .volumeMounts}],
  volumes: .spec.volumes
}'

# Inspect current readiness state for all running containers
kubectl get pod <pod-name> -n <namespace> -o json | jq '.status.containerStatuses[] | {
  name: .name,
  ready: .ready,
  started: .started,
  state: .state,
  restartCount: .restartCount
}'
```

Then inspect logs from both sides:
```bash
kubectl logs <pod-name> -n <namespace> -c <main-container>
kubectl logs <pod-name> -n <namespace> -c <sidecar-container>
```

**Best approach:** if the main container is failing against `localhost` or a shared file path, stop treating it as an application-only issue. This is usually a startup coordination problem.

---

### Pattern 1: Use an Init Container for Pre-Startup Work

If the sidecar’s real job is to prepare files, fetch config, or block until a dependency exists, it may not be a sidecar at all. It may be init-container work.

Use an init container when:
- Work must finish before the app starts
- Generated files should exist first
- The task is finite rather than continuously running

Examples:
- Download cert bundle
- Render config templates
- Wait for a service to resolve and accept connections

```yaml
initContainers:
- name: wait-for-local-dependency
  image: busybox:1.36
  command: ["/bin/sh", "-c"]
  args:
  - |
    until nc -z service-name 8080; do
      echo waiting
      sleep 2
    done
```

**Best approach:** do not use a long-running sidecar when the requirement is simply “make sure something exists before app start.” That is what init containers are for.

---

### Pattern 2: Gate the Main Container With Startup Logic

If the sidecar must keep running continuously, the main container may need to wait explicitly before starting its real process.

Examples:
- Loop until local proxy port is open
- Wait for a file to appear in a shared volume
- Poll a sidecar health endpoint on `localhost`

Example shell wrapper:
```sh
#!/bin/sh
until nc -z 127.0.0.1 15001; do
  echo waiting for sidecar
  sleep 1
done

exec /app/server
```

Tradeoffs:
- Works in any Kubernetes version
- Puts coordination logic into the main container startup path
- Can become brittle if not logged and tested carefully

---

### Pattern 3: Use Readiness to Protect Traffic, Not Startup Ordering

Readiness probes help ensure a pod does not receive traffic before it is usable, but they do not stop the main process from starting.

Use readiness when:
- The app can start before the sidecar is ready
- The app should avoid traffic until local dependencies are available

Example:
```yaml
readinessProbe:
  exec:
    command: ["/bin/sh", "-c", "nc -z 127.0.0.1 15001"]
```

This helps for traffic gating, but not for applications that crash immediately if the sidecar is unavailable.

---

### Native Sidecars in Kubernetes 1.33+

Kubernetes `1.33+` supports native sidecar behavior using `initContainers` with `restartPolicy: Always`, and this behavior is stable by default.

What this changes:
- The sidecar can start during init sequencing
- It remains running for the lifetime of the pod
- Kubernetes gives you a stronger startup ordering model than regular app containers

Conceptually, native sidecars solve the gap between:
- finite init containers that must complete
- long-running sidecars that also need to start before the main app

Example pattern:
```yaml
initContainers:
- name: sidecar-proxy
  image: my-sidecar:latest
  restartPolicy: Always
  ports:
  - containerPort: 15001
```

Important considerations:
- Requires Kubernetes `1.33+` semantics to be available end to end across the cluster and tooling
- Teams need to understand that this is different from both classic init containers and classic sidecars
- Manifest tooling and platform policies may lag behind cluster capability

**Best approach:** on Kubernetes `1.33+`, if you truly need the sidecar running before the main container, native sidecars are usually the cleanest pattern.

---

### Distinguish Sidecar Readiness From Sidecar Start

A sidecar process being `Running` does not mean it is ready to serve the main app.

Examples:
- Proxy process started, but certs are not loaded yet
- Agent started, but upstream control plane handshake is incomplete
- File watcher started, but initial files are not rendered yet

Check both status and behavior:
```bash
kubectl get pod <pod-name> -n <namespace> -o json | jq '.status.containerStatuses[] | {
  name: .name,
  ready: .ready,
  started: .started,
  state: .state
}'
```

**Tip:** “main container started after sidecar process existed” is not enough. What matters is whether the dependency was usable.

---

### Debug Shared Volume Sidecar Patterns

If the sidecar writes files for the app, inspect the shared volume handoff directly.

```bash
# Inspect the volume and mounts
kubectl get pod <pod-name> -n <namespace> -o json | jq '{
  volumes: .spec.volumes,
  mounts: [.spec.containers[] | {name: .name, volumeMounts: .volumeMounts}]
}'
```

Typical problems:
- Sidecar writes after the app has already checked for the file
- Sidecar writes with root ownership, app runs non-root
- App expects atomic file creation but sees partial writes

Fixes:
- Move pre-generation to an init container
- Add explicit wait logic in the main container
- Write files atomically and log completion clearly

---

### Best Practices and Prevention

#### 1. Do Not Assume Container List Order Is Startup Order
- Listing the sidecar first does not create a readiness dependency
- Treat ordering as undefined unless you build it explicitly

#### 2. Use the Right Primitive for the Dependency
- Finite pre-start work: init container
- Long-running dependency that must start first: native sidecar if supported
- Traffic gating only: readiness probe
- Last-resort startup coordination: main-container wait loop

#### 3. Keep Startup Dependencies Observable
- Log when the sidecar is actually usable
- Log when the main container begins waiting and when it proceeds
- Avoid silent coordination logic

#### 4. Design the App to Tolerate Local Dependency Warmup
- Retry local sidecar connections during startup
- Avoid immediate fatal exits if the sidecar is expected to come up seconds later

#### 5. Test on the Real Cluster Version
- Sidecar behavior patterns differ across Kubernetes versions
- Validate that native sidecar support is actually enabled and understood by your platform

---

### Quick Reference Checklist

When the sidecar does not start before the main container:

- [ ] Check whether the sidecar is a regular container or a native sidecar pattern
- [ ] Confirm the main container depends on sidecar readiness, not just existence
- [ ] Inspect logs from both main and sidecar containers
- [ ] Check whether the dependency is a local port, file, or agent handshake
- [ ] Use an init container if the work is finite and pre-start only
- [ ] Use readiness probes for traffic gating, not startup ordering
- [ ] Add explicit wait logic if the main process must not start early
- [ ] Consider native sidecars on Kubernetes 1.33+ as the default modern ordering pattern

---

### Practical Tips and Tricks

- A sidecar shown as `Running` can still be unusable for several seconds or minutes.
- If the main app only fails on cold starts but succeeds after restart, a sidecar readiness race is a strong suspect.
- Readiness protects traffic, not process startup order.
- If the sidecar is just creating files, it probably should not be a sidecar.
- Native sidecars reduce a lot of ad hoc startup scripting, but only if your cluster and tooling fully support them.

---

## 12. Pods Not Getting Scheduled After Node Added

### Overview
Adding a new node does not automatically mean pending pods can run there. Scheduling depends on constraints, not just raw capacity.

Common reasons pods remain `Pending` even after a node joins the cluster:
- The new node is missing required labels
- The new node has taints that the pods do not tolerate
- Node affinity or node selectors exclude the new node
- Topology spread, pod anti-affinity, or zone rules still block placement
- The node is registered but not actually `Ready` for scheduling yet
- The node lacks enough allocatable CPU, memory, ephemeral storage, or pod IP capacity
- Specialized resources such as GPU, hugepages, or extended resources are still unavailable

This is a scheduler decision problem first. The fastest path is usually to read the scheduler reason exactly as Kubernetes reports it.

---

### Fast Triage

Start with the pending pod events and the actual node state.

```bash
# Check pending pods and node placement status
kubectl get pod <pod-name> -n <namespace>

# Describe the pod to see scheduling failures
kubectl describe pod <pod-name> -n <namespace>

# Inspect node list and readiness
kubectl get nodes -o wide

# Inspect recent events for the pod in time order
kubectl get events -n <namespace> \
  --field-selector involvedObject.name=<pod-name> \
  --sort-by='.lastTimestamp'
```

**What to look for immediately:**
- `0/N nodes are available`
- `node(s) had taint ...`
- `node(s) didn't match Pod's node affinity/selector`
- `node(s) didn't satisfy existing pods anti-affinity rules`
- `Insufficient cpu`, `Insufficient memory`, or `Insufficient pods`
- `node not ready` or scheduling disabled

---

### Read the Scheduler Message Literally

Most incidents here are solved by trusting the exact scheduler reason instead of assuming “new node equals more capacity.”

Typical scheduler messages point directly to the constraint class:
- labels and selectors
- taints and tolerations
- affinity and anti-affinity
- topology spread
- resource shortage
- node readiness or cordon state

If the pod says `Pending`, the scheduler has usually already explained why in `kubectl describe pod`.

---

### Check Whether the New Node Is Actually Schedulable

A new node can appear in the cluster and still not be usable for pods.

```bash
# Inspect node readiness and scheduling state
kubectl get nodes

# Describe the new node in detail
kubectl describe node <node-name>
```

Things to verify:
- `Ready=True`
- not cordoned (`SchedulingDisabled` absent)
- expected labels are present
- expected taints are understood
- CNI is healthy and pod networking is initialized
- allocatable resources are available

**Tip:** a node that just joined may still be warming up daemonsets, networking, storage plugins, or cloud metadata integration.

---

### Debug Missing Node Labels

Many workloads use `nodeSelector` or `nodeAffinity`, and a fresh node may not yet carry the required labels.

#### Inspect Pod Placement Rules
```bash
kubectl get pod <pod-name> -n <namespace> -o json | jq '{
  nodeSelector: .spec.nodeSelector,
  affinity: .spec.affinity
}'
```

#### Inspect Node Labels
```bash
kubectl get node <node-name> --show-labels

# Or inspect full metadata
kubectl get node <node-name> -o json | jq '.metadata.labels'
```

Common causes:
- Autoscaled nodes came up without environment-specific labels
- Different node pool uses a different label key
- Helm values or manifests require labels no longer applied by bootstrap automation

Fixes:
- Add the expected labels to the node pool or bootstrap process
- Adjust `nodeSelector` or `nodeAffinity` if it is overly strict

Example:
```bash
kubectl label node <node-name> workload=general
```

---

### Debug Taints and Missing Tolerations

A new node may exist specifically to isolate certain workloads. If it has taints, pods without matching tolerations will never land there.

#### Inspect Node Taints
```bash
kubectl describe node <node-name> | grep -A 5 -i taints
```

#### Inspect Pod Tolerations
```bash
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.tolerations'
```

Typical causes:
- New node pool has taints like `dedicated=foo:NoSchedule`
- Existing pods were never given matching tolerations
- Bootstrap process added temporary taints that were not removed

Example toleration:
```yaml
tolerations:
- key: "dedicated"
  operator: "Equal"
  value: "foo"
  effect: "NoSchedule"
```

**Best approach:** if the new nodes were intentionally tainted, decide whether the workload should tolerate them before changing anything.

---

### Debug Node Affinity and Anti-Affinity

Affinity rules are stricter and more subtle than simple selectors.

```bash
# Inspect full affinity rules
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.affinity'
```

Common blockers:
- Required node affinity excludes the new node labels
- Required pod anti-affinity prevents co-location
- Zone or hostname spread rules still cannot be satisfied
- A new node was added in the wrong zone for the workload rules

Typical scheduler messages:
- `didn't match Pod's node affinity`
- `didn't satisfy existing pods anti-affinity rules`

**Tip:** one additional node is not enough if the workload also requires a new zone, topology domain, or anti-affinity spacing pattern.

---

### Debug Topology Spread Constraints

Topology spread constraints can keep pods pending even when capacity exists, because the scheduler is trying to maintain balance.

```bash
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.topologySpreadConstraints'
```

Common causes:
- New node added in a topology domain that still does not satisfy skew requirements
- Required spread across zones, but new capacity was added only in one zone
- Existing pod placement already makes strict skew impossible

If topology spread is involved, inspect:
- zone labels on nodes
- existing replica placement
- `whenUnsatisfiable` behavior

---

### Check Real Allocatable Capacity

A new node can exist and still not have enough usable allocatable resources for the pending pod.

```bash
# Inspect allocatable resources on the node
kubectl describe node <node-name>

# Inspect pod requests
kubectl get pod <pod-name> -n <namespace> -o json | jq '.spec.containers[] | {
  name: .name,
  requests: .resources.requests,
  limits: .resources.limits
}'
```

Pay attention to:
- CPU and memory requests
- ephemeral storage requests
- max pods per node
- extended resources such as GPUs
- daemonset overhead consuming expected capacity

Typical scheduler messages:
- `Insufficient cpu`
- `Insufficient memory`
- `Insufficient ephemeral-storage`
- `Too many pods`

**Best approach:** always compare pod `requests` to node `allocatable`, not node machine size alone.

---

### Check DaemonSet and Bootstrap Overhead

Fresh nodes often run several daemonsets before they can host ordinary workloads.

Examples:
- CNI pod
- CSI node plugin
- logging or monitoring agents
- service mesh components

These can affect scheduling because they:
- consume CPU and memory
- add taints temporarily during startup
- keep the node unready until networking is functional

Useful checks:
```bash
kubectl get pods -A -o wide --field-selector spec.nodeName=<node-name>
kubectl describe node <node-name>
```

---

### Compare a Pending Pod Against a Working Pod

If some workloads schedule and one does not, compare their constraints directly.

```bash
# Pending pod constraints
kubectl get pod <pending-pod> -n <namespace> -o json | jq '{
  nodeSelector: .spec.nodeSelector,
  affinity: .spec.affinity,
  tolerations: .spec.tolerations,
  topologySpreadConstraints: .spec.topologySpreadConstraints
}'

# Working pod constraints
kubectl get pod <working-pod> -n <namespace> -o json | jq '{
  nodeSelector: .spec.nodeSelector,
  affinity: .spec.affinity,
  tolerations: .spec.tolerations,
  topologySpreadConstraints: .spec.topologySpreadConstraints
}'
```

This is often the fastest way to spot a missing toleration, label expectation, or stricter affinity rule.

---

### Best Practices and Prevention

#### 1. Standardize Node Labels Across Node Pools
- Keep label keys and values predictable
- Validate bootstrap automation applies required labels consistently
- Avoid fragile ad hoc labels for critical placement rules

#### 2. Be Explicit About Taint Strategy
- Document which workloads tolerate which taints
- Remove temporary bootstrap taints when intended
- Avoid silent isolation of new capacity

#### 3. Review Scheduling Rules During Cluster Scaling
- New nodes must satisfy workload constraints, not just add vCPU and RAM
- Validate zone, hostname, and topology expectations during scaling events

#### 4. Monitor Allocatable, Not Just Node Count
- Watch pod density and daemonset overhead
- Track `allocatable` versus requested capacity
- Include extended resources in capacity planning

#### 5. Keep Scheduler Errors Visible
- Alert on long-pending pods
- Surface the actual scheduler reason in dashboards or incident tooling
- Teach teams to start with `kubectl describe pod`

---

### Quick Reference Checklist

When pods stay pending after a node is added:

- [ ] Read the exact scheduler message from `kubectl describe pod`
- [ ] Confirm the new node is `Ready` and schedulable
- [ ] Check node labels against `nodeSelector` and `nodeAffinity`
- [ ] Check node taints against pod tolerations
- [ ] Inspect pod anti-affinity and topology spread rules
- [ ] Compare pod resource requests to node `allocatable`
- [ ] Account for daemonset overhead and max pod count
- [ ] Verify the new node is in the right zone or topology domain
- [ ] Compare the pending pod to a similar working pod

---

### Practical Tips and Tricks

- “We added a node” is not a scheduling explanation; it is just a capacity event. Constraints still decide placement.
- If the pending reason says `didn't match Pod's node affinity`, more nodes without the right labels change nothing.
- If the new node pool is tainted, unschedulable workloads are often working exactly as designed.
- A fresh node may be visible in `kubectl get nodes` before it is truly ready for normal workloads.
- When only one deployment stays pending after scale-out, compare its scheduling rules to a deployment that successfully landed on the new node.

---

### Closing Note

This guide now covers the requested pod lifecycle troubleshooting topics from CrashLoopBackOff through scheduling failures after node scale-out. Keep using pod events, node state, and container lifecycle timing as the primary sources of truth during incidents.

For Kubernetes `1.33+`, also watch for newer signals such as `PodReadyToStartContainers`, stable native sidecars, and any kubelet-level restart backoff customization in your cluster configuration.
