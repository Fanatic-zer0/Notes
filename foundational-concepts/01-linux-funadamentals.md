# Linux Fundamentals for Kubernetes

## Overview

Kubernetes heavily relies on Linux kernel features to provide isolation, resource management, and networking for containers. Understanding these primitives is essential for troubleshooting and architecting Kubernetes solutions.

## 1. Linux Namespaces - Container Isolation

**What They Are:** Namespaces provide isolated views of system resources. Each container runs in its own set of namespaces, making it appear as if it has its own system.

### Types of Namespaces

```
┌─────────────────────────────────────────────────────────────────┐
│                    Container Process                            │
├─────────────────────────────────────────────────────────────────┤
│ PID Namespace    │ Own process tree (PID 1 inside)             │
│ Network Namespace│ Own network stack (interfaces, routes)      │
│ Mount Namespace  │ Own filesystem view                         │
│ UTS Namespace    │ Own hostname                                │
│ IPC Namespace    │ Own shared memory, semaphores               │
│ User Namespace   │ Own UID/GID mappings                        │
│ Cgroup Namespace │ Own cgroup view                             │
└─────────────────────────────────────────────────────────────────┘
```

### 1.1 PID Namespace

**Purpose:** Isolate process ID space. Process inside container sees itself as PID 1.

**Example:**
```bash
# On host: Create new PID namespace
sudo unshare --pid --fork --mount-proc bash

# Inside namespace
ps aux
# OUTPUT: Shows only processes in this namespace
# PID 1 is the bash process

# From host (different terminal)
ps aux | grep bash
# Shows the REAL PID (e.g., 12345)
```

**Kubernetes Usage:**
- Each container has its own PID namespace
- Init process (CMD/ENTRYPOINT) is PID 1
- `kubectl exec` creates new process in container's PID namespace

### 1.2 Network Namespace

**Purpose:** Isolate network stack - interfaces, routing tables, firewall rules.

**Example:**
```bash
# Create new network namespace
sudo ip netns add blue-ns

# List namespaces
ip netns list
# OUTPUT: blue-ns

# Execute command in namespace
sudo ip netns exec blue-ns ip addr
# OUTPUT: Only loopback interface (no eth0)

# Create veth pair (virtual ethernet cable)
sudo ip link add veth0 type veth peer name veth1

# Move one end to namespace
sudo ip link set veth1 netns blue-ns

# Configure interfaces
sudo ip addr add 10.0.0.1/24 dev veth0
sudo ip link set veth0 up

sudo ip netns exec blue-ns ip addr add 10.0.0.2/24 dev veth1
sudo ip netns exec blue-ns ip link set veth1 up
sudo ip netns exec blue-ns ip link set lo up

# Test connectivity
ping -c 2 10.0.0.2
# SUCCESS: Communicating across namespaces!
```

**Kubernetes Usage:**
- Each pod has its own network namespace
- All containers in a pod share the same network namespace
- CNI plugins manage veth pairs and routing

**Visual:**
```
┌──────────────────┐         ┌──────────────────┐
│   Host Network   │         │  Pod Network NS  │
│                  │         │                  │
│   ┌──────────┐   │  veth   │   ┌──────────┐   │
│   │   eth0   │   │  pair   │   │ Container│   │
│   │(physical)│   │◄───────►│   │   eth0   │   │
│   └──────────┘   │         │   └──────────┘   │
│   ┌──────────┐   │         │   10.244.1.5     │
│   │   cni0   │───┼─────────┤                  │
│   │ (bridge) │   │         │                  │
│   └──────────┘   │         │                  │
└──────────────────┘         └──────────────────┘
```

### 1.3 Mount Namespace

**Purpose:** Isolate filesystem mount points.

**Example:**
```bash
# Create mount namespace
sudo unshare --mount bash

# Mount in this namespace only
mkdir /tmp/test-mount
mount -t tmpfs tmpfs /tmp/test-mount

# Only visible in this namespace
df -h | grep test-mount

# Exit namespace - mount is gone
exit
df -h | grep test-mount  # No output
```

**Kubernetes Usage:**
- Containers see their own filesystem (image layers + volumes)
- ConfigMaps/Secrets mounted as files
- Persistent volumes attached

### 1.4 UTS Namespace

**Purpose:** Isolate hostname and domain name.

**Example:**
```bash
sudo unshare --uts bash
hostname my-container
hostname  # OUTPUT: my-container

# Exit - hostname reverts
exit
hostname  # OUTPUT: Original hostname
```

**Kubernetes Usage:**
- Pod hostname = pod name
- FQDN: `<pod-name>.<service-name>.<namespace>.svc.cluster.local`

### 1.5 IPC Namespace

**Purpose:** Isolate inter-process communication (shared memory, message queues).

**Example:**
```bash
# In host: Create shared memory segment
ipcmk -M 1024
# Shared memory id: 0

ipcs -m  # Shows shared memory

# In new IPC namespace
sudo unshare --ipc bash
ipcs -m  # Shows NOTHING (isolated)
```

**Kubernetes Usage:**
- Containers in same pod share IPC namespace (can use shared memory)
- Different pods have isolated IPC

### 1.6 User Namespace

**Purpose:** Map UIDs/GIDs between container and host.

**Example:**
```bash
# Run as root inside namespace, but unprivileged outside
unshare --user --map-root-user bash
id  # OUTPUT: uid=0(root) gid=0(root)

# But on host, you're still your normal user
# This is how rootless containers work!
```

**Kubernetes Usage:**
- Used for rootless containers (Podman, usernetes)
- Security: root inside container ≠ root on host

## 2. Control Groups (cgroups) - Resource Management

**What They Are:** Linux kernel feature to limit, account, and isolate resource usage (CPU, memory, I/O).

### cgroups v1 vs v2

| Feature | cgroups v1 | cgroups v2 |
|---------|-----------|-----------|
| Hierarchy | Multiple hierarchies | Single unified hierarchy |
| Controllers | Per-hierarchy | Unified |
| Status | Legacy | Modern (default in newer kernels) |

### Key Controllers

```
┌────────────────────────────────────────────────────────┐
│                    cgroup Controllers                  │
├────────────────────────────────────────────────────────┤
│ cpu         │ CPU time limits                          │
│ cpuset      │ CPU core pinning                         │
│ memory      │ Memory limits, OOM killer                │
│ blkio       │ Block I/O throttling                     │
│ net_cls     │ Network traffic classification           │
│ pids        │ Process ID limits                        │
└────────────────────────────────────────────────────────┘
```

### 2.1 CPU Limits

**Example:**
```bash
# Create cgroup
sudo cgcreate -g cpu:/limited-cpu

# Set CPU quota (50% of one core)
# period = 100ms, quota = 50ms
sudo cgset -r cpu.cfs_period_us=100000 limited-cpu
sudo cgset -r cpu.cfs_quota_us=50000 limited-cpu

# Run process in cgroup
sudo cgexec -g cpu:limited-cpu stress --cpu 1
# Process will use max 50% CPU

# Verify
top  # CPU usage ~50%
```

**Kubernetes Mapping:**
```yaml
resources:
  requests:
    cpu: "500m"      # 0.5 cores
  limits:
    cpu: "1000m"     # 1 core (cfs_quota / cfs_period)
```

**Behind the scenes:**
- `requests`: CPU shares (scheduling priority)
- `limits`: `cpu.cfs_quota_us` value

### 2.2 Memory Limits

**Example:**
```bash
# Create memory cgroup
sudo cgcreate -g memory:/limited-mem

# Set 100MB limit
sudo cgset -r memory.limit_in_bytes=104857600 limited-mem

# Run memory-hungry process
sudo cgexec -g memory:limited-mem stress --vm 1 --vm-bytes 200M

# Result: OOM killer terminates process when exceeding 100MB
dmesg | tail
# OUTPUT: Memory cgroup out of memory: Kill process...
```

**Kubernetes Mapping:**
```yaml
resources:
  requests:
    memory: "128Mi"  # memory.soft_limit_in_bytes
  limits:
    memory: "256Mi"  # memory.limit_in_bytes
```

**OOM Behavior:**
- Container exceeds limit → OOM killed
- Pod status: `OOMKilled`
- `RestartPolicy: Always` → kubelet restarts container

### 2.3 Viewing cgroups

```bash
# Find cgroup for a process
cat /proc/self/cgroup
# OUTPUT:
# 0::/user.slice/user-1000.slice/session-1.scope

# Find container cgroups
docker run -d --name test nginx
docker inspect test | grep Pid  # Get PID
cat /proc/<PID>/cgroup

# Kubernetes pod cgroups
# Located in: /sys/fs/cgroup/kubepods/...
```

### Kubernetes QoS Classes and cgroups

```
┌─────────────────────────────────────────────────────┐
│ QoS Class    │ requests == limits │ cgroup Priority│
├─────────────────────────────────────────────────────┤
│ Guaranteed   │ Yes                │ Highest        │
│ Burstable    │ No                 │ Medium         │
│ BestEffort   │ None specified     │ Lowest         │
└─────────────────────────────────────────────────────┘
```

**Eviction Order:** BestEffort → Burstable → Guaranteed

## 3. Virtual Ethernet (veth) Pairs

**What They Are:** Virtual network cables connecting two network namespaces.

**Analogy:** Like an ethernet cable with two ends - one in container, one on host.

### Creating veth Pairs

```bash
# Create pair
sudo ip link add veth-host type veth peer name veth-container

# Verify
ip link show type veth
# OUTPUT:
# veth-host@veth-container
# veth-container@veth-host

# Move one end to namespace
sudo ip netns add container-ns
sudo ip link set veth-container netns container-ns

# Configure
sudo ip addr add 10.0.0.1/24 dev veth-host
sudo ip link set veth-host up

sudo ip netns exec container-ns ip addr add 10.0.0.2/24 dev veth-container
sudo ip netns exec container-ns ip link set veth-container up
sudo ip netns exec container-ns ip link set lo up

# Test
ping -c 2 10.0.0.2  # SUCCESS!
```

**Visual Flow:**
```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  ┌─────────────┐           veth pair           ┌─────────┐│
│  │             │◄──────────────────────────────►│Container││
│  │    Host     │  veth0          veth1          │Namespace││
│  │   Bridge    │                                │         ││
│  │   (cni0)    │                                │  eth0   ││
│  └─────────────┘                                └─────────┘│
│        │                                                    │
│        └──► Other containers (via bridge)                  │
└────────────────────────────────────────────────────────────┘
```

### Kubernetes Usage

**Every pod:**
1. Has its own network namespace
2. Connected via veth pair to host bridge (cni0)
3. Bridge handles pod-to-pod communication on same node

**Inspect in Kubernetes:**
```bash
# Find pod's network namespace
kubectl get pod <pod-name> -o jsonpath='{.status.containerStatuses[0].containerID}'
# Extract container ID

# On node:
docker inspect <container-id> | grep Pid
sudo nsenter -t <PID> -n ip addr
# Shows pod's network interfaces (eth0 is veth endpoint)

# On host:
ip link | grep veth
# Shows host-side veth endpoints
```

## 4. Systemd - Service Management

**What It Is:** Init system and service manager for Linux.

### Systemd Units

**Types:**
- **service**: Daemons/processes
- **socket**: Socket-based activation
- **timer**: Scheduled tasks (replaces cron)
- **mount**: Filesystem mounts

### Managing Services

```bash
# View status
systemctl status kubelet

# Start/stop/restart
sudo systemctl start kubelet
sudo systemctl stop kubelet
sudo systemctl restart kubelet

# Enable (auto-start on boot)
sudo systemctl enable kubelet

# View logs
journalctl -u kubelet -f  # Follow logs
journalctl -u kubelet --since "10 minutes ago"
```

### Kubelet as Systemd Service

**Unit file:** `/etc/systemd/system/kubelet.service`

```ini
[Unit]
Description=Kubernetes Kubelet
Documentation=https://kubernetes.io/docs/
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/bin/kubelet \
  --config=/var/lib/kubelet/config.yaml \
  --container-runtime-endpoint=unix:///var/run/containerd/containerd.sock
Restart=always
StartLimitInterval=0
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Key Points:**
- `Restart=always`: Kubelet restarts on failure
- `After=network-online.target`: Waits for network
- `ExecStart`: Command to run

### Journalctl for Debugging

```bash
# View kubelet logs with priority
journalctl -u kubelet -p err  # Errors only

# Follow logs for multiple units
journalctl -u kubelet -u containerd -f

# Export logs
journalctl -u kubelet --since today --no-pager > kubelet.log

# Disk usage
journalctl --disk-usage
```

## 5. Hands-On Lab: Build a "Container" from Scratch

**Goal:** Create isolated environment using only Linux primitives (no Docker!).

```bash
#!/bin/bash
# container-from-scratch.sh

# 1. Create root filesystem
mkdir -p /tmp/container-root/{bin,lib,lib64}

# Copy bash and dependencies
cp /bin/bash /tmp/container-root/bin/
cp /bin/ls /tmp/container-root/bin/

# Copy shared libraries
ldd /bin/bash | grep -o '/lib[^ ]*' | xargs -I {} cp {} /tmp/container-root/lib/
ldd /bin/ls | grep -o '/lib[^ ]*' | xargs -I {} cp {} /tmp/container-root/lib/

# 2. Create namespaces and cgroups
sudo cgcreate -g cpu,memory:/my-container
sudo cgset -r memory.limit_in_bytes=104857600 my-container  # 100MB
sudo cgset -r cpu.cfs_quota_us=50000 my-container           # 50% CPU

# 3. Launch "container"
sudo cgexec -g cpu,memory:/my-container \
  unshare --uts --pid --mount --fork \
  chroot /tmp/container-root \
  /bin/bash

# Inside "container":
# - New UTS namespace (can change hostname)
# - New PID namespace (bash is PID 1)
# - New mount namespace (isolated filesystem)
# - Resource limits via cgroups
```

**Verify:**
```bash
# Inside container
hostname my-container
hostname  # OUTPUT: my-container

ps aux  # Only bash and ps visible

# Try to use too much memory
dd if=/dev/zero of=/dev/null bs=1M count=200  # OOM killed!
```

## 6. Key Takeaways

### Namespace Usage in Pods

```
Pod = Shared Namespaces Among Containers
┌─────────────────────────────────────┐
│ Pod: web-app                        │
├─────────────────────────────────────┤
│ Shared:                             │
│  • Network namespace (same IP)      │
│  • IPC namespace (shared memory)    │
│  • UTS namespace (same hostname)    │
├─────────────────────────────────────┤
│ Container 1: nginx                  │
│  • Own PID namespace                │
│  • Own mount namespace              │
├─────────────────────────────────────┤
│ Container 2: log-forwarder          │
│  • Own PID namespace                │
│  • Own mount namespace              │
└─────────────────────────────────────┘
```

### cgroups in Kubernetes

```yaml
# Pod YAML
resources:
  requests:
    cpu: "500m"        # cpu.shares (scheduling)
    memory: "256Mi"    # No hard limit, but influences eviction
  limits:
    cpu: "1"           # cpu.cfs_quota_us
    memory: "512Mi"    # memory.limit_in_bytes
```

**Result:**
- Container placed in cgroup hierarchy
- Path: `/sys/fs/cgroup/kubepods/<qos-class>/<pod-uid>/<container-id>/`
- Kernel enforces limits

### Networking Chain

```
Container Process
    ↓
Network Namespace (eth0 = veth endpoint)
    ↓
veth pair
    ↓
Host Bridge (cni0)
    ↓
Host Routing Table
    ↓
Physical Network / Overlay
```

## 7. Practice Exercises

1. **Create isolated web server:**
   - New network namespace
   - Run nginx in namespace
   - Connect via veth pair
   - Access from host

2. **Test cgroup limits:**
   - Create pod with memory limit
   - Run stress test
   - Observe OOM kill

3. **Inspect running pod:**
   - Find pod's PID
   - Check its namespaces (`ls -la /proc/<PID>/ns`)
   - Enter namespace with `nsenter`
   - View cgroup settings

4. **Build minimal container:**
   - Use `unshare` and `chroot`
   - Add only required binaries
   - Apply resource limits

## Next Steps

Now that you understand:
✅ How containers are isolated (namespaces)  
✅ How resources are limited (cgroups)  
✅ How pod networking works (veth pairs)  
✅ How services are managed (systemd)

**Move to:** [02-networking-fundamentals.md](02-networking-fundamentals.md) to understand IP addressing, routing, and how packets flow through Kubernetes networks.

## Additional Resources

- **Namespaces:** `man namespaces`, `man unshare`
- **cgroups:** `man cgroups`, `/sys/fs/cgroup/` exploration
- **veth:** `man ip-link`, `man bridge`
- **systemd:** `man systemctl`, `man journalctl`

**Practice environment:** Ubuntu 22.04 VM with root access
