# Containerization Deep Dive

## Overview

Containers are not magic — they are processes isolated using Linux kernel primitives. This guide covers how container images are built, how layers work, how the overlay filesystem enables efficient storage, and how Kubernetes interacts with container runtimes.

## 1. Container Images

### What is a Container Image?

A container image is a **read-only template** containing:
- Application code and dependencies
- Runtime (Python, Node.js, JVM, etc.)
- System libraries
- Configuration

**Key insight:** An image is NOT a virtual machine. It has no kernel, no init system, just files and metadata.

### Image Manifest

Every image has a **manifest** — a JSON document describing its contents:

```json
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
  "config": {
    "mediaType": "application/vnd.docker.container.image.v1+json",
    "digest": "sha256:abc123...",
    "size": 2345
  },
  "layers": [
    {
      "mediaType": "application/vnd.docker.image.rootfs.diff.tar.gzip",
      "digest": "sha256:layer1hash...",
      "size": 27091735
    },
    {
      "digest": "sha256:layer2hash...",
      "size": 1234567
    },
    {
      "digest": "sha256:layer3hash...",
      "size": 89012
    }
  ]
}
```

### Image Config (History + Entrypoint)

```json
{
  "architecture": "amd64",
  "os": "linux",
  "config": {
    "Entrypoint": ["/usr/bin/python3"],
    "Cmd": ["/app/main.py"],
    "Env": ["PATH=/usr/local/bin:/usr/bin:/bin", "PYTHONPATH=/app"],
    "WorkingDir": "/app",
    "ExposedPorts": {"8080/tcp": {}},
    "Labels": {"version": "1.0"}
  },
  "history": [
    {"created_by": "FROM debian:bullseye"},
    {"created_by": "RUN apt-get install python3"},
    {"created_by": "COPY app/ /app/"},
    {"created_by": "CMD [\"/app/main.py\"]", "empty_layer": true}
  ]
}
```

## 2. Image Layers

### The Layer System

**Key principle:** Every instruction in a Dockerfile creates a new layer. Layers are stacked to form the final filesystem.

```
Image: my-app:v1.0

Layer 5 (RW): Container runtime layer (ephemeral, per-container)
──────────────────────────────────────────────────────────────
Layer 4 (RO): ADD app.py /app/        sha256:aabbcc...
Layer 3 (RO): COPY requirements.txt   sha256:ddeeff...
Layer 2 (RO): RUN apt install python3 sha256:112233...
Layer 1 (RO): FROM debian:bullseye   sha256:445566...
```

### Layer Sharing

```
Image A: web-app
  Layer 1: FROM ubuntu:22.04  (sha256:111)
  Layer 2: RUN apt install nginx  (sha256:222)
  Layer 3: COPY html/ /var/www/  (sha256:333)

Image B: api-app
  Layer 1: FROM ubuntu:22.04  (sha256:111)  ← SAME LAYER!
  Layer 2: RUN apt install python3  (sha256:444)
  Layer 3: COPY src/ /app/  (sha256:555)

Disk usage:
  Without sharing: 2× ubuntu + 2× app layers
  With sharing:    1× ubuntu (shared!) + 2× app layers
```

**Result:** `docker pull` skips layers already present.

### What Each Dockerfile Instruction Creates

```dockerfile
FROM ubuntu:22.04        # Layer: base OS filesystem
                         # Size: ~77MB

RUN apt-get update && \  # Layer: apt metadata + downloaded packages
    apt-get install -y python3 python3-pip
                         # Size: ~120MB (all new files)

COPY requirements.txt /  # Layer: just requirements.txt
                         # Size: <1KB

RUN pip install -r /requirements.txt
                         # Layer: all Python packages
                         # Size: depends on packages

COPY app/ /app/         # Layer: application code
                         # Size: application size

CMD ["python3", "/app/main.py"]  # NO new layer (metadata only)
```

### Layer Caching

```dockerfile
# BAD: requirements.txt changes → reinstall packages every time
COPY . /app/
RUN pip install -r /app/requirements.txt

# GOOD: Only reinstall when requirements change
COPY requirements.txt /
RUN pip install -r /requirements.txt
COPY app/ /app/
```

**Cache invalidation:** If a layer changes, all subsequent layers are rebuilt.

## 3. OverlayFS - How Layers Are Mounted

### What is OverlayFS?

OverlayFS is a **union filesystem** that merges multiple directories into a single view. It enables container layers without copying data.

```
Container View (what process sees):
/usr/, /etc/, /app/, /tmp/ → merged view

UPPERDIR (read-write, container-specific changes)
WORKDIR  (internal use)
─────────────────────────────────────────────────
LOWERDIR (read-only image layers, from bottom):
  Layer 4: /app/main.py, /app/templates/
  Layer 3: /requirements.txt
  Layer 2: /usr/lib/python3/, /usr/bin/python3
  Layer 1: /bin/, /usr/, /etc/, /lib/   (base OS)
```

### OverlayFS Mount

```bash
# Manual OverlayFS example
mkdir -p /tmp/overlay/{lower,upper,work,merged}
echo "from lower" > /tmp/overlay/lower/file.txt

# Mount overlay
mount -t overlay overlay \
  -o lowerdir=/tmp/overlay/lower,upperdir=/tmp/overlay/upper,workdir=/tmp/overlay/work \
  /tmp/overlay/merged

# Read file (comes from lower)
cat /tmp/overlay/merged/file.txt
# OUTPUT: from lower

# Modify file (copy-on-write to upper)
echo "modified" > /tmp/overlay/merged/file.txt

# Original lower is unchanged!
cat /tmp/overlay/lower/file.txt  # "from lower"

# Change is in upper (container layer)
cat /tmp/overlay/upper/file.txt  # "modified"

# Cleanup
umount /tmp/overlay/merged
```

### Copy-on-Write (CoW)

```
Read file from container:
  1. Check upper layer → not there
  2. Check layer 4 → not there
  3. Check layer 3 → FOUND!
  4. Return file from layer 3

Modify file in container:
  1. Find file (same as above)
  2. Copy file to upper layer
  3. Modify the upper copy
  4. Original in lower layers unchanged

Delete file in container:
  1. Create "whiteout" file in upper layer
  2. OverlayFS hides the lower layer file
```

### Inspecting Container Layers

```bash
# Find container's overlay mounts
docker inspect <container-id> | grep -A5 "GraphDriver"
# OUTPUT:
# "GraphDriver": {
#   "Name": "overlay2",
#   "Data": {
#     "LowerDir": "/var/lib/docker/overlay2/abc.../diff:...",
#     "MergedDir": "/var/lib/docker/overlay2/xyz.../merged",
#     "UpperDir": "/var/lib/docker/overlay2/xyz.../diff",
#     "WorkDir": "/var/lib/docker/overlay2/xyz.../work"

# View layer contents on host
ls /var/lib/docker/overlay2/

# For containerd (Kubernetes)
ls /var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots/
```

## 4. Container Runtimes

### Runtime Architecture

```
kubectl
  │
  ▼ CRI (Container Runtime Interface)
kubelet
  │
  ▼ gRPC (CRI socket)
High-Level Runtime (containerd / CRI-O)
  │
  ▼ OCI spec
Low-Level Runtime (runc / gVisor / Kata)
  │
  ▼ Linux kernel
Namespaces + cgroups + OverlayFS
```

### CRI - Container Runtime Interface

**Purpose:** Kubernetes-defined gRPC API so kubelet doesn't depend on specific runtime.

**Services:**
- `RuntimeService`: Create/start/stop/remove containers
- `ImageService`: Pull/list/remove images

```bash
# Check what CRI is in use
kubectl get node <node-name> -o jsonpath='{.status.nodeInfo.containerRuntimeVersion}'
# OUTPUT: containerd://1.7.0

# CRI socket paths:
# containerd: unix:///var/run/containerd/containerd.sock
# CRI-O:      unix:///var/run/crio/crio.sock
```

### containerd

**Most common Kubernetes runtime (post-Docker).**

```bash
# containerd uses namespaces (not Linux namespaces, but its own)
ctr namespaces list
# NAME      LABELS
# k8s.io    (Kubernetes containers)
# default   (manual/docker-compatible)

# List containers
ctr -n k8s.io containers list

# List images
ctr -n k8s.io images list

# Or use crictl (CRI-compatible tool)
crictl images
crictl ps
crictl pods
```

### runc - The OCI Runtime

**runc** is the low-level runtime that actually creates containers.

```bash
# runc uses OCI (Open Container Initiative) bundle:
ls /path/to/oci-bundle/
# config.json  (container spec)
# rootfs/      (container filesystem)

# Run directly (not typical, but educational)
runc run my-container
```

**config.json (simplified):**
```json
{
  "ociVersion": "1.0.0",
  "process": {
    "terminal": false,
    "user": {"uid": 0, "gid": 0},
    "args": ["/app/main.py"],
    "env": ["PATH=/usr/bin:/bin"]
  },
  "root": {"path": "rootfs"},
  "namespaces": [
    {"type": "pid"},
    {"type": "network", "path": "/var/run/netns/pod-netns"},
    {"type": "ipc"},
    {"type": "uts"},
    {"type": "mount"}
  ],
  "linux": {
    "cgroupsPath": "/kubepods/burstable/pod-uid/container-id",
    "resources": {
      "memory": {"limit": 268435456},
      "cpu": {"quota": 100000, "period": 100000}
    }
  }
}
```

## 5. Dockerfile Best Practices

### Minimize Layer Size

```dockerfile
# BAD: Creates large layer with apt cache
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y pip

# GOOD: Single layer, cache cleaned
RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*
```

### Multi-Stage Builds

```dockerfile
# Stage 1: Build (includes build tools)
FROM golang:1.21 AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download                    # Layer: dependencies
COPY . .
RUN CGO_ENABLED=0 go build -o server . # Layer: compiled binary

# Stage 2: Run (minimal image)
FROM alpine:3.18
                                       # No Go toolchain!
COPY --from=builder /app/server /server  # Just the binary
EXPOSE 8080
CMD ["/server"]

# Result: 15MB image instead of 900MB!
```

### Security Considerations

```dockerfile
# Don't run as root
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
USER appuser

# Don't copy secrets into image
# BAD:
COPY .env /app/              # Now in image layer forever!

# GOOD:
# Pass at runtime via environment variables or secrets

# Scan image for vulnerabilities
# trivy image my-app:v1.0
```

## 6. Image Registries

### How `docker pull` (or containerd pull) Works

```
1. Parse image reference
   "nginx:1.25"
   Registry: index.docker.io (Docker Hub)
   Image:    library/nginx
   Tag:      1.25

2. Get manifest
   GET https://registry-1.docker.io/v2/library/nginx/manifests/1.25
   Authorization: Bearer <token>
   Response: manifest JSON with layer list

3. Check local cache
   For each layer: Is sha256:xxx... already in local store?

4. Pull missing layers (parallel)
   GET https://registry-1.docker.io/v2/library/nginx/blobs/sha256:xxx
   Decompress gzip
   Verify checksum

5. Unpack to snapshot (overlayfs layers)
   containerd stores in /var/lib/containerd/
```

### Kubernetes Image Pull Policy

```yaml
spec:
  containers:
  - name: app
    image: my-app:v1.0
    imagePullPolicy: Always     # Always pull (even if cached)
    # imagePullPolicy: IfNotPresent  # Only pull if not cached (default for tagged)
    # imagePullPolicy: Never         # Never pull (must be pre-loaded)
```

**Important:** Using `latest` tag with `IfNotPresent` → won't get updates!

### Private Registries

```bash
# Create pull secret
kubectl create secret docker-registry regcred \
  --docker-server=my-registry.example.com \
  --docker-username=user \
  --docker-password=password \
  --docker-email=user@example.com

# Reference in pod
spec:
  imagePullSecrets:
  - name: regcred
  containers:
  - name: app
    image: my-registry.example.com/my-app:v1.0
```

## 7. Container Image Inspection

```bash
# Inspect image layers
docker history nginx:1.25
# IMAGE         CREATED BY                              SIZE
# <layer>       CMD ["nginx" "-g" "daemon off;"]        0B
# <layer>       ENTRYPOINT ["/docker-entrypoint.sh"]    0B
# <layer>       COPY docker-entrypoint.sh /             4.62kB
# ...

# Export and inspect image
docker save nginx:1.25 | tar -tv
# Shows: manifest.json, config.json, layer.tar files

# Inspect with crane (OCI tool)
crane manifest nginx:1.25 | jq .

# With containerd
ctr -n k8s.io content ls
ctr -n k8s.io images export nginx.tar nginx:1.25
```

## 8. Kubernetes Container Lifecycle

### How Kubernetes Starts a Container

```
kubectl apply -f pod.yaml
          │
          ▼ API Server stores in etcd
Scheduler → Assigns pod to node
          │
          ▼ Kubelet detects pending pod
          │
          ▼ CRI: RunPodSandbox()
          │  Creates pause container (network namespace holder)
          │  Sets up pod network (CNI called here)
          │
          ▼ CRI: PullImage() → containerd pulls if needed
          │
          ▼ CRI: CreateContainer()
          │  Creates overlayfs snapshot
          │  Creates container config (OCI spec)
          │
          ▼ CRI: StartContainer()
             runc called → creates namespaces, cgroups
             Process starts as PID 1 in container
```

### The Pause Container

```bash
# Every pod has a "pause" container
docker ps | grep pause
# or
crictl ps | grep pause

# It holds the network namespace
# All other containers in pod join this namespace
# Why? If any container crashes, network namespace persists
```

### Container States

```
Waiting → Running → Terminated
            │
            ├── Crash → Waiting (CrashLoopBackOff)
            └── Complete → Terminated (Completed)
```

## 9. Container Security

### Read-Only Root Filesystem

```yaml
spec:
  containers:
  - name: app
    securityContext:
      readOnlyRootFilesystem: true
    volumeMounts:
    - name: tmp
      mountPath: /tmp           # App needs writable /tmp
  volumes:
  - name: tmp
    emptyDir: {}
```

### Capabilities

```yaml
spec:
  containers:
  - name: app
    securityContext:
      capabilities:
        drop:
        - ALL            # Drop all capabilities
        add:
        - NET_BIND_SERVICE  # Only add what's needed (bind <1024)
```

**Common capabilities:**
| Capability | Allows |
|-----------|--------|
| NET_ADMIN | Network configuration |
| NET_BIND_SERVICE | Bind to ports < 1024 |
| SYS_PTRACE | Process tracing |
| SYS_ADMIN | Many admin operations |
| CHOWN | Change file ownership |

## Next Steps

✅ Container images, manifests, and layers  
✅ OverlayFS and copy-on-write  
✅ Container runtimes (containerd, runc, CRI)  
✅ Dockerfile best practices and multi-stage builds  
✅ Container security fundamentals  

**Move to:** [06-yaml-rest-apis.md](06-yaml-rest-apis.md) to understand YAML syntax and the Kubernetes API that powers everything.
