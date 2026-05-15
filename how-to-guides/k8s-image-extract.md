# Kubernetes Node — Container Image Export Guide

A step-by-step guide to exporting a container image as a `.tar` from Kubernetes nodes (EKS), covering Bottlerocket and standard Amazon Linux nodes, with multiple fallback approaches.

---

## Table of Contents

1. [Find the Image Name](#1-find-the-image-name)
2. [Access the Node](#2-access-the-node)
   - [Standard Amazon Linux Node](#standard-amazon-linux-node)
   - [Bottlerocket Node](#bottlerocket-node)
3. [Export the Image to Tar on the Node](#3-export-the-image-to-tar-on-the-node)
   - [Using ctr (containerd)](#using-ctr-containerd)
   - [Using crictl](#using-crictl)
   - [Using nerdctl](#using-nerdctl)
   - [No tools found — download crane](#no-tools-found--download-crane)
4. [Copy the Tar to Local Machine](#4-copy-the-tar-to-local-machine)
   - [Method A: busybox Pod + kubectl cp (recommended)](#method-a-busybox-pod--kubectl-cp-recommended)
   - [Method B: Stream via kubectl exec cat](#method-b-stream-via-kubectl-exec-cat)
   - [Method C: Split large files](#method-c-split-large-files-if-file-is-very-large)
   - [Method D: Via S3](#method-d-via-s3)
5. [Skip the Node — Pull Directly from ECR](#5-skip-the-node--pull-directly-from-ecr-easiest)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Find the Image Name

From your local machine (requires `kubectl`):

```bash
# Get the node the pod is running on
kubectl get pod <pod-name> -n <namespace> -o wide

# Get the exact image name used by the pod
kubectl get pod <pod-name> -n <namespace> -o jsonpath='{.spec.containers[0].image}'
```

From inside the pod:

```bash
# Check environment or mounted metadata
cat /etc/hostname
```

From the node itself (if kubectl is unavailable):

```bash
# Using crictl
sudo crictl ps
sudo crictl inspect <container-id> | grep -i image

# Using ctr
sudo ctr -n k8s.io containers list
```

---

## 2. Access the Node

### Standard Amazon Linux Node

```bash
# SSM Session Manager (no SSH key needed — most common on EKS)
aws ssm start-session --target <ec2-instance-id>

# Or via SSH if enabled
ssh -i your-key.pem ec2-user@<node-ip>
```

### Bottlerocket Node

Bottlerocket has a layered shell model. You must go through each layer:

```bash
# Step 1: SSM drops you into the "control container" (very limited)
aws ssm start-session --target <ec2-instance-id>

# Step 2: Enter the admin container (more tools available)
sudo enter-admin-container

# Step 3: Drop into the actual host OS root shell via sheltie
sudo sheltie
```

> **Bottlerocket shell layers:**
> - SSM → **control container** (very limited, `apiclient` only)
> - `enter-admin-container` → **admin container** (curl, basic tools)
> - `sudo sheltie` → **host OS root shell** (containerd tools available)
> - Filesystem is mostly read-only except `/tmp`, `/var`, `/run`

---

## 3. Export the Image to Tar on the Node

All commands below assume you are in the host shell (`sheltie` on Bottlerocket, or direct SSH on Amazon Linux).

### Using `ctr` (containerd)

> EKS nodes use the `k8s.io` containerd namespace — always pass `-n k8s.io`.

```bash
# List all images on the node
ctr -n k8s.io images list

# List all containers and their images
ctr -n k8s.io containers list

# Inspect a specific container to find its image
ctr -n k8s.io containers info <container-id>

# Export image to tar
ctr -n k8s.io images export /tmp/image.tar <image-name>:<tag>
```

If `ctr` is not in PATH, try full paths:

```bash
which containerd
/usr/bin/ctr -n k8s.io images list
/usr/local/bin/ctr -n k8s.io images list
```

### Using `crictl`

```bash
# List all running containers
sudo crictl ps

# List all images
sudo crictl images

# Get image name from a container
sudo crictl inspect <container-id> | grep -i image

# Get all container → image mappings at once
sudo crictl ps -o json | python3 -c "
import json,sys
for c in json.load(sys.stdin)['containers']:
    print(c['metadata']['name'], c['image']['image'])
"
```

> Note: `crictl` does not support direct image export — use it to find the image name, then export with `ctr`.

### Using `nerdctl`

```bash
# Check if available
nerdctl --help

# List images
sudo nerdctl -n k8s.io images

# Save image to tar
sudo nerdctl -n k8s.io save <image>:<tag> -o /tmp/image.tar
```

### No tools found — download `crane`

If none of the above tools are available, download `crane` (Google's container tool) to `/tmp`:

```bash
curl -sL https://github.com/google/go-containerregistry/releases/latest/download/go-containerregistry_Linux_x86_64.tar.gz \
  | tar -xz -C /tmp crane

# Pull and save image to tar
/tmp/crane pull <image>:<tag> /tmp/image.tar
```

Or download `nerdctl`:

```bash
curl -sL https://github.com/containerd/nerdctl/releases/latest/download/nerdctl-linux-amd64.tar.gz \
  | sudo tar -xz -C /usr/local/bin nerdctl

sudo nerdctl -n k8s.io save <image>:<tag> -o /tmp/image.tar
```

---

## 4. Copy the Tar to Local Machine

### Method A: busybox Pod + kubectl cp (recommended)

This uses a `hostPath` volume to expose the node's `/tmp` into a pod, then copies it out with `kubectl`.

**Step 1:** Create a pod pinned to the exact node where the tar was exported:

```yaml
# busybox-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: image-transfer
  namespace: <your-namespace>
spec:
  nodeName: <your-node-name>    # pin to the exact node
  containers:
  - name: busybox
    image: busybox
    command: ["sleep", "3600"]
    volumeMounts:
    - name: host-tmp
      mountPath: /host-tmp
  volumes:
  - name: host-tmp
    hostPath:
      path: /tmp                # mounts node's /tmp into the pod
```

```bash
kubectl apply -f busybox-pod.yaml
kubectl wait --for=condition=Ready pod/image-transfer -n <your-namespace>
```

**Step 2:** Verify the tar is visible inside the pod:

```bash
kubectl exec -n <your-namespace> image-transfer -- ls -lh /host-tmp/image.tar
```

**Step 3:** Copy tar to local machine:

```bash
kubectl cp -n <your-namespace> image-transfer:/host-tmp/image.tar ./image.tar
```

**Step 4:** Cleanup:

```bash
kubectl delete pod image-transfer -n <your-namespace>
```

---

### Method B: Stream via `kubectl exec cat`

Use this if `kubectl cp` fails with `unexpected EOF` (common with large files):

```bash
kubectl exec -n <your-namespace> image-transfer -- cat /host-tmp/image.tar > ./image.tar
```

Or stream with `dd` to see progress:

```bash
kubectl exec -n <your-namespace> image-transfer -- dd if=/host-tmp/image.tar bs=1M | dd of=./image.tar bs=1M status=progress
```

> `kubectl cp` internally wraps with tar again, which can cause issues with already-tarred files or large transfers. The `cat` redirect is more reliable.

---

### Method C: Split large files (if file is very large)

If the image is too large to transfer in one shot, split it on the node first:

**On the node (inside sheltie):**

```bash
# Check file size first
ls -lh /tmp/image.tar

# Split into 500MB chunks
split -b 500m /tmp/image.tar /tmp/image_part_
ls /tmp/image_part_*
```

**From local machine:**

```bash
# Copy each chunk
kubectl exec -n <your-namespace> image-transfer -- cat /host-tmp/image_part_aa > image_part_aa
kubectl exec -n <your-namespace> image-transfer -- cat /host-tmp/image_part_ab > image_part_ab
# ... repeat for all parts

# Reassemble locally
cat image_part_* > image.tar

# Verify
docker load -i image.tar
```

---

### Method D: Via S3

If kubectl access is unavailable or the file is too large for kubectl:

```bash
# On the node (inside sheltie):
aws s3 cp /tmp/image.tar s3://your-bucket/image.tar

# From local machine:
aws s3 cp s3://your-bucket/image.tar ./image.tar
```

---

## 5. Skip the Node — Pull Directly from ECR (easiest)

If the image is stored in ECR, skip the node entirely and pull it straight to your local machine:

```bash
# Authenticate
aws ecr get-login-password --region <region> | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.<region>.amazonaws.com

# Pull the image
docker pull <account-id>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>

# Save to tar
docker save <account-id>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag> -o image.tar
```

---

## 6. Troubleshooting

### `unexpected EOF` on `kubectl cp`

```bash
# Use cat stream instead
kubectl exec -n <namespace> image-transfer -- cat /host-tmp/image.tar > ./image.tar
```

### `ctr: command not found`

```bash
# Try crictl
sudo crictl images

# Try full paths
ls /usr/bin/ctr /usr/local/bin/ctr 2>/dev/null

# Download crane to /tmp as fallback
curl -sL https://github.com/google/go-containerregistry/releases/latest/download/go-containerregistry_Linux_x86_64.tar.gz \
  | tar -xz -C /tmp crane
/tmp/crane pull <image>:<tag> /tmp/image.tar
```

### `kubectl: command not found` on the node

Normal — `kubectl` is not installed on nodes. Use `crictl` or `ctr` directly on the node to find image names.

### Verifying the exported tar is valid

```bash
# Load with Docker
docker load -i image.tar

# Load with containerd
ctr images import image.tar

# Inspect without loading
tar -tvf image.tar | head -20
```

### Bottlerocket — `enter-admin-container` not found

```bash
# Use apiclient to enable it first (from control container)
apiclient exec admin bash

# Or enable via SSM Run Command in AWS Console:
# Command: enable-admin-container
```

---

## Quick Reference

| Situation | Best Approach |
|---|---|
| ECR image | Pull directly from ECR locally |
| Have `ctr` in sheltie | `ctr -n k8s.io images export` → S3 or busybox pod |
| Have `crictl` only | Use `crictl` to find image name → export with `ctr` |
| No tools on node | Download `crane` to `/tmp` |
| Any registry + credentials | `skopeo copy docker://registry/img:tag docker-archive:/tmp/out.tar` |
| File too large for `kubectl cp` | `kubectl exec -- cat` stream or split + reassemble |
