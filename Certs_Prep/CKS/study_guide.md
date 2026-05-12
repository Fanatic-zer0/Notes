# Certified Kubernetes Security Specialist (CKS) Study Guide

## Overview

This guide is a practical, in-depth reference for preparing for the Certified Kubernetes Security Specialist exam. It focuses on the security ideas the exam expects you to understand operationally, not just memorize: what a control does, why it exists, how to implement it in a cluster, and how to verify or troubleshoot it quickly.

The CKS exam is hands-on. That changes how you should study. You are not being tested on whether you can recognize security buzzwords; you are being tested on whether you can harden a running cluster, write or fix manifests, inspect nodes and workloads, trace risk back to a concrete control, and make correct changes fast.

Use this guide in three layers:

1. Read each section for the conceptual model.
2. Reproduce the examples until the commands feel natural.
3. Practice under time pressure until the secure configuration becomes your default configuration.

Related notes in this repo:

- [RBAC](../../k8s-concepts/14-rbac.md)
- [Pod Security](../../k8s-concepts/20-pod-security.md)
- [Admission Controllers](../../k8s-concepts/13-admission-controllers.md)
- [Authn and Identity](../../k8s-concepts/19-authn-identity.md)
- [Network Policies](../../k8s-concepts/05-network-policies.md)
- [ConfigMaps and Secrets](../../k8s-concepts/07-configmaps-secrets.md)
- [Observability](../../k8s-concepts/21-observability.md)
- [Linux Fundamentals](../../k8s-concepts/foundational-concepts/01-linux-fundamentals.md)
- [Containerization Deep Dive](../../k8s-concepts/foundational-concepts/05-containerization-deep-dive.md)

---

## 1. What CKS Actually Tests

At a high level, CKS covers five security problem areas:

1. Cluster setup and hardening
2. System hardening
3. Minimizing microservice vulnerabilities
4. Supply chain security
5. Monitoring, logging, and runtime security

Those domains are not isolated. A realistic compromise usually spans multiple layers:

- An overly broad RBAC role lets an attacker list Secrets.
- A privileged Pod with host mounts turns that API compromise into node compromise.
- Unsigned images or unscanned dependencies introduce malicious code.
- Missing runtime detection lets the compromise continue unnoticed.

You should study each domain as part of a single control chain:

`prevent -> restrict -> detect -> investigate -> recover`

That mental model helps you choose the correct tool during the exam.

---

## 2. Study Strategy for a Hands-On Security Exam

### Why most people study inefficiently

Many candidates over-index on reading and under-index on repetition. Security controls feel understandable when you read them, but the exam measures whether you can apply them with precision.

Examples:

- Knowing that seccomp reduces syscall exposure is not enough.
- You need to know how to set `seccompProfile`, where to verify it, and what failure symptoms look like.

- Knowing that NetworkPolicy isolates traffic is not enough.
- You need to know that policies are additive, that enforcement depends on the CNI, and how to test ingress versus egress.

### How to study effectively

Train in this order:

1. Understand the threat model.
2. Learn the Kubernetes control that addresses it.
3. Apply the control in YAML or on the node.
4. Verify that it works.
5. Break it intentionally and debug it.

### Core habits to build

- Be fluent with `kubectl explain`.
- Be fluent with JSONPath and label selectors.
- Be able to edit manifests quickly with minimal typing.
- Learn the default-deny mindset for RBAC, NetworkPolicy, Pod Security, and Linux privileges.
- Always verify after you change something.

High-value commands to memorize:

```bash
# Quickly inspect security-relevant Pod fields
kubectl get pod <pod> -o yaml

# Generate a starter manifest and edit from there
kubectl run nginx --image=nginx --dry-run=client -o yaml

# Check a ServiceAccount on a Pod
kubectl get pod <pod> -o jsonpath='{.spec.serviceAccountName}'

# Check containers' security contexts
kubectl get pod <pod> -o jsonpath='{.spec.containers[*].securityContext}'

# Check effective capabilities from the container manifest
kubectl get pod <pod> -o jsonpath='{.spec.containers[*].securityContext.capabilities}'

# Find all ClusterRoleBindings
kubectl get clusterrolebindings

# Inspect NetworkPolicies in a namespace
kubectl get networkpolicies -n <ns>

# See recent events when a security control blocks a workload
kubectl get events -n <ns> --sort-by=.lastTimestamp
```

---

## 3. Cluster Setup and Hardening

This domain is about securing the Kubernetes control plane and the cluster's foundational configuration.

### 3.1 Threat Model

### What

Cluster hardening protects the control plane, PKI, node bootstrap paths, Secrets at rest, and the default trust boundaries between components.

### Why

If the control plane is weak, application-level hardening does not matter. A user or workload that can abuse the API server, kubelet, or etcd can often escalate to full cluster compromise.

### How

You reduce risk by:

- limiting who can authenticate
- limiting what authenticated identities can do
- encrypting sensitive data
- reducing insecure endpoints
- enabling visibility through audit logging
- benchmarking configuration against known guidance

---

### 3.2 CIS Benchmarks and kube-bench

### What

The CIS Kubernetes Benchmark is a security baseline describing recommended settings for control plane components, worker nodes, file permissions, anonymous access, audit settings, and more.

### Why

It gives you a repeatable hardening baseline. It does not guarantee security, but it reliably identifies common misconfigurations.

### How

Use `kube-bench` to compare a node or cluster against the CIS recommendations.

```bash
# Run kube-bench on a node or in a privileged diagnostic container
# Note: 'master' was renamed to 'controlplane' in kube-bench v0.6+
kube-bench run --targets controlplane,node
```

You should understand the types of findings it reports:

- overly permissive file permissions on PKI files
- insecure API server flags
- kubelet settings exposing unauthenticated access
- missing audit logging
- weak etcd protection

### Example

If `kube-bench` reports that anonymous authentication is enabled on the API server, the risk is clear: unauthenticated requests may reach code paths that should require identity. The fix is to ensure the API server starts with:

```text
--anonymous-auth=false
```

Then verify with the API server static Pod manifest, often under:

```text
/etc/kubernetes/manifests/kube-apiserver.yaml
```

### Exam angle

You may be asked to identify which component flag is wrong or apply a benchmark-aligned fix. Focus on the meaning of the setting, not just the flag name.

---

### 3.3 API Server Hardening

### What

The API server is the front door of the cluster. Every control-plane and user action eventually goes through it.

### Why

If the API server accepts insecure requests, logs insufficiently, or exposes powerful verbs to the wrong identities, the cluster is at risk.

### How

Common hardening themes:

- disable anonymous authentication
- use strong authorization modes
- enable audit logging
- restrict admission to required controllers
- secure TLS and client certificate trust
- protect etcd connectivity

Important flags and concepts:

```text
--anonymous-auth=false
--authorization-mode=Node,RBAC
--audit-log-path=/var/log/kubernetes/audit.log
--audit-policy-file=/etc/kubernetes/audit-policy.yaml
--enable-admission-plugins=NodeRestriction,...
--client-ca-file=<path>
--etcd-cafile=<path>
--etcd-certfile=<path>
--etcd-keyfile=<path>
```

### Example audit policy

```yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
- level: Metadata
  resources:
  - group: ""
    resources: ["secrets", "configmaps"]
- level: RequestResponse
  verbs: ["create", "update", "patch", "delete"]
  resources:
  - group: ""
    resources: ["pods"]
- level: Metadata
  omitStages:
  - RequestReceived
```

### Why this policy matters

- `Metadata` is cheaper than full payload logging and often enough for investigation.
- `RequestResponse` for mutating Pod actions gives more detail when tracking suspicious changes.
- `omitStages` reduces log noise.

### Verification

```bash
# Confirm the audit policy file is mounted into the apiserver static Pod
grep -n audit /etc/kubernetes/manifests/kube-apiserver.yaml

# Confirm audit log output exists
ls -l /var/log/kubernetes/
```

---

### 3.4 Authorization: Node + RBAC

### What

Authorization determines what an authenticated identity may do.

### Why

Authentication answers who you are. Authorization answers what you may do. A secure auth system without secure authorization is still unsafe.

### How

In Kubernetes, common modes include:

- `Node` authorizer for kubelet identities
- `RBAC` for users, groups, and service accounts

The secure pattern is:

```text
trusted identity -> minimum required privileges -> explicit binding
```

### Example

Bad pattern:

```yaml
kind: ClusterRoleBinding
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: bad-binding
subjects:
- kind: ServiceAccount
  name: default
  namespace: prod
roleRef:
  kind: ClusterRole
  name: cluster-admin
  apiGroup: rbac.authorization.k8s.io
```

Why it is dangerous:

- the `default` ServiceAccount is automatically used by Pods that do not specify a safer account
- `cluster-admin` effectively grants full cluster control

Safer pattern:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
  namespace: prod
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: pod-reader-binding
  namespace: prod
subjects:
- kind: ServiceAccount
  name: app-sa
  namespace: prod
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

### Verification

```bash
# Can this service account list secrets?
kubectl auth can-i list secrets --as=system:serviceaccount:prod:app-sa -n prod

# Can it read pods?
kubectl auth can-i get pods --as=system:serviceaccount:prod:app-sa -n prod
```

The `kubectl auth can-i` command is one of the highest-value security validation tools in the exam.

---

### 3.5 Restricting Service Account Token Exposure

### What

Pods can receive Kubernetes API credentials through mounted service account tokens.

### Why

If a workload is compromised and its token is mounted, the attacker can use the token against the API server within that ServiceAccount's permissions.

### How

Reduce risk by:

- disabling automatic token mount when not needed
- using dedicated ServiceAccounts
- giving each ServiceAccount only the minimum RBAC it needs

Example:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: secure-app
  namespace: prod
spec:
  serviceAccountName: app-sa
  automountServiceAccountToken: false
  containers:
  - name: app
    image: nginx:1.27
```

### Important nuance

`automountServiceAccountToken: false` prevents automatic credential mount, but it does not remove the Pod's identity conceptually. If the app needs API access, you must deliberately supply credentials another way or leave mounting enabled intentionally.

---

### 3.6 Secrets Encryption at Rest

### What

Kubernetes Secrets are base64-encoded in manifests, but base64 is not encryption. Without encryption at rest, Secret data is stored in etcd in a form that can be read by anyone with etcd access.

### Why

Protecting etcd is critical because etcd is the source of truth for cluster state. If etcd is compromised, plaintext Secrets may be exposed.

### How

Configure the API server with an `EncryptionConfiguration` file.

Example:

```yaml
apiVersion: apiserver.config.k8s.io/v1
kind: EncryptionConfiguration
resources:
- resources:
  - secrets
  providers:
  - aescbc:
      keys:
      - name: key1
        secret: c29tZTMyaXRlbWtleXRvYmU2NHRlc2xvbmcK
  - identity: {}
```

Then wire it into the API server:

```text
--encryption-provider-config=/etc/kubernetes/encryption-config.yaml
```

### Why provider order matters

Kubernetes uses providers in order. If `identity` comes first, data may be written unencrypted. The usual pattern is:

1. preferred encryption provider first
2. fallback providers later
3. `identity` last

### Re-encryption workflow

After enabling encryption, existing Secret objects may still be stored in the old format. A common operational step is to rewrite them so the new provider re-encrypts the data.

```bash
# Example pattern: rewrite secrets to trigger re-encryption
kubectl get secrets --all-namespaces -o json | kubectl replace -f -
```

Be careful with bulk rewrites in production. The exam may ask for concept or configuration rather than a risky mass rewrite.

---

### 3.7 Admission Control and NodeRestriction

### What

Admission controllers sit in the request path after authentication and authorization but before persistence.

### Why

They enforce policy on the shape and validity of objects being created or modified.

### How

Two especially important ideas for CKS:

- `NodeRestriction` stops kubelets from modifying objects they should not control.
- policy engines such as Pod Security Admission, Kyverno, or OPA Gatekeeper restrict insecure workload definitions.

Example reason `NodeRestriction` matters:

- a kubelet should update its own Node status
- it should not be able to mutate arbitrary Pods or other Nodes

---

### 3.8 TLS, PKI, and kubeconfig Security

### What

Kubernetes relies heavily on mutual TLS and certificate-based trust.

### Why

Without a trustworthy certificate chain and secure kubeconfig handling, attackers can impersonate users or components, intercept traffic, or steal administrative credentials.

### How

Protect:

- CA private keys
- admin kubeconfigs
- component client certificates
- etcd peer and client certificates

Practical guidance:

- keep `/etc/kubernetes/pki` permissions strict
- avoid copying admin kubeconfigs unnecessarily
- rotate credentials if exposure is suspected
- prefer separate credentials for separate roles

Example kubeconfig risks:

- a kubeconfig may embed client certificate material
- possession of the file may be enough to act as that user

---

### 3.9 Protecting Cloud Metadata Endpoints

### What

Cloud providers (AWS, GCP, Azure) expose an instance metadata API, typically at the link-local address `169.254.169.254`. This API returns IAM credentials, bootstrap tokens, SSH keys, and other sensitive data about the VM the node is running on.

### Why

A compromised Pod that can reach this endpoint may be able to retrieve node-level IAM credentials and use them to move laterally within the cloud environment, well beyond the Kubernetes cluster boundary.

### How

Block access to the metadata IP from Pods using NetworkPolicy. With a default-deny egress baseline in place, the metadata endpoint is never in any allowlist and cannot be reached. For namespaces that still allow broad egress, use an explicit block with `ipBlock.except`.

Example for a namespace without full default-deny:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: block-cloud-metadata
  namespace: prod
spec:
  podSelector: {}
  policyTypes:
  - Egress
  egress:
  - to:
    - ipBlock:
        cidr: 0.0.0.0/0
        except:
        - 169.254.169.254/32
```

### Why the default-deny approach is cleaner

With a default-deny egress policy, 169.254.169.254 is simply never included in an allow rule. The `ipBlock.except` pattern is useful when you cannot yet move a namespace to full deny-by-default.

### Exam angle

The exam may present a scenario where a Pod in a cloud cluster should not be able to access instance metadata. The correct answer is a NetworkPolicy that excludes `169.254.169.254/32` from reachable egress.

---

### 3.10 Verifying Platform Binary Integrity

### What

Kubernetes binaries (`kubectl`, `kubeadm`, `kubelet`) are distributed with accompanying SHA-256 checksum files. Verifying these checksums before use confirms the binary matches the officially published artifact.

### Why

A supply chain attack on a Kubernetes binary would compromise every node or operator workstation that downloads it unverified. Even a trusted distribution channel can be abused if binaries are not verified end-to-end.

### How

```bash
# Download a binary and its published checksum
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl.sha256"

# Verify the checksum — output must say OK
echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check
```

Expected output:

```text
kubectl: OK
```

If the output is anything other than `OK`, the binary should not be used.

### Example on an existing binary

For binaries already installed on a node, you can compute the hash and compare it to the published checksum for that specific version:

```bash
sha256sum $(which kubelet)
```

Then cross-reference the output against the official release checksum file for your Kubernetes version.

### Exam angle

You may be asked to verify a binary before using it or to identify why binary verification is a security control. The answer always involves `sha256sum` against the published checksum, not trust in a package manager alone.

---

## 4. System Hardening

This domain moves down a level, from Kubernetes objects to the node operating system and the container runtime environment.

### 4.1 Threat Model

### What

System hardening reduces the chance that a compromised process, container, or user session can escalate into broader node compromise.

### Why

Containers share a kernel. That means the kernel and host OS are part of your cluster's security boundary.

### How

You harden the system by controlling:

- host access
- privilege boundaries
- kernel attack surface
- runtime isolation
- file permissions
- allowed syscalls and Linux capabilities

---

### 4.2 Principle of Least Privilege on the Node

### What

Only required users, services, binaries, and kernel features should be present and enabled.

### Why

Every extra service or capability broadens the attack surface.

### How

Examples of node hardening tasks:

- disable password SSH login when possible
- restrict sudo access
- remove unused packages
- patch the OS and container runtime
- restrict file permissions on kubelet and PKI material
- limit direct root usage
- protect the kubelet configuration and kubeconfig files

Example files to care about:

```text
/var/lib/kubelet/config.yaml
/var/lib/kubelet/kubeconfig
/etc/kubernetes/kubelet.conf
/etc/kubernetes/pki/
```

### Example

If the kubelet kubeconfig is world-readable, any local user could potentially use its credentials or learn cluster endpoint details. The correct fix is not just changing permissions, but also understanding why the kubelet needs those credentials and ensuring only the kubelet service user can read them.

---

### 4.3 Kubelet Hardening

### What

The kubelet is the node agent that talks to the API server and instructs the container runtime to run Pods.

### Why

If the kubelet is weakly configured, attackers may bypass normal API controls or retrieve sensitive execution data directly from the node.

### How

Historically, insecure kubelet endpoints caused major security issues. You should understand these controls:

- disable anonymous auth
- use webhook authentication
- use webhook authorization
- protect read-only or debug endpoints

Example settings in kubelet config:

```yaml
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
authentication:
  anonymous:
    enabled: false
  webhook:
    enabled: true
authorization:
  mode: Webhook
readOnlyPort: 0
```

### Why this matters

- `anonymous.enabled: false` blocks unauthenticated requests.
- `authorization.mode: Webhook` delegates authorization to the API server.
- `readOnlyPort: 0` avoids exposing unauthenticated access on the old read-only interface.

### Verification

```bash
# Inspect the effective kubelet config
grep -n "readOnlyPort\|authentication\|authorization" /var/lib/kubelet/config.yaml
```

---

### 4.4 Seccomp

### What

Seccomp filters Linux syscalls. A container process may need only a subset of the full syscall surface.

### Why

Many exploits depend on reaching dangerous syscalls. Reducing the allowed syscall set reduces the impact of a compromise.

### How

Kubernetes exposes seccomp through `securityContext.seccompProfile`.

Example:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: seccomp-demo
spec:
  containers:
  - name: app
    image: nginx:1.27
    securityContext:
      seccompProfile:
        type: RuntimeDefault
```

Common values:

- `RuntimeDefault`: use the container runtime's default profile
- `Localhost`: use a node-local custom profile
- `Unconfined`: no seccomp filtering

### Why `RuntimeDefault` is usually the right baseline

It is safer than `Unconfined`, simpler than managing many custom profiles, and usually compatible with common workloads.

### Example troubleshooting

Symptom:

- a container fails unexpectedly after seccomp enforcement

Possible explanation:

- the app depends on a syscall blocked by the chosen profile

Response:

1. inspect Pod events and logs
2. identify whether a custom profile is too restrictive
3. adjust profile or workload behavior

---

### 4.5 AppArmor and SELinux

### What

AppArmor and SELinux are Linux Security Modules that enforce mandatory access control.

### Why

Traditional Unix permissions are not always enough. Mandatory access controls restrict what a process may do even if the process is running as root inside its namespace.

### How

You should understand the role of each:

- AppArmor: profile-based restrictions, often path-oriented
- SELinux: label-based enforcement across subjects and objects

In Kubernetes, these controls are tied to node support and workload configuration.

**AppArmor configuration (Kubernetes 1.30+):**

Since Kubernetes 1.30, AppArmor is configured via `securityContext.appArmorProfile` (the older annotation-based approach `container.apparmor.security.beta.kubernetes.io/<container>` is deprecated). The field mirrors the seccomp pattern:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: apparmor-demo
spec:
  containers:
  - name: app
    image: nginx:1.27
    securityContext:
      appArmorProfile:
        type: RuntimeDefault  # or: Localhost (with localhostProfile), or Unconfined
```

Profile types:

- `RuntimeDefault`: the container runtime's built-in AppArmor profile
- `Localhost`: a custom profile loaded on the node, referenced by name via `localhostProfile`
- `Unconfined`: no AppArmor enforcement

### Example use case

If a containerized process tries to write to sensitive host-mounted paths, an AppArmor or SELinux profile can block the operation even when the application itself is compromised.

---

### 4.6 Linux Capabilities

### What

Linux capabilities split root privileges into smaller units such as `NET_ADMIN`, `SYS_ADMIN`, or `CHOWN`.

### Why

A container should almost never need the full power of root. Capabilities let you grant only the narrow privileges required.

### How

The secure baseline is to drop all capabilities and add back only what is required.

Example:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: cap-demo
spec:
  containers:
  - name: app
    image: busybox:1.36
    command: ["sh", "-c", "sleep 3600"]
    securityContext:
      allowPrivilegeEscalation: false
      capabilities:
        drop: ["ALL"]
        add: ["NET_BIND_SERVICE"]
```

### Why `SYS_ADMIN` is dangerous

`SYS_ADMIN` is often described as the new root because it enables many powerful kernel-level operations. If you see it granted, treat it as high risk.

---

### 4.7 Privileged Containers and Host Namespaces

### What

Privileged containers and host namespace sharing weaken isolation between the container and the node.

### Why

These settings are common paths to node compromise.

### How

Security-sensitive fields to review:

- `privileged: true`
- `hostPID: true`
- `hostNetwork: true`
- `hostIPC: true`
- hostPath mounts
- `allowPrivilegeEscalation: true`

Example of a risky Pod:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: dangerous-pod
spec:
  hostPID: true
  containers:
  - name: shell
    image: busybox:1.36
    command: ["sh", "-c", "sleep 3600"]
    securityContext:
      privileged: true
    volumeMounts:
    - name: host-root
      mountPath: /host
  volumes:
  - name: host-root
    hostPath:
      path: /
```

### Why this is dangerous

- privileged mode gives broad kernel access
- host PID namespace reveals host processes
- host root mount exposes the node filesystem

This is a textbook escape-risk configuration.

---

### 4.8 readOnlyRootFilesystem and Non-Root Execution

### What

These controls reduce what an attacker can modify inside a container and avoid giving unnecessary user privileges.

### Why

Attackers prefer writable filesystems and root privileges because they make persistence and post-exploitation easier.

### How

Example secure container spec:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hardened-pod
spec:
  containers:
  - name: app
    image: nginx:1.27
    securityContext:
      runAsNonRoot: true
      runAsUser: 10001
      readOnlyRootFilesystem: true
      allowPrivilegeEscalation: false
      capabilities:
        drop: ["ALL"]
```

### Common failure mode

Some images assume write access to paths like `/tmp`, `/var/cache`, or `/run`. The secure fix is often:

- keep `readOnlyRootFilesystem: true`
- mount a small writable `emptyDir` only where needed

---

## 5. Minimizing Microservice Vulnerabilities

This domain is about reducing workload-level attack surface and enforcing safer defaults in application deployments.

### 5.1 Threat Model

### What

Microservice vulnerability reduction means controlling how Pods communicate, what they can access, what privileges they run with, and what policies gate deployment.

### Why

Even well-written applications get exploited. Your job is to limit blast radius.

### How

You use layered controls:

- NetworkPolicy
- Pod security settings
- admission policy
- RBAC
- Secret handling
- secure ingress and egress design

---

### 5.2 Pod Security Standards and Pod Security Admission

### What

Pod Security Standards define three policy levels:

- Privileged
- Baseline
- Restricted

Pod Security Admission enforces or audits those standards at the namespace level.

### Why

It replaces the older PodSecurityPolicy model with a simpler built-in admission mechanism.

### How

You configure namespace labels.

Example:

```bash
kubectl label ns prod \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted
```

### What `restricted` typically pushes you toward

- non-root containers
- no privilege escalation
- restricted volume types
- no privileged containers
- seccomp usage

### Example troubleshooting

If a Pod suddenly fails after enabling `restricted`, inspect:

```bash
kubectl describe pod <pod> -n <ns>
kubectl get events -n <ns> --sort-by=.lastTimestamp
```

The admission error will often tell you exactly which field violates policy.

---

### 5.3 Network Policies

### What

NetworkPolicies define which Pods may communicate with which other Pods or IP blocks, on which ports, and in which direction.

### Why

Without network segmentation, a single compromised Pod can laterally move across the cluster much more easily.

### How

Core rules to remember:

- policies are namespace-scoped
- policies select Pods, not Services
- ingress and egress are separate
- policies are additive
- enforcement depends on the CNI plugin supporting NetworkPolicy

### Default deny example

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: prod
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

### Why this matters

This does not "block the namespace" in an abstract way. It selects all Pods in the namespace and denies traffic unless another policy allows it.

### Allow frontend to backend on 8080

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
  namespace: prod
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080
```

### Example reasoning

What:

- backend Pods allow ingress only from Pods labeled `app=frontend` on TCP 8080

Why:

- shrinks lateral movement and avoids accidental wide-open east-west traffic

How to verify:

- use a temporary test Pod in the same namespace
- attempt allowed and disallowed connections

```bash
kubectl run testbox --rm -it --image=busybox:1.36 -- sh

# Inside the shell
nc -vz backend 8080
nc -vz backend 9090
```

---

### 5.4 Ingress, Egress, and Exposed Services

### What

Applications are vulnerable not only through Pod-to-Pod traffic but through publicly exposed entry points and unrestricted outbound access.

### Why

Ingress expands attack surface. Egress matters because malware often needs outbound communication for command-and-control, payload retrieval, or data exfiltration.

### How

Secure patterns:

- expose only what must be exposed
- terminate TLS correctly
- avoid broad NodePort usage when unnecessary
- restrict egress to required dependencies
- separate internal and external services clearly

Example restricted egress policy:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns-and-db-egress
  namespace: prod
spec:
  podSelector:
    matchLabels:
      app: api
  policyTypes:
  - Egress
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          kubernetes.io/metadata.name: kube-system
      podSelector:
        matchLabels:
          k8s-app: kube-dns
    ports:
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 53
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
```

### Common pitfall

People enable egress deny and forget DNS. Then applications fail with what looks like a service outage but is really blocked name resolution. Also remember to allow both UDP 53 and TCP 53: DNS uses UDP for standard queries and falls back to TCP for large responses (over 512 bytes) and zone transfers.

---

### 5.5 Secure Workload Specs

### What

A secure workload spec is one where the manifest itself avoids unnecessary privilege.

### Why

Many vulnerabilities are not code vulnerabilities. They are insecure deployment choices.

### How

Good defaults for most workloads:

- `runAsNonRoot: true`
- `allowPrivilegeEscalation: false`
- `readOnlyRootFilesystem: true`
- drop all capabilities
- use seccomp `RuntimeDefault`
- do not use host namespaces unless required
- use dedicated ServiceAccounts

Example deployment snippet:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  namespace: prod
spec:
  replicas: 2
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      serviceAccountName: api-sa
      automountServiceAccountToken: false
      containers:
      - name: api
        image: ghcr.io/example/api@sha256:1111111111111111111111111111111111111111111111111111111111111111
        ports:
        - containerPort: 8080
        securityContext:
          runAsNonRoot: true
          runAsUser: 10001
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false
          seccompProfile:
            type: RuntimeDefault
          capabilities:
            drop: ["ALL"]
        volumeMounts:
        - name: tmp
          mountPath: /tmp
      volumes:
      - name: tmp
        emptyDir: {}
```

### Why pin by digest

Using an image digest reduces tag drift. `:latest` or mutable tags make it harder to know what you actually deployed.

---

### 5.6 Secrets Management Inside Workloads

### What

Applications need credentials, but how those credentials are delivered affects security.

### Why

Secrets leak through overly broad RBAC, env vars in debug outputs, accidental logs, or plaintext storage.

### How

Safer patterns:

- mount Secrets as files when appropriate
- scope access by namespace and ServiceAccount
- avoid injecting unnecessary Secrets into many Pods
- use external secret systems when available
- rotate credentials and restart workloads safely

Secret volume example:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-secret
spec:
  containers:
  - name: app
    image: nginx:1.27
    volumeMounts:
    - name: db-creds
      mountPath: /var/run/secrets/db
      readOnly: true
  volumes:
  - name: db-creds
    secret:
      secretName: db-credentials
```

### Exam thinking

If asked to secure secret usage, do not stop at creating a Secret object. Ask:

- who can read it?
- where is it stored?
- how is it consumed?
- is it logged accidentally?

---

### 5.7 OPA Gatekeeper, Kyverno, and Policy-as-Code

### What

These tools extend admission control so you can enforce custom security policies.

### Why

Built-in controls cover common cases, but real organizations often need rules like:

- only allow images from approved registries
- require labels for ownership
- block privileged containers
- require read-only root filesystems

### How

Conceptually, the flow is:

```text
admission request -> policy evaluation -> allow or deny -> optional mutation
```

Kyverno often feels more Kubernetes-native for YAML-driven policies. Gatekeeper uses OPA/Rego and is powerful for constraint-style policy.

Example Kyverno-style requirement concept:

- deny Pods that do not set `runAsNonRoot: true`

The exam may not require large policy authoring, but you should understand why these tools exist and what kinds of controls they enforce.

---

### 5.8 Container Runtime Sandboxes

### What

Container runtime sandboxes provide a stronger isolation layer between workload processes and the host kernel. Standard containers share the host kernel directly; sandboxes add an intermediate layer.

### Why

If a container process exploits a kernel vulnerability, a kernel bug reached from a container affects the entire node. Sandboxes reduce the kernel attack surface by intercepting or virtualizing the syscall interface.

### How

In Kubernetes, sandboxes are configured using `RuntimeClass`. A `RuntimeClass` object maps a name to a container runtime handler. The Pod references the class by name.

Two dominant options:

**gVisor:**

- implements a user-space kernel (`runsc`)
- container syscalls are intercepted by gVisor's Go kernel, not passed directly to the host
- reduces host kernel exposure significantly
- trade-offs: some syscalls are unsupported or slower; not all workloads are compatible

**Kata Containers:**

- runs each Pod in a lightweight VM using hardware virtualization
- strongest kernel isolation because the workload kernel is separate from the host kernel
- more resource overhead than gVisor but better workload compatibility

Example `RuntimeClass` registration (cluster admin step):

```yaml
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: gvisor
handler: runsc
```

Pod referencing the sandbox:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: sandboxed-app
  namespace: prod
spec:
  runtimeClassName: gvisor
  containers:
  - name: app
    image: nginx:1.27
    securityContext:
      runAsNonRoot: true
      allowPrivilegeEscalation: false
      capabilities:
        drop: ["ALL"]
```

Verification:

```bash
# Confirm which runtime class the Pod is using
kubectl get pod sandboxed-app -n prod -o jsonpath='{.spec.runtimeClassName}'

# List available RuntimeClass objects
kubectl get runtimeclass
```

### What sandboxes do not replace

- they are a kernel isolation control, not a network or RBAC control
- you still need NetworkPolicy, securityContext, RBAC, and image scanning alongside a sandbox
- a sandbox does not prevent a misconfigured RBAC binding or a stolen Service Account token from being used

### Exam angle

- know what problem a sandbox solves: kernel isolation, not identity or network isolation
- know how to assign `runtimeClassName` in a Pod spec
- know that the `RuntimeClass` object must exist before a Pod can reference it

---

### 5.9 Pod-to-Pod Encryption

### What

Pod-to-pod encryption encrypts network traffic between Pods at the network or transport layer without requiring application-level changes.

### Why

NetworkPolicy controls which Pods may communicate but does not encrypt the data in transit. On a compromised or noisy node, plain traffic between Pods traversing the wire or host network stack could be observed. Encryption addresses confidentiality of data in motion.

### How

Two main approaches in the CKS context:

**Cilium transparent encryption (WireGuard or IPSec):**

- encrypts all node-to-node traffic carrying Pod traffic automatically
- no sidecar, no application change, no certificate rotation per workload
- configured at the Cilium level, not per Pod

Enable WireGuard encryption in Cilium (Helm values approach):

```yaml
encryption:
  enabled: true
  type: wireguard
```

Or via `cilium config`:

```bash
cilium config set encryption.enabled true
cilium config set encryption.type wireguard
```

**Service mesh mTLS (Istio, Linkerd):**

- encrypts traffic at the sidecar proxy layer
- also provides per-connection identity via certificates
- more operational complexity than transparent encryption
- Istio `PeerAuthentication` in `STRICT` mode enforces mTLS between all services in scope

### Key distinction for the exam

| Control | What it does |
|---|---|
| NetworkPolicy | controls which Pods may talk; does not encrypt |
| Cilium WireGuard | encrypts all node-to-node Pod traffic; does not restrict who may talk |
| Service mesh mTLS | encrypts and authenticates per-connection identity |

You almost always need both a NetworkPolicy (access control) and encryption (confidentiality). They address different threat models.

### Exam angle

- know that Cilium transparent encryption is the typical CKS answer for pod-to-pod encryption
- know the distinction between access control (NetworkPolicy) and confidentiality (encryption)
- know that WireGuard mode requires Cilium and Linux kernel 5.6+

---

## 6. Supply Chain Security

This domain covers the path from source code and dependencies to the image that runs in the cluster.

### 6.1 Threat Model

### What

Supply chain security protects against tampered images, vulnerable dependencies, malicious registries, unsafe build pipelines, and unverifiable artifacts.

### Why

If you deploy a malicious or compromised image, Kubernetes will faithfully run it. Orchestration does not make bad artifacts safe.

### How

You reduce risk through:

- trusted registries
- image scanning
- signed artifacts
- immutable references
- SBOMs
- admission policies for provenance

---

### 6.2 Image Scanning with Trivy or Similar Tools

### What

Image scanners detect known vulnerabilities, risky packages, secrets, and misconfigurations in container images or manifests.

### Why

You want to stop known-bad artifacts before deployment.

### How

Typical workflow:

1. scan image in CI
2. fail build above severity threshold
3. optionally scan manifests and IaC too
4. re-scan base images regularly because CVE knowledge changes over time

Example:

```bash
# Scan an image for vulnerabilities
trivy image nginx:1.27

# Scan Kubernetes manifests for misconfigurations
trivy config .
```

### Static analysis of manifests with kubesec

Beyond image CVE scanning, `kubesec` scores Kubernetes manifests for security risk and highlights dangerous fields like `privileged: true`, missing `securityContext` settings, or overly permissive capabilities.

```bash
# Score a manifest locally
kubesec scan deployment.yaml

# Or use the hosted API
curl -sSX POST --data-binary @deployment.yaml https://v2.kubesec.io/scan
```

Each result gives a numeric score and a list of `advise` and `critical` findings. A negative score indicates a manifest with high-risk settings. A score above zero and no critical findings is the target.

### Static analysis of Dockerfiles with hadolint

`hadolint` lints Dockerfiles against best practices and security guidelines.

```bash
hadolint Dockerfile
```

Common findings include: using `latest` tags, running as root, using `ADD` instead of `COPY`, and combining `RUN` commands in ways that leave sensitive data in image layers.

### Important nuance

Scanning is necessary but not sufficient.

- a clean scan today may become a bad scan tomorrow
- scanners find known issues, not all malicious logic
- severity alone is not risk; exploitability and exposure matter too

---

### 6.3 Image Provenance, Signing, and Verification

### What

Image signing proves that an artifact was produced and approved by a trusted identity or process.

### Why

Without provenance, anyone who can push an image with the expected name may be able to influence your deployments.

### How

Common pattern with Cosign/Sigstore:

1. build image
2. sign image
3. store signature and transparency metadata
4. verify signature before admission or deployment

Conceptual example:

```bash
# Sign an image (keyless with OIDC, e.g. in CI)
cosign sign ghcr.io/example/api:1.0.0

# Verify its signature — requires trust configuration:
# For key-based signing:
cosign verify --key cosign.pub ghcr.io/example/api:1.0.0

# For keyless/Sigstore signing (supply OIDC issuer and expected identity):
cosign verify \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  --certificate-identity-regexp 'https://github.com/example/.*' \
  ghcr.io/example/api:1.0.0
```

> A bare `cosign verify` with no trust parameters will fail. You must always provide either a public key or the OIDC issuer and identity that the signature was made with.

### Why verification at admission is strong

CI checks can be bypassed if someone manually deploys. Admission-time verification enforces trust at the cluster boundary.

---

### 6.4 Tags vs Digests

### What

Tags are mutable labels. Digests are immutable content-addressed identifiers.

### Why

Security depends on exact artifacts. If `api:prod` points to a different image tomorrow, your deployment may change without the manifest changing.

### How

Prefer:

```text
image: registry.example.com/team/api@sha256:<digest>
```

Instead of:

```text
image: registry.example.com/team/api:latest
```

### Exam rule of thumb

If the task is to reduce image supply chain risk, pin by digest unless the scenario explicitly requires tags.

---

### 6.5 Minimal Base Images and Attack Surface Reduction

### What

Base image choice affects package count, shell availability, libc choice, debugging tools, and vulnerability footprint.

### Why

Fewer packages usually means fewer CVEs and fewer tools available to an attacker.

### How

Prefer minimal and purpose-built images where operationally appropriate:

- distroless
- alpine where compatible
- slim runtime images
- multi-stage builds to avoid shipping build tooling

### Example reasoning

If your image contains package managers, shells, compilers, curl, and SSH tools that the app does not need, you have increased both attack surface and post-exploitation utility.

---

### 6.6 SBOMs and Dependency Visibility

### What

An SBOM, or Software Bill of Materials, lists the components and dependencies included in an artifact.

### Why

When a new vulnerability is disclosed, you need to answer quickly whether you are affected.

### How

Use SBOM generation and retention in your pipeline. Even if the exam does not ask you to generate one, understand the goal:

- inventory components
- support vulnerability response
- improve provenance and compliance

---

### 6.7 Registry and Pull Controls

### What

Registry security includes who may push, who may pull, what registries are allowed, and how credentials are handled.

### Why

An open or weakly governed registry path makes image trust meaningless.

### How

Practical controls:

- use private registries when appropriate
- restrict write access tightly
- use imagePullSecrets only where needed
- enforce approved registries with admission policy
- avoid embedding registry credentials broadly across namespaces

---

## 7. Monitoring, Logging, and Runtime Security

This domain focuses on detection and response once prevention controls are in place.

### 7.1 Threat Model

### What

Runtime security is about recognizing suspicious behavior in live systems: process execution, privilege changes, network anomalies, container escapes, unexpected file access, or API abuse.

### Why

No defensive stack is perfect. You need evidence, visibility, and a response path when something goes wrong.

### How

Build visibility across:

- Kubernetes audit logs
- application logs
- kubelet and runtime logs
- events
- syscall or eBPF-based detectors
- cluster metrics and alerts

---

### 7.2 Audit Logs

### What

Audit logs record API requests handled by the API server.

### Why

They answer critical incident-response questions:

- who created this privileged Pod?
- who read these Secrets?
- when was this RoleBinding changed?

### How

You configure policy, storage path, and log rotation strategy on the API server.

### Example investigation questions

If an attacker created an unexpected ClusterRoleBinding, audit logs can reveal:

- username or service account
- source IP
- verb used
- request object details if logged at that level

This is why audit logging is both a compliance and a forensic control.

---

### 7.3 Falco and Runtime Threat Detection

### What

Falco is a runtime security engine that detects suspicious behavior using syscall or kernel-event visibility.

### Why

Kubernetes knows desired state. It does not inherently know whether a process inside a container is spawning a shell, reading sensitive host files, or making an abnormal system call sequence.

### How

Falco evaluates runtime events against rules such as:

- shell spawned in container
- write below `/etc`
- access to sensitive paths
- unexpected network tools launched
- container started in privileged mode

Conceptual examples of suspicious behavior:

- `bash` started inside an nginx container that should only run nginx
- a container touches `/var/run/docker.sock` or other host control sockets
- process execution pattern differs from the workload baseline

### What a Falco rule looks like

Falco rules are YAML. Each rule has a condition written in the Falco filter language and an output template. You should be able to read a rule and understand what behavior it detects.

```yaml
- rule: Shell Spawned in Container
  desc: Detect a shell being spawned inside a running container.
  condition: >
    spawned_process
    and container
    and proc.name in (shell_binaries)
  output: >
    Shell spawned in a container
    (user=%user.name container=%container.name image=%container.image.repository
    shell=%proc.name parent=%proc.pname cmdline=%proc.cmdline)
  priority: WARNING
  tags: [container, shell, mitre_execution]
```

Key fields:

- `condition`: Falco filter expression. `spawned_process` fires when a new process starts; `container` scopes it to container context; `proc.name in (shell_binaries)` matches `sh`, `bash`, `zsh`, etc.
- `output`: the log line emitted on a match. Fields like `%container.name` and `%proc.cmdline` give you context for investigation.
- `priority`: severity level (`WARNING`, `ERROR`, `CRITICAL`, etc.).
- `tags`: metadata used for MITRE ATT&CK mapping and filtering.

Common Falco filter fields you should recognise:

| Field | Meaning |
|---|---|
| `proc.name` | process name |
| `proc.cmdline` | full command line |
| `proc.pname` | parent process name |
| `container.name` | container name |
| `container.image.repository` | image name without tag |
| `fd.name` | file or network descriptor path |
| `user.name` | user running the process |

### Falco output destinations

Falco can write alerts to:

- stdout
- a file
- syslog
- a webhook (e.g. to Slack, Falcosidekick, or a SIEM)

The exam is unlikely to require configuring a full pipeline but you should understand that Falco produces structured alert output per rule match.

### Why it matters for CKS

The exam may ask you to interpret a Falco alert, identify which rule triggered it, or understand what the condition expression is detecting.

---

### 7.4 eBPF-Based Visibility

### What

eBPF allows programs to run safely in the kernel for networking, tracing, and security observability.

### Why

It provides low-level insight into packet flow, syscalls, and process activity with less intrusion than older approaches in many cases.

### How

Security tools use eBPF to observe:

- network flows
- DNS activity
- process execution
- syscall behavior
- file operations

You are unlikely to implement eBPF programs in the exam, but you should understand why tools using it are powerful in runtime security.

---

### 7.5 Logs, Events, and Runtime Triage

### What

When something breaks or looks suspicious, the first response path usually involves correlating logs, events, and object state.

### Why

Security incidents often initially look like operational failures:

- a Pod crash might actually be blocked by seccomp or PSA
- failed DNS might actually be blocked egress
- a denied API call might actually be RBAC

### How

Fast triage loop:

```bash
# Check workload status
kubectl get pods -A

# Describe the problematic workload
kubectl describe pod <pod> -n <ns>

# Review recent logs
kubectl logs <pod> -n <ns>

# Review previous container logs if it crashed
kubectl logs <pod> -n <ns> --previous

# Review recent events
kubectl get events -n <ns> --sort-by=.lastTimestamp
```

### Runtime interpretation examples

Symptom:

- `CreateContainerConfigError`

Possible causes:

- missing Secret
- invalid config reference
- denied volume type or security policy mismatch

Symptom:

- `CrashLoopBackOff`

Possible causes:

- app bug
- denied filesystem writes due to read-only root filesystem
- blocked syscall from seccomp

---

### 7.6 Incident Response in Kubernetes

### What

Incident response in Kubernetes means containing the threat without destroying the evidence you need.

### Why

A bad response can either let the attacker persist or erase the clues needed to understand what happened.

### How

Core steps:

1. identify the affected workload, namespace, and node
2. isolate if necessary using NetworkPolicy, scaling, or cordoning
3. preserve logs and audit evidence
4. inspect ServiceAccount, mounts, privileges, and recent changes
5. rotate exposed credentials
6. remove or redeploy from a trusted artifact

### Example reasoning

If a compromised Pod used an over-privileged ServiceAccount:

- do not only delete the Pod
- identify what the token could access
- inspect audit logs for its actions
- revoke or change the RBAC binding
- rotate dependent credentials if Secrets were readable

---

## 8. High-Value Exam Topics and How to Think About Them

This section compresses common CKS tasks into fast decision rules.

### 8.1 If You See a Privileged Pod

Ask:

- does it really need privilege?
- can capability-based access replace it?
- does it also use hostPath or hostPID?
- will PSA restricted deny it?

Typical safer replacements:

- `privileged: false`
- `allowPrivilegeEscalation: false`
- drop `ALL` capabilities
- add only the one capability required

---

### 8.2 If You See the Default ServiceAccount

Ask:

- does the workload need Kubernetes API access at all?
- can token automount be disabled?
- should this workload have a dedicated ServiceAccount?

Default answer:

- create a dedicated ServiceAccount only if needed
- avoid using `default`
- disable automatic token mount unless API access is required

---

### 8.3 If Traffic Must Be Restricted

Ask:

- ingress, egress, or both?
- namespace scope or specific Pods?
- what about DNS?
- does the CNI enforce NetworkPolicy?

Default safe pattern:

1. apply default deny
2. add least-privilege allow rules
3. test allowed and denied flows

---

### 8.4 If an Image Must Be More Trustworthy

Ask:

- is it scanned?
- is it signed?
- is it pinned by digest?
- is the registry approved?

Default improvement path:

1. scan
2. pin by digest
3. sign and verify
4. enforce with admission policy

---

### 8.5 If a Workload Must Become More Restricted

Common secure defaults:

```yaml
securityContext:
  runAsNonRoot: true
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  seccompProfile:
    type: RuntimeDefault
  capabilities:
    drop: ["ALL"]
```

Then add only what is operationally necessary.

---

## 9. Practical Lab Checklist

Use these labs to turn the concepts into working reflexes.

### Lab 1: Lock Down a Deployment

Goal:

- convert an insecure Deployment into a hardened one

Practice:

1. create a Deployment that runs as root with default settings
2. add a dedicated ServiceAccount
3. disable automount token
4. enforce non-root, read-only root FS, no privilege escalation, dropped capabilities
5. validate the Pod still starts
6. mount `emptyDir` only where writes are required

---

### Lab 2: Build Namespace Isolation

Goal:

- implement default deny plus selective access

Practice:

1. create frontend, backend, and db Pods
2. apply a namespace-wide default deny
3. allow frontend to backend only on app port
4. allow backend to db only on DB port
5. allow DNS egress
6. test both allowed and denied paths

---

### Lab 3: Validate RBAC Least Privilege

Goal:

- replace broad privileges with minimal access

Practice:

1. create a ServiceAccount
2. bind it first to a too-broad role
3. use `kubectl auth can-i` to prove overreach
4. replace with a namespace-scoped Role
5. retest until only required verbs remain

---

### Lab 4: Enable and Use Pod Security Admission

Goal:

- understand what restricted mode blocks

Practice:

1. label a namespace with `enforce=restricted`
2. deploy an intentionally insecure Pod
3. read the admission failure
4. fix the Pod incrementally until admitted

---

### Lab 5: Scan and Pin Images

Goal:

- improve supply chain trust

Practice:

1. scan an image with Trivy
2. replace a mutable tag with a digest
3. compare the deployment manifest before and after
4. document why digest pinning improves traceability

---

### Lab 6: Runtime Investigation

Goal:

- practice distinguishing security from regular operational failure

Practice:

1. deploy a Pod blocked by PSA or seccomp
2. use `describe`, `logs`, and `events`
3. identify the exact denial reason
4. apply the smallest safe fix

---

## 10. Fast Command Reference

```bash
# RBAC checks
kubectl auth can-i --list --as=system:serviceaccount:default:default -n default
kubectl auth can-i get secrets --as=system:serviceaccount:prod:api-sa -n prod

# Security-relevant object inspection
kubectl get pod <pod> -n <ns> -o yaml
kubectl describe pod <pod> -n <ns>
kubectl get sa,role,rolebinding -n <ns>
kubectl get clusterrole,clusterrolebinding
kubectl get networkpolicy -n <ns>

# Troubleshooting
kubectl logs <pod> -n <ns>
kubectl logs <pod> -n <ns> --previous
kubectl get events -n <ns> --sort-by=.lastTimestamp

# Generate and edit manifests quickly
kubectl run tmp --image=busybox:1.36 --dry-run=client -o yaml
kubectl create role pod-reader --verb=get,list,watch --resource=pods -n prod --dry-run=client -o yaml

# Node-side config inspection
grep -n "anonymous\|authorization\|readOnlyPort" /var/lib/kubelet/config.yaml
grep -n "audit\|authorization-mode\|anonymous-auth\|encryption-provider-config" /etc/kubernetes/manifests/kube-apiserver.yaml
```

---

## 11. Common Mistakes That Cost Points

1. Fixing only the symptom, not the security issue.
2. Forgetting to verify RBAC with `kubectl auth can-i`.
3. Applying egress deny and forgetting DNS.
4. Using Pod labels incorrectly in NetworkPolicy selectors.
5. Leaving workloads on the `default` ServiceAccount.
6. Using `privileged: true` when a single added capability would suffice.
7. Enabling a control without understanding its operational impact.
8. Breaking a workload with `readOnlyRootFilesystem` and not adding a narrow writable mount.
9. Using mutable tags when the question is about supply chain integrity.
10. Forgetting that admission and runtime failures show up in events.

---

## 12. Exam Execution Strategy

### Before the exam

- be comfortable with `vim` or another fast editor available in the terminal
- practice jumping between clusters and namespaces quickly
- practice reading YAML faster than writing it from scratch

### During the exam

1. Read the task once for the security objective.
2. Read it again for the exact requested state.
3. Make the smallest correct change.
4. Verify immediately.
5. Move on if you are stuck; return later.

### Mental model during tasks

For each change, ask:

- what risk is being reduced?
- what Kubernetes or Linux control implements that reduction?
- how do I prove the new state is correct?

That is the difference between guessing and operating.

---

## 13. Final Review Sheet

If you only have a short time left before the exam, make sure you can explain and implement all of these from memory:

1. RBAC least privilege with `Role`, `RoleBinding`, and `kubectl auth can-i`
2. Pod Security Admission namespace labels and what `restricted` blocks
3. NetworkPolicy default deny plus targeted allow rules
4. secure container `securityContext` defaults
5. kubelet hardening settings and why insecure endpoints are dangerous
6. API server audit logging and encryption at rest for Secrets
7. why privileged Pods, hostPath mounts, and host namespaces are risky
8. seccomp, AppArmor, SELinux, and Linux capabilities at a conceptual level
9. image scanning, signing, digest pinning, and registry restrictions
10. how to investigate runtime failures with logs, events, and audit evidence

---

## 14. One-Sentence Summary of Each Domain

- Cluster hardening: secure the control plane and the default trust boundaries.
- System hardening: reduce node and kernel attack surface.
- Microservice security: make insecure workload behavior difficult by default.
- Supply chain security: trust the artifact before it runs.
- Runtime security: detect and investigate what prevention did not stop.

Master those five ideas operationally, and the exam becomes much more manageable.
