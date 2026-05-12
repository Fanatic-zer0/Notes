# Certified Kubernetes Security Specialist (CKS) Lab Workbook

## Overview

This workbook is the hands-on companion to the main study guide. It is built around realistic CKS-style scenarios where you must identify the security problem, choose the correct Kubernetes or Linux control, implement the fix, and verify the result.

Each lab is structured to train the exact workflow the exam rewards:

1. Read the scenario carefully.
2. Identify the security objective.
3. Make the smallest correct change.
4. Verify the new state.
5. Review the answer key and compare your reasoning.

Primary reference:

- [CKS-study-guide.md](CKS-study-guide.md)

---

## How To Use This Workbook

### Practice modes

Use the labs in three different modes.

### Mode 1: Untimed learning mode

Goal:

- understand the control deeply

How:

- do the task slowly
- use `kubectl explain`
- inspect objects before and after
- read the answer only after attempting the fix

### Mode 2: Timed drill mode

Goal:

- build exam speed

How:

- set a 10 to 15 minute timer per lab
- avoid reading the answer section until time expires
- verify quickly and move on

### Mode 3: Adversarial debug mode

Goal:

- learn to distinguish security misconfiguration from normal operational failure

How:

- intentionally break the scenario in multiple ways
- practice identifying the exact failing control from events, logs, and object state

---

## Lab Format

Each lab contains:

- scenario
- objective
- environment clues
- task list
- validation checklist
- hints
- answer key
- why the answer is correct
- common mistakes

---

## Lab 1. Remove Dangerous Workload Privileges

### Scenario

You inherit a namespace called `payments`. A debugging Pod was deployed there and left running. It uses a host mount, runs privileged, and shares the host PID namespace. Your task is to reduce its risk while preserving its basic purpose as a shell-based troubleshooting Pod.

### Objective

Harden the Pod so it no longer has obvious node-compromise paths unless they are strictly necessary.

### Environment clues

- namespace: `payments`
- Pod name: `debug-shell`
- image: `busybox:1.36`

### Tasks

1. Inspect the current Pod definition.
2. Remove unnecessary privilege.
3. Disable host namespace sharing unless the scenario explicitly requires it.
4. Remove any broad hostPath mount if not needed.
5. Prevent privilege escalation.
6. Drop all capabilities.
7. Run the container as non-root if the workload permits.

### Validation checklist

- Pod is running
- `privileged` is not enabled
- `allowPrivilegeEscalation` is false
- capabilities are dropped
- host PID sharing is not enabled
- host filesystem is not mounted

### Hints

- inspect the full YAML first
- do not assume the Pod needs host access just because it was used for debugging
- preserve only what is necessary for the shell to stay alive

### Answer key

Example hardened manifest:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: debug-shell
  namespace: payments
spec:
  containers:
  - name: shell
    image: busybox:1.36
    command: ["sh", "-c", "sleep 3600"]
    securityContext:
      runAsNonRoot: true
      runAsUser: 10001
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: false
      capabilities:
        drop: ["ALL"]
      seccompProfile:
        type: RuntimeDefault
```

Useful verification commands:

```bash
kubectl get pod debug-shell -n payments -o yaml
kubectl get pod debug-shell -n payments -o jsonpath='{.spec.hostPID}'
kubectl get pod debug-shell -n payments -o jsonpath='{.spec.containers[0].securityContext}'
```

### Why this answer is correct

The original risk is not that the Pod exists. The risk is that it has direct pathways into the host security boundary. Removing `privileged`, host PID sharing, and hostPath mounts eliminates the most dangerous escalation routes. Dropping capabilities and preventing privilege escalation make the shell far less dangerous if compromised.

### Common mistakes

- leaving `privileged: true` because it "might be needed"
- forgetting `allowPrivilegeEscalation: false`
- dropping capabilities but still keeping hostPath `/`

---

## Lab 2. Replace Default ServiceAccount Usage

### Scenario

An application in namespace `orders` is deployed with the default ServiceAccount. The app only serves HTTP traffic and does not need to call the Kubernetes API.

### Objective

Prevent unnecessary API credential exposure.

### Tasks

1. Inspect the Deployment and identify its current ServiceAccount behavior.
2. Disable service account token automount for the workload.
3. Optionally bind the Deployment to a dedicated ServiceAccount if the platform standard requires one.
4. Redeploy and verify the Pod no longer receives the token mount.

### Validation checklist

- the Pod is not relying on the `default` token mount
- `automountServiceAccountToken: false` is present at Pod spec level
- the application remains functional

### Answer key

Example Deployment fragment:

```yaml
spec:
  template:
    spec:
      serviceAccountName: orders-sa
      automountServiceAccountToken: false
      containers:
      - name: app
        image: nginx:1.27
```

Create a dedicated ServiceAccount if desired:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: orders-sa
  namespace: orders
```

Verification:

```bash
kubectl get pod <pod> -n orders -o jsonpath='{.spec.serviceAccountName}'
kubectl get pod <pod> -n orders -o jsonpath='{.spec.automountServiceAccountToken}'
```

### Why this answer is correct

The application does not need Kubernetes API access, so automatically mounting a token only increases blast radius. A compromised app should not get cluster credentials for free.

---

## Lab 3. Fix Overly Broad RBAC

### Scenario

A ServiceAccount named `reporter` in namespace `finance` is bound to a `ClusterRole` that permits listing Secrets cluster-wide. The app only needs read access to Pods in its own namespace.

### Objective

Reduce privileges to the minimum required.

### Tasks

1. Inspect the current RoleBinding or ClusterRoleBinding.
2. Prove the current permissions are too broad with `kubectl auth can-i`.
3. Replace the broad permission with a namespace-scoped `Role`.
4. Bind the Role to `reporter`.
5. Re-test effective permissions.

### Validation checklist

- `reporter` can get, list, and watch Pods in `finance`
- `reporter` cannot list Secrets in `finance` or cluster-wide
- no unnecessary `ClusterRoleBinding` remains

### Answer key

Example role and binding:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: reporter-pod-read
  namespace: finance
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: reporter-pod-read
  namespace: finance
subjects:
- kind: ServiceAccount
  name: reporter
  namespace: finance
roleRef:
  kind: Role
  name: reporter-pod-read
  apiGroup: rbac.authorization.k8s.io
```

Validation:

```bash
kubectl auth can-i list pods --as=system:serviceaccount:finance:reporter -n finance
kubectl auth can-i list secrets --as=system:serviceaccount:finance:reporter -n finance
kubectl auth can-i list secrets --as=system:serviceaccount:finance:reporter -A
```

### Why this answer is correct

The correct fix is not just removing one dangerous verb. It is narrowing both scope and resource type. A namespace-scoped `Role` matches the actual application need.

### Common mistakes

- replacing one broad `ClusterRole` with another `ClusterRole`
- forgetting to remove the old binding
- not verifying with impersonation

---

## Lab 4. Enforce Restricted Pod Security In a Namespace

### Scenario

The namespace `web-prod` currently allows insecure Pods. You need to enforce a restricted security baseline without changing cluster-wide admission behavior.

### Objective

Enable namespace-level Pod Security Admission and make a sample workload pass under the restricted profile.

### Tasks

1. Label `web-prod` to enforce restricted policy.
2. Deploy a sample Pod that initially violates the policy.
3. Observe the admission error.
4. Fix the manifest until it is compliant.

### Validation checklist

- namespace labels are present
- insecure Pod admission is denied
- corrected Pod is admitted and runs

### Answer key

Apply labels:

```bash
kubectl label ns web-prod \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted --overwrite
```

Example compliant Pod:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: web
  namespace: web-prod
spec:
  containers:
  - name: web
    image: nginx:1.27
    securityContext:
      runAsNonRoot: true
      runAsUser: 10001
      allowPrivilegeEscalation: false
      capabilities:
        drop: ["ALL"]
      seccompProfile:
        type: RuntimeDefault
```

Verification:

```bash
kubectl get ns web-prod --show-labels
kubectl get events -n web-prod --sort-by=.lastTimestamp
```

### Why this answer is correct

Pod Security Admission is namespace-scoped through labels. The secure path is to enforce the built-in restricted profile and then make workloads match that policy rather than weakening the policy to fit insecure workloads.

---

## Lab 5. Build Default-Deny Network Segmentation

### Scenario

Namespace `shop` contains `frontend`, `api`, and `db` workloads. Today everything can talk to everything. You need to reduce east-west movement.

### Objective

Implement least-privilege network connectivity.

### Tasks

1. Apply default deny for both ingress and egress.
2. Allow `frontend` to call `api` on TCP 8080.
3. Allow `api` to call `db` on TCP 5432.
4. Allow required DNS egress.
5. Confirm disallowed connections fail.

### Answer key

Default deny:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny
  namespace: shop
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

Allow frontend to api:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: frontend-to-api
  namespace: shop
spec:
  podSelector:
    matchLabels:
      app: api
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

Allow api to db:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-to-db
  namespace: shop
spec:
  podSelector:
    matchLabels:
      app: db
  policyTypes:
  - Ingress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: api
    ports:
    - protocol: TCP
      port: 5432
```

Allow DNS egress for all Pods:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-dns-egress
  namespace: shop
spec:
  podSelector: {}
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
```

### Why this answer is correct

A secure segmentation pattern starts with deny-by-default and then adds only the flows that map to actual application dependencies. DNS must be intentionally preserved or workloads may appear broken for the wrong reason.

### Common mistakes

- forgetting egress entirely
- writing policies against Services instead of Pod selectors
- forgetting DNS and misdiagnosing failures

---

## Lab 6. Secure a Workload With SecurityContext Defaults

### Scenario

A Deployment in `catalog` runs fine, but the security team requires a hardened baseline. The app writes temporary files to `/tmp` only.

### Objective

Apply a strong security context without breaking functionality.

### Tasks

1. Set `runAsNonRoot` and a non-zero user ID.
2. Set `allowPrivilegeEscalation: false`.
3. Drop all capabilities.
4. Enable `readOnlyRootFilesystem: true`.
5. Add `seccompProfile: RuntimeDefault`.
6. Provide a writable `emptyDir` at `/tmp`.

### Answer key

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: catalog
  namespace: catalog
spec:
  replicas: 2
  selector:
    matchLabels:
      app: catalog
  template:
    metadata:
      labels:
        app: catalog
    spec:
      containers:
      - name: catalog
        image: nginx:1.27
        securityContext:
          runAsNonRoot: true
          runAsUser: 10001
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
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

### Why this answer is correct

This is the standard CKS pattern for hardening a workload while preserving a narrow write path. The important thinking is not just "make it read-only" but "give it only the write path it truly needs."

---

## Lab 7. Enable Secrets Encryption At Rest

### Scenario

The cluster stores Secrets in etcd without a configured encryption provider. You need to enable encryption at rest for Secrets.

### Objective

Configure the API server to encrypt Secret resources in etcd.

### Tasks

1. Create an `EncryptionConfiguration` file on the control plane node.
2. Ensure an encryption provider such as `aescbc` is listed before `identity`.
3. Mount or reference the config from the API server static Pod manifest.
4. Restart the API server by updating the static Pod manifest if necessary.
5. Verify the flag is present.

### Answer key

Example configuration:

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

Relevant API server flag:

```text
--encryption-provider-config=/etc/kubernetes/encryption-config.yaml
```

Verification:

```bash
grep -n encryption-provider-config /etc/kubernetes/manifests/kube-apiserver.yaml
```

### Why this answer is correct

Without this configuration, base64-encoded Secrets are not meaningfully protected in etcd. Provider order matters because the first supported provider is used for writes.

### Common mistakes

- putting `identity` first
- editing the wrong control plane manifest
- assuming that creating the config file alone enables encryption

---

## Lab 8. Harden Kubelet Configuration

### Scenario

You are given node-level access. The kubelet configuration still exposes insecure defaults that allow unnecessary access paths.

### Objective

Disable insecure kubelet access patterns.

### Tasks

1. Inspect the kubelet config file.
2. Ensure anonymous auth is disabled.
3. Ensure webhook authn and authz are enabled.
4. Ensure `readOnlyPort` is set to `0`.
5. Restart kubelet if required.

### Answer key

Expected kubelet configuration shape:

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

Verification:

```bash
grep -n "anonymous\|webhook\|readOnlyPort\|authorization" /var/lib/kubelet/config.yaml
```

### Why this answer is correct

The kubelet is a high-value target on each node. Disabling anonymous access and delegating authorization to the API server removes legacy-style weak access paths.

---

## Lab 9. Investigate a Pod Blocked By Pod Security Admission

### Scenario

A Pod in namespace `restricted-apps` fails to start after the namespace was relabeled for restricted Pod Security Admission.

### Objective

Identify the violating fields and remediate the manifest.

### Tasks

1. Inspect Pod events.
2. Identify which restricted rules were violated.
3. Update the manifest.
4. Reapply and confirm admission succeeds.

### Answer key

Typical issues to remove or fix:

- privileged container
- root execution without non-root configuration
- missing seccomp profile
- added capabilities
- host namespace usage

Useful commands:

```bash
kubectl describe pod <pod> -n restricted-apps
kubectl get events -n restricted-apps --sort-by=.lastTimestamp
```

### Why this answer is correct

This lab trains a core CKS behavior: reading the denial closely rather than guessing. Admission messages usually point to the exact failing fields.

---

## Lab 10. Investigate a Pod Broken By readOnlyRootFilesystem

### Scenario

After hardening, an app in namespace `analytics` enters `CrashLoopBackOff`. Logs show a write failure under `/var/cache/app`.

### Objective

Preserve the hardening while restoring the app.

### Tasks

1. Confirm the failure path from logs.
2. Keep `readOnlyRootFilesystem: true`.
3. Add the narrowest writable mount needed.
4. Redeploy and verify.

### Answer key

Example fix:

```yaml
volumeMounts:
- name: app-cache
  mountPath: /var/cache/app
volumes:
- name: app-cache
  emptyDir: {}
```

### Why this answer is correct

The secure solution is not to remove the read-only root filesystem control. It is to add a minimal carve-out for the application's legitimate write path.

### Common mistakes

- deleting `readOnlyRootFilesystem: true`
- mounting a broad writable path instead of the exact directory

---

## Lab 11. Scan And Improve Image Trust

### Scenario

A Deployment uses `image: registry.example.com/payments/api:latest`. The security requirement is to improve trust and traceability of the running artifact.

### Objective

Reduce supply chain ambiguity.

### Tasks

1. Scan the image with a scanner such as Trivy.
2. Replace the mutable tag with a digest.
3. Document or verify the image provenance if signing exists.

### Answer key

Example commands:

```bash
trivy image registry.example.com/payments/api:latest
```

Manifest improvement:

```text
image: registry.example.com/payments/api@sha256:1111111111111111111111111111111111111111111111111111111111111111
```

Optional signature verification — always requires trust configuration:

```bash
# Key-based
cosign verify --key cosign.pub registry.example.com/payments/api:1.2.3

# Keyless / Sigstore (OIDC-based)
cosign verify \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  --certificate-identity-regexp 'https://github.com/example/.*' \
  registry.example.com/payments/api:1.2.3
```

### Why this answer is correct

Digest pinning gives you a stable identity for the exact artifact. Scanning and signature verification improve trust in both content and provenance. A bare `cosign verify` without trust parameters always fails — you must supply a key or OIDC issuer and identity.

---

## Lab 12. Use Audit Clues To Investigate Suspicious RBAC Change

### Scenario

A suspicious `ClusterRoleBinding` granting `cluster-admin` appears in the cluster. You need to determine what to inspect first and how to contain the issue.

### Objective

Practice security-focused investigation logic.

### Tasks

1. Inspect the binding object and identify its subject.
2. Determine whether the subject is a user, group, or ServiceAccount.
3. Review audit logs to find who created the binding.
4. Remove or replace the binding if it is unauthorized.
5. Assess whether any Secrets or privileged actions were possible through the subject.

### Answer key

Useful commands:

```bash
kubectl get clusterrolebinding <name> -o yaml
kubectl describe clusterrolebinding <name>
```

Audit log investigation should answer:

- who created the object
- when it was created
- from where the request originated
- what subject gained the access

Containment action:

- remove the unauthorized binding
- review what that identity could do with `kubectl auth can-i`
- rotate credentials or Secrets if exposure is plausible

### Why this answer is correct

Deleting the binding is only the containment step. The real security response includes identifying the creating actor and estimating blast radius.

---

## Lab 13. Runtime Suspicion: Unexpected Shell In Container

### Scenario

Your runtime detection system reports that a shell was spawned inside a container running a simple web server in namespace `marketing`.

### Objective

Practice runtime response logic.

### Tasks

1. Identify the Pod and node.
2. Inspect recent logs and events.
3. Determine which ServiceAccount and mounts the Pod has.
4. Check whether the Pod is privileged or has host access.
5. Decide how to isolate the workload with minimal evidence loss.

### Answer key

Useful inspection commands:

```bash
kubectl get pod <pod> -n marketing -o wide
kubectl describe pod <pod> -n marketing
kubectl get pod <pod> -n marketing -o yaml
kubectl logs <pod> -n marketing --previous
```

Security questions to answer:

- did the Pod have a mounted token?
- did it mount Secrets?
- did it have hostPath access?
- was it allowed outbound network access?

Containment options depend on the scenario:

- apply restrictive NetworkPolicy
- scale down the Deployment after evidence capture
- cordon the node only if node compromise is suspected

### Why this answer is correct

Runtime detection signals are starting points, not conclusions. The right response balances containment with preservation of evidence and blast-radius analysis.

---

## Lab 14. Combine Controls In A Full Hardening Exercise

### Scenario

You are given a namespace `legacy-apps` with a Deployment that has all of the following:

- default ServiceAccount
- mutable image tag
- writable root filesystem
- root execution
- added Linux capability
- no NetworkPolicy

### Objective

Apply layered hardening, not a single isolated fix.

### Tasks

1. Create or assign a dedicated ServiceAccount and disable token automount if not needed.
2. Replace the image tag with a digest.
3. Apply a secure container `securityContext`.
4. Add a narrow writable `emptyDir` if required.
5. Add default deny NetworkPolicy and minimal allow rules.
6. Verify the app still works.

### Answer key

This lab intentionally combines core CKS themes. The correct answer is a set of coordinated changes:

- least-privilege identity
- immutable image reference
- non-root, no privilege escalation, dropped capabilities, seccomp default
- least-privilege networking
- minimal writable filesystem

### Why this answer is correct

Real systems are rarely insecure in only one dimension. This lab trains you to think in layers and to prioritize changes that reduce both exploitability and blast radius.

---

## Lab 15. Block Cloud Metadata Endpoint Access

### Scenario

A cluster runs on a cloud provider. Pods in namespace `data-pipeline` currently have unrestricted egress. A security review flags that a compromised Pod could call `169.254.169.254` and retrieve node-level IAM credentials.

### Objective

Prevent Pods from reaching the instance metadata endpoint while preserving legitimate egress.

### Tasks

1. Write a NetworkPolicy for namespace `data-pipeline` that allows required egress (DNS, internet, internal services) but explicitly excludes `169.254.169.254/32`.
2. Use `ipBlock.except` to carve out the metadata IP.
3. Verify with a temporary Pod that the metadata endpoint is unreachable.

### Validation checklist

- egress to legitimate destinations succeeds
- `curl -s http://169.254.169.254/` from within a Pod times out or is refused
- the policy uses `ipBlock.except` correctly

### Answer key

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: block-cloud-metadata
  namespace: data-pipeline
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

Test block from a temporary Pod:

```bash
kubectl run testbox -n data-pipeline --rm -it --image=curlimages/curl:8.8.0 \
  -- curl -s --max-time 3 http://169.254.169.254/
# Expected: connection timeout or refused
```

### Why this answer is correct

`ipBlock.except` carves out a specific CIDR from an otherwise-allowed block. The metadata IP `169.254.169.254` is a link-local address and not reachable over normal internet routing, so this pattern only affects the specific cloud IMDS path.

### Common mistakes

- writing a policy that allows `0.0.0.0/0` without the `except` block
- forgetting to add DNS egress alongside the policy if default deny is also in place
- confusing `ipBlock.except` with a separate deny rule (NetworkPolicy has no explicit deny; only allow rules exist)

---

## Lab 16. Verify Kubernetes Binary Integrity

### Scenario

You are about to install `kubectl` v1.30.2 on a node. Before placing the binary on the system path, you must verify its integrity using the official checksum.

### Objective

Confirm the binary matches the published SHA-256 checksum.

### Tasks

1. Download the `kubectl` binary for the target version.
2. Download the corresponding `.sha256` checksum file from the official Kubernetes release URL.
3. Run `sha256sum --check` and confirm the output says `OK`.
4. Only proceed if verification passes.

### Validation checklist

- `sha256sum --check` output is `kubectl: OK`
- no error or mismatch is printed

### Answer key

```bash
# Download binary and checksum for a specific version
curl -LO https://dl.k8s.io/release/v1.30.2/bin/linux/amd64/kubectl
curl -LO https://dl.k8s.io/release/v1.30.2/bin/linux/amd64/kubectl.sha256

# Verify: output must be "kubectl: OK"
echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check

# Only then install
chmod +x kubectl
sudo mv kubectl /usr/local/bin/kubectl
```

For a binary already installed on a node, compute and compare manually:

```bash
# Get the hash of the installed binary
sha256sum $(which kubectl)

# Compare against the published checksum for that exact version
curl -sL https://dl.k8s.io/release/$(kubectl version --client -o json | jq -r .clientVersion.gitVersion)/bin/linux/amd64/kubectl.sha256
```

### Why this answer is correct

A package manager or CDN can be compromised. The SHA-256 checksum published by the Kubernetes project is the authoritative fingerprint of the expected binary content. If the hashes differ, the binary must not be used.

### Common mistakes

- assuming HTTPS download is sufficient without checksum verification
- verifying the wrong version's checksum against the downloaded binary
- not making the check part of automation or install scripts

---

## Lab 17. Sandbox a Workload With RuntimeClass

### Scenario

A high-sensitivity financial workload runs in namespace `finops`. The security team requires stronger kernel isolation than standard containers provide. gVisor is available on the nodes as the `runsc` handler.

### Objective

Assign the workload to a gVisor RuntimeClass.

### Tasks

1. Create a `RuntimeClass` named `gvisor` referencing the `runsc` handler.
2. Update the Deployment in `finops` to set `runtimeClassName: gvisor`.
3. Verify the Pod is using the correct runtime.

### Validation checklist

- `kubectl get runtimeclass gvisor` exists
- `kubectl get pod <pod> -n finops -o jsonpath='{.spec.runtimeClassName}'` returns `gvisor`
- Pod is Running

### Answer key

Create the RuntimeClass:

```yaml
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: gvisor
handler: runsc
```

Update the Deployment spec:

```yaml
spec:
  template:
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
kubectl get runtimeclass gvisor
kubectl get pod <pod> -n finops -o jsonpath='{.spec.runtimeClassName}'
kubectl get pod <pod> -n finops
```

### Why this answer is correct

`RuntimeClass` is the Kubernetes API for selecting a container runtime handler. Setting `runtimeClassName` in the Pod spec routes the workload to that handler. The `RuntimeClass` object must exist before the Pod references it or the Pod will fail to schedule.

### Common mistakes

- setting `handler` to `gvisor` instead of `runsc` (the handler name maps to the runtime binary, not the project name)
- creating the RuntimeClass in a namespace (it is cluster-scoped)
- expecting a RuntimeClass to replace securityContext — it is a kernel isolation control, not an identity or access control

---

## Lab 18. Read, Interpret, and Modify a Falco Rule

### Scenario

Your cluster runs Falco. A new rule fires constantly because it is too broad: it alerts on any new process in any container, including expected init processes. You need to understand what the rule does, interpret a sample alert, and narrow the condition to reduce noise without removing the detection.

### Objective

Read and modify a Falco rule at the condition level.

### Tasks

1. Read the following rule and explain what it detects.
2. Interpret the sample alert output.
3. Narrow the condition so it only triggers for shell binaries, not all new processes.

### Starting rule

```yaml
- rule: Any New Process In Container
  desc: Detects any new process spawned in a container.
  condition: spawned_process and container
  output: >
    New process in container
    (user=%user.name container=%container.name image=%container.image.repository
    proc=%proc.name cmdline=%proc.cmdline)
  priority: WARNING
```

### Sample alert

```text
New process in container (user=nginx container=web image=nginx proc=nginx cmdline=nginx: worker process)
```

### Tasks (continued)

4. The alert above is a false positive. Why?
5. Rewrite the condition to only alert on shell binaries.

### Answer key

What the original rule detects:

- `spawned_process`: fires whenever any new process is created
- `container`: scopes it to container context only
- combined: every single new process in every container triggers this rule

Why the alert is a false positive:

- `nginx: worker process` is expected and legitimate inside an nginx container
- the rule is too broad — it treats all process spawns as suspicious

Narrowed condition:

```yaml
- rule: Shell Spawned In Container
  desc: Detect a shell spawned inside a running container.
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

Key change: `and proc.name in (shell_binaries)` limits the match to shell executables only. `shell_binaries` is a built-in Falco macro that expands to a list including `sh`, `bash`, `zsh`, `ksh`, `dash`, etc.

Useful Falco filter fields:

| Field | Meaning |
|---|---|
| `proc.name` | process name |
| `proc.cmdline` | full command line |
| `proc.pname` | parent process name |
| `container.name` | container name |
| `container.image.repository` | image name |
| `user.name` | user running the process |

### Why this answer is correct

A rule that fires on everything is worse than no rule — it creates alert fatigue. The correct approach is the narrowest condition that still catches the suspicious behaviour you care about (shells in containers that shouldn't have them).

### Common mistakes

- removing the rule entirely instead of narrowing it
- confusing `proc.name` (binary name) with `proc.cmdline` (full invocation)
- not understanding that `shell_binaries` is a macro, not a literal field value

---

## Lab 19. Static Security Analysis of a Manifest With kubesec

### Scenario

A new Deployment manifest has been written by a developer. Before it goes to production, you must score it with `kubesec` and remediate any critical findings.

### Objective

Use static manifest analysis to identify and fix security issues before deployment.

### Starting manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  namespace: prod
spec:
  replicas: 1
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
      - name: api
        image: nginx:latest
        securityContext:
          privileged: true
```

### Tasks

1. Run `kubesec scan` on the manifest.
2. Identify all critical and advisory findings.
3. Fix the manifest until there are no critical findings and the score is positive.

### Answer key

Run analysis:

```bash
kubesec scan deployment.yaml

# Or using the hosted API without installing kubesec:
curl -sSX POST --data-binary @deployment.yaml https://v2.kubesec.io/scan | jq .
```

Expected critical findings on the starting manifest:

- `privileged: true` — critical, direct node compromise path
- missing `runAsNonRoot` — advisory
- missing `readOnlyRootFilesystem` — advisory
- missing `capabilities.drop` — advisory
- `image: nginx:latest` — mutable tag

Fixed manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  namespace: prod
spec:
  replicas: 1
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
      - name: api
        image: nginx:1.27
        securityContext:
          runAsNonRoot: true
          runAsUser: 10001
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          seccompProfile:
            type: RuntimeDefault
          capabilities:
            drop: ["ALL"]
```

### Why this answer is correct

kubesec catches structural problems before they reach a cluster. The critical findings map directly to the most dangerous securityContext settings. Clearing them raises the score and eliminates the highest-risk configurations.

### Common mistakes

- treating `kubesec` score as a pass/fail binary rather than a risk signal
- fixing the score without understanding why each field matters
- forgetting that a passing kubesec score does not replace runtime controls

---

## Timed Mock Drill Set

Use these as short practice prompts without reading the answer keys above.

1. A Pod in `prod` is privileged and mounts `/var/run/containerd/containerd.sock`. Reduce node compromise risk.
2. A namespace labeled `restricted` denies a Pod. Use events to identify the violating field and fix the manifest.
3. A ServiceAccount can list Secrets but should only watch Pods. Correct the RBAC and prove the result.
4. A hardened container fails because it needs `/tmp` writes. Keep the hardening and restore functionality.
5. An app loses all network access after NetworkPolicy changes. Determine whether the root cause is missing DNS egress (UDP and TCP 53).
6. A Deployment uses `:latest` from an unapproved registry. Improve artifact trust.
7. A Pod in a cloud namespace can reach `169.254.169.254`. Write a NetworkPolicy that blocks it while preserving internet egress.
8. You are about to deploy a new `kubelet` binary. Describe and perform the integrity check before installing it.
9. A high-risk workload must run with stronger kernel isolation. Create the `RuntimeClass` and assign it to the Pod.
10. A Falco rule fires on every nginx worker startup. Narrow the condition to shell processes only without removing the rule.

---

## Self-Assessment Rubric

After each lab, score yourself on these five questions.

1. Did I identify the actual security objective before editing anything?
2. Did I choose the narrowest correct control?
3. Did I avoid weakening one control to fix another issue?
4. Did I verify the final state directly?
5. Could I repeat the fix under time pressure without notes?

If the answer to question 5 is no, repeat the lab.

---

## Final Advice

The goal of this workbook is not just to help you pass one exam. It is to train a habit: every security change should be tied to a threat model, implemented with the least privilege possible, and verified with evidence.

That habit is exactly what CKS rewards.
