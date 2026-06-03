# Kubernetes RBAC Labs

**Setup:** Create three namespaces to simulate different environments:

```bash
kubectl create namespace dev
kubectl create namespace qa
kubectl create namespace prod
```

---

## Phase 1: Basic Namespace Roles & Bindings (Labs 1–5)

### Lab 1: The Scoped Reader

**Goal:** Grant read-only access to resources within a single namespace.

**Task:** Create a `Role` named `pod-reader` in the `dev` namespace that allows a user to `get`, `list`, and `watch` pods. Bind this role to a user named `alex` using a `RoleBinding` named `read-pods-alex`.

**Validation:**
```bash
kubectl auth can-i list pods --as alex -n dev    # Should be YES
kubectl auth can-i delete pods --as alex -n dev  # Should be NO
```

---

### Lab 2: Cross-Resource Developer Access

**Goal:** Grant write permissions to workload controllers but not the underlying pods directly.

**Task:** Create a `Role` in the `dev` namespace named `deployer`. Allow it to perform all actions (`*`) on `deployments` and `statefulsets`. Bind it to user `sam`.

**Validation:** Verify that `sam` can create a deployment but cannot delete a raw pod in the `dev` namespace.

---

### Lab 3: The Subresource Pitfall (Pod Logs & Exec)

**Goal:** Understand how to grant access to specific subresources like container logs or interactive shells.

**Task:** Create a `Role` in the `qa` namespace allowing user `tester` to read pod logs and execute commands inside pods.

> **Gotcha:** `log` and `exec` are subresources. Your YAML `resources` field must explicitly state `pods/log` and `pods/exec`.

---

### Lab 4: Multi-User Group Binding

**Goal:** Scale access management by targeting logical groups instead of single individuals.

**Task:** Create a `Role` named `config-manager` in the `prod` namespace that allows managing `configmaps` and `secrets`. Bind this role to an entire group named `engineering-leads`.

**Validation:**
```bash
kubectl auth can-i get secrets --as-group engineering-leads --as anyuser -n prod
```

---

### Lab 5: Restricting Access to Specific Named Resources

**Goal:** Implement absolute least privilege by locking access down to a singular, specific object name.

**Task:** Create a `Role` in the `prod` namespace that allows updating **only** the ConfigMap named `app-config`. No other ConfigMaps should be editable by user `taylor`.

> **YAML Key:** Use `resourceNames: ["app-config"]` in the Role rule block.

---

## Phase 2: Cluster-Wide Permissions & Scope (Labs 6–10)

### Lab 6: The Global Infrastructure Auditor

**Goal:** Grant view-only rights to non-namespaced, global resources.

**Task:** Create a `ClusterRole` named `node-inspector` that allows users to `get` and `list` nodes. Bind it to user `chris` using a `ClusterRoleBinding`.

**Validation:**
```bash
kubectl auth can-i list nodes --as chris   # must return yes
```

---

### Lab 7: Namespaced Escalation via ClusterRole

**Goal:** Reuse a global permission definition inside a limited namespace boundary.

**Task:** Take the built-in `ClusterRole` `view`. Bind it to user `morgan` using a standard `RoleBinding` targeting only the `dev` namespace.

**Validation:** Confirm `morgan` can view pods in `dev`, but is blocked from viewing pods in `prod`.

---

### Lab 8: Custom Persistent Volume Administrator

**Goal:** Create a dedicated administrative role for global storage resources.

**Task:** Create a `ClusterRole` named `pv-manager`. Give it full access to `persistentvolumes` and `persistentvolumeclaims`. Bind it globally to user `jordan`.

---

### Lab 9: Cluster-Wide CRD Management

**Goal:** Grant access to custom resource schemas added to the Kubernetes API.

**Task:** Assume a CRD named `certificates.cert-manager.io` exists. Create a `ClusterRole` allowing user `sec-ops` to `delete` these custom resources globally.

---

### Lab 10: Non-Resource URL Auditing

**Goal:** Grant access to raw, non-resource system health or metrics endpoints.

**Task:** Create a `ClusterRole` that permits user `monitoring-bot` to perform HTTP `GET` requests to the `/healthz` and `/metrics` paths of the API server.

> **YAML Key:** Use `nonResourceURLs: ["/healthz", "/metrics"]` instead of `resources`.

---

## Phase 3: Service Account Security & Hardening (Labs 11–14)

### Lab 11: Application Microservice Identity Isolation

**Goal:** Create a discrete identity for an automation controller inside a namespace.

**Task:** Create a `ServiceAccount` named `db-backup-sa` in the `prod` namespace. Create a local `Role` allowing it to read secrets. Bind the role directly to this ServiceAccount.

---

### Lab 12: Cross-Namespace Service Account Access

**Goal:** Allow an automated tool running in one namespace to manage resources in another.

**Task:** A CI/CD tool runs as `ServiceAccount` `jenkins-runner` in the `dev` namespace. Create a `RoleBinding` in the `prod` namespace that allows this ServiceAccount to modify deployments in `prod`.

---

### Lab 13: Auditing Dangerous Default Over-Privilege

**Goal:** Discover and fix wide-open system default permissions.

**CKS Scenario:** A legacy configuration bound the `cluster-admin` role to the `default` ServiceAccount of the `default` namespace.

**Task:** Write a query or scan command to find this binding and delete it immediately to pass your security compliance check.

---

### Lab 14: Restricting Pod Token Auto-Mounts

**Goal:** Harden workload configuration against credential theft.

**Task:** Write a Pod specification for a public-facing NGINX application. Ensure the default ServiceAccount token is **not** mounted at `/var/run/secrets/kubernetes.io/serviceaccount`, even if the container is compromised.

---

## Phase 4: Advanced Concepts & Aggregation (Labs 15–17)

### Lab 15: Dynamically Aggregated ClusterRoles

**Goal:** Combine multiple distinct access definitions using label selectors.

**Task:** Create an empty master `ClusterRole` named `custom-admin` with an `aggregationRule` matching the label `rbac.example.com: "true"`. Create two independent sub-ClusterRoles with that label (one for `deployments`, one for `services`) and verify that `custom-admin` automatically inherits both rule sets.

---

### Lab 16: Preventing Privilege Escalation (The "Bind" Verb Rule)

**Goal:** Understand why a user cannot grant permissions they do not already possess.

**Task:** Create a user `sub-admin` who has full rights to create `RoleBindings`, but no rights to read `Secrets`. Attempt to make `sub-admin` bind the `secret-reader` role to another user. Observe the API server blocking this unless the `bind` verb is explicitly granted.

---

### Lab 17: Eliminating Wildcard Exploits (`*`)

**Goal:** Replace insecure wildcard privileges with explicitly named targets.

**Task:** You find a role containing:
```yaml
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]
```
Refactor it so it explicitly targets only the core API group (`""`), limits resources to `pods` and `services`, and restricts verbs to `get` and `update`.

---

## Phase 5: Real-World Troubleshooting & Auditing (Labs 18–20)

### Lab 18: Fixing the "Forbidden" Webhook Connection

**Goal:** Debug a broken deployment failing due to missing API authorization.

**Task:** An application log shows:
```
User "system:serviceaccount:qa:app-sa" cannot list services in namespace "qa"
```
Fix this live error using the **minimum required** RBAC resources.

---

### Lab 19: Comprehensive RBAC Cluster Inventory Scan

**Goal:** Audit the environment for users holding dangerous global administrative power.

**Task:** Write a native `kubectl` command using custom columns or JSONPath to list every `User` or `Group` currently bound to the `cluster-admin` `ClusterRole`.

---

### Lab 20: The Grand CKS Multi-Tier Challenge

**Goal:** Synthesize multiple isolation techniques under exam-style constraints.

**Task:** Inside the `prod` namespace:

1. Create a `ServiceAccount` named `operator-sa`.
2. Ensure it **cannot** auto-mount API tokens.
3. Grant it access to `update` `deployments` (apps group), but block access to cluster-wide `nodes` or storage resources.
4. Ensure it can `get` a **single** Secret named `encryption-key` and nothing else.
