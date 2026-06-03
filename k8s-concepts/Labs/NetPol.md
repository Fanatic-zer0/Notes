# Kubernetes NetworkPolicy Labs

---

## Phase 1: Isolated Default Behaviors (Labs 1–3)

### Lab 1: The Global Black Hole (Default Deny All)

**Goal:** Completely isolate a namespace so no traffic can flow in or out.

**Task:** Create a namespace named `secure-zone`. Apply a single NetworkPolicy named `deny-all` that blocks all ingress and egress traffic for every pod in that namespace.

**Validation:**
- Deploy an `nginx` pod and a `busybox` pod in `secure-zone`.
- Ensure the `busybox` pod cannot ping the `nginx` pod.
- Ensure the `busybox` pod cannot ping `google.com`.

---

### Lab 2: Ingress Isolation Only

**Goal:** Block all incoming traffic but allow pods to reach external networks.

**Task:** In the `default` namespace, create a policy targeting pods labeled `app=isolated-receiver`. Block all incoming traffic but leave outgoing traffic unrestricted.

**Validation:**
- Run an outbound `curl` from a pod matching the label — it should succeed.
- Run a `curl` from a different pod to the matching pod — it must time out.

---

### Lab 3: Egress Isolation Only

**Goal:** Allow pods to receive traffic but block them from sending any traffic.

**Task:** Create a policy named `block-egress` matching pods with label `role=silent`. Block all outgoing connections from these pods.

**Validation:**
- Attempt to pull a webpage from inside the container using `wget`. It must fail immediately or time out.

---

## Phase 2: Targeted Ingress Restrictions (Labs 4–8)

### Lab 4: Simple Pod-to-Pod Communication

**Goal:** Allow traffic only from a specific sibling pod.

**Task:** Deploy two pods: `frontend` (`app=frontend`) and `backend` (`app=backend`). Create a NetworkPolicy so `backend` only accepts incoming traffic on port `80` from pods labeled `app=frontend`.

**Validation:**
- Exec into `frontend` and run `curl backend-ip:80` → **Success**
- Run the same command from a generic pod → **Fail**

---

### Lab 5: Namespace-Boundary Restrictions

**Goal:** Isolate traffic by originating namespace.

**Task:** Create two namespaces: `trusted` and `untrusted`. Label the `trusted` namespace with `purpose=trusted`. In the `default` namespace, deploy a pod labeled `app=database`. Protect `database` so it only accepts traffic from pods inside the `trusted` namespace.

**Validation:**
- Pod in `trusted` → connects successfully.
- Pod in `untrusted` → blocked.

---

### Lab 6: Combining Namespace and Pod Selectors (The "AND" Rule)

**Goal:** Restrict traffic to a specific pod inside a specific namespace.

**Task:** Create a policy targeting `app=protected-api`. Allow ingress only if the traffic originates from a pod labeled `tier=frontend` **AND** that pod resides in a namespace labeled `env=prod`.

> **Gotcha:** Watch your YAML array syntax closely (`- namespaceSelector` vs no hyphen — determines AND vs OR).

---

### Lab 7: Port-Specific Pinpointing

**Goal:** Expose certain application ports while keeping management ports locked.

**Task:** A pod labeled `app=multi-port` exposes port `80` (HTTP) and port `22` (SSH). Write a policy allowing traffic from any pod to port `80`, but completely deny access to port `22`.

**Validation:**
```bash
nc -zv <pod-ip> 80   # must succeed
nc -zv <pod-ip> 22   # must fail
```

---

### Lab 8: Named Ports Policy

**Goal:** Route policy rules using abstraction names instead of hardcoded numbers.

**Task:** Configure a deployment where the container specifies `ports.name: web-traffic` (mapping to `containerPort: 8080`). Write a NetworkPolicy that opens ingress using the named port `web-traffic` rather than the integer `8080`.

---

## Phase 3: Targeted Egress Restrictions (Labs 9–13)

### Lab 9: Pod-to-Pod Outbound Routing

**Goal:** Restrict where a client pod can send data.

**Task:** A processing pod labeled `app=worker` needs to send logs to a pod labeled `app=logger`. Create an egress policy on `worker` ensuring it can only send traffic to pods labeled `app=logger`.

**Validation:** Ensure `worker` cannot communicate with any other internal cluster service.

---

### Lab 10: The Essential DNS Rule

**Goal:** Prevent complete network failure when locking down egress.

**Task:** Write an egress policy for `app=client` that allows it to speak to `kube-dns` on port `53` (UDP and TCP) in the `kube-system` namespace, but blocks all other egress.

**Validation:**
```bash
nslookup kubernetes.default   # must succeed
```

---

### Lab 11: Controlled Namespace Egress

**Goal:** Restrict external pod communication to a dedicated namespace.

**Task:** Target pods labeled `app=mediator`. Allow them to initiate outbound traffic only to pods inside a namespace labeled `team=analytics`.

---

### Lab 12: External IP Whitelisting (CIDR)

**Goal:** Route data safely to an external database or third-party API.

**Task:** Write an egress policy for `app=payment-processor`:
- Allow connections to IP block `192.168.1.0/24`.
- Explicitly **except/block** the sub-range `192.168.1.50/32`.
- Block all other out-of-cluster internet traffic.

---

### Lab 13: Multi-Port Egress Rules

**Goal:** Secure a pod that must fetch patches and sync time externally.

**Task:** Create an egress policy for `app=node-updater`. Allow outbound traffic only on:
- Port `80` (HTTP)
- Port `443` (HTTPS)
- Port `123` (NTP)

---

## Phase 4: Advanced & Multi-Match Scenarios (Labs 14–17)

### Lab 14: The "OR" Condition Challenge

**Goal:** Allow access via two entirely separate criteria paths.

**Task:** Create a policy for `app=shared-storage`. Allow ingress if:

- **Path A:** Traffic comes from a pod labeled `type=admin` (any namespace).
- **OR**
- **Path B:** Traffic comes from any pod inside a namespace labeled `access=granted`.

> **Gotcha:** Pay attention to how arrays are grouped under `from` — separate list items act as OR; combined fields in one item act as AND.

---

### Lab 15: Overlapping Policies Resolution

**Goal:** Understand how Kubernetes handles multiple policies targeting the same pod.

**Task:**
1. Create **Policy 1**: Block all ingress to `app=webserver`.
2. Create **Policy 2**: Allow ingress to `app=webserver` from `app=load-balancer`.
3. Apply both to the same namespace.

**Validation:** Test if the load balancer can access the webserver. NetworkPolicies are **additive (union-based)**, so it should succeed.

---

### Lab 16: Protecting the Kubernetes Control Plane

**Goal:** Prevent compromised application pods from probing the control plane.

**Task:** Write a policy targeting all pods in the `default` namespace. Prevent them from communicating with the Kubernetes API server internal IP (typically the first IP in the service CIDR, or the `default/kubernetes` service endpoint).

---

### Lab 17: Host Network Pod Isolation

**Goal:** Understand the boundaries of NetworkPolicies.

**Task:** Deploy a pod with `hostNetwork: true`. Attempt to restrict its traffic using a standard NetworkPolicy.

**Validation:** Observe why standard NetworkPolicies **cannot** regulate pods using the host's network namespace — a critical CKS security design concept.

---

## Phase 5: Real-World Troubleshooting & Auditing (Labs 18–20)

### Lab 18: The Broken Microservice Debug

**Goal:** Identify and fix a broken connection caused by a misconfigured NetworkPolicy.

**Task:** Deploy the app using the manifest below. Find out why `frontend` cannot reach `backend` and fix it without deleting the policy.

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: strict-rules
spec:
  podSelector:
    matchLabels:
      app: backend
  ingress:
  - from:
    - podSelector:
        matchLabels:
          tier: frontend   # Bug: frontend pod actually has label "app: frontend"
```

---

### Lab 19: Auditing Unprotected Pods

**Goal:** Discover security holes in a live cluster.

**Task:** Write a native `kubectl` command using JSONPath or labels to list every pod in the `default` namespace that is **not** currently selected/protected by any NetworkPolicy.

---

### Lab 20: Comprehensive CKS Simulation Exam Question

**Goal:** Put all concepts together under exam pressure conditions.

**Task:** In namespace `production`:

1. Apply a **default deny-all** policy for ingress and egress.
2. Allow internal **DNS resolution** (port `53` UDP/TCP to `kube-system`).
3. Allow incoming traffic on **port 443** only from namespace `ingress-controllers`.
4. Allow outgoing **database traffic** to CIDR `10.0.0.0/16` on port `5432`.

> **Note:** Ensure all policies are deployed in the **same namespace** as the pods you want to protect.
