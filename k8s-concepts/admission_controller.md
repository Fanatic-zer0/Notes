# Admission Controllers & Webhooks in Kubernetes

## Overview

**Admission Controllers** are plugins that intercept requests to the Kubernetes API server **before** an object is persisted to etcd. They can **validate** or **mutate** (modify) requests.

## Request Flow Through Admission

```
kubectl apply -f pod.yaml
     │
     ├─→ 1. Authentication
     │       "Who are you?" (certificates, tokens)
     │
     ├─→ 2. Authorization (RBAC)
     │       "Are you allowed to do this?"
     │
     ├─→ 3. Mutating Admission
     │       "Modify the request if needed"
     │       - Add default values
     │       - Inject sidecars
     │       - Set resource limits
     │
     ├─→ 4. Schema Validation
     │       "Does this object match the CRD schema?"
     │
     ├─→ 5. Validating Admission
     │       "Is this request allowed?"
     │       - Policy enforcement
     │       - Security checks
     │       - Custom business rules
     │
     ├─→ 6. Persisted to etcd
     │       Object created/updated
     │
     └─→ 7. Response to client
```

> ⚠️ **Order matters:** Mutating runs BEFORE Validating. You can mutate to fix issues, then validate the result.

## Built-in Admission Controllers

| Controller | Purpose |
|------------|---------|
| `NamespaceLifecycle` | Reject operations in terminating namespaces |
| `LimitRanger` | Apply LimitRange defaults |
| `ServiceAccount` | Auto-inject service account |
| `DefaultStorageClass` | Set default StorageClass |
| `ResourceQuota` | Enforce resource quotas |
| `PodSecurity` | Enforce Pod Security Standards |
| `NodeRestriction` | Limit kubelet permissions |
| `MutatingAdmissionWebhook` | Call external mutating webhooks |
| `ValidatingAdmissionWebhook` | Call external validating webhooks |

## Webhook Types

| Type | Phase | Can modify? | Can reject? |
|------|-------|-------------|-------------|
| **MutatingAdmissionWebhook** | Mutating | ✅ Yes (JSONPatch) | ✅ Yes |
| **ValidatingAdmissionWebhook** | Validating | ❌ No | ✅ Yes |

## Mutating Webhook

**Use cases:**
- Auto-inject sidecars (Istio, logging agents)
- Set default resource limits
- Add required labels/annotations
- Modify image tags (latest → SHA)

### Example: Auto-inject Logging Sidecar

```yaml
# The webhook configuration
apiVersion: admissionregistration.k8s.io/v1
kind: MutatingWebhookConfiguration
metadata:
  name: sidecar-injector
webhooks:
- name: inject.sidecar.example.com
  
  # Call this service when webhook fires
  clientConfig:
    service:
      name: sidecar-injector
      namespace: injection-system
      path: /inject
    caBundle: <base64-ca-cert>  # Trust the webhook server cert
  
  # Which resources trigger this webhook
  rules:
  - operations: [CREATE, UPDATE]
    apiGroups: [""]
    apiVersions: ["v1"]
    resources: ["pods"]
  
  # Only inject if namespace has the label
  namespaceSelector:
    matchLabels:
      injection: enabled
  
  # Skip if pod has this label
  objectSelector:
    matchExpressions:
    - key: skip-injection
      operator: DoesNotExist
  
  # Behavior on webhook failure
  failurePolicy: Fail    # Reject pod creation if webhook fails
  # or: Ignore           # Allow pod creation even if webhook fails
  
  sideEffects: None      # Webhook has no side effects
  admissionReviewVersions: ["v1", "v1beta1"]
  timeoutSeconds: 5      # Webhook must respond in 5s
```

### Webhook Server Implementation (Go)

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
    
    admissionv1 "k8s.io/api/admission/v1"
    corev1 "k8s.io/api/core/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func main() {
    http.HandleFunc("/inject", handleInject)
    http.ListenAndServeTLS(":443", "/certs/tls.crt", "/certs/tls.key", nil)
}

func handleInject(w http.ResponseWriter, r *http.Request) {
    // Parse admission review
    var review admissionv1.AdmissionReview
    json.NewDecoder(r.Body).Decode(&review)
    
    // Parse the pod
    var pod corev1.Pod
    json.Unmarshal(review.Request.Object.Raw, &pod)
    
    // Build JSON patch to add sidecar
    sidecarContainer := corev1.Container{
        Name:  "log-shipper",
        Image: "fluentd:latest",
        VolumeMounts: []corev1.VolumeMount{
            {Name: "app-logs", MountPath: "/var/log/app"},
        },
    }
    
    sidecarJSON, _ := json.Marshal(sidecarContainer)
    patch := fmt.Sprintf(
        `[{"op":"add","path":"/spec/containers/-","value":%s}]`,
        sidecarJSON,
    )
    
    // Build response
    patchType := admissionv1.PatchTypeJSONPatch
    response := admissionv1.AdmissionReview{
        TypeMeta: metav1.TypeMeta{
            APIVersion: "admission.k8s.io/v1",
            Kind:       "AdmissionReview",
        },
        Response: &admissionv1.AdmissionResponse{
            UID:       review.Request.UID,
            Allowed:   true,  // Allow the pod
            PatchType: &patchType,
            Patch:     []byte(patch),  // With modifications
        },
    }
    
    json.NewEncoder(w).Encode(response)
}
```

### Complete Injection Flow

```
kubectl apply -f my-pod.yaml
(Namespace has: injection=enabled label)
     │
     ├─→ 1. API server receives CREATE pod request
     │
     ├─→ 2. Mutating Admission phase
     │       Webhook matches: operations=CREATE, resources=pods
     │
     ├─→ 3. API server calls webhook
     │       POST https://sidecar-injector.injection-system.svc/inject
     │       Body: AdmissionReview with pod spec
     │
     ├─→ 4. Webhook server processes request
     │       - Check: Should this pod be injected?
     │       - Check: annotation "skip-injection" absent
     │       - Build JSONPatch:
     │           + add log-shipper container
     │           + add shared volume
     │
     ├─→ 5. Webhook returns response
     │       Allowed: true
     │       Patch: [{"op":"add","path":"/spec/containers/-","value":{...}}]
     │
     ├─→ 6. API server applies patch to pod spec
     │       Pod now has EXTRA container: log-shipper
     │
     ├─→ 7. Validating admission phase
     │       Pod validated against policies
     │
     ├─→ 8. Pod stored in etcd (with injected sidecar)
     │
     └─→ 9. kubelet starts both containers
              - my-app
              - log-shipper (injected!)
```

## Validating Webhook

**Use cases:**
- Enforce security policies
- Require specific labels
- Prevent privileged pods
- Enforce image registry allowlist

### Example: Enforce Image Registry

```yaml
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  name: image-policy
webhooks:
- name: validate.images.example.com
  clientConfig:
    service:
      name: image-policy-webhook
      namespace: policy-system
      path: /validate
    caBundle: <base64-ca-cert>
  rules:
  - operations: [CREATE, UPDATE]
    apiGroups: [""]
    apiVersions: ["v1"]
    resources: ["pods"]
  - operations: [CREATE, UPDATE]
    apiGroups: ["apps"]
    apiVersions: ["v1"]
    resources: ["deployments", "replicasets", "daemonsets", "statefulsets"]
  failurePolicy: Fail
  sideEffects: None
  admissionReviewVersions: ["v1"]
```

```go
func handleValidate(w http.ResponseWriter, r *http.Request) {
    var review admissionv1.AdmissionReview
    json.NewDecoder(r.Body).Decode(&review)
    
    var pod corev1.Pod
    json.Unmarshal(review.Request.Object.Raw, &pod)
    
    allowedRegistries := []string{
        "registry.example.com",
        "gcr.io/my-project",
    }
    
    for _, container := range pod.Spec.Containers {
        allowed := false
        for _, registry := range allowedRegistries {
            if strings.HasPrefix(container.Image, registry) {
                allowed = true
                break
            }
        }
        if !allowed {
            // REJECT the request
            response := admissionv1.AdmissionReview{
                Response: &admissionv1.AdmissionResponse{
                    UID:     review.Request.UID,
                    Allowed: false,
                    Result: &metav1.Status{
                        Code:    403,
                        Message: fmt.Sprintf("Image %s is not from an allowed registry", container.Image),
                    },
                },
            }
            json.NewEncoder(w).Encode(response)
            return
        }
    }
    
    // ALLOW the request
    json.NewEncoder(w).Encode(admissionv1.AdmissionReview{
        Response: &admissionv1.AdmissionResponse{
            UID:     review.Request.UID,
            Allowed: true,
        },
    })
}
```

## Policy Engines (Admission Controller Frameworks)

### OPA/Gatekeeper

**Open Policy Agent** - Policy as code using **Rego** language.

```yaml
# 1. Define policy template (ConstraintTemplate)
apiVersion: templates.gatekeeper.sh/v1
kind: ConstraintTemplate
metadata:
  name: requirelabels
spec:
  crd:
    spec:
      names:
        kind: RequireLabels
      validation:
        openAPIV3Schema:
          properties:
            labels:
              type: array
              items:
                type: string
  targets:
  - target: admission.k8s.gatekeeper.sh
    rego: |
      package requirelabels
      
      violation[{"msg": msg}] {
        provided := {label | input.review.object.metadata.labels[label]}
        required := {label | label := input.parameters.labels[_]}
        missing := required - provided
        count(missing) > 0
        msg := sprintf("Missing required labels: %v", [missing])
      }
---
# 2. Create constraint (enforce policy)
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: RequireLabels
metadata:
  name: require-team-label
spec:
  match:
    kinds:
    - apiGroups: [""]
      kinds: ["Namespace"]
  parameters:
    labels:
    - team
    - environment
    - cost-center
```

**Effect:**
```bash
kubectl create namespace my-team
# Error from server: [require-team-label] Missing required labels: {cost-center, environment, team}

kubectl create namespace my-team \
  --dry-run=server \
  -o yaml \
  -- kubectl label namespace my-team team=backend environment=production cost-center=123
# Works!
```

### Kyverno

**Policy engine** designed specifically for Kubernetes (YAML-based, no Rego).

```yaml
# Enforce team label on all pods
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: require-team-label
spec:
  validationFailureAction: Enforce  # or Audit
  rules:
  - name: check-team-label
    match:
      any:
      - resources:
          kinds:
          - Pod
    validate:
      message: "Label 'team' is required"
      pattern:
        metadata:
          labels:
            team: "?*"  # Must exist with any value
---
# Auto-generate NetworkPolicy for each namespace
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: generate-namespace-policy
spec:
  rules:
  - name: default-deny
    match:
      any:
      - resources:
          kinds:
          - Namespace
    generate:
      kind: NetworkPolicy
      name: default-deny
      namespace: "{{request.object.metadata.name}}"
      data:
        spec:
          podSelector: {}
          policyTypes:
          - Ingress
          - Egress
---
# Mutate: Add resource limits if missing
apiVersion: kyverno.io/v1
kind: ClusterPolicy
metadata:
  name: add-default-resources
spec:
  rules:
  - name: add-limits
    match:
      any:
      - resources:
          kinds:
          - Pod
    mutate:
      patchStrategicMerge:
        spec:
          containers:
          - (name): "*"
            resources:
              limits:
                +(cpu): "500m"      # Add only if not set
                +(memory): "512Mi"
              requests:
                +(cpu): "100m"
                +(memory): "128Mi"
```

## ValidatingAdmissionPolicy (Native Kubernetes)

**CEL-based policies** without external webhooks (Kubernetes 1.26+).

```yaml
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata:
  name: no-privileged-pods
spec:
  failurePolicy: Fail
  matchConstraints:
    resourceRules:
    - apiGroups: [""]
      apiVersions: ["v1"]
      operations: [CREATE, UPDATE]
      resources: [pods]
  
  validations:
  - expression: "!has(object.spec.securityContext) || !object.spec.securityContext.runAsRoot"
    message: "Pods cannot run as root"
  
  - expression: >
      object.spec.containers.all(c, 
        !has(c.securityContext) || 
        !has(c.securityContext.privileged) || 
        !c.securityContext.privileged
      )
    message: "Privileged containers are not allowed"
  
  - expression: >
      object.spec.containers.all(c, 
        has(c.resources) && 
        has(c.resources.limits) && 
        has(c.resources.limits.cpu)
      )
    message: "CPU limits must be set"
---
# Bind policy to specific resources
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicyBinding
metadata:
  name: no-privileged-pods-binding
spec:
  policyName: no-privileged-pods
  validationActions: [Deny]  # Deny, Warn, or Audit
  matchResources:
    namespaceSelector:
      matchExpressions:
      - key: environment
        operator: In
        values: [production, staging]
```

## Webhook Security

### TLS Setup with cert-manager

```yaml
# Auto-managed TLS cert for webhook
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: webhook-cert
  namespace: webhook-system
spec:
  secretName: webhook-tls
  issuerRef:
    name: cluster-ca
    kind: ClusterIssuer
  dnsNames:
  - my-webhook.webhook-system.svc
  - my-webhook.webhook-system.svc.cluster.local
---
# Webhook references the cert
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingWebhookConfiguration
metadata:
  annotations:
    cert-manager.io/inject-ca-from: webhook-system/webhook-cert
webhooks:
- name: validate.example.com
  clientConfig:
    service:
      name: my-webhook
      namespace: webhook-system
      path: /validate
    # caBundle auto-injected by cert-manager!
```

## Troubleshooting

```bash
# Check webhook configurations
kubectl get mutatingwebhookconfigurations
kubectl get validatingwebhookconfigurations

# Describe a webhook
kubectl describe mutatingwebhookconfiguration sidecar-injector

# Test webhook (dry-run)
kubectl apply -f pod.yaml --dry-run=server

# View webhook events
kubectl get events --field-selector reason=FailedCreate

# Check webhook server logs
kubectl logs -n webhook-system deployment/my-webhook

# Temporarily disable a webhook (emergency)
kubectl patch mutatingwebhookconfiguration sidecar-injector \
  -p '{"webhooks":[{"name":"inject.sidecar.example.com","failurePolicy":"Ignore"}]}'

# View kube-apiserver audit logs for admission
# (requires audit policy configured)
grep "admission" /var/log/kubernetes/audit.log
```

