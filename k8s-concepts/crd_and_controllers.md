# CRDs & Custom Controllers in Kubernetes

## Overview

Kubernetes is **extensible by design**. **Custom Resource Definitions (CRDs)** let you add new resource types to the Kubernetes API, and **Custom Controllers** (Operators) implement the business logic to manage them.

### The Fundamental Pattern

```
Standard Kubernetes Resources (Built-in):
  
  Deployment (API resource)  ←──managed by──→  Deployment Controller
       │                                             │
       ├─ Desired state: 3 replicas                 ├─ Watches Deployments
       └─ Stored in: etcd                           ├─ Creates ReplicaSets
                                                     └─ Maintains desired state

Your Custom Resources (Extended):
  
  Database (CRD)            ←──managed by──→  Database Operator (Controller)
       │                                             │
       ├─ Desired state: postgres, 3 replicas       ├─ Watches Database CRs
       └─ Stored in: etcd                           ├─ Creates StatefulSets, Services
                                                     ├─ Manages backups
                                                     └─ Maintains desired state
```

### Why Extend Kubernetes?

**Without CRDs (Manual Operations):**
```
Developer: "I need a PostgreSQL database"
     │
     ├─→ 1. Create StatefulSet YAML (100+ lines)
     ├─→ 2. Create Service YAML
     ├─→ 3. Create ConfigMap for postgres config
     ├─→ 4. Create Secrets for credentials
     ├─→ 5. Create PVCs for storage
     ├─→ 6. Configure backup CronJob
     ├─→ 7. Set up monitoring
     └─→ 8. Remember to update all when scaling
     
Total: ~400 lines of YAML, error-prone, not reusable
```

**With CRDs + Operator:**
```
Developer: "I need a PostgreSQL database"
     │
     └─→ Create simple Database CR (20 lines):
     
         apiVersion: example.com/v1
         kind: Database
         metadata:
           name: my-db
         spec:
           engine: postgres
           replicas: 3
           storage: 100Gi
     
Operator handles all complexity automatically! ✅
```

## CRD Definition

### Step 1: Define the CRD Schema

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: databases.example.com  # Must be: <plural>.<group>
spec:
  group: example.com
  versions:
  - name: v1
    served: true     # This version is served by API server
    storage: true    # This version is stored in etcd
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            required: ["engine", "replicas", "storage"]
            properties:
              engine:
                type: string
                enum: [postgres, mysql, mongodb]
              version:
                type: string
                default: "15"
              replicas:
                type: integer
                minimum: 1
                maximum: 5
              storage:
                type: string
                pattern: '^[0-9]+Gi$'
              credentials:
                type: object
                properties:
                  secretName:
                    type: string
          status:
            type: object
            properties:
              phase:
                type: string
                enum: [Pending, Creating, Running, Updating, Failed]
              readyReplicas:
                type: integer
              connectionString:
                type: string
              conditions:
                type: array
                items:
                  type: object
                  properties:
                    type:
                      type: string
                    status:
                      type: string
                    lastTransitionTime:
                      type: string
                    message:
                      type: string
    subresources:
      status: {}      # Enable /status subresource
      scale:          # Enable /scale subresource (for HPA compatibility)
        specReplicasPath: .spec.replicas
        statusReplicasPath: .status.readyReplicas
    additionalPrinterColumns:
    - name: Engine
      type: string
      jsonPath: .spec.engine
    - name: Replicas
      type: integer
      jsonPath: .spec.replicas
    - name: Status
      type: string
      jsonPath: .status.phase
    - name: Age
      type: date
      jsonPath: .metadata.creationTimestamp
  scope: Namespaced    # or Cluster
  names:
    plural: databases
    singular: database
    kind: Database
    shortNames:
    - db
```

### Step 2: Create a Custom Resource

```yaml
apiVersion: example.com/v1
kind: Database
metadata:
  name: production-db
  namespace: default
spec:
  engine: postgres
  version: "15"
  replicas: 3
  storage: 100Gi
  credentials:
    secretName: db-credentials
```

```bash
# After creating CRD, you can use these:
kubectl get databases
kubectl get db                      # short name
kubectl get db production-db -o yaml
kubectl describe db production-db
```

## Custom Controller (Operator)

### Control Loop (Reconciliation)

```
Controller starts
     │
     ├─→ 1. Set up informers (watch for resource changes)
     │       Watch: databases.example.com
     │       Watch: StatefulSets, Services, Secrets
     │
     ├─→ 2. Work queue
     │       ADD/UPDATE/DELETE events → queue
     │
     ├─→ 3. Reconcile loop (per queued item)
     │       ┌──────────────────────────────────┐
     │       │         RECONCILE                 │
     │       │  1. Get current state            │
     │       │     kubectl get database prod-db  │
     │       │                                  │
     │       │  2. Get desired state            │
     │       │     spec.replicas: 3             │
     │       │                                  │
     │       │  3. Get observed state           │
     │       │     StatefulSet replicas: 1      │
     │       │                                  │
     │       │  4. Compute diff                 │
     │       │     Need 2 more replicas         │
     │       │                                  │
     │       │  5. Apply changes                │
     │       │     Update StatefulSet           │
     │       │                                  │
     │       │  6. Update status                │
     │       │     status.readyReplicas: 3      │
     │       └──────────────────────────────────┘
     │
     └─→ 4. Requeue on error (with backoff)
```

### Controller Code (Go with controller-runtime)

```go
package controllers

import (
    "context"
    "fmt"
    
    appsv1 "k8s.io/api/apps/v1"
    corev1 "k8s.io/api/core/v1"
    "k8s.io/apimachinery/pkg/api/errors"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/apimachinery/pkg/runtime"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/client"
    
    examplev1 "github.com/example/db-operator/api/v1"
)

// DatabaseReconciler reconciles Database objects
type DatabaseReconciler struct {
    client.Client
    Scheme *runtime.Scheme
}

// Reconcile is called on every Database change event
func (r *DatabaseReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    log := ctrl.LoggerFrom(ctx)
    
    // 1. Fetch the Database CR
    var db examplev1.Database
    if err := r.Get(ctx, req.NamespacedName, &db); err != nil {
        if errors.IsNotFound(err) {
            // Object deleted, nothing to do
            return ctrl.Result{}, nil
        }
        return ctrl.Result{}, err
    }
    
    // 2. Update status to Creating if new
    if db.Status.Phase == "" {
        db.Status.Phase = "Creating"
        if err := r.Status().Update(ctx, &db); err != nil {
            return ctrl.Result{}, err
        }
    }
    
    // 3. Create or update StatefulSet
    statefulSet := r.buildStatefulSet(&db)
    
    var existing appsv1.StatefulSet
    err := r.Get(ctx, client.ObjectKey{
        Name:      db.Name,
        Namespace: db.Namespace,
    }, &existing)
    
    if errors.IsNotFound(err) {
        // Create new StatefulSet
        log.Info("Creating StatefulSet", "name", statefulSet.Name)
        if err := ctrl.SetControllerReference(&db, statefulSet, r.Scheme); err != nil {
            return ctrl.Result{}, err
        }
        if err := r.Create(ctx, statefulSet); err != nil {
            return ctrl.Result{}, err
        }
    } else if err == nil {
        // Update existing StatefulSet
        existing.Spec = statefulSet.Spec
        if err := r.Update(ctx, &existing); err != nil {
            return ctrl.Result{}, err
        }
    } else {
        return ctrl.Result{}, err
    }
    
    // 4. Create Service for the database
    service := r.buildService(&db)
    // ... similar create/update logic
    
    // 5. Update status
    db.Status.Phase = "Running"
    db.Status.ReadyReplicas = int32(db.Spec.Replicas)
    db.Status.ConnectionString = fmt.Sprintf(
        "postgresql://%s.%s.svc.cluster.local:5432/mydb",
        db.Name, db.Namespace,
    )
    
    if err := r.Status().Update(ctx, &db); err != nil {
        return ctrl.Result{}, err
    }
    
    // 6. Requeue after 1 minute for health checks
    return ctrl.Result{RequeueAfter: time.Minute}, nil
}

// SetupWithManager registers the controller
func (r *DatabaseReconciler) SetupWithManager(mgr ctrl.Manager) error {
    return ctrl.NewControllerManagedBy(mgr).
        For(&examplev1.Database{}).            // Watch Database CRs
        Owns(&appsv1.StatefulSet{}).           // Watch owned StatefulSets
        Owns(&corev1.Service{}).               // Watch owned Services
        WithOptions(controller.Options{
            MaxConcurrentReconciles: 5,        // Parallel reconcilers
        }).
        Complete(r)
}
```

## Complete CRD + Controller Flow: From Creation to Running

### Phase 1: CRD Installation (Cluster Admin)

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: INSTALL CRD (One-time setup)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ kubectl apply -f database-crd.yaml                             │
│      │                                                          │
│      ├─→ 1. kubectl sends CRD to API server                    │
│      │                                                          │
│      ├─→ 2. API server validates CRD structure:                │
│      │      ✅ metadata.name = "databases.example.com"         │
│      │      ✅ spec.group = "example.com"                      │
│      │      ✅ spec.versions defined                           │
│      │      ✅ spec.schema.openAPIV3Schema valid               │
│      │                                                          │
│      ├─→ 3. CRD stored in etcd:                                │
│      │      Key: /registry/apiextensions.k8s.io/              │
│      │           customresourcedefinitions/                    │
│      │           databases.example.com                         │
│      │                                                          │
│      ├─→ 4. API server registers new API endpoint:            │
│      │      ✅ /apis/example.com/v1/databases                 │
│      │      ✅ /apis/example.com/v1/namespaces/*/databases    │
│      │                                                          │
│      └─→ 5. New resource type now available!                  │
│             You can now: kubectl get databases                 │
└─────────────────────────────────────────────────────────────────┘
```

**What Happens Behind the Scenes:**

```
API Server Components:
  
  ┌──────────────────────────────────────────┐
  │ API Aggregation Layer                    │
  ├──────────────────────────────────────────┤
  │                                          │
  │ BEFORE CRD installation:                 │
  │   /api/v1/pods                ✅         │
  │   /apis/apps/v1/deployments   ✅         │
  │   /apis/example.com/v1/databases  ❌     │
  │                                          │
  │ AFTER CRD installation:                  │
  │   /api/v1/pods                ✅         │
  │   /apis/apps/v1/deployments   ✅         │
  │   /apis/example.com/v1/databases  ✅ NEW!│
  │                                          │
  │ Schema Validation Rules Added:           │
  │   • spec.engine: enum[postgres,mysql]    │
  │   • spec.replicas: integer, min=1        │
  │   • spec.storage: string, pattern=Gi$    │
  └──────────────────────────────────────────┘
```

---

### Phase 2: Controller Deployment

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: DEPLOY CONTROLLER (Operator)                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ kubectl apply -f database-operator-deployment.yaml             │
│      │                                                          │
│      ├─→ 1. Creates Deployment for operator                    │
│      │      image: registry.example.com/db-operator:v1.0       │
│      │                                                          │
│      ├─→ 2. Pod starts, operator initialization:               │
│      │                                                          │
│      │      ┌──────────────────────────────────────┐          │
│      │      │ OPERATOR STARTUP SEQUENCE             │          │
│      │      ├──────────────────────────────────────┤          │
│      │      │                                       │          │
│      │      │ 1. Load kubeconfig (in-cluster)      │          │
│      │      │    ServiceAccount: db-operator       │          │
│      │      │    Namespace: operators              │          │
│      │      │                                       │          │
│      │      │ 2. Create Kubernetes client          │          │
│      │      │    API server: https://10.96.0.1:443│          │
│      │      │    Auth: ServiceAccount token        │          │
│      │      │                                       │          │
│      │      │ 3. Verify RBAC permissions:          │          │
│      │      │    ✅ Can list/watch databases       │          │
│      │      │    ✅ Can create/update StatefulSets│          │
│      │      │    ✅ Can create/update Services    │          │
│      │      │    ✅ Can update Database status    │          │
│      │      │                                       │          │
│      │      │ 4. Set up Informers (cache):         │          │
│      │      │    • DatabaseInformer                │          │
│      │      │    • StatefulSetInformer             │          │
│      │      │    • ServiceInformer                 │          │
│      │      │                                       │          │
│      │      │ 5. Start watching API server:        │          │
│      │      │    Watch databases.example.com/v1    │          │
│      │      │    Watch apps/v1/statefulsets        │          │
│      │      │    Watch v1/services                 │          │
│      │      │                                       │          │
│      │      │ 6. Start work queue                  │          │
│      │      │    Workers: 5 concurrent             │          │
│      │      │    Buffer: 100 items                 │          │
│      │      │                                       │          │
│      │      │ 7. Start event handlers              │          │
│      │      │    OnAdd: enqueue item               │          │
│      │      │    OnUpdate: enqueue item            │          │
│      │      │    OnDelete: enqueue item            │          │
│      │      │                                       │          │
│      │      │ 8. Operator READY ✅                  │          │
│      │      │    Waiting for Database CRs...       │          │
│      │      └──────────────────────────────────────┘          │
│      │                                                          │
│      └─→ 3. Leader election (if multiple replicas):            │
│             Only one operator actively reconciles               │
└─────────────────────────────────────────────────────────────────┘
```

**Informer Watch Connections:**

```
Operator Pod ────┬─→ WATCH /apis/example.com/v1/databases
                 │    (HTTP long-polling or WebSocket)
                 │    API server sends events when databases change
                 │
                 ├─→ WATCH /apis/apps/v1/statefulsets
                 │    (Tracks owned StatefulSets)
                 │
                 └─→ WATCH /api/v1/services
                      (Tracks owned Services)

Each watch maintains persistent connection to API server
Events arrive in real-time as resources change
```

---

### Phase 3: Creating a Custom Resource

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: USER CREATES DATABASE RESOURCE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ kubectl apply -f production-db.yaml                            │
│                                                                 │
│ Content:                                                        │
│   apiVersion: example.com/v1                                   │
│   kind: Database                                               │
│   metadata:                                                     │
│     name: production-db                                        │
│     namespace: default                                         │
│   spec:                                                         │
│     engine: postgres                                           │
│     version: "15"                                              │
│     replicas: 3                                                │
│     storage: 100Gi                                             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ REQUEST FLOW THROUGH API SERVER:                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 1. kubectl → API Server                                        │
│    POST /apis/example.com/v1/namespaces/default/databases     │
│                                                                 │
│ 2. Authentication (API server):                                │
│    ✅ Verify kubectl user credentials                          │
│    User: kubernetes-admin                                      │
│    Groups: system:masters                                      │
│                                                                 │
│ 3. Authorization (RBAC check):                                 │
│    ✅ Can user "kubernetes-admin" create databases?            │
│    ClusterRole: cluster-admin → Allows all                    │
│                                                                 │
│ 4. Admission Controllers (sequential pipeline):               │
│    │                                                            │
│    ├─→ MutatingAdmissionWebhook:                               │
│    │    • None configured for databases → Skip                 │
│    │                                                            │
│    ├─→ ValidatingAdmissionWebhook:                             │
│    │    • None configured for databases → Skip                 │
│    │                                                            │
│    ├─→ Schema Validation (from CRD):                           │
│    │    ┌──────────────────────────────────────┐             │
│    │    │ Validate against OpenAPI schema:     │             │
│    │    │                                        │             │
│    │    │ spec.engine: "postgres"                │             │
│    │    │   ✅ Value in enum [postgres, mysql]  │             │
│    │    │                                        │             │
│    │    │ spec.version: "15"                     │             │
│    │    │   ✅ Type: string                      │             │
│    │    │                                        │             │
│    │    │ spec.replicas: 3                       │             │
│    │    │   ✅ Type: integer                     │             │
│    │    │   ✅ Minimum: 1                        │             │
│    │    │   ✅ Maximum: 5                        │             │
│    │    │                                        │             │
│    │    │ spec.storage: "100Gi"                  │             │
│    │    │   ✅ Type: string                      │             │
│    │    │   ✅ Pattern: ^[0-9]+Gi$               │             │
│    │    │                                        │             │
│    │    │ ALL VALIDATIONS PASSED ✅               │             │
│    │    └──────────────────────────────────────┘             │
│    │                                                            │
│    └─→ ResourceQuota, LimitRanger, etc. (if configured)       │
│                                                                 │
│ 5. Persistence (etcd):                                         │
│    Write to etcd:                                              │
│    Key: /registry/example.com/databases/default/production-db │
│    Value: {metadata: {...}, spec: {...}, status: {}}          │
│    ResourceVersion: 1234567 (auto-incremented)                │
│                                                                 │
│ 6. Response to kubectl:                                        │
│    HTTP 201 Created                                            │
│    database.example.com/production-db created                  │
│                                                                 │
│ 7. API server sends WATCH event:                              │
│    Event type: ADDED                                           │
│    Resource: production-db                                     │
│    Sent to all watchers (including operator)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### Phase 4: Operator Receives Event

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: OPERATOR INFORMER RECEIVES WATCH EVENT                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ API Server ──→ Operator's DatabaseInformer                     │
│    Event: {                                                     │
│      type: "ADDED",                                            │
│      object: {                                                  │
│        kind: "Database",                                       │
│        metadata: {name: "production-db", namespace: "default"},│
│        spec: {engine: "postgres", replicas: 3, ...}           │
│      }                                                          │
│    }                                                            │
│                                                                 │
│ Informer Processing:                                           │
│   │                                                             │
│   ├─→ 1. Update local cache:                                   │
│   │      Store production-db in in-memory cache                │
│   │      (Reduces future API calls)                            │
│   │                                                             │
│   ├─→ 2. Trigger event handler:                                │
│   │      OnAdd callback registered by controller               │
│   │                                                             │
│   └─→ 3. Enqueue work item:                                    │
│         Add to work queue: "default/production-db"             │
│         Queue size: 1                                           │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ WORK QUEUE PROCESSING:                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Worker goroutine #1 dequeues item:                            │
│   Item: "default/production-db"                                │
│   Queue size now: 0                                            │
│                                                                 │
│ ┌─ Start processing                                            │
│ │  Call Reconcile(ctx, Request{                                │
│ │    NamespacedName: "default/production-db"                   │
│ │  })                                                           │
│ └─                                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

### Phase 5: Reconciliation Logic (The Core!)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ RECONCILE FUNCTION - STEP BY STEP                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ func (r *DatabaseReconciler) Reconcile(ctx, req) {                    │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐  │
│ │ STEP 1: FETCH DATABASE CUSTOM RESOURCE                           │  │
│ ├─────────────────────────────────────────────────────────────────┤  │
│ │                                                                   │  │
│ │ var db examplev1.Database                                        │  │
│ │ err := r.Get(ctx, req.NamespacedName, &db)                       │  │
│ │   // req.NamespacedName = "default/production-db"                │  │
│ │                                                                   │  │
│ │ Decision point:                                                   │  │
│ │   IF err == NotFound:                                             │  │
│ │     ✅ Resource was deleted → Return (nothing to do)             │  │
│ │     Controller will handle via finalizers if needed              │  │
│ │   ELSE IF err != nil:                                             │  │
│ │     ❌ API error → Return error, requeue                         │  │
│ │   ELSE:                                                           │  │
│ │     ✅ Resource found → Continue processing                      │  │
│ │                                                                   │  │
│ │ Retrieved Database CR:                                            │  │
│ │   metadata:                                                        │  │
│ │     name: production-db                                           │  │
│ │     namespace: default                                            │  │
│ │     resourceVersion: "1234567"                                    │  │
│ │     generation: 1                                                 │  │
│ │   spec:                                                            │  │
│ │     engine: postgres                                              │  │
│ │     version: "15"                                                 │  │
│ │     replicas: 3                                                   │  │
│ │     storage: 100Gi                                                │  │
│ │   status:                                                          │  │
│ │     {} (empty - new resource)                                     │  │
│ └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐  │
│ │ STEP 2: CHECK DELETION TIMESTAMP (Finalizer Logic)               │  │
│ ├─────────────────────────────────────────────────────────────────┤  │
│ │                                                                   │  │
│ │ IF !db.DeletionTimestamp.IsZero():                               │  │
│ │   // Resource is being deleted                                   │  │
│ │   IF db.Finalizers contains "example.com/db-finalizer":          │  │
│ │     1. Run cleanup logic:                                         │  │
│ │        • Take final backup                                        │  │
│ │        • Deregister from external services                        │  │
│ │        • Clean up cloud resources                                 │  │
│ │     2. Remove finalizer from list                                 │  │
│ │     3. Update Database CR                                         │  │
│ │     4. Return (API server will now delete object)                 │  │
│ │   ELSE:                                                           │  │
│ │     Return (already cleaned up)                                   │  │
│ │                                                                   │  │
│ │ Current case: DeletionTimestamp = nil → Resource is active       │  │
│ │ ✅ Continue to reconciliation                                     │  │
│ └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐  │
│ │ STEP 3: ADD FINALIZER (If Not Present)                           │  │
│ ├─────────────────────────────────────────────────────────────────┤  │
│ │                                                                   │  │
│ │ IF !containsString(db.Finalizers, "example.com/db-finalizer"):  │  │
│ │   db.Finalizers = append(db.Finalizers, "example.com/db-fin...")│  │
│ │   r.Update(ctx, &db)                                             │  │
│ │   ✅ Finalizer added - protects from accidental deletion         │  │
│ │                                                                   │  │
│ │ Why? Ensures cleanup logic runs before deletion                  │  │
│ └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐  │
│ │ STEP 4: INITIALIZE STATUS (If Empty)                             │  │
│ ├─────────────────────────────────────────────────────────────────┤  │
│ │                                                                   │  │
│ │ IF db.Status.Phase == "":                                        │  │
│ │   // This is a brand new Database resource                       │  │
│ │   db.Status.Phase = "Pending"                                    │  │
│ │   db.Status.ReadyReplicas = 0                                    │  │
│ │   db.Status.Conditions = []Condition{                            │  │
│ │     {Type: "Progressing", Status: "True",                        │  │
│ │      Reason: "ReconciliationStarted"}                            │  │
│ │   }                                                               │  │
│ │   r.Status().Update(ctx, &db)  // Update only /status subresource│  │
│ │                                                                   │  │
│ │ Result:                                                           │  │
│ │   Status updated in etcd                                          │  │
│ │   ✅ Phase: "Pending"                                             │  │
│ │   This triggers another reconcile (status changed)                │  │
│ └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐  │
│ │ STEP 5: BUILD DESIRED STATEFULSET                                │  │
│ ├─────────────────────────────────────────────────────────────────┤  │
│ │                                                                   │  │
│ │ desiredStatefulSet := r.buildStatefulSet(&db)                    │  │
│ │                                                                   │  │
│ │ // buildStatefulSet translates Database spec to StatefulSet:     │  │
│ │ func (r *DatabaseReconciler) buildStatefulSet(db *Database) {    │  │
│ │   return &appsv1.StatefulSet{                                    │  │
│ │     metadata: {                                                   │  │
│ │       name: db.Name,              // "production-db"             │  │
│ │       namespace: db.Namespace,    // "default"                   │  │
│ │       labels: {                                                   │  │
│ │         "app": db.Name,                                          │  │
│ │         "engine": db.Spec.Engine, // "postgres"                  │  │
│ │       }                                                           │  │
│ │     },                                                            │  │
│ │     spec: {                                                       │  │
│ │       replicas: db.Spec.Replicas,    // 3                        │  │
│ │       serviceName: db.Name,          // "production-db"          │  │
│ │       selector: {                                                 │  │
│ │         matchLabels: {"app": db.Name}                            │  │
│ │       },                                                          │  │
│ │       template: {                                                 │  │
│ │         spec: {                                                   │  │
│ │           containers: [{                                          │  │
│ │             name: "postgres",                                     │  │
│ │             image: "postgres:" + db.Spec.Version, // postgres:15 │  │
│ │             ports: [{containerPort: 5432}],                       │  │
│ │             env: [...database config...],                         │  │
│ │             volumeMounts: [...]                                   │  │
│ │           }]                                                       │  │
│ │         }                                                          │  │
│ │       },                                                           │  │
│ │       volumeClaimTemplates: [{                                    │  │
│ │         spec: {                                                    │  │
│ │           resources: {requests: {storage: db.Spec.Storage}}      │  │
│ │         }                                                          │  │
│ │       }]                                                           │  │
│ │     }                                                              │  │
│ │   }                                                                │  │
│ │ }                                                                  │  │
│ └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐  │
│ │ STEP 6: CHECK IF STATEFULSET EXISTS                              │  │
│ ├─────────────────────────────────────────────────────────────────┤  │
│ │                                                                   │  │
│ │ var existingStatefulSet appsv1.StatefulSet                       │  │
│ │ err := r.Get(ctx, client.ObjectKey{                              │  │
│ │   Name: db.Name,       // "production-db"                        │  │
│ │   Namespace: db.Namespace  // "default"                          │  │
│ │ }, &existingStatefulSet)                                         │  │
│ │                                                                   │  │
│ │ Decision Tree:                                                    │  │
│ │                                                                   │  │
│ │ ┌─ IF err == NotFound:                                           │  │
│ │ │    StatefulSet doesn't exist → CREATE IT                       │  │
│ │ │    ✅ BRANCH A (Creation path)                                 │  │
│ │ │                                                                  │  │
│ │ ├─ ELSE IF err == nil:                                           │  │
│ │ │    StatefulSet exists → UPDATE IF NEEDED                       │  │
│ │ │    ✅ BRANCH B (Update path)                                   │  │
│ │ │                                                                  │  │
│ │ └─ ELSE (other error):                                           │  │
│ │      API error → Return error, requeue                           │  │
│ │      ❌ BRANCH C (Error path)                                    │  │
│ │                                                                   │  │
│ │ Current scenario: StatefulSet not found                          │  │
│ │ → Follow BRANCH A                                                 │  │
│ └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐  │
│ │ STEP 7A: CREATE STATEFULSET (Branch A)                           │  │
│ ├─────────────────────────────────────────────────────────────────┤  │
│ │                                                                   │  │
│ │ 1. Set owner reference (critical!):                              │  │
│ │    ctrl.SetControllerReference(&db, desiredStatefulSet, r.Scheme)│  │
│ │                                                                   │  │
│ │    This adds to StatefulSet metadata:                            │  │
│ │      ownerReferences:                                             │  │
│ │      - apiVersion: example.com/v1                                │  │
│ │        kind: Database                                            │  │
│ │        name: production-db                                       │  │
│ │        uid: abc-123-def-456                                      │  │
│ │        controller: true      ← Database "owns" this StatefulSet  │  │
│ │        blockOwnerDeletion: true                                  │  │
│ │                                                                   │  │
│ │    Why important?                                                 │  │
│ │    • Garbage collection: If Database deleted, StatefulSet auto-  │  │
│ │      deleted                                                      │  │
│ │    • Watch relationship: Controller watches owned resources      │  │
│ │                                                                   │  │
│ │ 2. Create StatefulSet:                                           │  │
│ │    r.Create(ctx, desiredStatefulSet)                             │  │
│ │                                                                   │  │
│ │    API Server receives:                                           │  │
│ │      POST /apis/apps/v1/namespaces/default/statefulsets          │  │
│ │                                                                   │  │
│ │    StatefulSet Controller (built-in) now watches this resource   │  │
│ │    and creates pods: production-db-0, production-db-1, -2        │  │
│ │                                                                   │  │
│ │ 3. Update Database status:                                       │  │
│ │    db.Status.Phase = "Creating"                                  │  │
│ │    db.Status.Conditions = append(..., Condition{                 │  │
│ │      Type: "StatefulSetCreated",                                 │  │
│ │      Status: "True",                                             │  │
│ │      Reason: "StatefulSetCreatedSuccessfully"                    │  │
│ │    })                                                             │  │
│ │    r.Status().Update(ctx, &db)                                   │  │
│ │                                                                   │  │
│ │ Result: StatefulSet created ✅                                    │  │
│ └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐  │
│ │ STEP 7B: UPDATE STATEFULSET (Branch B - future reconciles)       │  │
│ ├─────────────────────────────────────────────────────────────────┤  │
│ │                                                                   │  │
│ │ User updates Database spec: replicas: 3 → 5                      │  │
│ │                                                                   │  │
│ │ On next reconcile:                                                │  │
│ │   Desired replicas: 5 (from db.Spec.Replicas)                    │  │
│ │   Existing replicas: 3 (from existingStatefulSet.Spec.Replicas) │  │
│ │                                                                   │  │
│ │   IF desired != existing:                                         │  │
│ │     // Update needed                                              │  │
│ │     existingStatefulSet.Spec.Replicas = &desiredReplicas         │  │
│ │     r.Update(ctx, &existingStatefulSet)                          │  │
│ │                                                                   │  │
│ │     db.Status.Phase = "Scaling"                                  │  │
│ │     r.Status().Update(ctx, &db)                                  │  │
│ │                                                                   │  │
│ │   ELSE:                                                           │  │
│ │     // No changes needed, skip update                            │  │
│ │                                                                   │  │
│ │ StatefulSet Controller sees the update and scales pods           │  │
│ └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐  │
│ │ STEP 8: CREATE/UPDATE SERVICE                                    │  │
│ ├─────────────────────────────────────────────────────────────────┤  │
│ │                                                                   │  │
│ │ desiredService := r.buildService(&db)                            │  │
│ │   // Headless service for StatefulSet DNS                        │  │
│ │   Service{                                                        │  │
│ │     metadata: {name: "production-db", namespace: "default"},     │  │
│ │     spec: {                                                       │  │
│ │       clusterIP: "None",  // Headless                            │  │
│ │       selector: {"app": "production-db"},                        │  │
│ │       ports: [{port: 5432, targetPort: 5432}]                    │  │
│ │     }                                                             │  │
│ │   }                                                               │  │
│ │                                                                   │  │
│ │ Same logic as StatefulSet:                                        │  │
│ │   • Check if exists                                               │  │
│ │   • Create if missing, update if changed                         │  │
│ │   • Set owner reference                                           │  │
│ │                                                                   │  │
│ │ Result: Service enables DNS:                                      │  │
│ │   production-db.default.svc.cluster.local → Pod IPs             │  │
│ └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐  │
│ │ STEP 9: CHECK POD READINESS & UPDATE STATUS                      │  │
│ ├─────────────────────────────────────────────────────────────────┤  │
│ │                                                                   │  │
│ │ // Fetch the StatefulSet again to get current status             │  │
│ │ r.Get(ctx, ..., &existingStatefulSet)                            │  │
│ │                                                                   │  │
│ │ readyReplicas := existingStatefulSet.Status.ReadyReplicas        │  │
│ │ desiredReplicas := db.Spec.Replicas                              │  │
│ │                                                                   │  │
│ │ Decision:                                                         │  │
│ │   IF readyReplicas == desiredReplicas:                           │  │
│ │     // All pods are ready!                                        │  │
│ │     db.Status.Phase = "Running"                                  │  │
│ │     db.Status.ReadyReplicas = readyReplicas  // 3                │  │
│ │     db.Status.ConnectionString = fmt.Sprintf(                    │  │
│ │       "postgresql://production-db.default.svc.cluster.local:5432"│  │
│ │     )                                                             │  │
│ │     db.Status.Conditions = [..., Condition{                      │  │
│ │       Type: "Ready",                                             │  │
│ │       Status: "True",                                            │  │
│ │       Reason: "AllReplicasReady"                                 │  │
│ │     }]                                                            │  │
│ │   ELSE:                                                           │  │
│ │     // Still creating/updating                                    │  │
│ │     db.Status.Phase = "Creating"                                 │  │
│ │     db.Status.ReadyReplicas = readyReplicas  // 1 (partial)      │  │
│ │                                                                   │  │
│ │ r.Status().Update(ctx, &db)                                      │  │
│ └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────┐  │
│ │ STEP 10: RETURN RECONCILE RESULT                                 │  │
│ ├─────────────────────────────────────────────────────────────────┤  │
│ │                                                                   │  │
│ │ Decision: Should we requeue?                                      │  │
│ │                                                                   │  │
│ │   IF db.Status.Phase != "Running":                               │  │
│ │     // Pods still creating, check again soon                     │  │
│ │     return ctrl.Result{                                           │  │
│ │       RequeueAfter: 30 * time.Second                             │  │
│ │     }, nil                                                         │  │
│ │                                                                   │  │
│ │   ELSE IF db.Status.Phase == "Running":                          │  │
│ │     // Everything healthy, check less frequently                 │  │
│ │     return ctrl.Result{                                           │  │
│ │       RequeueAfter: 5 * time.Minute                              │  │
│ │     }, nil                                                         │  │
│ │                                                                   │  │
│ │   IF error occurred:                                              │  │
│ │     // Automatic exponential backoff                             │  │
│ │     return ctrl.Result{}, err                                     │  │
│ │     // Retries: 0s, 1s, 2s, 4s, 8s, 16s... (capped at ~1min)    │  │
│ │                                                                   │  │
│ │ Result in this scenario:                                          │  │
│ │   RequeueAfter: 30 seconds (pods still creating)                 │  │
│ │   Item "default/production-db" added back to queue with timer    │  │
│ └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│ } // End of Reconcile function                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 6: StatefulSet Controller Creates Pods

```
┌─────────────────────────────────────────────────────────────────┐
│ STATEFULSET CONTROLLER (Built-in K8s Controller)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Watches StatefulSets, sees new "production-db" StatefulSet     │
│                                                                 │
│ 1. Create pods sequentially (StatefulSet guarantee):           │
│                                                                 │
│    ┌─→ Create production-db-0:                                 │
│    │     • Request PVC: production-db-pvc-production-db-0      │
│    │     • Wait for PVC bound                                  │
│    │     • Schedule pod to node                                │
│    │     • Pull image: postgres:15                             │
│    │     • Start container                                     │
│    │     • Wait for ReadinessProbe ✅                          │
│    │     • Pod Ready after ~15 seconds                         │
│    │                                                            │
│    ├─→ Create production-db-1: (only after -0 is Ready)        │
│    │     • Same process...                                     │
│    │     • Pod Ready after ~15 seconds                         │
│    │                                                            │
│    └─→ Create production-db-2: (only after -1 is Ready)        │
│          • Same process...                                     │
│          • Pod Ready after ~15 seconds                         │
│                                                                 │
│ 2. Update StatefulSet status:                                  │
│    status.replicas: 3                                          │
│    status.readyReplicas: 3                                     │
│    status.currentReplicas: 3                                   │
│                                                                 │
│ 3. This status change triggers Database reconcile again!       │
│    (Controller watches owned StatefulSets)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

### Phase 7: Final Status Update

```
┌─────────────────────────────────────────────────────────────────┐
│ DATABASE RECONCILER - 2ND RECONCILE (Triggered by StatefulSet) │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Reconcile called again after 30 seconds (or StatefulSet change)│
│                                                                 │
│ Steps 1-8: Same as before                                      │
│   • Fetch Database CR                                          │
│   • StatefulSet already exists → No changes needed            │
│   • Service already exists → No changes needed                │
│                                                                 │
│ Step 9: Check pod readiness                                    │
│   readyReplicas: 3 (now all ready!)                           │
│   desiredReplicas: 3                                           │
│   ✅ Match! All pods running                                   │
│                                                                 │
│ Update status to Running:                                      │
│   db.Status.Phase = "Running"                                  │
│   db.Status.ReadyReplicas = 3                                  │
│   db.Status.ConnectionString =                                 │
│     "postgresql://production-db.default.svc.cluster.local:5432"│
│                                                                 │
│ Step 10: Return with longer requeue:                           │
│   return ctrl.Result{RequeueAfter: 5 * time.Minute}, nil      │
│   (Check less frequently when healthy)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

### Complete Timeline

```
T+0s:     kubectl apply -f production-db.yaml
T+0.1s:   API server validates & stores Database CR in etcd
T+0.2s:   Operator receives WATCH event → Reconcile triggered
T+0.5s:   Reconcile creates StatefulSet + Service
T+1s:     StatefulSet controller creates production-db-0 pod
T+15s:    production-db-0 ready
T+16s:    StatefulSet controller creates production-db-1 pod
T+30s:    production-db-1 ready
T+31s:    StatefulSet controller creates production-db-2 pod
T+45s:    production-db-2 ready
T+45.5s:  StatefulSet status updated → Reconcile triggered again
T+46s:    Database status updated to "Running" ✅
T+5m46s:  Reconcile runs again (periodic health check)
T+10m46s: Reconcile runs again (periodic health check)
...continues every 5 minutes
```

---

### Handling Updates (User Changes Spec)

```
User runs: kubectl edit database production-db
Changes: spec.replicas: 3 → 5

┌─────────────────────────────────────────────────────────────────┐
│ UPDATE FLOW                                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 1. API server updates Database CR in etcd                      │
│    generation: 1 → 2 (spec changed)                            │
│                                                                 │
│ 2. Operator receives UPDATE event → Reconcile triggered        │
│                                                                 │
│ 3. Reconcile function runs:                                    │
│    Desired: 5 replicas (from db.Spec)                          │
│    Existing StatefulSet: 3 replicas                            │
│                                                                 │
│    Decision: UPDATE NEEDED                                     │
│                                                                 │
│ 4. Update StatefulSet:                                         │
│    existingStatefulSet.Spec.Replicas = 5                       │
│    r.Update(ctx, &existingStatefulSet)                         │
│                                                                 │
│ 5. Update Database status:                                     │
│    db.Status.Phase = "Scaling"                                 │
│    r.Status().Update(ctx, &db)                                 │
│                                                                 │
│ 6. StatefulSet controller sees update:                         │
│    Creates production-db-3 → Ready                             │
│    Creates production-db-4 → Ready                             │
│                                                                 │
│ 7. Status change triggers reconcile again:                     │
│    readyReplicas: 5 == desiredReplicas: 5                     │
│    db.Status.Phase = "Running" ✅                              │
└─────────────────────────────────────────────────────────────────┘
```

---

### Deletion with Finalizers

```
User runs: kubectl delete database production-db

┌─────────────────────────────────────────────────────────────────┐
│ DELETION FLOW WITH FINALIZER                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ 1. API server sets DeletionTimestamp:                          │
│    metadata.deletionTimestamp: "2026-05-02T10:00:00Z"         │
│    Finalizers: ["example.com/db-finalizer"]                    │
│    ❌ Object NOT deleted yet (finalizer blocks it)             │
│                                                                 │
│ 2. Operator receives UPDATE event → Reconcile triggered        │
│                                                                 │
│ 3. Reconcile detects deletion:                                 │
│    IF !db.DeletionTimestamp.IsZero():                          │
│      // Object being deleted                                   │
│                                                                 │
│ 4. Run cleanup logic:                                          │
│    a) Take final backup:                                       │
│       - Connect to postgres pods                               │
│       - Run pg_dump                                            │
│       - Upload to S3/GCS                                       │
│                                                                 │
│    b) Deregister from service discovery                        │
│                                                                 │
│    c) Notify monitoring system                                 │
│                                                                 │
│    IF cleanup fails:                                           │
│      return error → Retry with backoff                         │
│      Database deletion blocked until cleanup succeeds          │
│                                                                 │
│ 5. Remove finalizer:                                           │
│    db.Finalizers = []  // Remove finalizer                     │
│    r.Update(ctx, &db)                                          │
│                                                                 │
│ 6. API server sees no finalizers:                             │
│    ✅ Now safe to delete Database CR                          │
│    Deletes from etcd                                           │
│                                                                 │
│ 7. Garbage collector deletes owned resources:                 │
│    • StatefulSet (has ownerReference)                          │
│    • Service (has ownerReference)                              │
│    • Pods (owned by StatefulSet, cascade delete)              │
│    • PVCs (optional, depends on retainPolicy)                 │
│                                                                 │
│ Result: Clean deletion with backup preserved ✅                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Operator SDK and Kubebuilder

### Build an Operator with Kubebuilder

```bash
# Initialize project
kubebuilder init --domain example.com --repo github.com/example/db-operator

# Create API (CRD + Controller scaffolding)
kubebuilder create api --group apps --version v1 --kind Database

# Generate manifests (CRD YAML, RBAC)
make manifests

# Run locally (connects to current kubeconfig)
make run

# Build and push Docker image
make docker-build docker-push IMG=registry.example.com/db-operator:v1.0.0

# Deploy to cluster
make deploy IMG=registry.example.com/db-operator:v1.0.0
```

### Project Structure

```
db-operator/
├── api/
│   └── v1/
│       ├── database_types.go     # CRD Go types
│       └── zz_generated.deepcopy.go
├── config/
│   ├── crd/                      # Generated CRD YAML
│   ├── rbac/                     # RBAC for controller
│   └── manager/                  # Controller deployment
├── controllers/
│   └── database_controller.go    # Reconcile logic
├── main.go                       # Manager setup
└── Makefile
```

## Operator Lifecycle Manager (OLM)

**Manage operator lifecycle** in production.

```
OperatorHub (catalog)
     │
     ├─→ CatalogSource: "operatorhub.io"
     │
     ├─→ PackageManifest: "db-operator"
     │       versions: [1.0.0, 1.1.0, 1.2.0]
     │
     ├─→ Subscription: "install latest stable"
     │       channel: stable
     │       package: db-operator
     │       autoInstall: true
     │
     ├─→ InstallPlan created
     │       action: Install
     │       clusterServiceVersion: db-operator.v1.2.0
     │
     └─→ ClusterServiceVersion (CSV) installed
              CRDs created
              RBAC configured
              Operator pod started
```

## Finalizers (Cleanup Logic)

**Prevent premature deletion** and run cleanup logic.

```go
const dbFinalizer = "example.com/db-finalizer"

func (r *DatabaseReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    var db examplev1.Database
    r.Get(ctx, req.NamespacedName, &db)
    
    // Object being deleted
    if !db.DeletionTimestamp.IsZero() {
        if containsString(db.Finalizers, dbFinalizer) {
            // Run cleanup
            if err := r.cleanupExternalResources(&db); err != nil {
                return ctrl.Result{}, err
            }
            // Remove finalizer
            db.Finalizers = removeString(db.Finalizers, dbFinalizer)
            r.Update(ctx, &db)
        }
        return ctrl.Result{}, nil
    }
    
    // Add finalizer if not present
    if !containsString(db.Finalizers, dbFinalizer) {
        db.Finalizers = append(db.Finalizers, dbFinalizer)
        r.Update(ctx, &db)
    }
    
    // Normal reconciliation...
}
```

**Deletion Flow with Finalizer:**
```
kubectl delete database production-db
     │
     ├─→ 1. API server sets DeletionTimestamp
     │       (Object NOT deleted yet)
     │
     ├─→ 2. Controller reconciles
     │       Sees DeletionTimestamp
     │
     ├─→ 3. Runs cleanup:
     │       - Take final backup
     │       - Deregister from external service discovery
     │       - Release cloud resources (if static PVs)
     │
     ├─→ 4. Removes finalizer from object
     │
     └─→ 5. API server deletes object (no more finalizers)
```

## Real-World Operator Examples

### Prometheus Operator

```yaml
# Creates Prometheus instance
apiVersion: monitoring.coreos.com/v1
kind: Prometheus
metadata:
  name: cluster-monitoring
spec:
  replicas: 2
  serviceMonitorSelector:  # Auto-discover ServiceMonitors
    matchLabels:
      app: backend
  retention: 30d
---
# Auto-discovered scrape target
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: backend-metrics
  labels:
    app: backend
spec:
  selector:
    matchLabels:
      app: backend
  endpoints:
  - port: metrics
    interval: 30s
```

### cert-manager Operator

```yaml
# Creates and auto-renews TLS certificates
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: api-tls
spec:
  secretName: api-tls-secret
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
  - api.example.com
```

### Strimzi Kafka Operator

```yaml
# Creates production Kafka cluster
apiVersion: kafka.strimzi.io/v1beta2
kind: Kafka
metadata:
  name: production-kafka
spec:
  kafka:
    version: 3.5.0
    replicas: 3
    storage:
      type: persistent-claim
      size: 100Gi
  zookeeper:
    replicas: 3
    storage:
      type: persistent-claim
      size: 10Gi
```

## Troubleshooting Operators

```bash
# Check CRD is installed
kubectl get crds | grep example.com

# Verify CR is valid
kubectl apply -f database.yaml --dry-run=server

# Check operator logs
kubectl logs -n operators deployment/db-operator

# View CR status
kubectl describe database production-db

# Check operator RBAC
kubectl get clusterroles | grep db-operator
kubectl describe clusterrole db-operator-manager

# Watch reconcile events
kubectl get events --field-selector involvedObject.kind=Database

# Get all custom resources
kubectl get databases -A
```

