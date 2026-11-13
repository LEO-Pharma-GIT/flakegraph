{{/* Return the chart-qualified base resource name. */}}
{{- define "flakegraph.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Select an explicitly named or chart-managed service account. */}}
{{- define "flakegraph.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "flakegraph.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/* Select a release-scoped or explicitly external Spark service account. */}}
{{- define "flakegraph.sparkServiceAccountName" -}}
{{- $name := .Values.spark.serviceAccount.name -}}
{{- if .Values.spark.serviceAccount.create -}}
{{- default (printf "%s-spark" (include "flakegraph.fullname" . | trunc 57 | trimSuffix "-")) $name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- default "default" $name -}}
{{- end -}}
{{- end -}}

{{/* Keep chart-owned Spark RBAC distinct across releases. */}}
{{- define "flakegraph.sparkRbacName" -}}
{{- printf "%s-spark" (include "flakegraph.fullname" . | trunc 57 | trimSuffix "-") -}}
{{- end -}}

{{/* Resolve either an immutable digest or a conventional repository tag. */}}
{{- define "flakegraph.image" -}}
{{- if .Values.image.digest -}}
{{- printf "%s@%s" .Values.image.repository .Values.image.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.repository (.Values.image.tag | default .Chart.AppVersion) -}}
{{- end -}}
{{- end -}}

{{/* The sidecar's image: the application image, at its own tag when one is pinned. */}}
{{- define "flakegraph.sidecarImage" -}}
{{- if .Values.modelServing.sidecar.imageTag -}}
{{- printf "%s:%s" .Values.image.repository .Values.modelServing.sidecar.imageTag -}}
{{- else -}}
{{- include "flakegraph.image" . -}}
{{- end -}}
{{- end -}}

{{/*
The console's image.

It ships in the application image, but the pods that share that tag restart
with it: every worker, the document-parsing shim, the embedding server's seed
step and the finalizer's cache seed - which ends an in-flight finalization, the
better part of three hours of LLM work on a real corpus. A console change is a
web page; it should be able to move on its own. Empty follows image.tag.
*/}}
{{- define "flakegraph.controlPlaneImage" -}}
{{- if .Values.controlPlane.imageTag -}}
{{- printf "%s:%s" .Values.image.repository .Values.controlPlane.imageTag -}}
{{- else -}}
{{- include "flakegraph.image" . -}}
{{- end -}}
{{- end -}}

{{/*
The balancer's image.

It shares its code with the sidecar, but not its restart cost: the sidecar
lives beside every engine, so moving that pin reloads a 27B checkpoint on each
machine in turn - the better part of two hours - while the balancer is one
stateless Deployment that restarts in seconds and whose caps expire on their
own meanwhile. Empty means: follow the sidecar's pin, which is the right
default when both are moving together.
*/}}
{{- define "flakegraph.balancerImage" -}}
{{- if .Values.modelServing.balancer.imageTag -}}
{{- printf "%s:%s" .Values.image.repository .Values.modelServing.balancer.imageTag -}}
{{- else -}}
{{- include "flakegraph.sidecarImage" . -}}
{{- end -}}
{{- end -}}

{{/*
Whether the fleet routes by lanes: the balancer labels engines, parsing
replicas and - through executor anti-affinity - steers Spark work by them.
Renders "true" or nothing, so it reads as a condition.
*/}}
{{- define "flakegraph.lanesEnabled" -}}
{{- if and .Values.modelServing.enabled .Values.modelServing.balancer.enabled .Values.modelServing.balancer.lanes.enabled -}}
true
{{- end -}}
{{- end -}}

{{/* The headless Service resolving only parsing replicas in the batch lane. */}}
{{- define "flakegraph.mineruBatchName" -}}
{{- printf "%s-batch" (include "flakegraph.mineruName" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
The request header carrying a request's lane. Lowercase: the picker lowercases
every header name it receives, so any other spelling would never match.
*/}}
{{- define "flakegraph.laneHeader" -}}
x-flakegraph-lane
{{- end -}}

{{/* The embedding server's in-cluster Service name. */}}
{{- define "flakegraph.embeddingServingName" -}}
{{- printf "%s-embedding" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* The OpenAI-compatible base the gateway forwards embedding calls to. */}}
{{- define "flakegraph.embeddingServingEndpoint" -}}
{{- printf "http://%s:%v/v1" (include "flakegraph.embeddingServingName" .) .Values.embeddingServing.service.port -}}
{{- end -}}

{{/* Return the stable in-cluster vLLM Service name. */}}
{{- define "flakegraph.modelServingName" -}}
{{- printf "%s-vllm" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Return the OpenAI-compatible endpoint consumed by FlakeGraph workers. */}}
{{- define "flakegraph.modelServingEndpoint" -}}
{{- printf "http://%s:%v/v1" (include "flakegraph.modelServingName" .) .Values.modelServing.service.port -}}
{{- end -}}

{{/* Describe the attention shape that sets KV cost per token, as sidecar JSON. */}}
{{- define "flakegraph.modelServingGeometry" -}}
{{- printf "{\"kv_heads\":%v,\"head_dim\":%v,\"attention_layers\":%v,\"recurrent_layers\":%v,\"recurrent_state_bytes_per_layer\":%v,\"recurrent_state_slots_per_sequence\":%v,\"kv_cache_dtype\":\"%s\",\"weights_bytes\":%v}" .Values.modelServing.sizing.kvHeads .Values.modelServing.sizing.headDim .Values.modelServing.sizing.attentionLayers .Values.modelServing.sizing.recurrentLayers .Values.modelServing.sizing.recurrentStateBytesPerLayer .Values.modelServing.sizing.recurrentStateSlotsPerSequence .Values.modelServing.server.kvCacheDtype (mulf .Values.modelServing.sizing.weightsGiB 1073741824 | int64) -}}
{{- end -}}

{{/* Describe the memory an engine may spend, as sidecar JSON. */}}
{{- define "flakegraph.modelServingDeviceBudget" -}}
{{- printf "{\"device_memory_bytes\":%v,\"gpu_memory_utilization\":%v,\"overhead_bytes\":%v}" (mulf .Values.modelServing.sizing.deviceMemoryGiB 1073741824 | int64) .Values.modelServing.server.gpuMemoryUtilization (mulf .Values.modelServing.sizing.overheadGiB 1073741824 | int64) -}}
{{- end -}}

{{/* Return the one OpenAI-compatible URL every consumer resolves. */}}
{{- define "flakegraph.gatewayName" -}}
{{- printf "%s-litellm" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "flakegraph.gatewayEndpoint" -}}
{{- printf "http://%s:%v/v1" (include "flakegraph.gatewayName" .) .Values.gateway.litellm.service.port -}}
{{- end -}}

{{/* Return the Envoy listener the gateway forwards inference to. */}}
{{- define "flakegraph.inferenceRouterName" -}}
{{- printf "%s-inference-router" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "flakegraph.inferenceRouterEndpoint" -}}
{{- printf "http://%s:%v/v1" (include "flakegraph.inferenceRouterName" .) .Values.gateway.placement.service.port -}}
{{- end -}}

{{/* Resolve an immutable digest or a tag for any component image block. */}}
{{- define "flakegraph.componentImage" -}}
{{- if .digest -}}
{{- printf "%s@%s" .repository .digest -}}
{{- else -}}
{{- printf "%s:%s" .repository .tag -}}
{{- end -}}
{{- end -}}

{{/* The Spark image: published beside the application image, so an empty tag
     follows the chart's appVersion the way image.tag does. */}}
{{- define "flakegraph.sparkImage" -}}
{{- if .Values.spark.image.digest -}}
{{- printf "%s@%s" .Values.spark.image.repository .Values.spark.image.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.spark.image.repository (.Values.spark.image.tag | default .Chart.AppVersion) -}}
{{- end -}}
{{- end -}}

{{/* Whether the in-cluster placement layer and the engines are what the gateway
     forwards to. An external upstream has no replicas to place across and no
     sidecar to stamp priority, so nothing of that plane is rendered for it. */}}
{{- define "flakegraph.engineUpstream" -}}
{{- if and .Values.gateway.enabled (eq .Values.gateway.litellm.upstream.type "engine") -}}true{{- end -}}
{{- end -}}

{{/* Refuse an optional CRD-backed object on a cluster without the CRD.

     Takes a dict: "root", "api" (group/version/Kind as Capabilities spells
     it), "feature" (the values key that asked for it) and "install" (what
     puts the CRD there). An object whose API the cluster does not serve is
     not degraded, it is an apply-time error reported after everything else
     has been applied; failing here names the cause once, first, and tells an
     offline render which --api-versions to pass. */}}
{{- define "flakegraph.requireApi" -}}
{{- if not (.root.Capabilities.APIVersions.Has .api) -}}
{{- fail (printf "%s needs %s on the cluster (%s). Install it first, or pass --api-versions %s when rendering offline." .feature .api .install .api) -}}
{{- end -}}
{{- end -}}

{{/* The single sign-in gate in front of every routed service. */}}
{{- define "flakegraph.authProxyName" -}}
{{- printf "%s-auth" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* The control plane is a deployed component, not a script on a node. */}}
{{- define "flakegraph.controlPlaneName" -}}
{{- printf "%s-app" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Return the OCR shim Service and the endpoint the pipeline parses through. */}}
{{- define "flakegraph.ocrShimName" -}}
{{- printf "%s-ocr" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "flakegraph.ocrShimEndpoint" -}}
{{- printf "http://%s:%v" (include "flakegraph.ocrShimName" .) .Values.documentParsing.shim.service.port -}}
{{- end -}}

{{- define "flakegraph.mineruName" -}}
{{- printf "%s-mineru" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* The coordinates every FlakeGraph consumer needs to reach the shared planes.

     Defined once because the bootstrap Job validates the very profile the
     workers run: configured differently, it either passes a profile the workers
     cannot execute or fails one they can. Both have drifted from each other
     before, in both directions. */}}
{{/*
The coordination store and artifact store, as every process that submits,
claims, or reads a run must reach them. Workers and the control plane share
this block so the CLI the console shells out to sees the same fleet the
workers do.
*/}}
{{- define "flakegraph.coordinationEnv" -}}
- name: KG_DISTRIBUTED_DATABASE_URL
  valueFrom:
    secretKeyRef:
      name: {{ .Values.database.secretName }}
      key: {{ .Values.database.secretKey }}
{{- if .Values.artifactStorage.uri }}
- name: KG_DISTRIBUTED_ARTIFACT_URI
  value: {{ .Values.artifactStorage.uri | quote }}
- name: KG_DISTRIBUTED_ARTIFACT_ENDPOINT_URL
  value: {{ .Values.artifactStorage.endpointUrl | quote }}
- name: KG_DISTRIBUTED_ARTIFACT_REGION
  value: {{ .Values.artifactStorage.region | quote }}
{{- if .Values.artifactStorage.existingSecret }}
- name: KG_DISTRIBUTED_ARTIFACT_ACCESS_KEY_ID
  valueFrom:
    secretKeyRef:
      name: {{ .Values.artifactStorage.existingSecret }}
      key: {{ .Values.artifactStorage.accessKeyKey }}
- name: KG_DISTRIBUTED_ARTIFACT_SECRET_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Values.artifactStorage.existingSecret }}
      key: {{ .Values.artifactStorage.secretKeyKey }}
{{- end }}
{{- end }}
{{- end -}}

{{- define "flakegraph.consumerEnv" -}}
{{- if .Values.gateway.enabled }}
- name: KG_LLM_ENDPOINT
  value: {{ include "flakegraph.gatewayEndpoint" . | quote }}
- name: KG_LLM_MODEL
  value: {{ .Values.gateway.litellm.aliases.batch | quote }}
- name: KG_LLM_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Values.gateway.litellm.virtualKeySecret.name }}
      key: {{ .Values.gateway.litellm.virtualKeySecret.batchKey }}
{{- if .Values.embeddingServing.enabled }}
{{- /*
Embedding on the GPU through the gateway instead of on each process's CPU. The
model name stays the checkpoint's own, so a graph records what embedded it
rather than an alias, and the gateway serves it under that name.
*/}}
- name: KG_EMBEDDING_PROVIDER
  value: openai_compatible
- name: KG_EMBEDDING_ENDPOINT
  value: {{ include "flakegraph.gatewayEndpoint" . | quote }}
- name: KG_EMBEDDING_MODEL
  value: {{ .Values.embeddingServing.model.name | quote }}
- name: KG_EMBEDDING_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Values.gateway.litellm.virtualKeySecret.name }}
      key: {{ .Values.gateway.litellm.virtualKeySecret.batchKey }}
{{- end }}
{{- end }}
{{- if .Values.documentParsing.enabled }}
- name: KG_MINERU_API_URL
  value: {{ include "flakegraph.ocrShimEndpoint" . | quote }}
- name: KG_MINERU_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ .Values.documentParsing.shim.keySecret.name }}
      key: {{ .Values.documentParsing.shim.keySecret.batchKey }}
{{- end }}
{{- end -}}

{{/* Use an existing config map when configuration is managed externally. */}}
{{- define "flakegraph.configMapName" -}}
{{- default (printf "%s-config" (include "flakegraph.fullname" .)) .Values.config.existingConfigMap -}}
{{- end -}}

{{/* Use an existing ontology ConfigMap or the chart-owned content object. */}}
{{- define "flakegraph.ontologyConfigMapName" -}}
{{- default (printf "%s-ontology" (include "flakegraph.fullname" .)) .Values.ontology.existingConfigMap -}}
{{- end -}}

{{/* Cluster-scoped scheduling names include the namespace to avoid release collisions. */}}
{{- define "flakegraph.priorityClassPrefix" -}}
{{- printf "%s-%s" .Release.Namespace (include "flakegraph.fullname" .) | trunc 54 | trimSuffix "-" -}}
{{- end -}}

{{- define "flakegraph.workerPriorityClassName" -}}
{{- printf "%s-worker" (include "flakegraph.priorityClassPrefix" .) -}}
{{- end -}}

{{- define "flakegraph.sparkPriorityClassName" -}}
{{- printf "%s-spark" (include "flakegraph.priorityClassPrefix" .) -}}
{{- end -}}

{{- define "flakegraph.servingPriorityClassName" -}}
{{- printf "%s-serving" (include "flakegraph.priorityClassPrefix" .) -}}
{{- end -}}

{{- define "flakegraph.modelPriorityClassName" -}}
{{- printf "%s-model" (include "flakegraph.priorityClassPrefix" .) -}}
{{- end -}}

{{/* Common ownership labels used by every namespaced object. */}}
{{- define "flakegraph.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | quote }}
{{- end -}}

{{/* The labels a selector matches on, and nothing else.

     Selectors are immutable once applied, and a pod template that carries
     helm.sh/chart rolls every pod on every chart bump. Both are avoided by
     keeping this triple apart from the ownership labels above. Takes a dict:
     "root" is the top-level context, "component" the pool being selected. */}}
{{- define "flakegraph.selectorLabels" -}}
app.kubernetes.io/name: {{ .root.Chart.Name }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{/* A CPU-utilisation HorizontalPodAutoscaler for one workload.

     Takes a dict: "root" is the top-level context, "name" the workload and the
     HPA, "kind" the workload's kind, "component" its selector component, and
     "autoscaling" the values block with min/max replicas and the CPU target. */}}
{{- define "flakegraph.cpuHpa" -}}
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ .name }}
  labels:
    {{- include "flakegraph.labels" .root | nindent 4 }}
    app.kubernetes.io/component: {{ .component }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: {{ .kind }}
    name: {{ .name }}
  minReplicas: {{ .autoscaling.minReplicas }}
  maxReplicas: {{ .autoscaling.maxReplicas }}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .autoscaling.targetCpuUtilizationPercentage }}
{{- end -}}

{{/* The Middleware that removes every identity-shaped request header. */}}
{{- define "flakegraph.stripIdentityName" -}}
{{- printf "%s-strip-identity" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* The gate's middlewares, as the annotation every gated Ingress carries.

     Order matters. Middlewares run in the order listed, outermost first, so
     the identity strip comes first: whatever X-Auth-Request-* or
     X-Forwarded-* header a client sent is gone before the gate runs, and the
     only such headers the service sees are the ones forwardAuth copies from
     the gate's answer. Then the errors middleware, which has to wrap the gate
     in order to see its refusal and turn it into a redirect - listed the
     other way round the gate still refuses, and the browser is handed a bare
     401. Defined once so every host behind the gate is gated the same way. */}}
{{- define "flakegraph.authProxyMiddlewares" -}}
{{- printf "%s-%s@kubernetescrd,%s-%s-errors@kubernetescrd,%s-%s-auth@kubernetescrd" .Release.Namespace (include "flakegraph.stripIdentityName" .) .Release.Namespace (include "flakegraph.authProxyName" .) .Release.Namespace (include "flakegraph.authProxyName" .) -}}
{{- end -}}

{{/* The Middleware that compresses responses on their way to a browser. */}}
{{- define "flakegraph.compressionName" -}}
{{- printf "%s-compress" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
The middleware chain on the ingress people use: the gate first, when there
is one, then compression nearest the service so every response it sends -
the console's, and the gate's own sign-in redirects - is compressed alike.
*/}}
{{- define "flakegraph.ingressMiddlewares" -}}
{{- $chain := list -}}
{{- if .Values.ingress.authProxy.enabled -}}
{{- $chain = append $chain (include "flakegraph.authProxyMiddlewares" .) -}}
{{- end -}}
{{- if .Values.ingress.compression.enabled -}}
{{- $chain = append $chain (printf "%s-%s@kubernetescrd" .Release.Namespace (include "flakegraph.compressionName" .)) -}}
{{- end -}}
{{- join "," $chain -}}
{{- end -}}

{{/* The gate, expressed for ingress-nginx: the controller asks the gate about
     every request through auth-url, sends a refused browser to auth-signin,
     and copies the identity headers from the gate's answer onto the request
     it forwards - replacing whatever a client sent under those names. Used
     by every gated Ingress the chart renders on that controller. */}}
{{- define "flakegraph.nginxAuthAnnotations" -}}
nginx.ingress.kubernetes.io/auth-url: {{ printf "http://%s.%s.svc.cluster.local:4180/oauth2/auth" (include "flakegraph.authProxyName" .) .Release.Namespace | quote }}
nginx.ingress.kubernetes.io/auth-signin: {{ printf "https://%s.%s/oauth2/start?rd=$scheme://$host$request_uri" .Values.ingress.authProxy.host .Values.ingress.domain | quote }}
nginx.ingress.kubernetes.io/auth-response-headers: "X-Auth-Request-User,X-Auth-Request-Email,X-Auth-Request-Preferred-Username,X-Auth-Request-Groups"
{{- end -}}

{{/* The annotations a gated Ingress carries, per controller. The gate is
     refused on a controller that cannot express it, because an Ingress that
     silently rendered without its gate would publish the console open. */}}
{{- define "flakegraph.gatedIngressAnnotations" -}}
{{- $ingress := .Values.ingress -}}
{{- if or $ingress.authProxy.enabled $ingress.compression.enabled -}}
{{- if eq $ingress.controller "traefik" }}
traefik.ingress.kubernetes.io/router.middlewares: {{ include "flakegraph.ingressMiddlewares" . }}
{{- else if eq $ingress.controller "nginx" }}
{{- if $ingress.compression.enabled }}
{{- fail "ingress.compression.enabled renders a Traefik Middleware; with ingress.controller=nginx compress at the controller (use-gzip in its ConfigMap) and leave this off." }}
{{- end }}
{{- include "flakegraph.nginxAuthAnnotations" . }}
{{- else }}
{{- fail (printf "ingress.controller=%s cannot express the sign-in gate or edge compression. Set ingress.controller to traefik or nginx, or disable ingress.authProxy and ingress.compression." $ingress.controller) }}
{{- end }}
{{- end }}
{{- end -}}

{{/* Every operator-created Secret the selected features reference, as a JSON
     list of {name, key, purpose} - the contract the pre-install check and the
     control plane's read grant are both derived from, so neither can name a
     Secret the other does not. A key is empty for a Secret whose whole
     contents are mounted. */}}
{{- define "flakegraph.referencedSecrets" -}}
{{- $v := .Values -}}
{{- $refs := list -}}
{{- if not $v.database.cloudNativePG.enabled -}}
{{- $refs = append $refs (dict "name" $v.database.secretName "key" $v.database.secretKey "purpose" "PostgreSQL URI (database.secretName)") -}}
{{- end -}}
{{- if and $v.providerSecret.name (not $v.providerSecret.optional) -}}
{{- $refs = append $refs (dict "name" $v.providerSecret.name "key" "" "purpose" "provider credentials (providerSecret.name; every key is optional)") -}}
{{- end -}}
{{- if $v.artifactStorage.existingSecret -}}
{{- $refs = append $refs (dict "name" $v.artifactStorage.existingSecret "key" $v.artifactStorage.accessKeyKey "purpose" "object storage access key (artifactStorage.existingSecret)") -}}
{{- $refs = append $refs (dict "name" $v.artifactStorage.existingSecret "key" $v.artifactStorage.secretKeyKey "purpose" "object storage secret key (artifactStorage.existingSecret)") -}}
{{- end -}}
{{- if $v.gateway.enabled -}}
{{- $vk := $v.gateway.litellm.virtualKeySecret -}}
{{- range $key := list $vk.masterKey $vk.saltKey $vk.batchKey -}}
{{- $refs = append $refs (dict "name" $vk.name "key" $key "purpose" "gateway keys (gateway.litellm.virtualKeySecret)") -}}
{{- end -}}
{{- if include "flakegraph.engineUpstream" . -}}
{{- $uk := $v.gateway.litellm.upstreamKeySecret -}}
{{- range $key := list $uk.interactiveKey $uk.devKey $uk.batchKey -}}
{{- $refs = append $refs (dict "name" $uk.name "key" $key "purpose" "engine sidecar keys the gateway presents (gateway.litellm.upstreamKeySecret)") -}}
{{- end -}}
{{- else if $v.gateway.litellm.upstream.apiKeySecret.name -}}
{{- $refs = append $refs (dict "name" $v.gateway.litellm.upstream.apiKeySecret.name "key" $v.gateway.litellm.upstream.apiKeySecret.key "purpose" "external provider API key (gateway.litellm.upstream.apiKeySecret)") -}}
{{- end -}}
{{- end -}}
{{- if $v.modelServing.enabled -}}
{{- $refs = append $refs (dict "name" $v.modelServing.sidecar.keySecret.name "key" $v.modelServing.sidecar.keySecret.key "purpose" "sidecar key-to-class map (modelServing.sidecar.keySecret)") -}}
{{- if and $v.modelServing.huggingFaceTokenSecret.name (not $v.modelServing.huggingFaceTokenSecret.optional) -}}
{{- $refs = append $refs (dict "name" $v.modelServing.huggingFaceTokenSecret.name "key" $v.modelServing.huggingFaceTokenSecret.key "purpose" "Hugging Face token (modelServing.huggingFaceTokenSecret)") -}}
{{- end -}}
{{- end -}}
{{- if $v.documentParsing.enabled -}}
{{- $refs = append $refs (dict "name" $v.documentParsing.shim.keySecret.name "key" $v.documentParsing.shim.keySecret.key "purpose" "parsing shim keyring (documentParsing.shim.keySecret)") -}}
{{- $refs = append $refs (dict "name" $v.documentParsing.shim.keySecret.name "key" $v.documentParsing.shim.keySecret.batchKey "purpose" "the workers' parsing key (documentParsing.shim.keySecret.batchKey)") -}}
{{- end -}}
{{- if and $v.controlPlane.enabled $v.controlPlane.ask.secretName -}}
{{- $refs = append $refs (dict "name" $v.controlPlane.ask.secretName "key" $v.controlPlane.ask.secretKey "purpose" "the console's gateway key (controlPlane.ask.secretName)") -}}
{{- end -}}
{{- if and $v.ingress.enabled $v.ingress.authProxy.enabled -}}
{{- $refs = append $refs (dict "name" $v.ingress.authProxy.existingSecret "key" $v.ingress.authProxy.clientSecretKey "purpose" "OIDC client secret (ingress.authProxy.existingSecret)") -}}
{{- $refs = append $refs (dict "name" $v.ingress.authProxy.existingSecret "key" $v.ingress.authProxy.cookieSecretKey "purpose" "gate cookie secret (ingress.authProxy.existingSecret)") -}}
{{- end -}}
{{- if and $v.ingress.enabled $v.ingress.tls.secretName -}}
{{- $refs = append $refs (dict "name" $v.ingress.tls.secretName "key" "tls.crt" "purpose" "TLS certificate (ingress.tls.secretName)") -}}
{{- end -}}
{{- $refs | toJson -}}
{{- end -}}

{{/* The distinct Secret names above, plus the ones the chart creates itself
     that the console may still need to read (the CloudNativePG application
     Secret, and the provider Secret even when optional). */}}
{{- define "flakegraph.referencedSecretNames" -}}
{{- $names := list -}}
{{- range include "flakegraph.referencedSecrets" . | fromJsonArray -}}
{{- $names = append $names .name -}}
{{- end -}}
{{- $names = append $names .Values.database.secretName -}}
{{- with .Values.providerSecret.name -}}
{{- $names = append $names . -}}
{{- end -}}
{{- $names | uniq | toJson -}}
{{- end -}}

{{/* Names produced by the kube-prometheus-stack release monitoring addresses. */}}
{{- define "flakegraph.grafanaServiceName" -}}
{{- printf "%s-grafana" .Values.monitoring.release | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* The namespace the monitoring stack runs in; empty means this release's. */}}
{{- define "flakegraph.monitoringNamespace" -}}
{{- .Values.monitoring.namespace | default .Release.Namespace -}}
{{- end -}}

{{/* Whether the stack shares this release's namespace, which is what lets the
     chart put Grafana behind its own gate: an Ingress backs only onto a Service
     in its own namespace. */}}
{{- define "flakegraph.monitoringInNamespace" -}}
{{- if eq (include "flakegraph.monitoringNamespace" .) .Release.Namespace -}}true{{- end -}}
{{- end -}}

{{/* The NetworkPolicy peer that is the stack's Prometheus, wherever it runs. */}}
{{- define "flakegraph.prometheusPeer" -}}
{{- if include "flakegraph.monitoringInNamespace" . }}
- podSelector:
    matchLabels:
      app.kubernetes.io/name: prometheus
{{- else }}
- namespaceSelector:
    matchLabels:
      kubernetes.io/metadata.name: {{ include "flakegraph.monitoringNamespace" . }}
  podSelector:
    matchLabels:
      app.kubernetes.io/name: prometheus
{{- end }}
{{- end -}}

{{/* The ConfigMap of SQL the CloudNativePG exporter runs as metrics. */}}
{{- define "flakegraph.postgresQueriesName" -}}
{{- printf "%s-postgres-queries" (include "flakegraph.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Refuse to render monitoring objects on a cluster that cannot hold them.

     Without the operator's CRDs a ServiceMonitor is not a degraded object, it
     is an install error at apply time, reported per object and after everything
     else has been applied. Failing here names the cause once, first. */}}
{{- define "flakegraph.monitoringRequired" -}}
{{- if not (.Capabilities.APIVersions.Has "monitoring.coreos.com/v1/ServiceMonitor") -}}
{{- fail "monitoring.enabled needs the Prometheus operator CRDs (monitoring.coreos.com/v1). Install a kube-prometheus-stack release on the cluster first (deploy/spark/install-cluster.sh shows one way), or pass --api-versions monitoring.coreos.com/v1/ServiceMonitor,monitoring.coreos.com/v1/PodMonitor,monitoring.coreos.com/v1/PrometheusRule when rendering offline." -}}
{{- end -}}
{{- end -}}

{{/*
Provider credentials from the operator's Secret, for every pod that reads a
source or calls a provider: the workers, and the control plane that lists
sources and runs preflight on their behalf.
*/}}
{{- define "flakegraph.providerSecretEnv" -}}
{{- if .Values.providerSecret.name }}
{{- range $mapping := .Values.providerSecret.env }}
{{- /*
When the gateway is enabled its virtual key *is* the LLM credential, and it
was already emitted by consumerEnv. Declaring the same variable twice leaves
which one survives up to the kubelet's ordering rather than the chart's
intent, and an absent optional key can blank a value the gateway just set.
Skip the duplicate instead.
*/ -}}
{{- if not (and $.Values.gateway.enabled (eq $mapping.name "KG_LLM_API_KEY")) }}
- name: {{ $mapping.name }}
  valueFrom:
    secretKeyRef:
      name: {{ $.Values.providerSecret.name }}
      key: {{ $mapping.key }}
      optional: {{ or $.Values.providerSecret.optional $mapping.optional }}
{{- end }}
{{- end }}
{{- end }}
{{- end -}}

{{/*
The most pods a worker pool scales to. A pool that names maxReplicas gets it.
One that leaves it empty is sized from the model sequences it feeds - the
engines this release runs, or the external provider's declared share - so that
its slots, all together, oversubscribe those sequences by the declared factor:

  ceil(sequences x sequenceOversubscription / slots)

Called with (dict "root" $ "name" $poolName "pool" $pool).
*/}}
{{- define "flakegraph.workerMaxReplicas" -}}
{{- $pool := .pool -}}
{{- $scaling := $pool.autoscaling -}}
{{- if $scaling.maxReplicas -}}
{{- $scaling.maxReplicas | int -}}
{{- else -}}
{{- $sequences := 0 -}}
{{- if .root.Values.modelServing.enabled -}}
{{- $sequences = mul (int .root.Values.modelServing.replicas) (int .root.Values.modelServing.server.maxNumSeqs) -}}
{{- else -}}
{{- $sequences = int ($scaling.providerSequences | default 0) -}}
{{- end -}}
{{- $oversubscription := $scaling.sequenceOversubscription | default 0 -}}
{{- if not (and (gt $sequences 0) $oversubscription) -}}
{{- fail (printf "workers.%s.autoscaling.maxReplicas is empty, and the chart has no model sequences to size the pool from: set it, or set sequenceOversubscription with modelServing enabled or providerSequences" .name) -}}
{{- end -}}
{{- max 1 (ceil (divf (mulf $sequences $oversubscription) (int $pool.slots)) | int) -}}
{{- end -}}
{{- end -}}

{{/*
One --serve-stage flag per stage of every enabled worker pool, for the
database-bootstrap Job to declare what this release's fleet serves.
*/}}
{{- define "flakegraph.servedStageFlags" -}}
{{- range $poolName, $pool := .Values.workers }}
{{- if $pool.enabled }}
{{- range $stage := $pool.stages }} --serve-stage {{ $stage }}{{- end }}
{{- end }}
{{- end }}
{{- end }}

{{/*
The engine's resource block, with its device reservation charged to the
scheduler on unified-memory hardware.

On a discrete GPU the engine's weights and KV cache live in device memory the
kubelet never hands out, so the pod's memory request covers only the host-side
process and the block is used as written. On a unified-memory part - GB10,
Grace-Hopper - that same allocation comes out of the one pool the kubelet is
handing to every other pod, and it appears in neither the pod's cgroup nor the
kubelet's accounting. The scheduler therefore reads a node holding a 69 GiB
engine as nearly empty, packs workers and Spark executors onto it, and the
kernel - not the scheduler - decides who dies, by global OOM, with the cluster's
own datastore on the node as a candidate.

Naming the pool's size here charges gpuMemoryUtilization of it, per device, to
both the request and the limit, on top of whatever the block already asks for
the process itself. The limit is charged as well because it is the same RAM:
leaving it as written would put the pod's limit below its own request, which
the API server rejects outright. Zero means discrete-GPU accounting.
*/}}
{{- define "flakegraph.modelServingResources" -}}
{{- $resources := deepCopy .Values.modelServing.resources -}}
{{- $pool := .Values.modelServing.server.unifiedMemoryPerDeviceGi | default 0 | int -}}
{{- if gt $pool 0 -}}
{{- $devices := .Values.modelServing.server.tensorParallelSize | default 1 | int -}}
{{- $charged := ceil (mulf (float64 $pool) (float64 .Values.modelServing.server.gpuMemoryUtilization) (float64 $devices)) | int -}}
{{- range $section := list "requests" "limits" -}}
{{- $declared := (dig $section "memory" "" $resources) -}}
{{- if $declared -}}
{{- $host := include "flakegraph.gibibytes" $declared | int -}}
{{- $_ := set (get $resources $section) "memory" (printf "%dGi" (add $host $charged)) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- toYaml $resources -}}
{{- end -}}

{{/*
A Kubernetes memory quantity in whole gibibytes, rounded up. Only the suffixes
a resource block can carry are accepted; anything else is a values error rather
than a silently wrong reservation.
*/}}
{{- define "flakegraph.gibibytes" -}}
{{- $value := . | toString -}}
{{- if hasSuffix "Gi" $value -}}
{{- ceil (float64 (trimSuffix "Gi" $value)) | int -}}
{{- else if hasSuffix "Mi" $value -}}
{{- divf (float64 (trimSuffix "Mi" $value)) 1024.0 | ceil | int -}}
{{- else if hasSuffix "G" $value -}}
{{- divf (mulf (float64 (trimSuffix "G" $value)) 1000000000.0) 1073741824.0 | ceil | int -}}
{{- else if hasSuffix "M" $value -}}
{{- divf (mulf (float64 (trimSuffix "M" $value)) 1000000.0) 1073741824.0 | ceil | int -}}
{{- else if regexMatch "^[0-9]+$" $value -}}
{{- divf (float64 $value) 1073741824.0 | ceil | int -}}
{{- else -}}
{{- fail (printf "modelServing.resources.requests.memory: %q is not a memory quantity this chart can charge a device reservation against; use Gi, Mi, G, M or bytes" $value) -}}
{{- end -}}
{{- end -}}

{{/* Preserve the driver's actual provider Secret references for Spark executors. */}}
{{- define "flakegraph.sparkProviderSecretRefs" -}}
{{- $refs := dict -}}
{{- if .Values.providerSecret.name -}}
{{- range $mapping := .Values.providerSecret.env -}}
{{- $_ := set $refs $mapping.name (printf "%s:%s" $.Values.providerSecret.name $mapping.key) -}}
{{- end -}}
{{- end -}}
{{- if .Values.gateway.enabled -}}
{{- $gatewayKey := printf "%s:%s" .Values.gateway.litellm.virtualKeySecret.name .Values.gateway.litellm.virtualKeySecret.batchKey -}}
{{- $_ := set $refs "KG_LLM_API_KEY" $gatewayKey -}}
{{- if .Values.embeddingServing.enabled -}}
{{- $_ := set $refs "KG_EMBEDDING_API_KEY" $gatewayKey -}}
{{- end -}}
{{- end -}}
{{- $refs | toJson -}}
{{- end -}}
