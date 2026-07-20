# CropGuard Kanban bootstrap
# Creates a GitHub Project board linked to the repo, with all project tasks
# (Spanish titles + undergraduate-level technical descriptions) as issues,
# placed in Done / In progress columns.
#
# PREREQUISITE (one time, interactive):
#   gh auth login -s project
#
# Then run:  powershell -File scripts\create_kanban.ps1

$ErrorActionPreference = "Stop"
$gh = "C:\Program Files\GitHub CLI\gh.exe"
$repo = "jVeyZ/AI_SAMSUNG-CapstoneProy"
$projectTitle = "CropGuard - Plan de desarrollo"

# ---------------------------------------------------------------- tasks ----
# status: done | progress
$tasks = @(
    @{ t = "Descargar y organizar los datasets de los tres cultivos"; s = "done";
       b = "Script ``setup.py`` con ``kagglehub`` para tomate (PlantVillage) y arroz (Kaggle). La naranja requiere descarga manual desde Mendeley (Cloudflare bloquea la automatizacion). Renombrado de las carpetas de clase a un formato comun." },
    @{ t = "Construir el pipeline de entrenamiento inicial con Keras"; s = "done";
       b = "ResNet50 congelada como extractor de caracteristicas (1000-D) y cabezales densos por cultivo con Keras sobre backend torch. Split 75/15/10 con semilla fija." },
    @{ t = "Crear la app web de demostracion con Streamlit"; s = "done";
       b = "Subida de foto de la hoja, selector de cultivo, diagnostico con barra de confianza y grafica matplotlib de probabilidades por clase." },
    @{ t = "Integrar consejos de tratamiento con LLM y voz"; s = "done";
       b = "Consejos de tratamiento generados con la API de Groq (llama-3.3-70b) y lectura en voz alta con gTTS en espanol. Degradacion elegante si faltan paquetes." },
    @{ t = "Corregir el orden de clases de arroz en crop_config.py"; s = "done";
       b = "ImageFolder asigna indices de clase alfabeticamente; la lista de arroz no seguia ese orden y la app mostraba nombres de enfermedad incorrectos en los indices 3-7." },
    @{ t = "Corregir la fuga de data augmentation en validacion y test"; s = "done";
       b = "El transform de augmentation se aplicaba al dataset compartido, asi que val y test tambien se evaluaban con imagenes aumentadas. Solucion: dos ImageFolder (train/eval) partidos con la misma semilla." },
    @{ t = "Mejorar las caracteristicas a 2048 dimensiones"; s = "done";
       b = "Se elimino la capa fc de ResNet50 para usar el avgpool (2048-D) en lugar de los logits de ImageNet, y se anadio una capa de Normalization dentro del cabezal. Arroz: 61.6% a 74.5%." },
    @{ t = "Reestructurar las figuras de resultados por cultivo"; s = "done";
       b = "De ``results/figures/`` a ``results/<cultivo>/`` con 7 figuras: historial, matriz de confusion (cuenta y normalizada), metricas por clase, histograma de confianza, muestras mal clasificadas y distribucion de clases." },
    @{ t = "Implementar fine-tuning de layer4 en PyTorch puro"; s = "done";
       b = "Dos fases por cultivo: calentamiento del cabezal (backbone congelada) y fine-tuning de layer4 con LR bajo, label smoothing, class weights y early stopping. Resultados: arroz 96.25%, tomate 99.0%, naranja 98.6%." },
    @{ t = "Migrar la inferencia a PyTorch puro y eliminar Keras"; s = "done";
       b = "``model_def.py`` compartido entre entrenamiento e inferencia (arquitectura + transforms con normalizacion ImageNet). app.py y predict_worker.py ya no dependen de Keras/TensorFlow." },
    @{ t = "Documentar el proyecto en AGENTS.md"; s = "done";
       b = "Guia para futuras sesiones: comandos, arquitectura, trampas conocidas (split con misma semilla, guard de main en Windows, deprecation de use_container_width)." },
    @{ t = "Crear el backend REST con FastAPI"; s = "progress";
       b = "``server.py`` con endpoints ``/predict`` (multipart), ``/crops``, ``/treatment`` y ``/chat``. Modelos cargados perezosamente en CPU; CORS abierto; ``CROPGUARD_MODELS_DIR`` permite inyectar modelos de prueba." },
    @{ t = "Redactar las fichas de tratamiento en tres idiomas"; s = "progress";
       b = "``treatments.json``: explicacion, sintomas, tratamiento y prevencion para las 25 enfermedades en ingles, espanol y valenciano. Es el consejo por defecto (no requiere IA)." },
    @{ t = "Anadir preguntas de seguimiento con IA gratuita"; s = "progress";
       b = "``llm_advice.py`` usa Gemini (gemini-2.0-flash, nivel gratuito, ``GEMINI_API_KEY``) con el contenido estatico como contexto de agronomo experto. Sin clave, devuelve el contenido guardado." },
    @{ t = "Desarrollar la app Android con Jetpack Compose"; s = "progress";
       b = "Modulo ``android/``: pantalla de captura (camara/galeria), pantalla de resultado con barra de confianza animada, tarjeta de tratamiento y chat de seguimiento. Retrofit + OkHttp contra el backend." },
    @{ t = "Implementar el selector de idioma EN/ES/Valenciano"; s = "progress";
       b = "Diccionario en memoria (evita recrear la Activity y perder el estado) con preferencia persistida en SharedPreferences. El backend sirve el contenido en el idioma elegido (parametro lang)." },
    @{ t = "Escribir tests unitarios con pytest"; s = "progress";
       b = "29 tests en ``tests/unit``: alineacion de clases con ImageFolder, model_def, completitud de treatments.json y endpoints con modelos de pesos aleatorios (sin modelos reales de 92 MB)." },
    @{ t = "Escribir tests e2e del flujo completo de diagnostico"; s = "progress";
       b = "``tests/e2e``: levanta un servidor uvicorn real y recorre el flujo por HTTP: /crops -> /predict -> /treatment (valenciano) -> /chat con fallback." },
    @{ t = "Configurar CI con GitHub Actions"; s = "progress";
       b = "``.github/workflows/ci.yml``: job Python (pytest unit + e2e con torch CPU) y job Android (testDebugUnitTest + assembleDebug con JDK 17). Sin modelos ni claves en CI." },
    @{ t = "Crear el tablero Kanban del proyecto"; s = "progress";
       b = "Tablero GitHub Projects vinculado al repo con todas las tareas desde el inicio del proyecto, en espanol. Creado con ``gh`` CLI (este script)." }
)

# ------------------------------------------------------------ 1. labels ----
& $gh label create completada --repo $repo --color 2da44e --description "Tarea finalizada" 2>$null
& $gh label create expansion --repo $repo --color 0969da --description "Expansion Android + backend + CI" 2>$null

# ------------------------------------------------------------ 2. issues ----
$created = @()
foreach ($task in $tasks) {
    $label = if ($task.s -eq "done") { "completada" } else { "expansion" }
    Write-Host "Creando issue: $($task.t)"
    $url = & $gh issue create --repo $repo --title $task.t --body $task.b --label $label
    $created += @{ url = $url; status = $task.s }
    if ($task.s -eq "done") {
        & $gh issue close $url --repo $repo | Out-Null
    }
}

# ----------------------------------------------------------- 3. project ----
Write-Host "Creando proyecto: $projectTitle"
& $gh project create --owner "@me" --title $projectTitle | Out-Null
$proj = (& $gh project list --owner "@me" --format json | ConvertFrom-Json).projects |
        Where-Object { $_.title -eq $projectTitle } | Select-Object -First 1

# Link project to the repository
& $gh project link $proj.number --owner "@me" --repo $repo 2>$null

$fields = & $gh project field-list $proj.number --owner "@me" --format json | ConvertFrom-Json
$statusField = $fields.fields | Where-Object { $_.name -eq "Status" }
$doneOpt = $statusField.options | Where-Object { $_.name -eq "Done" }
$progOpt = $statusField.options | Where-Object { $_.name -eq "In progress" }

# ------------------------------------------------- 4. add items + status ----
foreach ($item in $created) {
    Write-Host "Anadiendo al tablero: $($item.url)"
    $itemId = (& $gh project item-add $proj.number --owner "@me" --url $item.url --format json | ConvertFrom-Json).id
    $opt = if ($item.status -eq "done") { $doneOpt.id } else { $progOpt.id }
    & $gh project item-edit --project-id $proj.id --id $itemId --field-id $statusField.id --single-select-option-id $opt | Out-Null
}

Write-Host ""
Write-Host "Tablero creado: $($proj.url)"
Write-Host "Hecho: $(($created | Where-Object { $_.status -eq 'done' }).Count) tareas | En curso: $(($created | Where-Object { $_.status -eq 'progress' }).Count) tareas"
