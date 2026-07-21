package com.cropguard.app.ui

/**
 * In-memory i18n for EN / ES / VA (Valencian).
 *
 * Dictionary-based instead of res/values-XX on purpose: switching Android
 * resource locales recreates the Activity and would wipe the ViewModel state
 * (selected image, diagnosis). The dictionary approach recomposes instantly
 * and maps 1:1 to the backend `lang` parameter (en/es/va).
 */
object L {

    fun t(key: String, lang: String): String =
        STRINGS[lang]?.get(key) ?: STRINGS["en"]!!.getValue(key)

    fun t(key: String, lang: String, crop: String): String {
        val part = cropPart(crop)
        val cropKey = "${key}_$part"
        return STRINGS[lang]?.get(cropKey) ?: STRINGS["en"]?.get(cropKey)
            ?: t(key, lang)
    }

    fun cropName(crop: String, lang: String): String =
        STRINGS[lang]?.get("crop_${crop.lowercase()}") ?: crop

    /** Orange dataset uses fruit images; others use leaf. */
    fun cropPart(crop: String): String = if (crop.equals("Orange", ignoreCase = true)) "fruit" else "leaf"

    fun disease(crop: String, disease: String, lang: String): String {
        val slug = disease.lowercase()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        val key = "disease_${crop.lowercase()}_$slug"
        return STRINGS[lang]?.get(key) ?: disease
    }

    private val UI_EN = mapOf(
        "select_crop" to "Select crop",
        "take_photo" to "Camera",
        "choose_gallery" to "Gallery",
        "analyze_leaf" to "Analyze leaf",
        "analyze_fruit" to "Analyze fruit",
        "analyzing" to "Analyzing…",
        "diagnosis_result" to "Diagnosis",
        "confidence" to "confidence",
        "treatment_advice" to "Treatment advice",
        "symptoms" to "Symptoms",
        "treatment" to "Treatment",
        "prevention" to "Prevention",
        "followup_hint" to "Ask a follow-up question…",
        "ask_ai" to "Send",
        "ai_unavailable" to "AI unavailable — showing stored recommendations.",
        "error_network" to "Network error — is the backend server running?",
        "pick_image_leaf" to "Pick a leaf photo first",
        "pick_image_fruit" to "Pick a fruit photo first",
        "language" to "Language",
        "no_treatment" to "No stored recommendations for this disease.",
        "you" to "You",
        "agronomist_ai" to "Agronomist AI",
        "crop_tomato" to "Tomato",
        "crop_rice" to "Rice",
        "crop_orange" to "Orange",
        "settings" to "Settings",
        "ai_provider" to "AI Provider",
        "ai_provider_desc" to "Choose which AI service answers follow-up questions.",
        "provider_gemini" to "Google Gemini",
        "provider_opencode" to "OpenCode",
        "server_url" to "Server URL",
        "server_url_hint" to "http://192.168.1.101:8000",
        "server_url_desc" to "IP address and port of the CropGuard backend server.",
        "save" to "Save",
        "saved" to "Saved",
    )

    private val UI_ES = mapOf(
        "select_crop" to "Selecciona cultivo",
        "take_photo" to "Cámara",
        "choose_gallery" to "Galería",
        "analyze_leaf" to "Analizar hoja",
        "analyze_fruit" to "Analizar fruta",
        "analyzing" to "Analizando…",
        "diagnosis_result" to "Diagnóstico",
        "confidence" to "de confianza",
        "treatment_advice" to "Consejos de tratamiento",
        "symptoms" to "Síntomas",
        "treatment" to "Tratamiento",
        "prevention" to "Prevención",
        "followup_hint" to "Haz una pregunta de seguimiento…",
        "ask_ai" to "Enviar",
        "ai_unavailable" to "IA no disponible — mostrando recomendaciones guardadas.",
        "error_network" to "Error de red — ¿está el servidor en marcha?",
        "pick_image_leaf" to "Elige primero una foto de la hoja",
        "pick_image_fruit" to "Elige primero una foto de la fruta",
        "language" to "Idioma",
        "no_treatment" to "No hay recomendaciones guardadas para esta enfermedad.",
        "you" to "Tú",
        "agronomist_ai" to "IA agrónoma",
        "crop_tomato" to "Tomate",
        "crop_rice" to "Arroz",
        "crop_orange" to "Naranja",
        "settings" to "Configuración",
        "ai_provider" to "Proveedor de IA",
        "ai_provider_desc" to "Elige qué servicio de IA responde preguntas de seguimiento.",
        "provider_gemini" to "Google Gemini",
        "provider_opencode" to "OpenCode",
        "server_url" to "URL del servidor",
        "server_url_hint" to "http://192.168.1.101:8000",
        "server_url_desc" to "Dirección IP y puerto del servidor backend de CropGuard.",
        "save" to "Guardar",
        "saved" to "Guardado",
    )

    private val UI_VA = mapOf(
        "select_crop" to "Selecciona cultiu",
        "take_photo" to "Càmera",
        "choose_gallery" to "Galeria",
        "analyze_leaf" to "Analitza fulla",
        "analyze_fruit" to "Analitza fruita",
        "analyzing" to "Analitzant…",
        "diagnosis_result" to "Diagnòstic",
        "confidence" to "de confiança",
        "treatment_advice" to "Consells de tractament",
        "symptoms" to "Símptomes",
        "treatment" to "Tractament",
        "prevention" to "Prevenció",
        "followup_hint" to "Fes una pregunta de seguiment…",
        "ask_ai" to "Envia",
        "ai_unavailable" to "IA no disponible — mostrant recomanacions guardades.",
        "error_network" to "Error de xarxa — el servidor està en marxa?",
        "pick_image_leaf" to "Tria primer una foto de la fulla",
        "pick_image_fruit" to "Tria primer una foto de la fruita",
        "language" to "Idioma",
        "no_treatment" to "No hi ha recomanacions guardades per a esta malaltia.",
        "you" to "Tu",
        "agronomist_ai" to "IA agrònoma",
        "crop_tomato" to "Tomàtiga",
        "crop_rice" to "Arròs",
        "crop_orange" to "Taronja",
        "settings" to "Configuració",
        "ai_provider" to "Proveïdor d'IA",
        "ai_provider_desc" to "Tria quin servei d'IA respon preguntes de seguiment.",
        "provider_gemini" to "Google Gemini",
        "provider_opencode" to "OpenCode",
        "server_url" to "URL del servidor",
        "server_url_hint" to "http://192.168.1.101:8000",
        "server_url_desc" to "Adreça IP i port del servidor backend de CropGuard.",
        "save" to "Guarda",
        "saved" to "Guardat",
    )

    private val DISEASES_EN = mapOf(
        "disease_tomato_bacterial_spot" to "Bacterial Spot",
        "disease_tomato_early_blight" to "Early Blight",
        "disease_tomato_healthy" to "Healthy",
        "disease_tomato_late_blight" to "Late Blight",
        "disease_tomato_leaf_mold" to "Leaf Mold",
        "disease_tomato_mosaic_virus" to "Mosaic Virus",
        "disease_tomato_septoria_leaf_spot" to "Septoria Leaf Spot",
        "disease_tomato_spider_mites" to "Spider Mites",
        "disease_tomato_target_spot" to "Target Spot",
        "disease_tomato_yellow_leaf_curl_virus" to "Yellow Leaf Curl Virus",
        "disease_rice_bacterial_leaf_blight" to "Bacterial Leaf Blight",
        "disease_rice_bacterial_leaf_streak" to "Bacterial Leaf Streak",
        "disease_rice_bacterial_panicle_blight" to "Bacterial Panicle Blight",
        "disease_rice_brown_spot" to "Brown Spot",
        "disease_rice_dead_heart" to "Dead Heart",
        "disease_rice_downy_mildew" to "Downy Mildew",
        "disease_rice_healthy_rice" to "Healthy Rice",
        "disease_rice_rice_blast" to "Rice Blast",
        "disease_rice_rice_hispa" to "Rice Hispa",
        "disease_rice_tungro" to "Tungro",
        "disease_orange_black_spot" to "Black Spot",
        "disease_orange_canker" to "Canker",
        "disease_orange_greening_hlb" to "Greening (HLB)",
        "disease_orange_healthy_orange" to "Healthy Orange",
        "disease_orange_scab" to "Scab",
    )

    private val DISEASES_ES = mapOf(
        "disease_tomato_bacterial_spot" to "Mancha bacteriana",
        "disease_tomato_early_blight" to "Tizón temprano",
        "disease_tomato_healthy" to "Sana",
        "disease_tomato_late_blight" to "Tizón tardío",
        "disease_tomato_leaf_mold" to "Moho de la hoja",
        "disease_tomato_mosaic_virus" to "Virus del mosaico",
        "disease_tomato_septoria_leaf_spot" to "Mancha foliar por Septoria",
        "disease_tomato_spider_mites" to "Araña roja",
        "disease_tomato_target_spot" to "Mancha diana",
        "disease_tomato_yellow_leaf_curl_virus" to "Virus del rizado amarillo",
        "disease_rice_bacterial_leaf_blight" to "Tizón bacteriano de la hoja",
        "disease_rice_bacterial_leaf_streak" to "Raya bacteriana de la hoja",
        "disease_rice_bacterial_panicle_blight" to "Tizón bacteriano de la panícula",
        "disease_rice_brown_spot" to "Mancha marrón",
        "disease_rice_dead_heart" to "Corazón muerto",
        "disease_rice_downy_mildew" to "Mildiu velloso",
        "disease_rice_healthy_rice" to "Arroz sano",
        "disease_rice_rice_blast" to "Piriculariosis",
        "disease_rice_rice_hispa" to "Hispa del arroz",
        "disease_rice_tungro" to "Tungro",
        "disease_orange_black_spot" to "Mancha negra",
        "disease_orange_canker" to "Cancro",
        "disease_orange_greening_hlb" to "Greening (HLB)",
        "disease_orange_healthy_orange" to "Naranjo sano",
        "disease_orange_scab" to "Roña",
    )

    private val DISEASES_VA = mapOf(
        "disease_tomato_bacterial_spot" to "Taca bacteriana",
        "disease_tomato_early_blight" to "Peste primerenca",
        "disease_tomato_healthy" to "Sana",
        "disease_tomato_late_blight" to "Peste tardana",
        "disease_tomato_leaf_mold" to "Floridura de la fulla",
        "disease_tomato_mosaic_virus" to "Virus del mosaic",
        "disease_tomato_septoria_leaf_spot" to "Taca foliar per Septoria",
        "disease_tomato_spider_mites" to "Aranya roja",
        "disease_tomato_target_spot" to "Taca diana",
        "disease_tomato_yellow_leaf_curl_virus" to "Virus de l'enrotllament groc",
        "disease_rice_bacterial_leaf_blight" to "Peste bacteriana de la fulla",
        "disease_rice_bacterial_leaf_streak" to "Estria bacteriana de la fulla",
        "disease_rice_bacterial_panicle_blight" to "Peste bacteriana de la panícula",
        "disease_rice_brown_spot" to "Taca marró",
        "disease_rice_dead_heart" to "Cor mort",
        "disease_rice_downy_mildew" to "Míldiu vellutat",
        "disease_rice_healthy_rice" to "Arròs sa",
        "disease_rice_rice_blast" to "Piriculariosi",
        "disease_rice_rice_hispa" to "Hispa de l'arròs",
        "disease_rice_tungro" to "Tungro",
        "disease_orange_black_spot" to "Taca negra",
        "disease_orange_canker" to "Càncer",
        "disease_orange_greening_hlb" to "Greening (HLB)",
        "disease_orange_healthy_orange" to "Taronger sa",
        "disease_orange_scab" to "Ronya",
    )

    private val STRINGS: Map<String, Map<String, String>> = mapOf(
        "en" to (UI_EN + DISEASES_EN),
        "es" to (UI_ES + DISEASES_ES),
        "va" to (UI_VA + DISEASES_VA),
    )
}
