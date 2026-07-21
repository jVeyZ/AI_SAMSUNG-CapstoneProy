package com.cropguard.app.vm

import android.app.Application
import android.content.Context
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.cropguard.app.data.ApiClient
import com.cropguard.app.data.ChatRequest
import com.cropguard.app.data.PredictResponse
import com.cropguard.app.data.TreatmentResponse
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody

data class ChatMessage(val question: String, val answer: String?, val note: String?)

data class UiState(
    val crops: List<String> = listOf("Tomato", "Rice", "Orange"),
    val selectedCrop: String = "Tomato",
    val imageBytes: ByteArray? = null,
    val imageMime: String = "image/jpeg",
    val loading: Boolean = false,
    val errorNetwork: Boolean = false,
    val prediction: PredictResponse? = null,
    val treatment: TreatmentResponse? = null,
    val chat: List<ChatMessage> = emptyList(),
    val lang: String = "en",
)

class CropViewModel(app: Application) : AndroidViewModel(app) {

    private val prefs = app.getSharedPreferences("cropguard", Context.MODE_PRIVATE)

    private val _state = MutableStateFlow(UiState(lang = prefs.getString("lang", "en") ?: "en"))
    val state: StateFlow<UiState> = _state

    val serverUrl: String
        get() = ApiClient.getServerUrl(getApplication())

    fun selectCrop(crop: String) {
        _state.value = _state.value.copy(selectedCrop = crop)
    }

    fun setImage(bytes: ByteArray, mime: String) {
        _state.value = _state.value.copy(imageBytes = bytes, imageMime = mime, errorNetwork = false)
    }

    fun setLanguage(lang: String) {
        prefs.edit().putString("lang", lang).apply()
        _state.value = _state.value.copy(lang = lang)
        // Refresh treatment text in the new language if we already have a diagnosis
        val pred = _state.value.prediction
        if (pred != null) {
            viewModelScope.launch {
                runCatching { ApiClient.api.treatment(pred.crop, pred.disease, lang) }
                    .onSuccess { _state.value = _state.value.copy(treatment = it) }
            }
        }
    }

    fun setServerUrl(url: String) {
        ApiClient.setServerUrl(getApplication(), url)
    }

    fun analyze(onDone: () -> Unit) {
        val s = _state.value
        val bytes = s.imageBytes ?: return
        _state.value = s.copy(loading = true, errorNetwork = false)
        viewModelScope.launch {
            try {
                val filePart = MultipartBody.Part.createFormData(
                    "file", "leaf.jpg", bytes.toRequestBody(s.imageMime.toMediaType()))
                val cropPart = s.selectedCrop.toRequestBody("text/plain".toMediaType())
                val pred = ApiClient.api.predict(filePart, cropPart)
                val treat = runCatching {
                    ApiClient.api.treatment(pred.crop, pred.disease, _state.value.lang)
                }.getOrNull()
                _state.value = _state.value.copy(
                    loading = false, prediction = pred, treatment = treat, chat = emptyList())
                onDone()
            } catch (e: Exception) {
                _state.value = _state.value.copy(loading = false, errorNetwork = true)
            }
        }
    }

    fun ask(question: String) {
        val s = _state.value
        val pred = s.prediction ?: return
        if (question.isBlank()) return
        _state.value = s.copy(loading = true)
        viewModelScope.launch {
            try {
                val resp = ApiClient.api.chat(
                    ChatRequest(pred.crop, pred.disease, question, _state.value.lang))
                _state.value = _state.value.copy(
                    loading = false,
                    chat = _state.value.chat + ChatMessage(question, resp.answer, resp.note))
            } catch (e: Exception) {
                _state.value = _state.value.copy(loading = false, errorNetwork = true)
            }
        }
    }
}
