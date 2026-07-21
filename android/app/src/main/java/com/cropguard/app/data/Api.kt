package com.cropguard.app.data

import android.content.Context
import com.cropguard.app.BuildConfig
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.RequestBody
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Path
import retrofit2.http.Query
import java.util.concurrent.TimeUnit

data class PredictResponse(
    val crop: String,
    val disease: String,
    val confidence: Float,
    val probabilities: Map<String, Float>,
)

data class TreatmentResponse(
    val crop: String,
    val disease: String,
    val lang: String,
    val explanation: String,
    val symptoms: List<String>,
    val treatment: List<String>,
    val prevention: List<String>,
)

data class TreatmentFallback(
    val explanation: String,
    val symptoms: List<String>,
    val treatment: List<String>,
    val prevention: List<String>,
)

data class ChatRequest(val crop: String, val disease: String, val question: String, val lang: String)

data class ChatResponse(val answer: String?, val fallback: TreatmentFallback?, val note: String?)

interface CropGuardApi {
    @Multipart
    @POST("predict")
    suspend fun predict(
        @Part file: MultipartBody.Part,
        @Part("crop") crop: RequestBody,
    ): PredictResponse

    @GET("treatment/{crop}/{disease}")
    suspend fun treatment(
        @Path("crop") crop: String,
        @Path("disease") disease: String,
        @Query("lang") lang: String,
    ): TreatmentResponse

    @POST("chat")
    suspend fun chat(@Body req: ChatRequest): ChatResponse
}

object ApiClient {

    private const val PREFS_NAME = "cropguard"
    private const val KEY_SERVER_URL = "server_url"

    @Volatile
    private var _api: CropGuardApi? = null

    val api: CropGuardApi
        get() = _api ?: synchronized(this) {
            _api ?: buildApi(BuildConfig.BASE_URL).also { _api = it }
        }

    private fun buildApi(baseUrl: String): CropGuardApi {
        val normalized = if (baseUrl.endsWith("/")) baseUrl else "$baseUrl/"
        val client = OkHttpClient.Builder()
            .connectTimeout(15, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .build()
        return Retrofit.Builder()
            .baseUrl(normalized)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(CropGuardApi::class.java)
    }

    fun init(context: Context) {
        val saved = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getString(KEY_SERVER_URL, null)
        if (saved != null) {
            _api = buildApi(saved)
        }
    }

    fun setServerUrl(context: Context, url: String) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit().putString(KEY_SERVER_URL, url).apply()
        synchronized(this) {
            _api = buildApi(url)
        }
    }

    fun getServerUrl(context: Context): String {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getString(KEY_SERVER_URL, BuildConfig.BASE_URL) ?: BuildConfig.BASE_URL
    }
}
