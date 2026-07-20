package com.cropguard.app

import com.cropguard.app.data.ChatRequest
import com.cropguard.app.data.CropGuardApi
import kotlinx.coroutines.test.runTest
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

class ApiTest {

    private lateinit var server: MockWebServer
    private lateinit var api: CropGuardApi

    @Before
    fun setUp() {
        server = MockWebServer()
        server.start()
        api = Retrofit.Builder()
            .baseUrl(server.url("/"))
            .client(OkHttpClient())
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(CropGuardApi::class.java)
    }

    @After
    fun tearDown() {
        server.shutdown()
    }

    @Test
    fun predict_parses_response() = runTest {
        server.enqueue(
            MockResponse().setBody(
                """{"crop":"Rice","disease":"Brown Spot","confidence":0.91,
                   "probabilities":{"Brown Spot":0.91,"Tungro":0.09}}"""
            )
        )
        val part = MultipartBody.Part.createFormData(
            "file", "leaf.png", ByteArray(16).toRequestBody("image/png".toMediaType()))
        val resp = api.predict(part, "Rice".toRequestBody("text/plain".toMediaType()))

        assertEquals("Brown Spot", resp.disease)
        assertEquals(0.91f, resp.confidence, 0.001f)
        assertEquals(2, resp.probabilities.size)

        val recorded = server.takeRequest()
        assertEquals("POST", recorded.method)
        assertTrue(recorded.path!!.startsWith("/predict"))
    }

    @Test
    fun treatment_parses_response() = runTest {
        server.enqueue(
            MockResponse().setBody(
                """{"crop":"Rice","disease":"Brown Spot","lang":"va",
                   "explanation":"Text","symptoms":["a"],"treatment":["b"],"prevention":["c"]}"""
            )
        )
        val resp = api.treatment("Rice", "Brown Spot", "va")

        assertEquals("va", resp.lang)
        assertEquals(listOf("a"), resp.symptoms)
        assertEquals(listOf("b"), resp.treatment)

        val recorded = server.takeRequest()
        assertTrue(recorded.path!!.startsWith("/treatment/Rice/Brown%20Spot"))
        assertTrue(recorded.path!!.contains("lang=va"))
    }

    @Test
    fun chat_parses_ai_unavailable_fallback() = runTest {
        server.enqueue(
            MockResponse().setBody(
                """{"answer":null,"note":"AI unavailable",
                   "fallback":{"explanation":"E","symptoms":["s"],"treatment":["t"],"prevention":["p"]}}"""
            )
        )
        val resp = api.chat(ChatRequest("Tomato", "Late Blight", "Can I use copper?", "en"))

        assertNull(resp.answer)
        assertEquals("AI unavailable", resp.note)
        assertNotNull(resp.fallback)
        assertEquals("E", resp.fallback!!.explanation)
    }
}
