package com.cropguard.app.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import coil.compose.AsyncImage
import com.cropguard.app.ui.theme.CropGuardTheme
import com.cropguard.app.vm.CropViewModel
import java.io.File

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CaptureScreen(vm: CropViewModel, onResult: () -> Unit, onSettings: () -> Unit = {}) {
    val state by vm.state.collectAsState()
    val ctx = LocalContext.current
    var cameraUri by remember { mutableStateOf<Uri?>(null) }

    val ALLOWED_IMAGE_TYPES = setOf("image/jpeg", "image/png", "image/bmp", "image/webp")

    fun readUri(uri: Uri) {
        val mime = ctx.contentResolver.getType(uri) ?: "image/jpeg"
        if (mime !in ALLOWED_IMAGE_TYPES) {
            vm.setImageError()
            return
        }
        val bytes = ctx.contentResolver.openInputStream(uri)?.use { it.readBytes() }
        if (bytes != null) vm.setImage(bytes, mime)
    }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let { readUri(it) }
    }
    val camera = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok) cameraUri?.let { readUri(it) }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("CropGuard") },
                actions = {
                    SettingsButton(onClick = onSettings, lang = state.lang)
                    LanguageMenu(vm)
                },
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .padding(padding)
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(L.t("select_crop", state.lang), style = MaterialTheme.typography.titleMedium)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                state.crops.forEach { crop ->
                    FilterChip(
                        selected = state.selectedCrop == crop,
                        onClick = { vm.selectCrop(crop) },
                        label = { Text(L.cropName(crop, state.lang)) },
                    )
                }
            }

            Card(
                shape = RoundedCornerShape(24.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
            ) {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    if (state.imageBytes != null) {
                        AsyncImage(
                            model = state.imageBytes,
                            contentDescription = null,
                            modifier = Modifier.fillMaxSize(),
                            contentScale = ContentScale.Crop,
                        )
                    } else {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Icon(
                                Icons.Default.AddPhotoAlternate,
                                contentDescription = null,
                                modifier = Modifier.size(64.dp),
                                tint = MaterialTheme.colorScheme.outline,
                            )
                            Text(
                                L.t("pick_image", state.lang, state.selectedCrop),
                                color = MaterialTheme.colorScheme.outline,
                            )
                        }
                    }
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(
                    onClick = {
                        picker.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                    },
                    modifier = Modifier.weight(1f),
                ) {
                    Icon(Icons.Default.PhotoLibrary, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text(L.t("choose_gallery", state.lang))
                }
                OutlinedButton(
                    onClick = {
                        val file = File(ctx.cacheDir, "capture_${System.currentTimeMillis()}.jpg")
                        val uri = FileProvider.getUriForFile(ctx, "com.cropguard.app.fileprovider", file)
                        cameraUri = uri
                        camera.launch(uri)
                    },
                    modifier = Modifier.weight(1f),
                ) {
                    Icon(Icons.Default.PhotoCamera, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text(L.t("take_photo", state.lang))
                }
            }

            if (state.errorNetwork) {
                Text(L.t("error_network", state.lang), color = MaterialTheme.colorScheme.error)
            }
            if (state.errorBadImage) {
                Text(L.t("error_bad_image", state.lang), color = MaterialTheme.colorScheme.error)
            }

            Button(
                onClick = { vm.analyze(onResult) },
                enabled = !state.loading && state.imageBytes != null,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                shape = RoundedCornerShape(16.dp),
            ) {
                if (state.loading) {
                    CircularProgressIndicator(modifier = Modifier.size(24.dp), strokeWidth = 2.dp)
                    Spacer(Modifier.width(8.dp))
                    Text(L.t("analyzing", state.lang))
                } else {
                    Icon(Icons.Default.Search, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text(L.t("analyze", state.lang, state.selectedCrop))
                }
            }
        }
    }
}

// ---- Preview -----------------------------------------------------------------

@OptIn(ExperimentalMaterial3Api::class)
@Preview(showBackground = true, name = "Capture Screen")
@Composable
private fun PreviewCaptureScreen() {
    CropGuardTheme {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text("CropGuard") },
                    actions = {
                        SettingsButton(onClick = {}, lang = "en")
                    },
                )
            }
        ) { padding ->
            Column(
                modifier = Modifier
                    .padding(padding)
                    .fillMaxSize()
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                Text("Select crop", style = MaterialTheme.typography.titleMedium)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf("Tomato", "Rice", "Orange").forEach { crop ->
                        FilterChip(
                            selected = crop == "Tomato",
                            onClick = {},
                            label = { Text(L.cropName(crop, "en")) },
                        )
                    }
                }

                Card(
                    shape = RoundedCornerShape(24.dp),
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                ) {
                    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Icon(
                                Icons.Default.AddPhotoAlternate,
                                contentDescription = null,
                                modifier = Modifier.size(64.dp),
                                tint = MaterialTheme.colorScheme.outline,
                            )
                            Text(
                                "Pick or take a photo first",
                                color = MaterialTheme.colorScheme.outline,
                            )
                        }
                    }
                }

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = {}, modifier = Modifier.weight(1f)) {
                        Icon(Icons.Default.PhotoLibrary, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("Gallery")
                    }
                    OutlinedButton(onClick = {}, modifier = Modifier.weight(1f)) {
                        Icon(Icons.Default.PhotoCamera, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("Camera")
                    }
                }

                Button(
                    onClick = {},
                    enabled = false,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(56.dp),
                    shape = RoundedCornerShape(16.dp),
                ) {
                    Icon(Icons.Default.Search, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Analyze")
                }
            }
        }
    }
}
