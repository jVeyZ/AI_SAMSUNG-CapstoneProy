package com.cropguard.app.ui

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.ProgressIndicatorDefaults
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
import androidx.compose.ui.unit.dp
import com.cropguard.app.vm.CropViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultScreen(vm: CropViewModel, onBack: () -> Unit) {
    val state by vm.state.collectAsState()
    var question by remember { mutableStateOf("") }
    val pred = state.prediction

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(L.t("diagnosis_result", state.lang)) },
                navigationIcon = {
                    IconButton(onClick = onBack) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = null)
                    }
                },
                actions = { LanguageMenu(vm) },
            )
        }
    ) { padding ->
        if (pred == null) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator()
            }
            return@Scaffold
        }

        LazyColumn(
            modifier = Modifier
                .padding(padding)
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            item {
                Card(shape = RoundedCornerShape(24.dp), modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Text(
                            L.disease(pred.crop, pred.disease, state.lang),
                            style = MaterialTheme.typography.headlineMedium,
                        )
                        Text(
                            pred.crop,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.outline,
                        )
                        val animated by animateFloatAsState(targetValue = pred.confidence, label = "conf")
                        LinearProgressIndicator(
                            progress = { animated },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(8.dp),
                            strokeCap = ProgressIndicatorDefaults.LinearStrokeCap,
                        )
                        Text(
                            "${(pred.confidence * 100).toInt()}% ${L.t("confidence", state.lang)}",
                            style = MaterialTheme.typography.labelLarge,
                        )
                    }
                }
            }

            state.treatment?.let { t ->
                item {
                    TreatmentCard(
                        title = L.t("treatment_advice", state.lang),
                        explanation = t.explanation,
                        sections = listOf(
                            L.t("symptoms", state.lang) to t.symptoms,
                            L.t("treatment", state.lang) to t.treatment,
                            L.t("prevention", state.lang) to t.prevention,
                        ),
                    )
                }
            }

            items(state.chat.size) { i ->
                ChatBubble(state.chat[i], state.lang)
            }

            item {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    OutlinedTextField(
                        value = question,
                        onValueChange = { question = it },
                        placeholder = { Text(L.t("followup_hint", state.lang)) },
                        modifier = Modifier.weight(1f),
                        enabled = !state.loading,
                        shape = RoundedCornerShape(24.dp),
                        maxLines = 3,
                    )
                    IconButton(
                        onClick = {
                            vm.ask(question)
                            question = ""
                        },
                        enabled = !state.loading && question.isNotBlank(),
                    ) {
                        Icon(Icons.AutoMirrored.Filled.Send, contentDescription = L.t("ask_ai", state.lang))
                    }
                }
            }
        }
    }
}
