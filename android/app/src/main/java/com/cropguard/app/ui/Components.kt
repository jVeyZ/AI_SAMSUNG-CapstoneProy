package com.cropguard.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Language
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.cropguard.app.vm.ChatMessage
import com.cropguard.app.vm.CropViewModel

@Composable
fun LanguageMenu(vm: CropViewModel) {
    var open by remember { mutableStateOf(false) }
    val current = vm.state.value.lang
    IconButton(onClick = { open = true }) {
        Icon(Icons.Default.Language, contentDescription = L.t("language", current))
    }
    DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
        listOf("en" to "English", "es" to "Español", "va" to "Valencià").forEach { (tag, label) ->
            DropdownMenuItem(
                text = {
                    Text(
                        label + if (tag == current) " ✓" else "",
                        fontWeight = if (tag == current) FontWeight.Bold else FontWeight.Normal,
                    )
                },
                onClick = {
                    vm.setLanguage(tag)
                    open = false
                },
            )
        }
    }
}

@Composable
fun TreatmentCard(
    title: String,
    explanation: String,
    sections: List<Pair<String, List<String>>>,
) {
    Card(
        shape = RoundedCornerShape(24.dp),
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
    ) {
        Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(title, style = MaterialTheme.typography.titleLarge)
            Text(explanation, style = MaterialTheme.typography.bodyMedium)
            sections.forEach { (heading, items) ->
                Spacer(Modifier.height(2.dp))
                Text(heading, style = MaterialTheme.typography.titleSmall)
                items.forEach { item ->
                    Row {
                        Text("• ", style = MaterialTheme.typography.bodyMedium)
                        Text(item, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
        }
    }
}

@Composable
fun ChatBubble(msg: ChatMessage, lang: String) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
            Card(
                shape = RoundedCornerShape(18.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primary),
            ) {
                Text(
                    msg.question,
                    color = MaterialTheme.colorScheme.onPrimary,
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp),
                )
            }
        }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Start) {
            Card(shape = RoundedCornerShape(18.dp)) {
                Column(Modifier.padding(horizontal = 14.dp, vertical = 10.dp)) {
                    Text(
                        msg.answer ?: msg.note ?: L.t("ai_unavailable", lang),
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }
        }
    }
}
