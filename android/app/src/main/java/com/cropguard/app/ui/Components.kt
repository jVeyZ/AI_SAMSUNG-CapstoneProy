package com.cropguard.app.ui

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
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
import androidx.compose.material.icons.filled.Settings
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
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.cropguard.app.ui.theme.CropGuardTheme
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
fun SettingsButton(onClick: () -> Unit, lang: String) {
    IconButton(onClick = onClick) {
        Icon(Icons.Default.Settings, contentDescription = L.t("settings", lang))
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

/** Parse **bold** markers into an AnnotatedString with bold spans. */
fun parseBasicMarkdown(text: String): AnnotatedString = buildAnnotatedString {
    val regex = Regex("""\*\*(.+?)\*\*""")
    var lastEnd = 0
    for (match in regex.findAll(text)) {
        append(text.substring(lastEnd, match.range.first))
        withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append(match.groupValues[1]) }
        lastEnd = match.range.last + 1
    }
    append(text.substring(lastEnd))
}

@Composable
private fun TypingIndicator() {
    val transition = rememberInfiniteTransition(label = "typing")
    val alpha by transition.animateFloat(
        initialValue = 0.3f, targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(400, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "dot",
    )
    Row(horizontalArrangement = Arrangement.spacedBy(4.dp), verticalAlignment = Alignment.CenterVertically) {
        listOf(0f, 0.15f, 0.3f).forEach { delay ->
            val dotAlpha by transition.animateFloat(
                initialValue = 0.3f, targetValue = 1f,
                animationSpec = infiniteRepeatable(
                    animation = tween(400, delayMillis = (delay * 1000).toInt(), easing = LinearEasing),
                    repeatMode = RepeatMode.Reverse,
                ),
                label = "dot$delay",
            )
            Text(
                "\u2022",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.alpha(dotAlpha),
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
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
                    if (msg.isLoading) {
                        TypingIndicator()
                    } else {
                        val answerText = msg.answer ?: msg.note ?: L.t("ai_unavailable", lang)
                        Text(
                            parseBasicMarkdown(answerText),
                            style = MaterialTheme.typography.bodyMedium,
                        )
                    }
                }
            }
        }
    }
}

// ---- Previews ---------------------------------------------------------------

@Preview(showBackground = true, name = "Treatment Card")
@Composable
private fun PreviewTreatmentCard() {
    CropGuardTheme {
        TreatmentCard(
            title = "Treatment Advice",
            explanation = "Bacterial Leaf Blight is caused by Xanthomonas oryzae. It causes water-soaked lesions that turn yellow and then dry.",
            sections = listOf(
                "Symptoms" to listOf("Water-soaked lesions on leaves", "Yellowing from leaf tips", "Bacterial ooze on leaf surface"),
                "Treatment" to listOf("Remove infected plants", "Apply copper-based bactericide", "Use resistant varieties"),
                "Prevention" to listOf("Use certified seeds", "Avoid overhead irrigation", "Maintain field sanitation"),
            ),
        )
    }
}

@Preview(showBackground = true, name = "Chat Bubble")
@Composable
private fun PreviewChatBubble() {
    CropGuardTheme {
        ChatBubble(
            msg = ChatMessage(
                question = "Can I use copper spray on this?",
                answer = "Yes, **copper-based bactericides** are effective. Apply every **7-10 days** during wet conditions.",
                note = null,
            ),
            lang = "en",
        )
    }
}

@Preview(showBackground = true, name = "Chat Bubble (fallback)")
@Composable
private fun PreviewChatBubbleFallback() {
    CropGuardTheme {
        ChatBubble(
            msg = ChatMessage(
                question = "How do I prevent this?",
                answer = null,
                note = "AI unavailable — showing stored recommendations.",
            ),
            lang = "en",
        )
    }
}
