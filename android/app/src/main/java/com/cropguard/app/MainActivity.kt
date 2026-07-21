package com.cropguard.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.cropguard.app.data.ApiClient
import com.cropguard.app.ui.CaptureScreen
import com.cropguard.app.ui.ResultScreen
import com.cropguard.app.ui.SettingsScreen
import com.cropguard.app.ui.theme.CropGuardTheme
import com.cropguard.app.vm.CropViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ApiClient.init(this)
        setContent {
            CropGuardTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background,
                ) {
                    val nav = rememberNavController()
                    val vm: CropViewModel = viewModel()
                    NavHost(navController = nav, startDestination = "capture") {
                        composable("capture") {
                            CaptureScreen(
                                vm,
                                onResult = { nav.navigate("result") },
                                onSettings = { nav.navigate("settings") },
                            )
                        }
                        composable("result") {
                            ResultScreen(vm, onBack = { nav.popBackStack() }, onSettings = { nav.navigate("settings") })
                        }
                        composable("settings") {
                            SettingsScreen(vm, onBack = { nav.popBackStack() })
                        }
                    }
                }
            }
        }
    }
}
