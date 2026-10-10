package com.upscaler.app

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.GenericShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

const val MAX_INPUT = 1000   // longest side of the photo that gets enhanced

object Engine {
    private const val PAD = 8

    private fun to255(x: Float) = (x.coerceIn(0f, 1f) * 255f + 0.5f).toInt()

    // Runs the model tile by tile so the phone never runs out of memory.
    fun run(ctx: Context, src: Bitmap, onProgress: (Float) -> Unit): Bitmap {
        val bytes = ctx.assets.open("model.tflite").use { it.readBytes() }
        val model = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
        model.put(bytes); model.rewind()
        val interp = Interpreter(model, Interpreter.Options().setNumThreads(4))
        try {
            val inT = interp.getInputTensor(0)
            val outT = interp.getOutputTensor(0)
            check(inT.dataType() == DataType.FLOAT32 && outT.dataType() == DataType.FLOAT32) {
                "This model is not float32"
            }
            val ins = inT.shape(); val outs = outT.shape()
            val nchw = ins[1] == 3 && ins[3] != 3
            val tH = if (nchw) ins[2] else ins[1]
            val tW = if (nchw) ins[3] else ins[2]
            val oH = if (nchw) outs[2] else outs[1]
            val oW = if (nchw) outs[3] else outs[2]
            val s = oH / tH

            val w = src.width; val h = src.height
            val px = IntArray(w * h); src.getPixels(px, 0, w, 0, 0, w, h)
            val result = Bitmap.createBitmap(w * s, h * s, Bitmap.Config.ARGB_8888)

            val inBuf = ByteBuffer.allocateDirect(4 * 3 * tH * tW).order(ByteOrder.nativeOrder())
            val outBuf = ByteBuffer.allocateDirect(4 * 3 * oH * oW).order(ByteOrder.nativeOrder())
            val inF = inBuf.asFloatBuffer()
            val coreW = tW - 2 * PAD; val coreH = tH - 2 * PAD
            val tilesX = (w + coreW - 1) / coreW; val tilesY = (h + coreH - 1) / coreH
            val total = tilesX * tilesY
            var done = 0

            for (ty in 0 until tilesY) for (tx in 0 until tilesX) {
                val cx = tx * coreW; val cy = ty * coreH
                for (j in 0 until tH) {
                    val yy = (cy - PAD + j).coerceIn(0, h - 1)
                    for (i in 0 until tW) {
                        val xx = (cx - PAD + i).coerceIn(0, w - 1)
                        val p = px[yy * w + xx]
                        val r = ((p shr 16) and 255) / 255f
                        val g = ((p shr 8) and 255) / 255f
                        val b = (p and 255) / 255f
                        if (nchw) {
                            val k = j * tW + i
                            inF.put(k, r); inF.put(tH * tW + k, g); inF.put(2 * tH * tW + k, b)
                        } else {
                            val k = (j * tW + i) * 3
                            inF.put(k, r); inF.put(k + 1, g); inF.put(k + 2, b)
                        }
                    }
                }
                inBuf.rewind(); outBuf.rewind()
                interp.run(inBuf, outBuf)
                outBuf.rewind()
                val outF = outBuf.asFloatBuffer()

                val cw = minOf(coreW, w - cx); val ch = minOf(coreH, h - cy)
                val rw = cw * s; val rh = ch * s
                val chunk = IntArray(rw * rh)
                for (v in 0 until rh) for (u in 0 until rw) {
                    val oy = PAD * s + v; val ox = PAD * s + u
                    val r: Float; val g: Float; val b: Float
                    if (nchw) {
                        val k = oy * oW + ox
                        r = outF.get(k); g = outF.get(oH * oW + k); b = outF.get(2 * oH * oW + k)
                    } else {
                        val k = (oy * oW + ox) * 3
                        r = outF.get(k); g = outF.get(k + 1); b = outF.get(k + 2)
                    }
                    chunk[v * rw + u] = (255 shl 24) or (to255(r) shl 16) or (to255(g) shl 8) or to255(b)
                }
                result.setPixels(chunk, 0, rw, cx * s, cy * s, rw, rh)
                done++
                onProgress(done / total.toFloat())
            }
            return result
        } finally {
            interp.close()
        }
    }
}

fun loadBitmap(ctx: Context, uri: Uri): Bitmap? = runCatching {
    ImageDecoder.decodeBitmap(ImageDecoder.createSource(ctx.contentResolver, uri)) { dec, info, _ ->
        dec.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
        val big = maxOf(info.size.width, info.size.height)
        if (big > MAX_INPUT) {
            val k = MAX_INPUT.toFloat() / big
            dec.setTargetSize((info.size.width * k).toInt(), (info.size.height * k).toInt())
        }
    }
}.getOrNull()

fun savePhoto(ctx: Context, bmp: Bitmap): Boolean = runCatching {
    val v = ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, "Upscaled_${System.currentTimeMillis()}.jpg")
        put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
        put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Upscaler")
    }
    val uri = ctx.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, v)!!
    ctx.contentResolver.openOutputStream(uri)!!.use { bmp.compress(Bitmap.CompressFormat.JPEG, 95, it) }
    true
}.getOrDefault(false)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                Surface(Modifier.fillMaxSize()) { App() }
            }
        }
    }
}

@Composable
fun App() {
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var orig by remember { mutableStateOf<Bitmap?>(null) }
    var full by remember { mutableStateOf<Bitmap?>(null) }
    var preview by remember { mutableStateOf<Bitmap?>(null) }
    var factor by remember { mutableStateOf(4) }
    var progress by remember { mutableStateOf<Float?>(null) }
    var msg by remember { mutableStateOf("") }
    var frac by remember { mutableStateOf(0.5f) }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) {
            orig = loadBitmap(ctx, uri); full = null; preview = null; msg = ""; frac = 0.5f
        }
    }

    Column(Modifier.fillMaxSize().systemBarsPadding().padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            OutlinedButton(onClick = {
                picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            }) { Text("Open photo") }
            Button(enabled = full != null, onClick = {
                val f = full ?: return@Button
                scope.launch {
                    val ok = withContext(Dispatchers.IO) { savePhoto(ctx, f) }
                    msg = if (ok) "Saved to Pictures/Upscaler" else "Save failed"
                }
            }) { Text("Save") }
        }

        Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
            val o = orig
            if (o == null) {
                Text("Open a photo to start")
            } else {
                val oi = remember(o) { o.asImageBitmap() }
                val pv = preview
                Box(Modifier.aspectRatio(o.width.toFloat() / o.height)) {
                    if (pv != null) {
                        val pi = remember(pv) { pv.asImageBitmap() }
                        Image(pi, null, Modifier.fillMaxSize())
                        Image(
                            oi, null,
                            Modifier.fillMaxSize().clip(GenericShape { size, _ ->
                                addRect(Rect(0f, 0f, size.width * frac, size.height))
                            })
                        )
                    } else {
                        Image(oi, null, Modifier.fillMaxSize())
                    }
                }
            }
        }

        if (preview != null) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Original"); Text("Enhanced")
            }
            Slider(value = frac, onValueChange = { frac = it })
        }

        if (msg.isNotEmpty()) Text(msg)
        progress?.let { p -> LinearProgressIndicator(progress = { p }, modifier = Modifier.fillMaxWidth()) }

        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilterChip(selected = factor == 2, onClick = { factor = 2 }, label = { Text("2x") })
            FilterChip(selected = factor == 4, onClick = { factor = 4 }, label = { Text("4x") })
            Spacer(Modifier.weight(1f))
            Button(enabled = orig != null && progress == null, onClick = {
                val o = orig ?: return@Button
                scope.launch {
                    msg = ""; progress = 0f
                    try {
                        val out = withContext(Dispatchers.Default) {
                            val r = Engine.run(ctx, o) { progress = it }
                            val target = o.width * factor
                            if (r.width != target) {
                                Bitmap.createScaledBitmap(r, target, o.height * factor, true)
                            } else r
                        }
                        full = out
                        val k = minOf(1f, 2048f / maxOf(out.width, out.height))
                        preview = if (k < 1f) {
                            Bitmap.createScaledBitmap(out, (out.width * k).toInt(), (out.height * k).toInt(), true)
                        } else out
                        msg = "Done: ${out.width} x ${out.height}"
                    } catch (t: Throwable) {
                        msg = "Problem: ${t.message}"
                    }
                    progress = null
                }
            }) { Text("Enhance") }
        }
    }
}
