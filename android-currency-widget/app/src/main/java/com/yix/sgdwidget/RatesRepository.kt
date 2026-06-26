package com.yix.sgdwidget

import android.content.Context
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/**
 * Fetches the live exchange rate of one foreign currency into SGD.
 *
 * Uses the Frankfurter API (https://www.frankfurter.app), which is free,
 * key-less and backed by European Central Bank reference rates. The result is
 * cached via [Prefs] so the widget can still show the last known rate while
 * offline.
 */
object RatesRepository {

    data class Result(
        val code: String,
        val rate: Double,        // 1 [code] = rate SGD
        val timeMillis: Long,
        val fromNetwork: Boolean,
    )

    /**
     * Returns the SGD rate for [code]. Tries the network first; on any failure
     * falls back to the cached value if one exists. May return null only when
     * there is neither network nor cache.
     *
     * Must be called off the main thread.
     */
    fun fetchToSgd(ctx: Context, code: String): Result? {
        try {
            val url = URL("https://api.frankfurter.app/latest?from=$code&to=SGD")
            val conn = (url.openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 8000
                readTimeout = 8000
                setRequestProperty("Accept", "application/json")
            }
            try {
                if (conn.responseCode == 200) {
                    val body = conn.inputStream.bufferedReader().use { it.readText() }
                    val rates = JSONObject(body).getJSONObject("rates")
                    val rate = rates.getDouble("SGD")
                    val now = System.currentTimeMillis()
                    Prefs.cacheRate(ctx, code, rate, now)
                    return Result(code, rate, now, fromNetwork = true)
                }
            } finally {
                conn.disconnect()
            }
        } catch (_: Exception) {
            // fall through to cache
        }

        val cached = Prefs.cachedRate(ctx, code) ?: return null
        return Result(code, cached, Prefs.cachedRateTime(ctx, code), fromNetwork = false)
    }
}
