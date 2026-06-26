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
            val body = httpGet("https://api.frankfurter.dev/v1/latest?from=$code&to=SGD")
            if (body != null) {
                val rates = JSONObject(body).getJSONObject("rates")
                val rate = rates.getDouble("SGD")
                val now = System.currentTimeMillis()
                Prefs.cacheRate(ctx, code, rate, now)
                return Result(code, rate, now, fromNetwork = true)
            }
        } catch (_: Exception) {
            // fall through to cache
        }

        val cached = Prefs.cachedRate(ctx, code) ?: return null
        return Result(code, cached, Prefs.cachedRateTime(ctx, code), fromNetwork = false)
    }

    /**
     * Minimal GET that returns the response body, or null on a non-2xx result.
     *
     * Follows up to a few HTTP redirects manually. Java's HttpURLConnection
     * silently refuses to follow a redirect that switches protocol (e.g.
     * http -> https) or, on some Android builds, that switches host — which is
     * exactly what broke us when the API moved domains. Handling it ourselves
     * keeps the widget working if the endpoint is relocated again.
     */
    private fun httpGet(startUrl: String): String? {
        var current = startUrl
        repeat(5) {
            val conn = (URL(current).openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 8000
                readTimeout = 8000
                instanceFollowRedirects = false
                setRequestProperty("Accept", "application/json")
            }
            try {
                when (conn.responseCode) {
                    in 200..299 ->
                        return conn.inputStream.bufferedReader().use { it.readText() }
                    301, 302, 303, 307, 308 -> {
                        val location = conn.getHeaderField("Location") ?: return null
                        // Resolve relative redirects against the current URL.
                        current = URL(URL(current), location).toString()
                    }
                    else -> return null
                }
            } finally {
                conn.disconnect()
            }
        }
        return null
    }
}
