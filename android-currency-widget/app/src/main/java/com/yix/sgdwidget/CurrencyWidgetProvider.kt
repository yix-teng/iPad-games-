package com.yix.sgdwidget

import android.app.PendingIntent
import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.widget.RemoteViews
import java.text.DecimalFormat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Home-screen / lock-screen widget: a calculator-style currency converter.
 *
 * The widget carries its own numeric keypad, so you can key an amount of the
 * persisted foreign currency directly on the home screen and see the SGD value
 * update live (using the last fetched rate — no network needed per keypress).
 *
 * - Tapping the header (flag/code) opens the picker to change the currency.
 * - Tapping ⟳ re-fetches the rate from the network.
 * - The keypad keys broadcast back here to edit the amount.
 */
class CurrencyWidgetProvider : AppWidgetProvider() {

    companion object {
        const val ACTION_REFRESH = "com.yix.sgdwidget.ACTION_REFRESH"
        const val ACTION_KEY = "com.yix.sgdwidget.ACTION_KEY"
        const val EXTRA_KEY = "key"
        private const val MAX_LEN = 14
        private val IO = Executors.newCachedThreadPool()

        // viewId -> key token, used to wire up the keypad.
        private val KEYS: List<Pair<Int, String>> = listOf(
            R.id.key0 to "0", R.id.key1 to "1", R.id.key2 to "2",
            R.id.key3 to "3", R.id.key4 to "4", R.id.key5 to "5",
            R.id.key6 to "6", R.id.key7 to "7", R.id.key8 to "8",
            R.id.key9 to "9", R.id.keyDot to ".",
            R.id.keyBack to "back", R.id.keyClear to "clear",
        )

        /** Render using cached data, then fetch a fresh rate in the background. */
        fun render(ctx: Context, mgr: AppWidgetManager, widgetId: Int) {
            pushViews(ctx, mgr, widgetId, loading = true)
            val code = Prefs.getWidgetCurrency(ctx, widgetId)
            IO.execute {
                RatesRepository.fetchToSgd(ctx, code)
                pushViews(ctx, mgr, widgetId, loading = false)
            }
        }

        /** Render from cached rate only — instant, used for keypad edits. */
        fun renderCached(ctx: Context, mgr: AppWidgetManager, widgetId: Int) {
            pushViews(ctx, mgr, widgetId, loading = false)
        }

        /** Apply a keypad token to the current input string. */
        fun applyKey(current: String, key: String): String = when (key) {
            "back" -> if (current.isNotEmpty()) current.dropLast(1) else current
            "clear" -> ""
            "." -> when {
                current.contains(".") -> current
                current.isEmpty() -> "0."
                current.length < MAX_LEN -> "$current."
                else -> current
            }
            else -> when {            // a digit
                current == "0" -> key  // replace a lone leading zero
                current.length >= MAX_LEN -> current
                else -> current + key
            }
        }

        private fun pushViews(
            ctx: Context,
            mgr: AppWidgetManager,
            widgetId: Int,
            loading: Boolean,
        ) {
            val code = Prefs.getWidgetCurrency(ctx, widgetId)
            val cur = Currencies.byCode(code)
            val input = Prefs.getWidgetInput(ctx, widgetId)
            val rate = Prefs.cachedRate(ctx, code)
            val views = RemoteViews(ctx.packageName, R.layout.widget_currency)

            views.setTextViewText(R.id.wFlag, cur.flag)
            views.setTextViewText(R.id.wCode, "${cur.code} → SGD")

            // Display: the typed amount and its SGD value.
            val shown = if (input.isEmpty()) "0" else input
            views.setTextViewText(R.id.wInput, "$shown ${cur.code}")

            if (rate != null) {
                views.setTextViewText(R.id.wSub, "1 ${cur.code} = ${fmtRate(rate)} SGD")
                val amount = input.toDoubleOrNull() ?: 0.0
                views.setTextViewText(R.id.wValue, "S$" + money(amount * rate))
            } else {
                views.setTextViewText(R.id.wSub, "1 ${cur.code} → SGD")
                views.setTextViewText(R.id.wValue, if (loading) "S$ …" else "Tap ⟳ for rate")
            }

            val time = Prefs.cachedRateTime(ctx, code).takeIf { it > 0 }
            views.setTextViewText(
                R.id.wFooter,
                if (time != null)
                    "Updated " + SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date(time))
                else "Tap ⟳ to update"
            )

            wireClicks(ctx, views, widgetId)
            mgr.updateAppWidget(widgetId, views)
        }

        private fun wireClicks(ctx: Context, views: RemoteViews, widgetId: Int) {
            // Header -> change currency.
            val configIntent = Intent(ctx, WidgetConfigActivity::class.java).apply {
                action = Intent.ACTION_MAIN
                putExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, widgetId)
            }
            views.setOnClickPendingIntent(
                R.id.wHeader,
                PendingIntent.getActivity(
                    ctx, widgetId, configIntent,
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
                )
            )

            // Refresh -> re-fetch rate.
            views.setOnClickPendingIntent(
                R.id.wRefresh,
                broadcast(ctx, widgetId, ACTION_REFRESH, null, requestCode = widgetId * 100)
            )

            // Keypad keys.
            KEYS.forEachIndexed { i, (viewId, token) ->
                views.setOnClickPendingIntent(
                    viewId,
                    broadcast(ctx, widgetId, ACTION_KEY, token, requestCode = widgetId * 100 + 1 + i)
                )
            }
        }

        private fun broadcast(
            ctx: Context,
            widgetId: Int,
            action: String,
            key: String?,
            requestCode: Int,
        ): PendingIntent {
            val intent = Intent(ctx, CurrencyWidgetProvider::class.java).apply {
                this.action = action
                putExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, widgetId)
                if (key != null) putExtra(EXTRA_KEY, key)
            }
            return PendingIntent.getBroadcast(
                ctx, requestCode, intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
        }

        private fun fmtRate(rate: Double): String {
            val pattern = when {
                rate >= 100 -> "#,##0.##"
                rate >= 1 -> "0.0000"
                else -> "0.000000"
            }
            return DecimalFormat(pattern).format(rate)
        }

        private fun money(v: Double): String = DecimalFormat("#,##0.00").format(v)
    }

    override fun onUpdate(ctx: Context, mgr: AppWidgetManager, widgetIds: IntArray) {
        widgetIds.forEach { render(ctx, mgr, it) }
    }

    override fun onReceive(ctx: Context, intent: Intent) {
        super.onReceive(ctx, intent)
        val mgr = AppWidgetManager.getInstance(ctx)
        val id = intent.getIntExtra(
            AppWidgetManager.EXTRA_APPWIDGET_ID, AppWidgetManager.INVALID_APPWIDGET_ID
        )
        when (intent.action) {
            ACTION_KEY -> {
                if (id == AppWidgetManager.INVALID_APPWIDGET_ID) return
                val key = intent.getStringExtra(EXTRA_KEY) ?: return
                val next = applyKey(Prefs.getWidgetInput(ctx, id), key)
                Prefs.setWidgetInput(ctx, id, next)
                renderCached(ctx, mgr, id)
            }
            ACTION_REFRESH -> {
                if (id != AppWidgetManager.INVALID_APPWIDGET_ID) {
                    render(ctx, mgr, id)
                } else {
                    mgr.getAppWidgetIds(ComponentName(ctx, CurrencyWidgetProvider::class.java))
                        .forEach { render(ctx, mgr, it) }
                }
            }
        }
    }

    override fun onDeleted(ctx: Context, widgetIds: IntArray) {
        widgetIds.forEach { Prefs.clearWidget(ctx, it) }
    }
}
