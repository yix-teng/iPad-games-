package com.yix.sgdwidget

import android.app.PendingIntent
import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.widget.RemoteViews
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Home-screen / lock-screen widget that shows the live value of one unit of a
 * persisted foreign currency in SGD.
 *
 * - Tapping the body opens the picker to change (and persist) the currency.
 * - Tapping the refresh glyph re-fetches the rate.
 */
class CurrencyWidgetProvider : AppWidgetProvider() {

    companion object {
        const val ACTION_REFRESH = "com.yix.sgdwidget.ACTION_REFRESH"
        private val IO = Executors.newCachedThreadPool()

        /** Push a fresh render for [widgetId], fetching the rate in background. */
        fun render(ctx: Context, mgr: AppWidgetManager, widgetId: Int) {
            val code = Prefs.getWidgetCurrency(ctx, widgetId)

            // Immediately show whatever we know (cached) so the widget never
            // looks blank while the network call runs.
            pushViews(ctx, mgr, widgetId, code, loading = true)

            IO.execute {
                val result = RatesRepository.fetchToSgd(ctx, code)
                pushViews(ctx, mgr, widgetId, code, loading = false, result = result)
            }
        }

        private fun pushViews(
            ctx: Context,
            mgr: AppWidgetManager,
            widgetId: Int,
            code: String,
            loading: Boolean,
            result: RatesRepository.Result? = null,
        ) {
            val cur = Currencies.byCode(code)
            val views = RemoteViews(ctx.packageName, R.layout.widget_currency)

            views.setTextViewText(R.id.wFlag, cur.flag)
            views.setTextViewText(R.id.wCode, cur.code)

            val rate = result?.rate ?: Prefs.cachedRate(ctx, code)
            if (rate != null) {
                views.setTextViewText(R.id.wValue, "S$" + fmt(rate))
                views.setTextViewText(R.id.wSub, "1 ${cur.code} = ${fmt(rate)} SGD")
            } else {
                views.setTextViewText(R.id.wValue, if (loading) "…" else "—")
                views.setTextViewText(R.id.wSub, "1 ${cur.code} → SGD")
            }

            val time = result?.timeMillis?.takeIf { it > 0 }
                ?: Prefs.cachedRateTime(ctx, code).takeIf { it > 0 }
            val stamp = if (time != null) {
                "Updated " + SimpleDateFormat("HH:mm", Locale.getDefault()).format(Date(time))
            } else {
                "Tap ⟳ to update"
            }
            val offline = result != null && !result.fromNetwork
            views.setTextViewText(
                R.id.wFooter,
                if (offline) "$stamp (offline)" else stamp
            )

            // Tap body -> change currency (config activity, reused standalone).
            val configIntent = Intent(ctx, WidgetConfigActivity::class.java).apply {
                action = Intent.ACTION_MAIN
                putExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, widgetId)
            }
            views.setOnClickPendingIntent(
                R.id.wBody,
                PendingIntent.getActivity(
                    ctx, widgetId, configIntent,
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
                )
            )

            // Tap refresh -> broadcast back to this provider.
            val refreshIntent = Intent(ctx, CurrencyWidgetProvider::class.java).apply {
                action = ACTION_REFRESH
                putExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, widgetId)
            }
            views.setOnClickPendingIntent(
                R.id.wRefresh,
                PendingIntent.getBroadcast(
                    ctx, widgetId, refreshIntent,
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
                )
            )

            mgr.updateAppWidget(widgetId, views)
        }

        private fun fmt(rate: Double): String {
            // Small rates (e.g. IDR, KRW) need more decimals to be meaningful.
            val pattern = when {
                rate >= 100 -> "#,##0.##"
                rate >= 1 -> "0.0000"
                else -> "0.000000"
            }
            return java.text.DecimalFormat(pattern).format(rate)
        }
    }

    override fun onUpdate(ctx: Context, mgr: AppWidgetManager, widgetIds: IntArray) {
        widgetIds.forEach { render(ctx, mgr, it) }
    }

    override fun onReceive(ctx: Context, intent: Intent) {
        super.onReceive(ctx, intent)
        if (intent.action == ACTION_REFRESH) {
            val mgr = AppWidgetManager.getInstance(ctx)
            val id = intent.getIntExtra(
                AppWidgetManager.EXTRA_APPWIDGET_ID,
                AppWidgetManager.INVALID_APPWIDGET_ID
            )
            if (id != AppWidgetManager.INVALID_APPWIDGET_ID) {
                render(ctx, mgr, id)
            } else {
                // Refresh every placed widget.
                val ids = mgr.getAppWidgetIds(ComponentName(ctx, CurrencyWidgetProvider::class.java))
                ids.forEach { render(ctx, mgr, it) }
            }
        }
    }

    override fun onDeleted(ctx: Context, widgetIds: IntArray) {
        widgetIds.forEach { Prefs.clearWidget(ctx, it) }
    }
}
