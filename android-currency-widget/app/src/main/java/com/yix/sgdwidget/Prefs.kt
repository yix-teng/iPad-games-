package com.yix.sgdwidget

import android.content.Context

/**
 * Tiny SharedPreferences wrapper.
 *
 * The selected foreign currency is persisted here so it survives widget
 * redraws, app restarts and device reboots. It is stored per app-widget id
 * (each placed widget can track a different currency), with a global default
 * used by the full-screen converter and any widget that has no value yet.
 */
object Prefs {
    private const val FILE = "sgd_widget_prefs"
    private const val KEY_GLOBAL_CURRENCY = "global_currency"
    private const val KEY_WIDGET_PREFIX = "widget_currency_"
    private const val KEY_INPUT_PREFIX = "widget_input_"
    private const val KEY_RATE_PREFIX = "rate_"
    private const val KEY_RATE_TIME_PREFIX = "rate_time_"
    private const val DEFAULT_CURRENCY = "USD"

    private fun prefs(ctx: Context) =
        ctx.getSharedPreferences(FILE, Context.MODE_PRIVATE)

    /** The currency a specific widget should display. */
    fun getWidgetCurrency(ctx: Context, widgetId: Int): String =
        prefs(ctx).getString(KEY_WIDGET_PREFIX + widgetId, null)
            ?: getGlobalCurrency(ctx)

    fun setWidgetCurrency(ctx: Context, widgetId: Int, code: String) {
        prefs(ctx).edit()
            .putString(KEY_WIDGET_PREFIX + widgetId, code)
            .putString(KEY_GLOBAL_CURRENCY, code)
            .apply()
    }

    fun clearWidget(ctx: Context, widgetId: Int) {
        prefs(ctx).edit()
            .remove(KEY_WIDGET_PREFIX + widgetId)
            .remove(KEY_INPUT_PREFIX + widgetId)
            .apply()
    }

    /** The amount the user has keyed into a specific widget's keypad. */
    fun getWidgetInput(ctx: Context, widgetId: Int): String =
        prefs(ctx).getString(KEY_INPUT_PREFIX + widgetId, "") ?: ""

    fun setWidgetInput(ctx: Context, widgetId: Int, value: String) {
        prefs(ctx).edit().putString(KEY_INPUT_PREFIX + widgetId, value).apply()
    }

    /** App-wide default, used by the converter screen. */
    fun getGlobalCurrency(ctx: Context): String =
        prefs(ctx).getString(KEY_GLOBAL_CURRENCY, DEFAULT_CURRENCY) ?: DEFAULT_CURRENCY

    fun setGlobalCurrency(ctx: Context, code: String) {
        prefs(ctx).edit().putString(KEY_GLOBAL_CURRENCY, code).apply()
    }

    /** Cache the last fetched rate so the widget can show something offline. */
    fun cacheRate(ctx: Context, code: String, rate: Double, epochMillis: Long) {
        prefs(ctx).edit()
            .putString(KEY_RATE_PREFIX + code, rate.toString())
            .putLong(KEY_RATE_TIME_PREFIX + code, epochMillis)
            .apply()
    }

    fun cachedRate(ctx: Context, code: String): Double? =
        prefs(ctx).getString(KEY_RATE_PREFIX + code, null)?.toDoubleOrNull()

    fun cachedRateTime(ctx: Context, code: String): Long =
        prefs(ctx).getLong(KEY_RATE_TIME_PREFIX + code, 0L)
}
