package com.yix.sgdwidget

import android.app.Activity
import android.appwidget.AppWidgetManager
import android.content.Intent
import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.ListView
import android.widget.TextView

/**
 * Lets the user pick the foreign currency to convert into SGD.
 *
 * Serves two roles:
 *  1. As the widget's configuration activity (launched when a widget is first
 *     placed) — it must return RESULT_OK with the widget id.
 *  2. As a standalone "change currency" screen launched by tapping a placed
 *     widget — it just persists the choice and refreshes that widget.
 *
 * Either way the selected currency is persisted via [Prefs] so it sticks.
 */
class WidgetConfigActivity : Activity() {

    private var widgetId = AppWidgetManager.INVALID_APPWIDGET_ID
    private var isConfigFlow = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_config)

        isConfigFlow = intent?.action == AppWidgetManager.ACTION_APPWIDGET_CONFIGURE
        widgetId = intent?.extras?.getInt(
            AppWidgetManager.EXTRA_APPWIDGET_ID,
            AppWidgetManager.INVALID_APPWIDGET_ID
        ) ?: AppWidgetManager.INVALID_APPWIDGET_ID

        // If launched for configuration, default the result to CANCELED so that
        // backing out removes the half-placed widget.
        if (isConfigFlow) {
            setResult(RESULT_CANCELED, Intent().putExtra(
                AppWidgetManager.EXTRA_APPWIDGET_ID, widgetId))
        }

        val current = if (widgetId != AppWidgetManager.INVALID_APPWIDGET_ID) {
            Prefs.getWidgetCurrency(this, widgetId)
        } else {
            Prefs.getGlobalCurrency(this)
        }
        findViewById<TextView>(R.id.cfgTitle).text =
            getString(R.string.config_title, current)

        val items = Currencies.ALL.map { "${it.flag}  ${it.code} — ${it.name}" }
        val list = findViewById<ListView>(R.id.cfgList)
        list.adapter = ArrayAdapter(this, R.layout.item_currency, R.id.itemText, items)
        list.setOnItemClickListener { _, _, position, _ ->
            choose(Currencies.ALL[position].code)
        }
    }

    private fun choose(code: String) {
        Prefs.setGlobalCurrency(this, code)
        if (widgetId != AppWidgetManager.INVALID_APPWIDGET_ID) {
            Prefs.setWidgetCurrency(this, widgetId, code)
            val mgr = AppWidgetManager.getInstance(this)
            CurrencyWidgetProvider.render(this, mgr, widgetId)
            if (isConfigFlow) {
                setResult(RESULT_OK, Intent().putExtra(
                    AppWidgetManager.EXTRA_APPWIDGET_ID, widgetId))
            }
        }
        finish()
    }
}
