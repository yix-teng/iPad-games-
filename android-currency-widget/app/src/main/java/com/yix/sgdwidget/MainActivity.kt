package com.yix.sgdwidget

import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.text.DecimalFormat
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Full-screen converter. Mirrors the widget but adds a free amount field, so
 * you can convert any quantity of the persisted foreign currency into SGD.
 */
class MainActivity : AppCompatActivity() {

    private val io = Executors.newSingleThreadExecutor()
    private var rate: Double? = null
    private var code: String = "USD"

    private lateinit var amount: EditText
    private lateinit var result: TextView
    private lateinit var rateLine: TextView
    private lateinit var status: TextView
    private lateinit var changeBtn: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        amount = findViewById(R.id.amount)
        result = findViewById(R.id.result)
        rateLine = findViewById(R.id.rateLine)
        status = findViewById(R.id.status)
        changeBtn = findViewById(R.id.changeBtn)

        amount.addTextChangedListener(object : TextWatcher {
            override fun afterTextChanged(s: Editable?) = recompute()
            override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
        })

        changeBtn.setOnClickListener {
            startActivity(android.content.Intent(this, WidgetConfigActivity::class.java))
        }

        findViewById<View>(R.id.refreshBtn).setOnClickListener { loadRate() }
    }

    override fun onResume() {
        super.onResume()
        // Re-read the persisted currency every time we return (it may have been
        // changed on the picker screen) and refresh the rate.
        code = Prefs.getGlobalCurrency(this)
        loadRate()
    }

    private fun loadRate() {
        val cur = Currencies.byCode(code)
        findViewById<TextView>(R.id.curLine).text = "${cur.flag}  ${cur.code} — ${cur.name}"
        rate = Prefs.cachedRate(this, code)
        status.text = getString(R.string.updating)
        recompute()

        io.execute {
            val r = RatesRepository.fetchToSgd(this, code)
            runOnUiThread {
                if (r != null) {
                    rate = r.rate
                    val stamp = SimpleDateFormat("HH:mm", Locale.getDefault())
                        .format(Date(r.timeMillis))
                    status.text = if (r.fromNetwork) {
                        getString(R.string.updated_at, stamp)
                    } else {
                        getString(R.string.updated_offline, stamp)
                    }
                } else {
                    status.text = getString(R.string.no_rate)
                }
                recompute()
            }
        }
    }

    private fun recompute() {
        val r = rate
        if (r == null) {
            rateLine.text = "1 $code → SGD"
            result.text = "S$ —"
            return
        }
        rateLine.text = "1 $code = ${money(r)} SGD"
        val amt = amount.text.toString().toDoubleOrNull() ?: 0.0
        result.text = "S$" + money(amt * r)
    }

    private fun money(v: Double): String = DecimalFormat("#,##0.00##").format(v)

    override fun onDestroy() {
        io.shutdownNow()
        super.onDestroy()
    }
}
