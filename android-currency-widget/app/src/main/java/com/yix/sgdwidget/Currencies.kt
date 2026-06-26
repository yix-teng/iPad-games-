package com.yix.sgdwidget

/**
 * Foreign currencies the user can convert FROM (target is always SGD).
 * Codes here are the ones the Frankfurter / ECB feed supports. SGD itself is
 * excluded because converting SGD -> SGD is meaningless.
 */
object Currencies {
    data class Currency(val code: String, val name: String, val flag: String)

    val ALL: List<Currency> = listOf(
        Currency("USD", "US Dollar", "🇺🇸"),
        Currency("EUR", "Euro", "🇪🇺"),
        Currency("GBP", "British Pound", "🇬🇧"),
        Currency("JPY", "Japanese Yen", "🇯🇵"),
        Currency("AUD", "Australian Dollar", "🇦🇺"),
        Currency("CNY", "Chinese Yuan", "🇨🇳"),
        Currency("HKD", "Hong Kong Dollar", "🇭🇰"),
        Currency("MYR", "Malaysian Ringgit", "🇲🇾"),
        Currency("THB", "Thai Baht", "🇹🇭"),
        Currency("IDR", "Indonesian Rupiah", "🇮🇩"),
        Currency("INR", "Indian Rupee", "🇮🇳"),
        Currency("KRW", "South Korean Won", "🇰🇷"),
        Currency("PHP", "Philippine Peso", "🇵🇭"),
        Currency("CHF", "Swiss Franc", "🇨🇭"),
        Currency("CAD", "Canadian Dollar", "🇨🇦"),
        Currency("NZD", "New Zealand Dollar", "🇳🇿"),
    )

    fun byCode(code: String): Currency =
        ALL.firstOrNull { it.code == code } ?: ALL.first()
}
