# SGD Currency Widget (Android)

A native Android **home-screen / lock-screen widget** that shows the live value
of a foreign currency in **Singapore Dollars (SGD)**, plus a full converter
screen for entering any amount.

- Pick the foreign currency once — it is **persisted** (SharedPreferences) and
  survives redraws, app restarts and reboots.
- Live rates from the free, key-less [Frankfurter](https://frankfurter.dev)
  API (`https://api.frankfurter.dev/v1/latest`, European Central Bank reference
  rates). The fetch follows HTTP redirects, so it keeps working if the endpoint
  moves. The last rate is cached so the widget still shows a value when offline.
- Tap the widget body to change the currency; tap **⟳** to refresh.

## What's inside

| Piece | File |
|-------|------|
| Widget provider (renders + refreshes) | `app/.../CurrencyWidgetProvider.kt` |
| Currency picker / widget config | `app/.../WidgetConfigActivity.kt` |
| Full converter screen | `app/.../MainActivity.kt` |
| Rate fetching + offline cache | `app/.../RatesRepository.kt` |
| Persistence | `app/.../Prefs.kt` |
| Supported currencies | `app/.../Currencies.kt` |
| Widget layout | `app/src/main/res/layout/widget_currency.xml` |

## Build & install

This is a standard Android Studio project.

1. Open the `android-currency-widget` folder in **Android Studio** (Giraffe or
   newer). It will sync Gradle and download the wrapper automatically.
2. Or from the command line (with the Android SDK installed and
   `local.properties` pointing at it):
   ```bash
   ./gradlew assembleDebug          # build the APK
   ./gradlew installDebug           # install on a connected device/emulator
   ```
   > If `gradlew` is missing, run `gradle wrapper` once (or let Android Studio
   > generate it) — the binary wrapper jar is not committed.

## Using it

1. Launch the app once. Tap **Change currency**, choose e.g. 🇺🇸 USD — the
   choice is saved.
2. Add the widget: long-press an empty area of the home screen → **Widgets** →
   **SGD Currency Widget** → drop it. The config screen appears so you can pick
   the currency for that widget too.
3. The widget shows `1 USD = 1.35 SGD` and refreshes every ~30 min (and on tap).

## Lock screen

The widget declares `android:widgetCategory="home_screen|keyguard"`, so it is
eligible for the lock screen where the OS/launcher supports lock-screen widgets:

- **Android 16+ tablets/foldables**: add via lock-screen customization.
- **Older Android (4.2–4.4)**: native keyguard widgets.
- **Other versions/phones**: Google removed system lock-screen widgets between
  Android 5 and 15, so behaviour depends on your OEM skin (e.g. some Samsung /
  Xiaomi builds) or a launcher that re-adds the feature. On those, the widget
  still works fully on the home screen.

This is an OS limitation, not an app one — the widget itself is lock-screen
ready wherever the platform exposes the slot.
